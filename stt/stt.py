"""
Speech-to-text layer supporting:
1. faster-whisper (default, fast local inference on CPU with int8 quantization — zero API key required)
2. Sarvam AI STT (cloud API alternative)
"""

import io
import re
import wave
import numpy as np

import config
from utils.text_utils import normalize_lang as _normalize_lang_shared  # Fix #18: shared util

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class STT:
    _whisper_model = None

    @classmethod
    def get_whisper_model(cls):
        """Lazily initialize and return the cached faster-whisper model."""
        if cls._whisper_model is None:
            if not FASTER_WHISPER_AVAILABLE:
                raise RuntimeError(
                    "faster-whisper is not installed. Install with: uv pip install faster-whisper"
                )
            model_size = getattr(config, "WHISPER_MODEL_SIZE", "base")
            device = getattr(config, "WHISPER_DEVICE", "cpu")
            compute_type = getattr(config, "WHISPER_COMPUTE_TYPE", "int8")
            print(f"[stt] Loading faster-whisper ({model_size}, {device}, {compute_type})...")
            cls._whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            print("[stt] faster-whisper model ready.")
        return cls._whisper_model

    def __init__(self):
        self.provider = getattr(config, "STT_PROVIDER", "faster-whisper")
        self.sarvam_client = None
        self._sarvam_quota_exceeded = False

        if self.provider == "sarvam":
            if not config.SARVAM_API_KEY:
                print("[stt] SARVAM_API_KEY not set. Falling back to faster-whisper.")
                self.provider = "faster-whisper"
            else:
                try:
                    from sarvamai import SarvamAI
                    self.sarvam_client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
                    print("[stt] Sarvam STT client ready.")
                except Exception as e:
                    print(f"[stt] Sarvam initialization failed ({e}), falling back to faster-whisper.")
                    self.provider = "faster-whisper"

        if self.provider == "faster-whisper":
            self.whisper_model = self.get_whisper_model()

    def _normalize_lang(self, lang: str | None) -> str:
        """Delegates to the shared normalize_lang utility (Fix #18)."""
        if not lang or lang == "unknown":
            return "unknown"
        return _normalize_lang_shared(lang, default="unknown")

    def _whisper_lang(self, lang: str | None) -> str | None:
        if not lang or lang == "unknown":
            return None
        code = lang.split("-")[0].lower()
        return code if code in ("en", "hi", "gu", "mr", "ta", "te", "bn", "pa", "ur") else None

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> tuple[str, str]:
        """
        Transcribe audio bytes to text.
        Supports WAV or PCM binary streams.
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return "", "unknown"

        active_provider = "faster-whisper" if (self.provider == "faster-whisper" or self._sarvam_quota_exceeded) else "sarvam"

        if active_provider == "faster-whisper":
            transcript, detected_lang = self._transcribe_whisper(audio_bytes, language_code)
        else:
            transcript, detected_lang = self._transcribe_sarvam(audio_bytes, language_code)

        if transcript.strip():
            try:
                print("\n" + "=" * 60)
                print(f"🎙️  SPEECH TRANSCRIPTION [{detected_lang.upper()}] (Engine: {active_provider}):")
                print(f"👉  \"{transcript.strip()}\"")
                print("=" * 60 + "\n", flush=True)
            except Exception:
                print(f"[STT {active_provider}] Transcribed [{detected_lang.upper()}]: {transcript.strip()[:60]}", flush=True)
        else:
            try:
                print(f"[STT {active_provider}] No words detected in audio snippet", flush=True)
            except Exception:
                pass

        return transcript, detected_lang

    def _transcribe_whisper(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> tuple[str, str]:
        try:
            whisper_model = self.get_whisper_model()
            target_lang = self._whisper_lang(language_code)
            bio = io.BytesIO(audio_bytes)

            # Enable vad_filter with greedy low-latency decoding
            try:
                segments, info = whisper_model.transcribe(
                    bio,
                    language=target_lang,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 250},
                )
            except Exception:
                bio.seek(0)
                segments, info = whisper_model.transcribe(
                    bio,
                    language=target_lang,
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    vad_filter=False,
                )
            transcript = " ".join(s.text.strip() for s in segments if s.text).strip()
            detected_lang = self._normalize_lang(info.language or language_code)
            return transcript, detected_lang
        except Exception as e:
            print(f"[stt] faster-whisper transcription error: {e}")
            if self.sarvam_client and not self._sarvam_quota_exceeded:
                print("[stt] Attempting Sarvam STT fallback...")
                return self._transcribe_sarvam(audio_bytes, language_code)
            return "", "unknown"

    def _transcribe_sarvam(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> tuple[str, str]:
        try:
            kwargs = {
                "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
                "model": config.STT_MODEL,
                "mode": config.STT_MODE,
            }
            lang = self._normalize_lang(language_code)
            if lang and lang != "unknown":
                kwargs["language_code"] = lang

            response = self.sarvam_client.speech_to_text.transcribe(**kwargs)
            transcript = response.transcript.strip() if hasattr(response, "transcript") and response.transcript else ""
            detected_lang = getattr(response, "language_code", None) or lang or "en-IN"
            norm_lang = self._normalize_lang(detected_lang)

            # Prevent Sarvam from hallucinating non-supported Indic languages (e.g. Odia 'or-IN', Kannada 'kn-IN')
            # or non-supported scripts (Odia \u0b00-\u0b7f, Bengali \u0980-\u09ff, etc.)
            # on short English words like "hello", "hi", "yes". The university agent only serves Gujarati, Hindi, English.
            has_unsupported_script = bool(re.search(r"[\u0980-\u0a7f\u0b00-\u0d7f]", transcript))
            if (norm_lang not in ("gu-IN", "hi-IN", "en-IN") or has_unsupported_script) and not language_code:
                try:
                    retry_kwargs = {
                        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
                        "model": config.STT_MODEL,
                        "mode": config.STT_MODE,
                        "language_code": "en-IN",
                    }
                    retry_resp = self.sarvam_client.speech_to_text.transcribe(**retry_kwargs)
                    retry_trans = retry_resp.transcript.strip() if hasattr(retry_resp, "transcript") and retry_resp.transcript else ""
                    if retry_trans and not re.search(r"[\u0980-\u0a7f\u0b00-\u0d7f]", retry_trans):
                        print(f"[stt] Corrected non-target language '{norm_lang}' ('{transcript}') -> en-IN: '{retry_trans}'")
                        return retry_trans, "en-IN"
                except Exception as retry_err:
                    print(f"[stt] en-IN retry notice: {retry_err}")

                try:
                    retry_kwargs = {
                        "file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav"),
                        "model": config.STT_MODEL,
                        "mode": config.STT_MODE,
                        "language_code": "gu-IN",
                    }
                    retry_resp = self.sarvam_client.speech_to_text.transcribe(**retry_kwargs)
                    retry_trans = retry_resp.transcript.strip() if hasattr(retry_resp, "transcript") and retry_resp.transcript else ""
                    if retry_trans and not re.search(r"[\u0980-\u0a7f\u0b00-\u0d7f]", retry_trans):
                        print(f"[stt] Corrected non-target language '{norm_lang}' ('{transcript}') -> gu-IN: '{retry_trans}'")
                        return retry_trans, "gu-IN"
                except Exception as retry_err:
                    print(f"[stt] gu-IN retry notice: {retry_err}")

            return transcript, norm_lang
        except Exception as e:
            err_str = str(e)
            print(f"[stt] Sarvam STT notice: {err_str}")
            if "402" in err_str or "insufficient_quota" in err_str:
                print("[stt] Sarvam STT quota reached (402). Switching automatically to faster-whisper local STT.")
                self._sarvam_quota_exceeded = True
            print("[stt] Attempting faster-whisper fallback...")
            return self._transcribe_whisper(audio_bytes, language_code)

    @staticmethod
    def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert a float32 numpy array to WAV bytes (16-bit PCM)."""
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()