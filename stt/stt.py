"""
Speech-to-text layer supporting:
1. faster-whisper (default, fast local inference on CPU with int8 quantization — zero API key required)
2. Sarvam AI STT (cloud API alternative)
"""

import io
import wave
import numpy as np

import config

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False


class STT:
    _whisper_model = None

    def __init__(self):
        self.provider = getattr(config, "STT_PROVIDER", "faster-whisper")
        self.sarvam_client = None

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
            if not FASTER_WHISPER_AVAILABLE:
                raise RuntimeError(
                    "faster-whisper is not installed. Install with: pip install faster-whisper"
                )
            if STT._whisper_model is None:
                model_size = getattr(config, "WHISPER_MODEL_SIZE", "base")
                device = getattr(config, "WHISPER_DEVICE", "cpu")
                compute_type = getattr(config, "WHISPER_COMPUTE_TYPE", "int8")
                print(f"[stt] Loading faster-whisper ({model_size}, {device}, {compute_type})...")
                STT._whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                )
                print("[stt] faster-whisper model ready.")
            self.whisper_model = STT._whisper_model

    def _normalize_lang(self, lang: str | None) -> str:
        if not lang or lang == "unknown":
            return "unknown"
        if lang in ("en", "en-IN"):
            return "en-IN"
        if lang in ("hi", "hi-IN"):
            return "hi-IN"
        if lang in ("gu", "gu-IN"):
            return "gu-IN"
        if len(lang) == 2:
            return f"{lang}-IN"
        return lang

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

        if self.provider == "faster-whisper":
            transcript, detected_lang = self._transcribe_whisper(audio_bytes, language_code)
        else:
            transcript, detected_lang = self._transcribe_sarvam(audio_bytes, language_code)

        if transcript.strip():
            print("\n" + "=" * 60)
            print(f"🎙️  SPEECH TRANSCRIPTION [{detected_lang.upper()}] (Engine: {self.provider}):")
            print(f"👉  \"{transcript.strip()}\"")
            print("=" * 60 + "\n", flush=True)
        else:
            print(f"⚠️  [STT {self.provider}] No words detected in audio snippet", flush=True)

        return transcript, detected_lang

    def _transcribe_whisper(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> tuple[str, str]:
        try:
            target_lang = self._whisper_lang(language_code)
            bio = io.BytesIO(audio_bytes)

            # vad_filter=False because frontend VAD or push-to-talk already isolates the speech segment
            segments, info = self.whisper_model.transcribe(
                bio,
                language=target_lang,
                beam_size=5,
                vad_filter=False,
            )
            transcript = " ".join(s.text.strip() for s in segments if s.text).strip()
            detected_lang = self._normalize_lang(info.language or language_code)
            return transcript, detected_lang
        except Exception as e:
            print(f"[stt] faster-whisper transcription error: {e}")
            if self.sarvam_client:
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
            return transcript, self._normalize_lang(detected_lang)
        except Exception as e:
            print(f"[stt] Sarvam STT error: {e}")
            if hasattr(self, "whisper_model") and self.whisper_model:
                print("[stt] Attempting faster-whisper fallback...")
                return self._transcribe_whisper(audio_bytes, language_code)
            return "", "unknown"

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