"""
Text-to-speech via Sarvam AI with resilient Edge-TTS fallback.

Converts text into WAV audio bytes using Sarvam TTS API.
If Sarvam quota is exhausted (402) or offline, automatically falls back
to Microsoft Edge Neural TTS (gu-IN-DhwaniNeural, hi-IN-SwaraNeural, en-IN-NeerjaNeural).
"""

import asyncio
import base64
import concurrent.futures
import io
import soundfile as sf

import config
from utils.text_utils import normalize_lang  # Fix #18: shared util

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False


EDGE_VOICE_MAP = {
    "gu-IN": "gu-IN-DhwaniNeural",
    "gu": "gu-IN-DhwaniNeural",
    "hi-IN": "hi-IN-SwaraNeural",
    "hi": "hi-IN-SwaraNeural",
    "en-IN": "en-IN-NeerjaNeural",
    "en": "en-IN-NeerjaNeural",
}


class TTS:
    def __init__(self):
        self.client = None
        self._sarvam_quota_exceeded = False

        if config.SARVAM_API_KEY:
            try:
                from sarvamai import SarvamAI
                self.client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
                print("[tts] Sarvam TTS client ready.")
            except Exception as e:
                print(f"[tts] Sarvam TTS initialization notice: {e}")
        else:
            print("[tts] SARVAM_API_KEY not set. Using Edge-TTS neural voices.")

    async def _edge_coro(self, text: str, voice: str) -> bytes:
        """Asynchronously synthesizes speech using edge-tts and converts to 16kHz PCM WAV."""
        try:
            comm = edge_tts.Communicate(text, voice)
            data = bytearray()
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    data.extend(chunk["data"])
            if not data:
                return b""

            audio, sr = sf.read(io.BytesIO(data))
            out = io.BytesIO()
            sf.write(out, audio, 16000, format="WAV", subtype="PCM_16")
            return out.getvalue()
        except Exception as e:
            print(f"[tts] Edge-TTS synthesis error: {e}")
            return b""

    def _synthesize_edge(self, text: str, lang: str = "gu-IN") -> bytes:
        """Synchronous wrapper for edge-tts."""
        if not EDGE_TTS_AVAILABLE:
            print("[tts] Edge-TTS is not installed.")
            return b""

        voice = EDGE_VOICE_MAP.get(lang, "gu-IN-DhwaniNeural")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self._edge_coro(text, voice)).result()
        return asyncio.run(self._edge_coro(text, voice))

    def synthesize(
        self,
        text: str,
        language_code: str = "gu-IN",  # Fix #15: default to Gujarati (primary)
    ) -> bytes:
        """
        Convert text to audio.

        Parameters
        ----------
        text : str
            The text to speak.
        language_code : str
            BCP-47 code — ``"gu-IN"``, ``"hi-IN"``, or ``"en-IN"``.

        Returns
        -------
        bytes
            WAV audio data (can be sent over WebSocket or played locally).
        """
        if not text.strip():
            return b""

        # Fix #3: Hard cap at 300 chars to avoid silent Sarvam API failures
        text = text.strip()[:300]
        lang = normalize_lang(language_code, default="gu-IN")

        # 1. Try Sarvam AI if configured and quota is intact
        if self.client and not self._sarvam_quota_exceeded:
            try:
                response = self.client.text_to_speech.convert(
                    model=config.TTS_MODEL,
                    text=text,
                    language_code=lang,
                    speaker=config.TTS_SPEAKER,
                )
                if response.audios and len(response.audios) > 0:
                    return base64.b64decode(response.audios[0])
            except Exception as e:
                err_str = str(e)
                if "402" in err_str or "insufficient_quota" in err_str:
                    print("[tts] Sarvam TTS quota reached (402). Switching automatically to Edge-TTS neural voices.")
                    self._sarvam_quota_exceeded = True
                else:
                    print(f"[tts] Sarvam notice for '{text[:40]}…': {e}")

        # 2. Resilient Edge-TTS fallback
        return self._synthesize_edge(text, lang)

    def speak_local(self, text: str, language_code: str = "gu-IN"):
        """Synthesize and play through local speakers (CLI mode only)."""
        try:
            import sounddevice as sd

            audio_bytes = self.synthesize(text, language_code)
            if not audio_bytes:
                return

            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            sd.play(audio_data, sample_rate)
            sd.wait()
        except Exception as err:
            print(f"[tts] Local audio device error (headless environment?): {err}")