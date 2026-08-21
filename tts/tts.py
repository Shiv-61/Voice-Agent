"""
Text-to-speech via Sarvam AI.

Converts text into WAV audio bytes using the Sarvam TTS API.
Supports English, Hindi, and Gujarati.
"""

import base64

from sarvamai import SarvamAI

import config


class TTS:
    def __init__(self):
        if not config.SARVAM_API_KEY:
            raise RuntimeError(
                "SARVAM_API_KEY is not set. "
                "Export it as an environment variable before running."
            )
        self.client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
        print("[tts] Sarvam TTS client ready.")

    def _normalize_lang(self, lang: str | None) -> str:
        if not lang or lang == "unknown":
            return "en-IN"
        if lang in ("en", "en-IN"):
            return "en-IN"
        if lang in ("hi", "hi-IN"):
            return "hi-IN"
        if lang in ("gu", "gu-IN"):
            return "gu-IN"
        if len(lang) == 2:
            return f"{lang}-IN"
        return lang

    def synthesize(
        self,
        text: str,
        language_code: str = "en-IN",
    ) -> bytes:
        """
        Convert text to audio.

        Parameters
        ----------
        text : str
            The text to speak.
        language_code : str
            BCP-47 code — ``"en-IN"``, ``"hi-IN"``, or ``"gu-IN"``.

        Returns
        -------
        bytes
            WAV audio data (can be sent over WebSocket or played locally).
        """
        if not text.strip():
            return b""

        lang = self._normalize_lang(language_code)

        try:
            response = self.client.text_to_speech.convert(
                model=config.TTS_MODEL,
                text=text.strip(),
                language_code=lang,
                speaker=config.TTS_SPEAKER,
            )

            # Sarvam returns a list of base64-encoded WAV strings
            if response.audios and len(response.audios) > 0:
                return base64.b64decode(response.audios[0])
        except Exception as e:
            print(f"[tts] Synthesis error for '{text[:40]}...': {e}")
        return b""

    def speak_local(self, text: str, language_code: str = "en-IN"):
        """Synthesize and play through local speakers (CLI mode only)."""
        import sounddevice as sd
        import soundfile as sf
        import io

        audio_bytes = self.synthesize(text, language_code)
        if not audio_bytes:
            return

        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sd.play(audio_data, sample_rate)
        sd.wait()