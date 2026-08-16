"""
Speech-to-text via Sarvam AI.

Accepts raw audio bytes (WAV or PCM-16kHz), sends them to the Sarvam
STT API, and returns the transcript along with the detected language.
"""

import io
import wave

import numpy as np
from sarvamai import SarvamAI

import config


class STT:
    def __init__(self):
        if not config.SARVAM_API_KEY:
            raise RuntimeError(
                "SARVAM_API_KEY is not set. "
                "Export it as an environment variable before running."
            )
        self.client = SarvamAI(api_subscription_key=config.SARVAM_API_KEY)
        print("[stt] Sarvam STT client ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> tuple[str, str]:
        """
        Transcribe audio bytes to text.

        Parameters
        ----------
        audio_bytes : bytes
            Raw audio in WAV format (16-bit PCM, 16 kHz, mono).
        language_code : str | None
            BCP-47 language code (e.g. "hi-IN"). Pass ``None`` or
            ``"unknown"`` to let Sarvam auto-detect.

        Returns
        -------
        tuple[str, str]
            ``(transcript, detected_language_code)``
            e.g. ``("What are the placements like?", "en-IN")``
        """
        lang = language_code or "unknown"

        response = self.client.speech_to_text.transcribe(
            file=audio_bytes,
            model=config.STT_MODEL,
            mode=config.STT_MODE,
            language_code=lang,
        )

        transcript = response.transcript.strip()
        detected_lang = response.language_code or lang

        return transcript, detected_lang

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
        """Convert a float32 numpy array to WAV bytes (16-bit PCM)."""
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()