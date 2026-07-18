"""
Speech-to-text: push-to-talk recording + faster-whisper transcription.

Push-to-talk (press Enter to start/stop) is the simplest thing that works
for an MVP. It avoids the extra complexity of voice-activity-detection
(VAD) tuning while you're still validating the STT -> LLM -> TTS pipeline
itself. Swap in VAD later for a true hands-free feel.
"""

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

import config


class STT:
    def __init__(self):
        print(f"[stt] loading faster-whisper '{config.STT_MODEL_SIZE}' "
              f"({config.STT_DEVICE}/{config.STT_COMPUTE_TYPE})...")
        self.model = WhisperModel(
            config.STT_MODEL_SIZE,
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )
        print("[stt] ready.")

    def record(self) -> np.ndarray:
        """Record from the default mic between two Enter presses."""
        input("\n🎙️  Press Enter to start speaking...")
        print("🔴 Recording... press Enter again to stop.")

        frames = []

        def callback(indata, frame_count, time_info, status):
            frames.append(indata.copy())

        stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        )
        with stream:
            input()  # blocks until Enter is pressed again

        if not frames:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(frames, axis=0).flatten()
        return audio

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _info = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,        # trims leading/trailing silence, cuts hallucination
            beam_size=1,             # greedy decoding = fastest, fine for short utterances
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text