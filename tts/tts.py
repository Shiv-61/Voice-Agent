"""
Text-to-speech via Kokoro-82M — tiny (82M params), Apache-2.0, runs fast
on CPU. We synthesize sentence-by-sentence so playback can start almost
immediately instead of waiting for the whole reply to be generated.
"""

import sounddevice as sd
from kokoro import KPipeline

import config

KOKORO_SAMPLE_RATE = 24000


class TTS:
    def __init__(self):
        print("[tts] loading Kokoro...")
        # lang_code='a' = American English. See Kokoro docs for other langs.
        self.pipeline = KPipeline(lang_code="a")
        print("[tts] ready.")

    def speak(self, text: str):
        if not text.strip():
            return
        generator = self.pipeline(
            text,
            voice=config.TTS_VOICE,
            speed=config.TTS_SPEED,
        )
        for _graphemes, _phonemes, audio in generator:
            sd.play(audio, KOKORO_SAMPLE_RATE)
            sd.wait()