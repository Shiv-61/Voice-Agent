"""
Conversational Acoustic Bridge & Micro-Filler Manager.
Pre-caches warm, realistic spoken acknowledgment snippets in memory
(English, Hindi, Gujarati). Streams within 200ms of user speech completion
to eliminate caller-perceived latency during complex RAG or DB retrievals.
"""

import os
from tts.tts import TTS

FILLER_TEXTS = {
    "en": "Sure, let me check that for you.",
    "hi": "जी, मैं अभी चेक करके बताती हूँ।",
    "gu": "હા, હું હમણાં જ વિગતો જોઈ લઉં છું.",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "audio_cache")


class FillerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, tts: TTS | None = None):
        if self._initialized:
            return
        self.tts = tts or TTS()
        self.cache: dict[str, bytes] = {}
        os.makedirs(CACHE_DIR, exist_ok=True)
        self._warmup_cache()
        self._initialized = True

    def _warmup_cache(self):
        """Loads cached filler audio from disk or synthesizes once on startup."""
        for lang_prefix, text in FILLER_TEXTS.items():
            lang_code = f"{lang_prefix}-IN" if lang_prefix in ("hi", "gu", "en") else "en-IN"
            cache_file = os.path.join(CACHE_DIR, f"filler_{lang_prefix}.wav")

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        self.cache[lang_prefix] = f.read()
                    continue
                except Exception as e:
                    print(f"[filler] Error reading {cache_file}: {e}")

            # Synthesize once and persist
            try:
                audio = self.tts.synthesize(text, lang_code)
                if audio:
                    self.cache[lang_prefix] = audio
                    with open(cache_file, "wb") as f:
                        f.write(audio)
                    print(f"[filler] Pre-cached {lang_prefix} micro-filler audio ({len(audio)} bytes).")
            except Exception as e:
                print(f"[filler] Warning: failed to pre-cache {lang_prefix} filler: {e}")

    def get_filler(self, language_code: str | None = None) -> bytes | None:
        """Returns pre-cached audio bytes in 0.1ms."""
        code = (language_code or "en").split("-")[0].lower()
        if code in self.cache:
            return self.cache[code]
        return self.cache.get("en")
