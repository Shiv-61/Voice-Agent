"""
All the knobs in one place. Tune these before touching the pipeline code.
"""

# ---- STT (faster-whisper) ----
# "small.en" = good accuracy/speed balance on CPU, English-only, ~250MB.
# Bump to "distil-large-v3" if you have a GPU and want higher accuracy.
STT_MODEL_SIZE = "small.en"
STT_DEVICE = "cpu"          # "cuda" if you have an NVIDIA GPU
STT_COMPUTE_TYPE = "int8"   # int8 = fastest on CPU, small quality tradeoff
SAMPLE_RATE = 16000          # Whisper expects 16kHz mono

# ---- LLM (Ollama, local) ----
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:3b"   # swap to "llama3.2:3b" if preferred
LLM_MAX_TOKENS = 120          # keeps replies short -> keeps latency + cost down
LLM_TEMPERATURE = 0.7

# ---- TTS (Kokoro) ----
TTS_VOICE = "af_heart"   # one of Kokoro's built-in voice presets
TTS_SPEED = 1.0

# ---- Conversation ----
MAX_HISTORY_TURNS = 8   # how many past turns to keep in context (cost control)