"""
All configuration knobs in one place.

Environment variables:
    SARVAM_API_KEY  — Sarvam AI API subscription key (required for STT & TTS)
    DATABASE_URL    — PostgreSQL connection string (default: localhost)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ---- Sarvam AI (STT + TTS) ----
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# ---- STT Configuration (sarvam by default / faster-whisper) ----
STT_PROVIDER = os.getenv("STT_PROVIDER", "sarvam").lower()  # sarvam | faster-whisper
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")       # tiny | base | small
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Sarvam STT settings
STT_MODEL = os.getenv("STT_MODEL", "saaras:v3")
STT_MODE = os.getenv("STT_MODE", "transcribe")              # transcribe | translate | verbatim


# TTS
TTS_MODEL = os.getenv("TTS_MODEL", "bulbul:v3")
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "priya")             # female voices: priya, shreya, neha, pooja
TTS_SPEED = 1.0

# ---- Language ----
# BCP-47 codes used by Sarvam
SUPPORTED_LANGUAGES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
}
DEFAULT_LANGUAGE = "en"               # fallback language code

# ---- LLM Configuration ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()  # openrouter | ollama
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "350"))                # more room for tool-call reasoning
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))               # lower for factual accuracy

# ---- Conversation ----
MAX_HISTORY_TURNS = 8

# ---- PostgreSQL ----
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/university_agent",
)

# ---- Server & Network (Render & Local) ----
WS_HOST = os.getenv("HOST", "0.0.0.0")
WS_PORT = int(os.getenv("PORT", "8765"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# ---- ChromaDB Vector Store ----
CHROMA_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent / "db" / "chroma_db"),
)

# ---- Audio ----
SAMPLE_RATE = 16000                   # Sarvam expects 16kHz for PCM