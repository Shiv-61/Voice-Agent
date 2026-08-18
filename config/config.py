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

# STT
STT_MODEL = "saaras:v3"
STT_MODE = "transcribe"              # transcribe | translate | verbatim

# TTS
TTS_MODEL = "bulbul:v3"
TTS_SPEAKER = "anushka"              # see Sarvam docs for available speakers
TTS_SPEED = 1.0

# ---- Language ----
# BCP-47 codes used by Sarvam
SUPPORTED_LANGUAGES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "gu": "gu-IN",
}
DEFAULT_LANGUAGE = "en"               # fallback language code

# ---- LLM (Ollama, local) ----
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:3b"
LLM_MAX_TOKENS = 250                 # more room for tool-call reasoning
LLM_TEMPERATURE = 0.4                # lower for factual accuracy

# ---- Conversation ----
MAX_HISTORY_TURNS = 8

# ---- PostgreSQL ----
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/university_agent",
)

# ---- WebSocket Server ----
WS_HOST = "0.0.0.0"
WS_PORT = 8765

# ---- Audio ----
SAMPLE_RATE = 16000                   # Sarvam expects 16kHz for PCM