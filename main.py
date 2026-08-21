"""
University Admission & Student Info Voice Call Agent & RAG Portal.

Pipeline:
  1. STT (Sarvam) -> Multi-lingual Speech-to-Text (English, Hindi, Gujarati)
  2. LLM (Qwen 2.5 via Ollama) -> Context-aware, tri-lingual agent with PostgreSQL/SQLite DB & RAG Vector Tool Calls
  3. TTS (Sarvam) -> Natural multi-lingual Speech Synthesis
  4. RAG (ChromaDB + PyPDF) -> Dynamic PDF Knowledge Base ingestion & retrieval

Modes:
  - Unified Web & Voice Gateway (default): `python main.py` -> Runs Web UI & WebSocket Server on http://0.0.0.0:8765
  - Interactive CLI Mode: `python main.py --cli` -> Push-to-talk mic & speaker test in terminal
"""

import argparse
import re
import sys
import uvicorn

import config
from stt import STT
from llm import LLM
from tts import TTS

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def split_ready_sentences(buffer: str):
    parts = SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    return complete, remainder


def run_cli_mode():
    """Interactive local Push-to-talk CLI mode."""
    import sounddevice as sd
    import numpy as np

    stt = STT()
    llm = LLM()
    tts = TTS()

    print("\n🎓 University Admission Voice Agent (CLI Mode) Ready.")
    print("Supports queries in English, Hindi, and Gujarati.")
    print("Press Ctrl+C to exit.\n")

    current_language = config.DEFAULT_LANGUAGE

    try:
        while True:
            input("🎙️ Press Enter to start speaking...")
            print("🔴 Recording... Press Enter again to stop.")

            frames = []

            def callback(indata, frame_count, time_info, status):
                frames.append(indata.copy())

            stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="float32", callback=callback
            )
            with stream:
                input()

            if not frames:
                print("(didn't catch anything, try again)")
                continue

            audio_np = np.concatenate(frames, axis=0).flatten()
            audio_bytes = stt.numpy_to_wav_bytes(audio_np, sample_rate=16000)

            print("⏳ Transcribing with Sarvam...")
            user_text, detected_lang = stt.transcribe(
                audio_bytes, language_code="unknown"
            )

            if not user_text:
                print("(didn't catch anything, try again)")
                continue

            current_language = detected_lang or current_language
            print(f"🧑 User [{current_language}]: {user_text}")
            print("🤖 Agent: ", end="", flush=True)

            buffer = ""
            for piece in llm.reply_stream(user_text):
                print(piece, end="", flush=True)
                buffer += piece
                ready_sentences, buffer = split_ready_sentences(buffer)

                for sentence in ready_sentences:
                    tts.speak_local(sentence, language_code=current_language)

            if buffer.strip():
                tts.speak_local(buffer.strip(), language_code=current_language)

            print("\n")

    except KeyboardInterrupt:
        print("\n👋 Call ended.")


def main():
    parser = argparse.ArgumentParser(
        description="University Admission & Student Info Voice Call Agent"
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in local interactive push-to-talk terminal mode",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=config.WS_HOST,
        help="Host address to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=config.WS_PORT,
        help="Port number to bind (default: 8765)",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli_mode()
    else:
        print("\n=======================================================")
        print(f"🚀 UniVoice AI Portal & Voice Gateway is Active!")
        print(f"👉 Open in your browser: http://localhost:{args.port} (or http://127.0.0.1:{args.port})")
        print("=======================================================\n")
        uvicorn.run("web.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()