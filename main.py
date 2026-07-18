"""
MVP voice call agent: STT -> LLM -> TTS, all local, all free.

Loop:
  1. Push-to-talk record the caller's voice.
  2. Transcribe with faster-whisper.
  3. Stream a reply from a local LLM (Ollama).
  4. As soon as each sentence of the reply is complete, synthesize and
     play it with Kokoro — so the agent starts talking before it has
     finished "thinking" the whole reply, which is what makes it feel
     like a real conversation instead of a request/response bot.
"""

import re

from stt import STT
from llm import LLM
from tts import TTS

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_ready_sentences(buffer: str):
    """
    Given the text accumulated so far, split off any complete sentences
    and return (list_of_complete_sentences, remaining_incomplete_buffer).
    """
    parts = SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    return complete, remainder


def main():
    stt = STT()
    llm = LLM()
    tts = TTS()

    print("\n✅ Voice agent ready. Ctrl+C to quit.\n")

    try:
        while True:
            audio = stt.record()
            print("⏳ Transcribing...")
            user_text = stt.transcribe(audio)

            if not user_text:
                print("(didn't catch anything, try again)")
                continue

            print(f"🧑 You said: {user_text}")
            print("🤖 Agent: ", end="", flush=True)

            buffer = ""
            for piece in llm.reply_stream(user_text):
                print(piece, end="", flush=True)
                buffer += piece
                ready, buffer = split_ready_sentences(buffer)
                for sentence in ready:
                    tts.speak(sentence)

            if buffer.strip():
                tts.speak(buffer)

            print()  # newline after the printed reply

    except KeyboardInterrupt:
        print("\n👋 Call ended.")


if __name__ == "__main__":
    main()