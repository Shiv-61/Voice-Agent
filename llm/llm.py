"""
LLM layer — talks to a local Ollama server (free, no API cost).

The system prompt is doing most of the "sound human" work. A few
deliberate choices, explained inline below.
"""

import json
import requests

import config


SYSTEM_PROMPT = """\
You are a warm, competent person having a real phone conversation — not a \
chatbot reading out an answer. Follow these rules on every single turn:

1. Talk the way people actually talk on the phone: short sentences, \
contractions (I'm, that's, don't), everyday words. Never use bullet points, \
numbered lists, headers, or markdown — this is audio, not a document.
2. Keep it brief. One to three sentences per turn unless the caller clearly \
asked for detail. Say the one thing that matters most first.
3. Sound present: briefly acknowledge what the caller just said before \
answering ("Got it", "Ah, okay", "Sure, that makes sense") — but vary it, \
don't repeat the same phrase every turn.
4. Ask at most one question at a time, and only when you actually need the \
answer to help them.
5. If you're not sure or need to check something, say so plainly ("I'm not \
totally sure, but I think...") instead of inventing a confident-sounding \
answer.
6. Never say you are an AI/assistant/language model unless directly asked. \
Just be a helpful person on the call.
"""


class LLM:
    def __init__(self):
        self.history = []  # list of {"role": ..., "content": ...}

    def _trim_history(self):
        max_msgs = config.MAX_HISTORY_TURNS * 2  # user+assistant per turn
        if len(self.history) > max_msgs:
            self.history[:] = self.history[-max_msgs:]

    def reply_stream(self, user_text: str):
        """
        Yields response text incrementally (token chunks) from Ollama's
        streaming API, and appends the full turn to history at the end.
        """
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        }

        full_reply = ""
        with requests.post(config.OLLAMA_URL, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    full_reply += piece
                    yield piece
                if chunk.get("done"):
                    break

        self.history.append({"role": "assistant", "content": full_reply})