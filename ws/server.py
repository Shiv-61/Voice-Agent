"""
WebSocket Media Gateway Server for University Voice Call Agent.

Protocol:
- Caller connects via WebSocket (ws://host:port).
- Client sends audio payload (WAV or raw PCM bytes).
- Server processes: STT (Sarvam) -> LLM (Qwen + DB Tool Call) -> TTS (Sarvam).
- Server streams back audio payload (WAV bytes) for real-time speech response.
"""

import asyncio
import json
import re
import websockets
from websockets.asyncio.server import ServerConnection

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


class VoiceCallSession:
    def __init__(self):
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.language_code = config.DEFAULT_LANGUAGE

    async def handle_audio(self, audio_bytes: bytes) -> list[bytes]:
        """
        Process incoming audio payload:
        1. STT -> Transcribe text + detect language
        2. LLM -> Generate answer (+ DB queries if needed)
        3. TTS -> Convert to spoken audio bytes
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return []

        # 1. Transcribe audio with Sarvam STT
        try:
            transcript, detected_lang = self.stt.transcribe(
                audio_bytes, language_code=self.language_code
            )
        except Exception as e:
            print(f"[ws-session] STT error: {e}")
            return []

        if not transcript.strip():
            print("[ws-session] (empty transcription)")
            return []

        print(f"[ws-session] User [{detected_lang}]: {transcript}")
        self.language_code = detected_lang  # remember language preference for current call

        # 2. Query LLM & Stream TTS Chunks
        audio_responses = []
        buffer = ""

        for piece in self.llm.reply_stream(transcript):
            buffer += piece
            ready_sentences, buffer = split_ready_sentences(buffer)

            for sentence in ready_sentences:
                if sentence.strip():
                    audio_chunk = self.tts.synthesize(
                        sentence.strip(), language_code=detected_lang
                    )
                    if audio_chunk:
                        audio_responses.append(audio_chunk)

        if buffer.strip():
            audio_chunk = self.tts.synthesize(
                buffer.strip(), language_code=detected_lang
            )
            if audio_chunk:
                audio_responses.append(audio_chunk)

        return audio_responses


async def connection_handler(websocket: ServerConnection):
    """Handles an active caller WebSocket connection."""
    client_ip = websocket.remote_address
    print(f"📞 [ws] New call connected from {client_ip}")
    session = VoiceCallSession()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"🎙️ [ws] Received audio stream ({len(message)} bytes)")
                audio_chunks = await session.handle_audio(message)
                
                for chunk in audio_chunks:
                    await websocket.send(chunk)
                    print(f"🔊 [ws] Sent audio response chunk ({len(chunk)} bytes)")
            elif isinstance(message, str):
                # JSON control frame
                try:
                    data = json.loads(message)
                    if data.get("event") == "start":
                        session.language_code = data.get("language_code", config.DEFAULT_LANGUAGE)
                        print(f"⚙️ [ws] Session language set to: {session.language_code}")
                except Exception:
                    pass

    except websockets.exceptions.ConnectionClosed:
        print(f"👋 [ws] Call disconnected from {client_ip}")
    except Exception as err:
        print(f"⚠️ [ws] Unexpected connection error: {err}")


async def start_websocket_server(host: str = config.WS_HOST, port: int = config.WS_PORT):
    """Starts the WebSocket media server."""
    print(f"🚀 Starting Voice Agent WebSocket Gateway on ws://{host}:{port}")
    async with websockets.serve(connection_handler, host, port):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(start_websocket_server())
