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
from llm.llm import LLM, WELCOME_MESSAGE
from tts import TTS
from db.database import Database
from rag import RAGStore
from utils.filler_manager import FillerManager
from utils import split_ready_sentences


class VoiceCallSession:
    def __init__(self, db: Database = None, rag: RAGStore = None):
        self.db = db or Database()
        self.rag = rag or RAGStore()
        self.stt = STT()
        self.llm = LLM(db=self.db, rag=self.rag)
        self.tts = TTS()
        self.filler_mgr = FillerManager(tts=self.tts)
        self.language_code = "gu-IN"

    async def handle_audio_stream(self, audio_bytes: bytes, websocket: ServerConnection):
        """
        Process incoming audio payload with real-time streaming:
        1. STT -> Transcribe text in worker thread
        2. LLM -> Stream tokens asynchronously with anticipatory context
        3. TTS -> Synthesize each sentence in worker thread and stream chunk immediately!
        """
        if not audio_bytes or len(audio_bytes) < 100:
            return

        # Acoustic filler bridge for telephony callers (disabled by default)
        if getattr(config, "ENABLE_ACOUSTIC_FILLER", False) and len(audio_bytes) >= 16000:
            filler = self.filler_mgr.get_filler(self.language_code)
            if filler:
                try:
                    await websocket.send(filler)
                    print(f"⚡ [ws] Sent instant acoustic filler ({self.language_code})")
                except Exception as fe:
                    print(f"[ws] Filler notice: {fe}")

        try:
            transcript, detected_lang = await asyncio.to_thread(
                self.stt.transcribe, audio_bytes, None
            )
        except Exception as e:
            print(f"[ws-session] STT error: {e}")
            return

        if not transcript.strip():
            print("[ws-session] (empty transcription)")
            return

        turn_lang, bcp47, _ = self.llm.detect_turn_language(transcript, self.language_code, stt_lang=detected_lang)
        self.language_code = bcp47
        print(f"[ws-session] User [{bcp47}]: {transcript}")

        buffer = ""
        sentence_queue = asyncio.Queue()

        async def llm_producer():
            nonlocal buffer
            try:
                async for piece in self.llm.areply_stream(transcript, stt_lang=detected_lang):
                    buffer += piece
                    ready_sentences, buffer = split_ready_sentences(buffer)

                    for sentence in ready_sentences:
                        s = sentence.strip()
                        if s:
                            synth_task = asyncio.create_task(
                                asyncio.to_thread(self.tts.synthesize, s, bcp47)
                            )
                            await sentence_queue.put(synth_task)

                if buffer.strip():
                    rem = buffer.strip()
                    synth_task = asyncio.create_task(
                        asyncio.to_thread(self.tts.synthesize, rem, bcp47)
                    )
                    await sentence_queue.put(synth_task)
            finally:
                await sentence_queue.put(None)

        async def tts_consumer():
            while True:
                task = await sentence_queue.get()
                if task is None:
                    break
                try:
                    audio_chunk = await task
                    if audio_chunk:
                        await websocket.send(audio_chunk)
                        print(f"🔊 [ws] Streamed audio response chunk ({len(audio_chunk)} bytes)")
                except Exception as err:
                    print(f"[ws] Pipelined TTS chunk error: {err}")

        await asyncio.gather(llm_producer(), tts_consumer())


async def connection_handler(websocket: ServerConnection):
    """Handles an active caller WebSocket connection."""
    client_ip = websocket.remote_address
    print(f"📞 [ws] New call connected from {client_ip}")
    session = VoiceCallSession()

    try:
        async for message in websocket:
            if isinstance(message, bytes):
                print(f"🎙️ [ws] Received audio stream ({len(message)} bytes)")
                await session.handle_audio_stream(message, websocket)
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
