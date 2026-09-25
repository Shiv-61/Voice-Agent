"""
FastAPI Unified Web & WebSocket Gateway Server.
Serves the Glassmorphic Web UI, PDF RAG Upload APIs, DB Explorer APIs, and Real-time Voice WebSocket.
"""

import asyncio
import base64
import json
import os
import re
import traceback
import uuid
from typing import Any
import requests

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from db.database import Database
from llm.llm import LLM, WELCOME_MESSAGE
from rag import RAGStore
from stt import STT
from tts import TTS

from utils import split_ready_sentences, is_hangup_intent, clean_speech_text


# Initialize application
app = FastAPI(
    title="University Voice Agent Gateway",
    description="Multilingual Voice Agent & Document Intelligence Portal",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if "*" not in config.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global shared resources
db = Database()
rag = RAGStore()
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/healthz")
async def health_check():
    """Lightweight zero-dependency health check endpoint for Render monitoring."""
    return {"status": "ok", "service": "university-voice-agent", "version": "1.1.0"}


@app.get("/")
async def get_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"message": "Voice Agent API is active. Static UI not yet deployed."})



# -----------------------------------------------------------------------------
# REST API: System Status & Diagnostics
# -----------------------------------------------------------------------------

@app.get("/api/system/status")
async def get_system_status():
    """Checks the health of Ollama LLM, Sarvam AI API, Database, and Vector Store."""
    # 1. LLM status
    llm_status = "unavailable"
    provider_label = "OpenRouter (Cloud)" if config.LLM_PROVIDER == "openrouter" else "Ollama (Local)"
    active_model = config.LLM_MODEL

    if config.LLM_PROVIDER == "openrouter":
        if config.OPENROUTER_API_KEY and len(config.OPENROUTER_API_KEY) > 10:
            llm_status = "ready"
        else:
            llm_status = "missing API key"
    else:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                if any(config.LLM_MODEL in m for m in models):
                    llm_status = "ready"
                else:
                    llm_status = f"connected (model {config.LLM_MODEL} not loaded)"
        except Exception:
            llm_status = "offline"

    # 2. Sarvam status
    sarvam_configured = bool(config.SARVAM_API_KEY and len(config.SARVAM_API_KEY) > 10)

    # 3. Database status
    db_mode = "SQLite (Local Fallback)" if db.use_sqlite else "PostgreSQL (Production)"

    # 4. RAG store metrics
    doc_count = len(rag.list_documents())
    chunk_count = rag.collection.count()

    return {
        "stt": {
            "provider": "Faster-Whisper (Open-Source Local)" if config.STT_PROVIDER == "faster-whisper" else "Sarvam AI (Cloud)",
            "model": config.WHISPER_MODEL_SIZE if config.STT_PROVIDER == "faster-whisper" else config.STT_MODEL,
            "status": "ready",
        },
        "llm": {
            "status": llm_status,
            "model": active_model,
            "provider": provider_label,
        },
        "sarvam_ai": {
            "configured": sarvam_configured,
            "tts_model": config.TTS_MODEL,
            "speaker": config.TTS_SPEAKER,
        },
        "database": {
            "mode": db_mode,
            "status": "connected",
        },
        "vector_store": {
            "status": "ready",
            "indexed_documents": doc_count,
            "total_chunks": chunk_count,
        },
    }


# -----------------------------------------------------------------------------
# REST API: PDF Document Knowledge Base (RAG)
# -----------------------------------------------------------------------------

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, chunks text, and indexes into ChromaDB."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = rag.ingest_pdf(content, file.filename)
        return {
            "success": True,
            "message": f"Successfully indexed '{file.filename}'",
            "data": result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.get("/api/documents")
async def list_documents():
    """Lists all indexed PDF documents in the vector store."""
    docs = rag.list_documents()
    return {"documents": docs, "total": len(docs), "total_chunks": rag.collection.count()}


@app.get("/api/documents/{doc_id}/chunks")
async def get_document_chunks_endpoint(doc_id: str):
    """Returns all indexed chunks for a specific document."""
    chunks = rag.get_document_chunks(doc_id)
    return {"doc_id": doc_id, "chunks": chunks, "total": len(chunks)}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Deletes a document and its embeddings from the vector store."""
    success = rag.delete_document(doc_id)
    if success:
        return {"success": True, "message": f"Document {doc_id} deleted."}
    raise HTTPException(status_code=500, detail="Failed to delete document.")


class SearchRequest(BaseModel):
    query: str
    n_results: int = 4


@app.post("/api/documents/search")
async def search_documents(req: SearchRequest):
    """Tests semantic search on indexed documents."""
    results = rag.query_documents(req.query, n_results=req.n_results)
    return {"query": req.query, "results": results, "count": len(results)}


# -----------------------------------------------------------------------------
# REST API: Database Explorer & CRUD
# -----------------------------------------------------------------------------

@app.get("/api/data/dashboard")
async def get_dashboard_summary():
    """Returns overview KPI metrics for the student desk dashboard."""
    students = db._execute_query("SELECT COUNT(*) as count FROM students")
    admissions = db._execute_query("SELECT COUNT(*) as count FROM admission_info")
    placements = db._execute_query("SELECT COUNT(*) as count FROM placement_stats")

    student_count = students[0]["count"] if students else 0
    admission_count = admissions[0]["count"] if admissions else 0
    placement_count = placements[0]["count"] if placements else 0

    return {
        "total_students": student_count,
        "total_programs": admission_count,
        "placement_records": placement_count,
        "indexed_documents": len(rag.list_documents()),
        "total_chunks": rag.collection.count(),
    }


@app.get("/api/data/students")
async def get_all_students():
    """Lists students with details, marks, and attendance."""
    query = """
        SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
        FROM students s
        JOIN departments d ON s.department_id = d.department_id
        ORDER BY s.student_id
    """
    students = db._execute_query(query)
    for s in students:
        s_id = s["student_id"]
        s["marks"] = db.get_student_marks(s_id)
        s["attendance"] = db.get_student_attendance(s_id)

    return {"students": students}


class StudentCreateRequest(BaseModel):
    student_id: str
    name: str
    department_id: str
    semester: int
    parent_phone: str = ""
    marks: list[dict] = []
    attendance: list[dict] = []


@app.post("/api/data/students")
async def create_student(req: StudentCreateRequest):
    """Inserts a new student into the active database."""
    success = db.add_student(
        student_id=req.student_id.strip().upper(),
        name=req.name.strip(),
        department_id=req.department_id.strip().upper(),
        semester=req.semester,
        parent_phone=req.parent_phone.strip(),
        marks_list=req.marks,
        attendance_list=req.attendance,
    )
    if success:
        return {"success": True, "message": f"Student {req.name} ({req.student_id}) added successfully."}
    raise HTTPException(status_code=500, detail="Failed to add student to database.")


@app.get("/api/data/departments")
async def get_departments_list():
    """Lists departments."""
    return {"departments": db.get_departments()}


@app.get("/api/data/placements")
async def get_placements():
    """Lists placement stats."""
    return {"placements": db.get_placement_stats()}


@app.get("/api/data/admissions")
async def get_admissions():
    """Lists admission info and eligibility."""
    return {"admissions": db.get_admission_info()}


# -----------------------------------------------------------------------------
# REST API: Call History
# -----------------------------------------------------------------------------

@app.get("/api/calls/history")
async def get_call_history(limit: int = 50):
    """Returns logged voice calls, most recent calls first."""
    calls = db.get_call_history(limit=max(1, min(limit, 200)))
    return {"calls": calls, "total": len(calls)}


# -----------------------------------------------------------------------------
# WebSocket: Real-time Voice & Text Call Gateway
# -----------------------------------------------------------------------------

class WebVoiceSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.language_code = config.DEFAULT_LANGUAGE
        self.active_task = None
        self.call_id = None
        self.caller_number = "Web"
        self._hook_llm_tools()

    def ensure_call_log(self):
        """Opens a call log entry if the session doesn't have one yet."""
        if not self.call_id:
            self.call_id = "CALL-" + uuid.uuid4().hex[:6].upper()
            db.log_call_start(
                self.call_id,
                caller_number=self.caller_number,
                language=self.language_code,
            )
        return self.call_id

    def end_call_log(self):
        """Closes the session's open call log entry, if any."""
        if self.call_id:
            db.log_call_end(self.call_id)
            self.call_id = None

    def _hook_llm_tools(self):
        """Wraps LLM tool execution to broadcast live tool events to the frontend."""
        original_execute = self.llm._execute_tool

        def wrapped_execute(tool_name: str, kwargs: dict) -> str:
            # Broadcast tool execution event to client
            import asyncio
            try:
                preview = f"{tool_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
                asyncio.create_task(self.ws.send_json({
                    "event": "tool_executed",
                    "tool": tool_name,
                    "args": kwargs,
                    "preview": preview,
                }))
            except Exception as err:
                print(f"[web-ws] Error sending tool event: {err}")
            return original_execute(tool_name, kwargs)

        self.llm._execute_tool = wrapped_execute

    async def process_user_query(self, user_text: str, detected_lang: str):
        """Processes a transcribed or typed query through LLM, tools, and TTS streaming."""
        if not user_text.strip():
            return

        self.language_code = detected_lang

        # 1. Send Thinking Event
        await self.ws.send_json({"event": "agent_thinking"})

        # Check call hangup intent via zero-latency evaluator
        is_hangup = is_hangup_intent(user_text)

        # 2. Query LLM & Stream TTS Chunks Asynchronously
        buffer = ""
        full_agent_reply = ""

        try:
            async for piece in self.llm.areply_stream(user_text):
                buffer += piece
                full_agent_reply += piece
                ready_sentences, buffer = split_ready_sentences(buffer)

                for sentence in ready_sentences:
                    if sentence.strip():
                        await self.ws.send_json({
                            "event": "agent_partial_text",
                            "text": sentence.strip(),
                        })
                        try:
                            audio_chunk = await asyncio.to_thread(
                                self.tts.synthesize, sentence.strip(), detected_lang
                            )
                            if audio_chunk:
                                await self.ws.send_bytes(audio_chunk)
                        except Exception as err:
                            print(f"[web-ws] TTS chunk error: {err}")

            # Process buffer remainder
            if buffer.strip():
                await self.ws.send_json({
                    "event": "agent_partial_text",
                    "text": buffer.strip(),
                })
                try:
                    audio_chunk = await asyncio.to_thread(
                        self.tts.synthesize, buffer.strip(), detected_lang
                    )
                    if audio_chunk:
                        await self.ws.send_bytes(audio_chunk)
                except Exception as err:
                    print(f"[web-ws] TTS buffer error: {err}")

            # Signal completion with call_hangup flag
            await self.ws.send_json({
                "event": "agent_done",
                "full_text": full_agent_reply.strip(),
                "language": detected_lang,
                "call_hangup": is_hangup,
            })

            # If hangup condition satisfied, signal call disconnection
            if is_hangup:
                print("📞 [web-ws] Call hangup condition met. Initiating disconnect...")
                await asyncio.sleep(1.0)  # Allow final farewell speech chunk to play
                await self.ws.send_json({
                    "event": "call_ended",
                    "call_hangup": True,
                    "reason": "Caller concluded conversation",
                })
        except asyncio.CancelledError:
            print("🛑 [web-ws] Active query pipeline cancelled due to user barge-in.")
            raise

    async def process_audio_payload(self, audio_bytes: bytes):
        """Transcribes input audio bytes and runs pipeline."""
        if len(audio_bytes) < 200:
            print(f"⚠️ [web-ws] Dropping tiny audio payload: {len(audio_bytes)} bytes")
            return

        # Check audio properties & normalize to 16kHz mono PCM for Sarvam AI
        audio_dur = 0
        audio_rms = 0
        try:
            import io, wave
            import numpy as np
            with io.BytesIO(audio_bytes) as bio, wave.open(bio, "rb") as wf:
                framerate = wf.getframerate()
                channels = wf.getnchannels()
                n_frames = wf.getnframes()
                audio_dur = n_frames / framerate if framerate else 0
                raw_frames = wf.readframes(n_frames)
                pcm = np.frombuffer(raw_frames, dtype=np.int16)
                if len(pcm) > 0:
                    audio_rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
                print(f"🎙️ [web-ws] Received audio: {len(audio_bytes)} bytes, {audio_dur:.2f}s, {framerate}Hz, RMS: {audio_rms:.4f}")

                if framerate != 16000 and framerate > 0:
                    print(f"🎙️ [web-ws] Normalizing audio from {framerate}Hz ({channels}ch) to 16000Hz mono...")
                    if channels > 1:
                        pcm = pcm[::channels]
                    new_len = int(len(pcm) * 16000 / framerate)
                    indices = np.linspace(0, len(pcm) - 1, new_len)
                    resampled = np.interp(indices, np.arange(len(pcm)), pcm).astype(np.int16)
                    out_bio = io.BytesIO()
                    with wave.open(out_bio, "wb") as out_wf:
                        out_wf.setnchannels(1)
                        out_wf.setsampwidth(2)
                        out_wf.setframerate(16000)
                        out_wf.writeframes(resampled.tobytes())
                    audio_bytes = out_bio.getvalue()
        except Exception as e:
            print(f"⚠️ [web-ws] Audio normalization notice: {e}")

        try:
            transcript, detected_lang = await asyncio.to_thread(
                self.stt.transcribe, audio_bytes, self.language_code
            )

        except Exception as e:
            err_str = str(e)
            if "invalid_api_key" in err_str.lower() or "403" in err_str or "forbidden" in err_str.lower():
                user_msg = "⚠️ Speech-to-Text Notice: Sarvam API key is invalid or expired. You can still type queries directly in the chat bar below!"
            else:
                user_msg = f"Speech-to-text error: {err_str}"
            print(f"[web-ws] STT error: {e}")
            await self.ws.send_json({"event": "error", "message": user_msg})
            return

        if not transcript.strip():
            print(f"⚠️ [web-ws] STT ({self.stt.provider}) returned empty transcript for {audio_dur:.2f}s audio (RMS: {audio_rms:.4f})")
            await self.ws.send_json({"event": "empty_transcript", "rms": audio_rms, "duration": audio_dur})
            return

        print(f"🎙️ [web-ws] User [{detected_lang}]: {transcript}")


        await self.ws.send_json({
            "event": "user_transcript",
            "text": transcript,
            "language": detected_lang,
        })

        self.ensure_call_log()
        db.log_call_query(self.call_id, transcript)

        await self.process_user_query(transcript, detected_lang)


@app.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"📞 [web-ws] Client connected from {client_host}")

    session = WebVoiceSession(websocket)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                if session.active_task and not session.active_task.done():
                    session.active_task.cancel()
                session.active_task = asyncio.create_task(session.process_audio_payload(audio_bytes))

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    event = payload.get("event")
                    if event == "start":
                        session.language_code = payload.get("language_code", config.DEFAULT_LANGUAGE)
                        await websocket.send_json({
                            "event": "session_started",
                            "language": session.language_code,
                        })
                        # Immediately greet caller with welcome message and audio
                        await websocket.send_json({
                            "event": "agent_partial_text",
                            "text": WELCOME_MESSAGE,
                        })
                        await websocket.send_json({
                            "event": "agent_done",
                            "full_text": WELCOME_MESSAGE,
                            "language": "hi-IN",
                            "call_hangup": False,
                        })
                        try:
                            welcome_audio = session.tts.synthesize(WELCOME_MESSAGE, language_code="hi-IN")
                            if welcome_audio:
                                await websocket.send_bytes(welcome_audio)
                        except Exception as e:
                            print(f"[web-ws] Notice sending welcome audio: {e}")
                    elif event == "call_started":
                        # Frontend signals mic-on: open a call log entry
                        session.caller_number = payload.get("caller_number", "Web") or "Web"
                        session.language_code = payload.get("language_code", session.language_code)
                        session.end_call_log()
                        session.ensure_call_log()
                        await websocket.send_json({
                            "event": "call_started_ack",
                            "call_id": session.call_id,
                        })
                    elif event == "call_ended":
                        # Frontend signals mic-off: close the call log entry
                        session.end_call_log()
                        await websocket.send_json({"event": "call_ended_ack"})
                    elif event == "text_query":
                        # Dual text chat input support
                        text = payload.get("text", "")
                        lang = payload.get("language_code", session.language_code)
                        await websocket.send_json({
                            "event": "user_transcript",
                            "text": text,
                            "language": lang,
                        })
                        session.ensure_call_log()
                        db.log_call_query(session.call_id, text)
                        if session.active_task and not session.active_task.done():
                            session.active_task.cancel()
                        session.active_task = asyncio.create_task(session.process_user_query(text, lang))

                    elif event == "interrupt":
                        # Client signal to interrupt playback and current processing
                        print(f"🛑 [web-ws] Barge-in interruption from {client_host}")
                        if session.active_task and not session.active_task.done():
                            session.active_task.cancel()
                        await websocket.send_json({"event": "interrupted"})

                    elif event == "ping":
                        await websocket.send_json({"event": "pong"})
                except Exception as e:
                    print(f"[web-ws] JSON message parse error: {e}")

    except WebSocketDisconnect:
        print(f"👋 [web-ws] Client disconnected ({client_host})")
    except Exception as e:
        print(f"⚠️ [web-ws] Connection error: {e}")
    finally:
        session.end_call_log()


# -----------------------------------------------------------------------------
# Telephony Integration: Vobiz Audio Streams (https://vobiz.ai)
# -----------------------------------------------------------------------------

@app.api_route("/api/vobiz/answer", methods=["GET", "POST"])
async def vobiz_answer_endpoint(request: Request):
    """
    Vobiz Answer URL webhook.
    Returns XML instruction telling Vobiz to connect a bidirectional WebSocket stream.
    """
    host = request.headers.get("host", f"localhost:{config.WS_PORT}")
    proto = "wss" if "render.com" in host or request.url.scheme == "https" else "ws"
    ws_url = f"{proto}://{host}/ws/vobiz"
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-l16;rate=16000">
    {ws_url}
  </Stream>
</Response>"""
    return Response(content=xml_content, media_type="application/xml")


@app.websocket("/ws/vobiz")
async def websocket_vobiz_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket endpoint for Vobiz Voice AI Telephony.
    Streams 16kHz linear PCM audio back and forth, supporting:
    - Welcome message on call connection
    - Continuous speech recognition & 2s silence commit
    - Full-duplex barge-in (flushing playback queue via clearAudio)
    - Sarvam STT & TTS (female voice: Priya)
    - Automatic call disconnection on hangup detection
    """
    import asyncio, io, wave, numpy as np
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"📞 [vobiz-ws] Telephony call connected from {client_host}")

    stt = STT()
    llm = LLM()
    tts = TTS()
    stream_id = None
    call_id = None

    audio_chunks: list[bytes] = []
    is_speech_active = False
    last_speech_time = 0.0
    vad_threshold = 0.008
    active_turn_task = None
    is_playing_audio = False

    async def send_vobiz_audio(text: str, lang: str):
        nonlocal is_playing_audio
        if not stream_id or not text.strip():
            return
        try:
            is_playing_audio = True
            wav_bytes = await asyncio.to_thread(tts.synthesize, text.strip(), lang)
            if not wav_bytes:
                return
            # Strip 44-byte WAV header to extract raw Linear16 PCM
            pcm_bytes = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes
            b64_payload = base64.b64encode(pcm_bytes).decode("utf-8")
            msg = {
                "event": "playAudio",
                "streamId": stream_id,
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": 16000,
                    "payload": b64_payload,
                },
            }
            await websocket.send_text(json.dumps(msg))
        except Exception as err:
            print(f"[vobiz-ws] Error sending playAudio: {err}")
        finally:
            is_playing_audio = False

    async def process_caller_audio(raw_pcm: bytes):
        nonlocal active_turn_task
        if len(raw_pcm) < 16000 * 2 * 0.4:
            return
        # Package into 16kHz mono WAV
        out_bio = io.BytesIO()
        with wave.open(out_bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(raw_pcm)
        wav_payload = out_bio.getvalue()

        try:
            transcript, detected_lang = await asyncio.to_thread(stt.transcribe, wav_payload)
            if not transcript.strip():
                return

            print(f"🎙️ [vobiz-ws] Caller [{detected_lang}]: {transcript}")
            is_hangup = is_hangup_intent(transcript)

            buffer = ""
            async for piece in llm.areply_stream(transcript):
                buffer += piece
                ready_sentences, buffer = split_ready_sentences(buffer)
                for sentence in ready_sentences:
                    if sentence.strip():
                        await send_vobiz_audio(sentence.strip(), detected_lang)

            if buffer.strip():
                await send_vobiz_audio(buffer.strip(), detected_lang)

            if is_hangup:
                print("📞 [vobiz-ws] Caller hangup detected ({call_hangup: true}). Hanging up call.")
                await asyncio.sleep(1.0)
                await websocket.send_text(json.dumps({
                    "event": "stop",
                    "streamId": stream_id,
                }))
        except asyncio.CancelledError:
            print("🛑 [vobiz-ws] Turn cancelled by caller interruption.")
            raise
        except Exception as err:
            print(f"[vobiz-ws] Error processing turn: {err}")

    try:
        while True:
            raw_msg = await websocket.receive_text()
            data = json.loads(raw_msg)
            event = data.get("event")

            if event == "start":
                stream_id = data.get("streamId") or data.get("stream_id")
                call_id = data.get("callId") or data.get("call_id")
                print(f"🚀 [vobiz-ws] Phone stream started: streamId={stream_id}, callId={call_id}")
                # Greet caller with opening welcome message over the phone
                asyncio.create_task(send_vobiz_audio(WELCOME_MESSAGE, "hi-IN"))

            elif event == "media":
                media = data.get("media", {})
                b64_chunk = media.get("payload", "")
                if not b64_chunk:
                    continue

                chunk_bytes = base64.b64decode(b64_chunk)
                pcm = np.frombuffer(chunk_bytes, dtype=np.int16)
                if len(pcm) == 0:
                    continue

                rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))

                # Barge-in: if caller interrupts while agent is speaking, clear audio
                if is_playing_audio and rms > vad_threshold * 1.3:
                    print("🛑 [vobiz-ws] Barge-in from caller. Clearing playback queue.")
                    if active_turn_task and not active_turn_task.done():
                        active_turn_task.cancel()
                    is_playing_audio = False
                    await websocket.send_text(json.dumps({
                        "event": "clearAudio",
                        "streamId": stream_id,
                    }))
                    audio_chunks = [chunk_bytes]
                    is_speech_active = True
                    last_speech_time = asyncio.get_event_loop().time()
                    continue

                if rms > vad_threshold:
                    if not is_speech_active:
                        is_speech_active = True
                    audio_chunks.append(chunk_bytes)
                    last_speech_time = asyncio.get_event_loop().time()
                elif is_speech_active:
                    audio_chunks.append(chunk_bytes)
                    # Check 0.5s silence commit (reduced from 2.0s for sub-second turnaround)
                    now = asyncio.get_event_loop().time()
                    if now - last_speech_time >= 0.5:
                        is_speech_active = False
                        total_pcm = b"".join(audio_chunks)
                        audio_chunks = []
                        if active_turn_task and not active_turn_task.done():
                            active_turn_task.cancel()
                        active_turn_task = asyncio.create_task(process_caller_audio(total_pcm))

            elif event == "playedStream":
                is_playing_audio = False

            elif event == "stop":
                print(f"👋 [vobiz-ws] Stream stopped: {stream_id}")
                break

    except WebSocketDisconnect:
        print(f"👋 [vobiz-ws] Telephony client disconnected ({client_host})")
    except Exception as e:
        print(f"⚠️ [vobiz-ws] Telephony stream error: {e}")


def create_app():
    return app
