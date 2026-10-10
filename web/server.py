"""
FastAPI Unified Web & WebSocket Gateway Server.
Serves the Glassmorphic Web UI, PDF RAG Upload APIs, DB Explorer APIs, and Real-time Voice WebSocket.
"""

import asyncio
import base64
import collections
import json
import os
import re
import time
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
from utils.filler_manager import FillerManager

from utils import (
    split_ready_sentences,
    is_hangup_intent,
    is_agent_farewell,
    clean_speech_text,
    is_prompt_leak,
    is_noise_hallucination,
    is_filler_phrase,
    SILENCE_CHECK_PROMPTS,
    FAREWELL_PROMPTS,
)
from db import get_shared_mongo_store


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

# Shared STT & TTS instances (lazy initialized to save memory and avoid concurrent leak)
_shared_tts = None
_shared_stt = None
_active_ws_connections: dict[str, int] = {}

def get_shared_tts() -> TTS:
    global _shared_tts
    if _shared_tts is None:
        _shared_tts = TTS()
    return _shared_tts

def get_shared_stt() -> STT:
    global _shared_stt
    if _shared_stt is None:
        _shared_stt = STT()
    return _shared_stt


@app.on_event("startup")
async def startup_event():
    print("🚀 Initializing Voice Agent Subsystems...")
    stt_inst = get_shared_stt()
    tts_inst = get_shared_tts()
    print(f"[stt] Active Provider: {stt_inst.provider.upper()} (Sarvam client: {'ready' if stt_inst.sarvam_client else 'not configured'})")
    print(f"[tts] Active Provider: {'SARVAM' if tts_inst.client else 'Edge-TTS fallback'} (Speaker: {config.TTS_SPEAKER})")
    print(f"[llm] Active Provider: {config.LLM_PROVIDER.upper()} | Model: {config.LLM_MODEL} | Temp: {config.LLM_TEMPERATURE}")
    print("✅ All systems ready and listening for calls.")



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


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Fix #23: Serve favicon to eliminate 404 on every browser page load."""
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/x-icon")
    # Return a minimal 1x1 transparent ICO so browsers don't log 404
    ico_bytes = (
        b"\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00"
        b"\x30\x00\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x01\x00"
        b"\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00"
        b"\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x4e\x6c\xe8\x00\x00\x00"
    )
    return Response(content=ico_bytes, media_type="image/x-icon")


# -----------------------------------------------------------------------------
# REST API: System Status & Diagnostics
# -----------------------------------------------------------------------------

@app.get("/api/system/status")
async def get_system_status():
    """Checks the health of Ollama LLM, Sarvam AI API, Database, and Vector Store."""
    # 1. LLM status
    llm_status = "unavailable"
    active_model = config.LLM_MODEL
    if config.LLM_PROVIDER == "sarvam":
        provider_label = "Sarvam AI (Cloud)"
        llm_status = "ready" if config.SARVAM_API_KEY and len(config.SARVAM_API_KEY) > 10 else "missing API key"
    elif config.LLM_PROVIDER == "groq":
        provider_label = "Groq (Cloud)"
        llm_status = "ready" if config.GROQ_API_KEY and len(config.GROQ_API_KEY) > 10 else "missing API key"
    elif config.LLM_PROVIDER == "openrouter":
        provider_label = "OpenRouter (Cloud)"
        llm_status = "ready" if config.OPENROUTER_API_KEY and len(config.OPENROUTER_API_KEY) > 10 else "missing API key"
    else:
        provider_label = "Ollama (Local)"
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


class TextDocumentUploadRequest(BaseModel):
    text: str
    title: str = "custom_knowledge.txt"
    category: str = "general_campus"


@app.post("/api/documents/text")
async def upload_text_document(req: TextDocumentUploadRequest):
    """
    Ingests raw text or markdown content into ChromaDB RAG.
    Enables adding FAQs, university guidelines, fee updates, or policies directly without a PDF.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
    try:
        result = rag.ingest_text(req.text, req.title.strip(), req.category.strip())
        return {
            "success": True,
            "message": f"Successfully indexed text document '{req.title}'",
            "data": result,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process text document: {str(e)}")


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
    if db.use_sqlite:
        query = """
            SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            ORDER BY s.student_id
        """
    else:
        query = """
            SELECT COALESCE(s.student_id, 'STU' || s.id::text) AS student_id,
                   COALESCE(s.name, s.student_name) AS name,
                   COALESCE(d.department_name, s.course, s.department_id, 'General') AS department_name,
                   COALESCE(s.semester, 4) AS semester,
                   COALESCE(s.parent_phone, s.mobile_number, '') AS parent_phone,
                   s.cpi, s.attendance_percentage
            FROM students s
            LEFT JOIN departments d ON s.department_id = d.department_id
            ORDER BY s.id
        """
    students = db._execute_query(query)
    for s in students:
        s_id = str(s.get("student_id") or "")
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
# REST API: Call Transcripts (MongoDB with Postgres fallback)
# -----------------------------------------------------------------------------

@app.get("/api/transcripts/recent")
async def get_recent_transcripts(limit: int = 10):
    """
    Fetches the recent 10 call transcripts from MongoDB (with Postgres fallback).
    """
    mongo_store = get_shared_mongo_store()
    transcripts = mongo_store.get_recent_transcripts(limit=max(1, min(limit, 50)))
    if transcripts:
        return {"transcripts": transcripts, "source": "mongodb", "total": len(transcripts)}

    # Fallback: synthesize from Postgres call_history if MongoDB is empty or offline
    calls = db.get_call_history(limit=limit)
    fallback_transcripts = []
    for c in calls:
        cid = c.get("call_id")
        queries = []
        try:
            queries = json.loads(c.get("queries_json") or "[]")
        except Exception:
            queries = []
        turns = []
        lines = []
        for q in queries:
            role = q.get("role", "user")
            text = q.get("text", "")
            turns.append({"role": role, "text": text})
            lines.append(f"{'User' if role == 'user' else 'Priya'}: {text}")
        fallback_transcripts.append({
            "call_id": cid,
            "caller_number": c.get("caller_number", "Web"),
            "language": c.get("language", "gu-IN"),
            "duration_seconds": c.get("duration_seconds", 0),
            "status": "completed",
            "turn_count": len(turns),
            "turns": turns,
            "full_transcript": "\n".join(lines),
            "created_at": c.get("started_at"),
        })
    return {"transcripts": fallback_transcripts, "source": "postgres_fallback", "total": len(fallback_transcripts)}


@app.get("/api/transcripts/{call_id}")
async def get_call_transcript_detail(call_id: str):
    """Returns full transcript dialogue for a given call ID."""
    mongo_store = get_shared_mongo_store()
    doc = mongo_store.get_transcript_by_call_id(call_id)
    if doc:
        return {"transcript": doc, "source": "mongodb"}

    turns = db.get_call_queries(call_id)
    call_log = None
    try:
        calls = db.get_call_history(limit=100)
        call_log = next((c for c in calls if c.get("call_id") == call_id), None)
    except Exception as e:
        print(f"[transcripts] Notice fetching call history: {e}")

    if turns or call_log:
        formatted = [{"role": t.get("role", "user"), "text": t.get("query_text", "")} for t in (turns or [])]
        caller = call_log.get("caller_number", "Caller") if call_log else "Caller"
        dur = call_log.get("duration", 0) if call_log else 0
        return {
            "transcript": {
                "call_id": call_id,
                "caller_number": caller,
                "duration_seconds": dur,
                "turns": formatted,
                "full_transcript": "\n".join(f"{'User' if t.get('role') == 'user' else 'Priya'}: {t.get('query_text', '')}" for t in formatted) if formatted else "(No spoken turns recorded)",
            },
            "source": "postgres_fallback",
        }
    return JSONResponse(status_code=404, content={"error": "Transcript not found"})


# -----------------------------------------------------------------------------
# WebSocket: Real-time Voice & Text Call Gateway
# -----------------------------------------------------------------------------

class WebVoiceSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.send_lock = asyncio.Lock()
        self.pending_synth_tasks: set[asyncio.Task] = set()
        self.stt = get_shared_stt()
        self.llm = LLM(db=db, rag=rag)
        self.tts = get_shared_tts()
        self.filler_mgr = FillerManager(tts=self.tts)
        self.language_code = config.DEFAULT_LANGUAGE
        self.last_stt_lang = config.DEFAULT_LANGUAGE   # Fix #5: track per-turn STT hint
        self.active_task = None
        self.call_id = None
        self.caller_number = "Web"
        self.last_filler_time = 0.0
        self._turn_buffer: list[dict] = []  # Fix #12: buffer turns, bulk-flush at end
        self.is_processing = False
        self.last_user_activity = 0.0
        self.silence_prompt_active = False
        self.inactivity_task = asyncio.create_task(self._inactivity_watchdog())
        self._hook_llm_tools()

    async def _inactivity_watchdog(self):
        """Monitors 7s silence inactivity in active web call sessions."""
        try:
            while True:
                await asyncio.sleep(0.5)
                if not self.call_id or self.is_processing:
                    continue
                now = asyncio.get_running_loop().time()
                if self.last_user_activity == 0.0:
                    self.last_user_activity = now
                    continue

                idle = now - self.last_user_activity
                if not self.silence_prompt_active:
                    if idle >= 7.0:
                        print("⏱️ [web-ws] 7s silence detected. Asking: 'kya aap abhi bhi line par hai'")
                        self.silence_prompt_active = True
                        self.last_user_activity = asyncio.get_running_loop().time()
                        cur_l = getattr(self.llm, "current_lang", "gu")
                        prompt_text = SILENCE_CHECK_PROMPTS.get(cur_l, SILENCE_CHECK_PROMPTS["gu"])
                        bcp47 = self.language_code
                        await self.send_json_safe({
                            "event": "agent_partial_text",
                            "text": prompt_text,
                        })
                        audio = await asyncio.to_thread(self.tts.synthesize, prompt_text, bcp47)
                        if audio:
                            await self.send_bytes_safe(audio)
                        await self.send_json_safe({
                            "event": "agent_done",
                            "full_text": prompt_text,
                            "language": bcp47,
                            "call_hangup": False,
                        })
                else:
                    if idle >= 7.0:
                        print("⏱️ [web-ws] Inactivity timeout: No response after 7s prompt. Hanging up call.")
                        cur_l = getattr(self.llm, "current_lang", "gu")
                        farewell_text = FAREWELL_PROMPTS.get(cur_l, FAREWELL_PROMPTS["gu"])
                        bcp47 = self.language_code
                        audio = await asyncio.to_thread(self.tts.synthesize, farewell_text, bcp47)
                        if audio:
                            await self.send_bytes_safe(audio)
                        await self.send_json_safe({
                            "event": "agent_done",
                            "full_text": farewell_text,
                            "language": bcp47,
                            "call_hangup": True,
                        })
                        await asyncio.sleep(1.5)
                        await self.send_json_safe({
                            "event": "call_ended",
                            "call_hangup": True,
                            "reason": "Inactivity timeout (no response after 7s check)",
                        })
                        self.end_call_log()
                        await fire_hangup_url(call_id=self.call_id, caller_num="Web Portal", reason="inactivity_timeout")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[web-ws] Inactivity watchdog notice: {e}")

    async def send_json_safe(self, data: dict):
        """Thread-safe and concurrency-safe JSON transmission over WebSocket."""
        async with self.send_lock:
            try:
                await self.ws.send_json(data)
            except Exception as e:
                print(f"[web-ws] send_json_safe notice: {e}")

    async def send_bytes_safe(self, data: bytes):
        """Thread-safe and concurrency-safe binary audio transmission over WebSocket."""
        async with self.send_lock:
            try:
                await self.ws.send_bytes(data)
            except Exception as e:
                print(f"[web-ws] send_bytes_safe notice: {e}")

    def cancel_active_pipeline(self):
        """Cancels inflight transcription, LLM, and synthesis worker tasks immediately."""
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
        for t in list(self.pending_synth_tasks):
            if not t.done():
                t.cancel()
        self.pending_synth_tasks.clear()

    def ensure_call_log(self):
        """Opens a call log entry if the session doesn't have one yet."""
        if not self.call_id:
            self.call_id = "CALL-" + uuid.uuid4().hex[:6].upper()
            db.log_call_start(
                self.call_id,
                caller_number=self.caller_number,
                language=self.language_code,
            )
            self.last_user_activity = asyncio.get_running_loop().time()
        return self.call_id

    def end_call_log(self):
        """Closes the session's open call log entry, saves transcript to MongoDB, and runs async post-call CRM extraction."""
        if hasattr(self, "inactivity_task") and self.inactivity_task and not self.inactivity_task.done():
            self.inactivity_task.cancel()
        if self.call_id:
            cid = self.call_id
            # Fix #12: single bulk flush instead of N per-turn writes
            if self._turn_buffer:
                db.flush_queries_bulk(cid, self._turn_buffer)
                self._turn_buffer = []
            db.log_call_end(cid)
            try:
                mongo_store = get_shared_mongo_store()
                turns_formatted = []
                for h in self.llm.history:
                    turns_formatted.append({
                        "role": h.get("role", "user"),
                        "text": h.get("content", ""),
                    })
                mongo_store.save_call_transcript(
                    call_id=cid,
                    caller_number=self.caller_number,
                    turns=turns_formatted,
                    language=self.language_code,
                    status="completed",
                )
            except Exception as me:
                print(f"[web-ws] Mongo transcript save notice: {me}")
            self.call_id = None
            try:
                from intelligence.post_call import analyze_and_record_call
                asyncio.create_task(analyze_and_record_call(cid, self.llm.history.copy(), db))
            except Exception as e:
                print(f"[web-ws] Post-call analyzer launch notice: {e}")

    def _hook_llm_tools(self):
        """Wraps LLM tool execution to broadcast live tool events to the frontend."""
        original_execute = self.llm._execute_tool

        def wrapped_execute(tool_name: str, kwargs: dict) -> str:
            # Broadcast tool execution event to client thread-safely via running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                except Exception:
                    loop = None

            if loop and loop.is_running():
                preview = f"{tool_name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
                coro = self.send_json_safe({
                    "event": "tool_executed",
                    "tool": tool_name,
                    "args": kwargs,
                    "preview": preview,
                })
                try:
                    asyncio.run_coroutine_threadsafe(coro, loop)
                except Exception as err:
                    print(f"[web-ws] Error sending tool event: {err}")
            return original_execute(tool_name, kwargs)

        self.llm._execute_tool = wrapped_execute

    async def process_user_query(self, user_text: str, detected_lang: str):
        """Processes a transcribed or typed query through LLM, tools, and TTS streaming."""
        if not user_text.strip():
            return

        self.is_processing = True
        self.silence_prompt_active = False
        self.last_user_activity = asyncio.get_running_loop().time()

        # Dynamically determine the active turn language for TTS and UI synchronization
        turn_lang, bcp47, _ = self.llm.detect_turn_language(user_text, self.language_code, stt_lang=detected_lang)
        self.language_code = bcp47
        active_lang = bcp47

        # 1. Send Thinking Event
        await self.send_json_safe({"event": "agent_thinking"})

        # Check call hangup intent via zero-latency evaluator
        is_hangup = is_hangup_intent(user_text)

        # 2. Query LLM & Stream TTS Chunks Concurrently (Pipelined Synthesis)
        buffer = ""
        full_agent_reply = ""
        sentence_queue = asyncio.Queue()

        async def llm_producer():
            nonlocal buffer, full_agent_reply
            try:
                async for piece in self.llm.areply_stream(user_text, stt_lang=detected_lang):
                    buffer += piece
                    full_agent_reply += piece
                    ready_sentences, buffer = split_ready_sentences(buffer)

                    for sentence in ready_sentences:
                        s = clean_speech_text(sentence.strip())
                        if s:
                            if is_prompt_leak(s) or is_filler_phrase(s):
                                print(f"🛑 [web-ws] Suppressed prompt leak/filler from TTS: {s}")
                                continue
                            await self.send_json_safe({
                                "event": "agent_partial_text",
                                "text": s,
                            })
                            # Schedule synthesis in worker threadpool immediately using active turn language
                            synth_task = asyncio.create_task(
                                asyncio.to_thread(self.tts.synthesize, s, active_lang)
                            )
                            self.pending_synth_tasks.add(synth_task)
                            synth_task.add_done_callback(self.pending_synth_tasks.discard)
                            await sentence_queue.put(synth_task)

                # Process buffer remainder
                rem = clean_speech_text(buffer.strip())
                if rem:
                    if not (is_prompt_leak(rem) or is_filler_phrase(rem)):
                        await self.send_json_safe({
                            "event": "agent_partial_text",
                            "text": rem,
                        })
                        synth_task = asyncio.create_task(
                            asyncio.to_thread(self.tts.synthesize, rem, active_lang)
                        )
                        self.pending_synth_tasks.add(synth_task)
                        synth_task.add_done_callback(self.pending_synth_tasks.discard)
                        await sentence_queue.put(synth_task)
            finally:
                await sentence_queue.put(None)  # Sentinel

        is_first_chunk = True  # kept for filler logic only
        async def tts_consumer():
            nonlocal is_first_chunk
            while True:
                task = await sentence_queue.get()
                if task is None:
                    break
                try:
                    audio_chunk = await task
                    if audio_chunk:
                        # Fix #6: removed dead is_first_chunk/last_filler_time guard
                        # (ENABLE_ACOUSTIC_FILLER is False so filler is never sent)
                        await self.send_bytes_safe(audio_chunk)
                except Exception as err:
                    print(f"[web-ws] Pipelined TTS streaming error: {err}")

        try:
            await asyncio.gather(llm_producer(), tts_consumer())

            should_hangup = is_hangup or is_agent_farewell(full_agent_reply)

            # Signal completion with call_hangup flag and active turn language
            await self.send_json_safe({
                "event": "agent_done",
                "full_text": full_agent_reply.strip(),
                "language": active_lang,
                "call_hangup": should_hangup,
            })

            # If hangup condition satisfied (by caller intent or agent farewell), disconnect call
            if should_hangup:
                print(f"📞 [web-ws] Farewell delivered ({'agent farewell: Thank you for your time. Have a great day!' if is_agent_farewell(full_agent_reply) else 'caller intent'}). Initiating disconnect...")
                await asyncio.sleep(2.0)  # Allow final farewell speech chunk to finish playing in browser
                await self.send_json_safe({
                    "event": "call_ended",
                    "call_hangup": True,
                    "reason": "Agent concluded conversation with farewell",
                })
                self.end_call_log()
                await fire_hangup_url(call_id=self.call_id, caller_num="Web Portal", reason="caller_hangup_requested")
        except asyncio.CancelledError:
            print("🛑 [web-ws] Active query pipeline cancelled due to user barge-in.")
            for t in list(self.pending_synth_tasks):
                if not t.done():
                    t.cancel()
            self.pending_synth_tasks.clear()
            raise
        finally:
            self.is_processing = False
            self.last_user_activity = asyncio.get_running_loop().time()

    async def process_audio_payload(self, audio_bytes: bytes):
        """Transcribes input audio bytes and runs pipeline."""
        if len(audio_bytes) < 200:
            print(f"⚠️ [web-ws] Dropping tiny audio payload: {len(audio_bytes)} bytes")
            return

        # Check audio properties & normalize to 16kHz mono PCM for Sarvam AI
        audio_dur = 0
        audio_rms = 0
        try:
            import io, wave, math
            import numpy as np
            try:
                with io.BytesIO(audio_bytes) as bio, wave.open(bio, "rb") as wf:
                    framerate = wf.getframerate()
                    channels = wf.getnchannels()
                    n_frames = wf.getnframes()
                    audio_dur = n_frames / framerate if framerate else 0
                    raw_frames = wf.readframes(n_frames)
                    pcm = np.frombuffer(raw_frames, dtype=np.int16)
            except (wave.Error, Exception):
                # Fallback: treat as raw 16kHz mono PCM
                framerate = 16000
                channels = 1
                pcm = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_dur = len(pcm) / 16000.0

            if len(pcm) > 0:
                audio_rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))
            print(f"🎙️ [web-ws] Received audio: {len(audio_bytes)} bytes, {audio_dur:.2f}s, {framerate}Hz, RMS: {audio_rms:.4f}")

            # Server-side acoustic noise gate to reject ambient room hum, breathing, and mic thuds
            if audio_dur < 0.30 or audio_rms < 0.010:
                print(f"🔇 [web-ws] Rejected ambient noise snippet: {audio_dur:.2f}s, RMS: {audio_rms:.4f}")
                await self.send_json_safe({
                    "event": "empty_transcript",
                    "rms": audio_rms,
                    "duration": audio_dur,
                    "had_filler": False,
                })
                return

            if framerate != 16000 and framerate > 0:
                print(f"🎙️ [web-ws] Normalizing audio from {framerate}Hz ({channels}ch) to 16000Hz mono with polyphase anti-aliasing...")
                if channels > 1:
                    pcm = pcm[::channels]
                try:
                    from scipy.signal import resample_poly
                    gcd = math.gcd(16000, framerate)
                    up = 16000 // gcd
                    down = framerate // gcd
                    resampled = resample_poly(pcm, up, down).astype(np.int16)
                except Exception as re_err:
                    print(f"⚠️ [web-ws] Scipy resample fallback notice: {re_err}")
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

        filler_sent = False
        if getattr(config, "ENABLE_ACOUSTIC_FILLER", False) and audio_dur >= 0.8 and audio_rms >= 0.024:
            filler_audio = self.filler_mgr.get_filler(self.language_code)
            if filler_audio:
                try:
                    await self.send_json_safe({
                        "event": "agent_filler",
                        "language": self.language_code,
                    })
                    await self.send_bytes_safe(filler_audio)
                    filler_sent = True
                    self.last_filler_time = time.time()
                    print(f"⚡ [web-ws] Streamed instant acoustic filler ({self.language_code}) within 150ms.")
                except Exception as fe:
                    print(f"[web-ws] Notice sending filler audio: {fe}")

        # Allow STT engine (Sarvam saaras / faster-whisper) to dynamically detect the spoken language on every turn
        try:
            transcript, detected_lang = await asyncio.to_thread(
                self.stt.transcribe, audio_bytes, None
            )

        except Exception as e:
            from utils import get_error_message  # Fix #22
            err_str = str(e)
            if "invalid_api_key" in err_str.lower() or "403" in err_str or "forbidden" in err_str.lower():
                user_msg = get_error_message("stt_error", self.last_stt_lang)
            else:
                user_msg = get_error_message("stt_error", self.last_stt_lang)
            print(f"[web-ws] STT error: {e}")
            await self.send_json_safe({"event": "error", "message": user_msg})
            return

        if not transcript.strip() or is_noise_hallucination(transcript, audio_dur, audio_rms):
            print(f"🔇 [web-ws] Filtered noise/silence artifact: \"{transcript.strip()}\" ({audio_dur:.2f}s, RMS: {audio_rms:.4f})")
            await self.send_json_safe({
                "event": "empty_transcript",
                "rms": audio_rms,
                "duration": audio_dur,
                "had_filler": filler_sent,
            })
            if filler_sent:
                # Politely follow up so caller is never stranded after filler
                clarification = {
                    "hi-IN": "माफ़ कीजिए, मुझे आपकी आवाज़ साफ़ नहीं आई। क्या आप दोबारा कह सकते हैं?",
                    "gu-IN": "માફ કરશો, મને તમારો અવાજ સ્પષ્ટ ન સંભળાયો. કૃપા કરીને ફરીથી કહેશો?",
                    "en-IN": "I'm sorry, I couldn't hear that clearly. Could you please repeat your question?",
                }.get(self.language_code, "I'm sorry, I couldn't hear that clearly. Could you please repeat your question?")
                await self.send_json_safe({
                    "event": "agent_partial_text",
                    "text": clarification,
                })
                await self.send_json_safe({
                    "event": "agent_done",
                    "full_text": clarification,
                    "language": self.language_code,
                    "call_hangup": False,
                })
                try:
                    followup_audio = await asyncio.to_thread(self.tts.synthesize, clarification, self.language_code)
                    if followup_audio:
                        await self.send_bytes_safe(followup_audio)
                except Exception as ce:
                    print(f"[web-ws] Clarification audio notice: {ce}")
            return

        print(f"🎙️ [web-ws] User [{detected_lang}]: {transcript}")

        await self.send_json_safe({
            "event": "user_transcript",
            "text": transcript,
            "language": detected_lang,
        })

        self.ensure_call_log()
        # Fix #12: buffer to turn list (bulk-flushed at end_call_log)
        import datetime as _dt
        self._turn_buffer.append({
            "text": transcript,
            "lang": detected_lang,
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "role": "user",
        })
        # Fix #5: update STT hint so next audio chunk is primed with caller's detected language
        self.last_stt_lang = detected_lang or self.last_stt_lang

        await self.process_user_query(transcript, detected_lang)


@app.websocket("/ws/call")
async def websocket_call_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    if _active_ws_connections.get(client_host, 0) >= 8:
        print(f"⚠️ [web-ws] Connection rejected: Rate limit reached for host {client_host}")
        await websocket.close(code=1008)
        return
    _active_ws_connections[client_host] = _active_ws_connections.get(client_host, 0) + 1
    print(f"📞 [web-ws] Client connected from {client_host} (active for host: {_active_ws_connections[client_host]})")

    session = WebVoiceSession(websocket)
    try:
        while True:
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                session.cancel_active_pipeline()
                session.active_task = asyncio.create_task(session.process_audio_payload(audio_bytes))

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    event = payload.get("event")
                    if event == "start":
                        session.language_code = payload.get("language_code", config.DEFAULT_LANGUAGE)
                        is_reconnect = payload.get("is_reconnect", False)
                        await session.send_json_safe({
                            "event": "session_started",
                            "language": session.language_code,
                            "reconnected": is_reconnect,
                        })
                    elif event == "call_started":
                        # Frontend signals mic-on: open a call log entry
                        session.caller_number = payload.get("caller_number", "Web") or "Web"
                        session.language_code = payload.get("language_code", session.language_code)
                        session.end_call_log()
                        # Reset LLM conversation history for the fresh call
                        session.llm.history = [{"role": "assistant", "content": WELCOME_MESSAGE}]
                        session.llm.active_student_id = None
                        session.llm.active_student_name = None
                        session.ensure_call_log()
                        await session.send_json_safe({
                            "event": "call_started_ack",
                            "call_id": session.call_id,
                        })
                        # Greet caller with spoken welcome greeting when call is actively started
                        await session.send_json_safe({
                            "event": "agent_partial_text",
                            "text": WELCOME_MESSAGE,
                        })
                        await session.send_json_safe({
                            "event": "agent_done",
                            "full_text": WELCOME_MESSAGE,
                            "language": session.language_code or "gu-IN",
                            "call_hangup": False,
                        })
                        try:
                            cached_greeting_path = os.path.join(static_dir, "welcome_greeting.wav")
                            welcome_audio = None
                            if os.path.exists(cached_greeting_path):
                                with open(cached_greeting_path, "rb") as wf:
                                    welcome_audio = wf.read()
                            if not welcome_audio:
                                welcome_audio = await asyncio.to_thread(
                                    session.tts.synthesize, WELCOME_MESSAGE, language_code=session.language_code or "gu-IN"
                                )
                            if welcome_audio:
                                await session.send_bytes_safe(welcome_audio)
                        except Exception as e:
                            print(f"[web-ws] Notice sending welcome audio on call_started: {e}")
                    elif event == "call_ended":
                        # Frontend signals mic-off: close the call log entry
                        session.end_call_log()
                        await session.send_json_safe({"event": "call_ended_ack"})
                    elif event == "text_query":
                        # Dual text chat input support
                        text = payload.get("text", "")
                        lang = payload.get("language_code", session.language_code)
                        await session.send_json_safe({
                            "event": "user_transcript",
                            "text": text,
                            "language": lang,
                        })
                        session.ensure_call_log()
                        db.log_call_query(session.call_id, text)
                        session.cancel_active_pipeline()
                        session.active_task = asyncio.create_task(session.process_user_query(text, lang))

                    elif event == "interrupt":
                        # Client signal to interrupt playback and current processing
                        session.cancel_active_pipeline()
                        await session.send_json_safe({"event": "interrupted"})

                    elif event == "ping":
                        await session.send_json_safe({"event": "pong"})
                except Exception as e:
                    print(f"[web-ws] JSON message parse error: {e}")

    except WebSocketDisconnect:
        print(f"👋 [web-ws] Client disconnected ({client_host})")
    except Exception as e:
        if "disconnect message has been received" in str(e).lower() or "closed" in str(e).lower():
            print(f"👋 [web-ws] Client session ended cleanly ({client_host})")
        else:
            print(f"⚠️ [web-ws] Connection error: {e}")
    finally:
        new_count = max(0, _active_ws_connections.get(client_host, 1) - 1)
        if new_count == 0:
            _active_ws_connections.pop(client_host, None)  # Fix #14: prevent unbounded dict growth
        else:
            _active_ws_connections[client_host] = new_count
        session.end_call_log()


# -----------------------------------------------------------------------------
# Telephony Integration: Vobiz & Plivo Audio Streams (Primary Answer & Hangup URLs)
# -----------------------------------------------------------------------------

@app.api_route("/api/vobiz/answer", methods=["GET", "POST"])
@app.api_route("/api/telephony/answer", methods=["GET", "POST"])
@app.api_route("/answer", methods=["GET", "POST"])
@app.api_route("/primary-answer", methods=["GET", "POST"])
async def primary_answer_endpoint(request: Request):
    """
    Primary Answer URL webhook.
    Called when an inbound telephone call starts (HTTP POST/GET).
    Must return valid call instructions (XML) instructing the telephony provider (Vobiz/Plivo)
    to connect a bidirectional 16kHz linear PCM WebSocket stream.
    """
    caller_num = "Telephony"
    call_uuid = ""
    try:
        content_type = request.headers.get("content-type", "").lower()
        form_data = {}
        if "form" in content_type:
            form = await request.form()
            form_data = dict(form)
        elif "json" in content_type:
            form_data = await request.json()
        if not form_data:
            form_data = dict(request.query_params)

        call_uuid = form_data.get("CallUUID") or form_data.get("call_uuid") or form_data.get("CallSid") or ""
        caller_num = form_data.get("From") or form_data.get("caller") or form_data.get("from") or "Telephony"
    except Exception as parse_err:
        print(f"[answer-webhook] Notice parsing incoming call payload: {parse_err}")

    if call_uuid:
        try:
            db.log_call_start(call_uuid, caller_number=caller_num, language="gu-IN")
            print(f"📞 [telephony] Logged incoming call start: UUID={call_uuid}, From={caller_num}")
        except Exception as db_err:
            print(f"[answer-webhook] DB log notice: {db_err}")

    host = request.headers.get("host", f"localhost:{config.WS_PORT}")
    fwd_proto = request.headers.get("x-forwarded-proto", "").lower()
    proto = "wss" if fwd_proto == "https" or "render.com" in host or request.url.scheme == "https" else "ws"
    ws_url = f"{proto}://{host}/ws/vobiz"
    status_proto = "https" if proto == "wss" else "http"
    hangup_callback_url = f"{status_proto}://{host}/api/vobiz/hangup"

    # Support both audio/x-mulaw;rate=8000 (standard Vobiz telephony default) and audio/x-l16;rate=16000
    requested_fmt = request.query_params.get("format", "").lower()
    if requested_fmt in ("l16", "linear16", "pcm", "16000"):
        content_type_attr = 'contentType="audio/x-l16;rate=16000"'
    else:
        content_type_attr = 'contentType="audio/x-mulaw;rate=8000"'

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Stream bidirectional="true" keepCallAlive="true" {content_type_attr} statusCallbackUrl="{hangup_callback_url}" statusCallbackMethod="POST">
    {ws_url}
  </Stream>
</Response>"""
    return Response(content=xml_content, media_type="application/xml")


@app.api_route("/api/vobiz/hangup", methods=["GET", "POST"])
@app.api_route("/api/telephony/hangup", methods=["GET", "POST"])
@app.api_route("/hangup", methods=["GET", "POST"])
async def hangup_endpoint(request: Request):
    """
    Hangup URL webhook.
    Notified via HTTP POST when an inbound telephone call terminates.
    Logs call end metrics (duration, hangup cause) and launches post-call CRM intelligence.
    """
    call_uuid = ""
    caller_num = ""
    duration = 0
    hangup_cause = "normal_clearing"

    try:
        content_type = request.headers.get("content-type", "").lower()
        form_data = {}
        if "form" in content_type:
            form = await request.form()
            form_data = dict(form)
        elif "json" in content_type:
            form_data = await request.json()
        if not form_data:
            form_data = dict(request.query_params)

        call_uuid = form_data.get("CallUUID") or form_data.get("call_uuid") or form_data.get("CallSid") or ""
        caller_num = form_data.get("From") or form_data.get("caller") or form_data.get("from") or ""
        duration_raw = form_data.get("Duration") or form_data.get("BillDuration") or form_data.get("duration") or 0
        try:
            duration = int(duration_raw)
        except (ValueError, TypeError):
            duration = 0
        hangup_cause = form_data.get("HangupCause") or form_data.get("hangup_cause") or form_data.get("reason") or "normal_clearing"
    except Exception as parse_err:
        print(f"[hangup-webhook] Notice parsing hangup payload: {parse_err}")

    print(f"📞 [telephony-hangup] Call ended notification: UUID={call_uuid}, Duration={duration}s, Cause={hangup_cause}")

    if call_uuid:
        try:
            db.log_call_end(call_uuid)
            try:
                turns = db.get_call_queries(call_uuid)
                if turns:
                    mongo_store = get_shared_mongo_store()
                    turns_formatted = [
                        {"role": t.get("role", "user"), "text": t.get("query_text", "")}
                        for t in turns
                    ]
                    mongo_store.save_call_transcript(
                        call_id=call_uuid,
                        caller_number=caller_num or "Telephony",
                        turns=turns_formatted,
                        duration_seconds=float(duration),
                        status="completed",
                    )
            except Exception as me:
                print(f"[hangup-webhook] Mongo transcript save notice: {me}")
            try:
                from intelligence.post_call import analyze_and_record_call
                turns = db.get_call_queries(call_uuid)
                if turns:
                    formatted_history = [
                        {"role": t.get("role", "user"), "content": t.get("query_text", "")}
                        for t in turns
                    ]
                    asyncio.create_task(analyze_and_record_call(call_uuid, formatted_history, db))
            except Exception as post_err:
                print(f"[hangup-webhook] Post-call analyzer launch notice: {post_err}")
        except Exception as db_err:
            print(f"[hangup-webhook] DB log notice: {db_err}")

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Call hangup recorded successfully",
            "call_id": call_uuid,
            "duration": duration,
            "hangup_cause": hangup_cause,
        },
    )


async def fire_hangup_url(call_id: str, caller_num: str = "Telephony", reason: str = "normal_clearing", duration: int = 0):
    """
    Fires the hangup URL webhook (/api/vobiz/hangup) to record call conclusion,
    update metrics in Postgres, and persist transcripts in MongoDB.
    """
    if not call_id:
        return
    try:
        import httpx
        port = config.WS_PORT
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"http://127.0.0.1:{port}/api/vobiz/hangup",
                data={
                    "CallUUID": call_id,
                    "From": caller_num or "Telephony",
                    "Duration": str(duration),
                    "HangupCause": reason,
                }
            )
            print(f"🔥 [telephony] Hangup URL fired successfully for CallUUID={call_id} (reason: {reason})")
    except Exception as e:
        print(f"⚠️ [telephony] Notice firing hangup URL: {e}")


@app.websocket("/ws/vobiz")
@app.websocket("/ws")
@app.websocket("/ws/telephony")
async def websocket_vobiz_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket endpoint for Vobiz Voice AI Telephony.
    Streams 16kHz linear PCM audio back and forth, supporting:
    - Welcome message on call connection
    - Continuous speech recognition & 0.55s silence commit
    - Full-duplex barge-in (flushing playback queue via clearAudio)
    - Sarvam STT & TTS (female voice: Priya)
    - Automatic call disconnection on hangup detection
    - Real-time CRM logging to PostgreSQL / SQLite
    """
    import asyncio, io, wave, numpy as np
    await websocket.accept()
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"📞 [vobiz-ws] Telephony call connected from {client_host}")

    stt = get_shared_stt()
    llm = LLM(db=db, rag=rag)
    tts = get_shared_tts()
    # Fix #8: asyncio.Queue ensures sequential TTS playback (no interleaving)
    # Fix #20: asyncio.Event replaces nonlocal bool for thread-safe is_playing tracking
    stream_id = None
    call_id = None
    caller_num = "Telephony"
    audio_chunks: list[bytes] = []
    pre_speech_buffer: collections.deque = collections.deque(maxlen=12)  # ~240ms of pre-speech audio
    is_speech_active = False
    last_speech_time = 0.0
    vad_threshold = 0.018  # Optimal sensitivity for telephony
    consecutive_speech = 0
    consecutive_barge = 0
    active_turn_task = None
    audio_queue: asyncio.Queue = asyncio.Queue()
    is_playing_event = asyncio.Event()  # set = currently playing audio
    is_mulaw = True  # Dynamically updated on 'start' event; default is standard telephony mu-law
    last_user_activity = 0.0
    silence_prompt_active = False
    inactivity_task = None

    async def telephony_inactivity_monitor():
        """Monitors 7s silence inactivity during live telephone calls."""
        nonlocal silence_prompt_active, last_user_activity, stream_id, call_id, caller_num
        try:
            while True:
                await asyncio.sleep(0.5)
                now = asyncio.get_running_loop().time()
                if last_user_activity == 0.0:
                    continue
                # If audio is playing to caller, audio is queued, or caller is speaking:
                if is_playing_event.is_set() or not audio_queue.empty() or is_speech_active:
                    last_user_activity = now
                    continue

                idle_seconds = now - last_user_activity
                if not silence_prompt_active:
                    if idle_seconds >= 7.0:
                        print(f"⏱️ [vobiz-ws] Inactivity timeout: 7s silence detected. Asking caller: 'Are you still on the line?'")
                        silence_prompt_active = True
                        last_user_activity = asyncio.get_running_loop().time()
                        cur_l = getattr(llm, "current_lang", "gu")
                        prompt_msg = SILENCE_CHECK_PROMPTS.get(cur_l, SILENCE_CHECK_PROMPTS["gu"])
                        bcp47_code = "hi-IN" if cur_l == "hi" else ("en-IN" if cur_l == "en" else "gu-IN")
                        await send_vobiz_audio(prompt_msg, bcp47_code)
                else:
                    if idle_seconds >= 7.0:
                        print(f"⏱️ [vobiz-ws] Inactivity timeout: No response after 7s prompt. Concluding call and hanging up...")
                        cur_l = getattr(llm, "current_lang", "gu")
                        farewell_msg = FAREWELL_PROMPTS.get(cur_l, FAREWELL_PROMPTS["gu"])
                        bcp47_code = "hi-IN" if cur_l == "hi" else ("en-IN" if cur_l == "en" else "gu-IN")
                        await send_vobiz_audio(farewell_msg, bcp47_code)
                        while not audio_queue.empty() or is_playing_event.is_set():
                            await asyncio.sleep(0.2)
                        await asyncio.sleep(0.8)
                        print(f"📞 [vobiz-ws] Inactivity hangup: Disconnecting stream {stream_id}...")
                        await websocket.send_text(json.dumps({
                            "event": "stop",
                            "streamId": stream_id,
                        }))
                        await fire_hangup_url(call_id=call_id, caller_num=caller_num, reason="inactivity_timeout")
                        break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[vobiz-ws] Inactivity monitor notice: {e}")

    async def vobiz_audio_sender():
        """Drains the audio queue sequentially so sentences never interleave."""
        while True:
            item = await audio_queue.get()
            if item is None:  # sentinel
                break
            text, lang = item
            if not text.strip():
                continue
            is_playing_event.set()
            try:
                print(f"🔊 [vobiz-ws] Synthesizing speech ({lang}): '{text.strip()[:60]}...'")
                wav_bytes = await asyncio.to_thread(tts.synthesize, text.strip(), lang)
                if not wav_bytes:
                    print(f"⚠️ [vobiz-ws] Empty audio synthesized for: '{text.strip()[:40]}'")
                    continue
                try:
                    with io.BytesIO(wav_bytes) as bio, wave.open(bio, "rb") as wf:
                        in_sr = wf.getframerate()
                        in_ch = wf.getnchannels()
                        in_sw = wf.getsampwidth()
                        pcm_bytes = wf.readframes(wf.getnframes())
                except Exception:
                    in_sr = 22050
                    in_ch = 1
                    in_sw = 2
                    pcm_bytes = wav_bytes[44:] if wav_bytes.startswith(b"RIFF") else wav_bytes

                if is_mulaw:
                    from utils import pcm16_16k_to_mulaw
                    out_bytes = pcm16_16k_to_mulaw(wav_bytes)
                    out_content_type = "audio/x-mulaw"
                    out_sample_rate = 8000
                    chunk_size = 160  # 20ms at 8kHz mono mu-law (Vobiz standard)
                else:
                    import audioop
                    if in_sr != 16000:
                        out_bytes, _ = audioop.ratecv(pcm_bytes, in_sw, in_ch, in_sr, 16000, None)
                    else:
                        out_bytes = pcm_bytes
                    out_content_type = "audio/x-l16"
                    out_sample_rate = 16000
                    chunk_size = 320  # 20ms at 16kHz linear PCM

                # Send audio in 20ms chunks matching Vobiz Voice API specification
                num_chunks = 0
                for i in range(0, len(out_bytes), chunk_size):
                    chunk = out_bytes[i:i + chunk_size]
                    b64_payload = base64.b64encode(chunk).decode("utf-8")
                    msg = {
                        "event": "playAudio",
                        "media": {
                            "contentType": out_content_type,
                            "sampleRate": out_sample_rate,
                            "payload": b64_payload,
                        },
                    }
                    if stream_id:
                        msg["streamId"] = stream_id
                    await websocket.send_text(json.dumps(msg))
                    num_chunks += 1

                # Send checkpoint event so Vobiz acknowledges when audio has fully played
                checkpoint_msg = {
                    "event": "checkpoint",
                    "name": f"turn_complete_{int(time.time() * 1000)}",
                }
                if stream_id:
                    checkpoint_msg["streamId"] = stream_id
                await websocket.send_text(json.dumps(checkpoint_msg))
                print(f"🔊 [vobiz-ws] Streamed {num_chunks} audio frames ({len(out_bytes)} bytes {out_content_type}) to caller: '{text.strip()[:50]}...'")
            except Exception as err:
                print(f"[vobiz-ws] Error sending playAudio: {err}")
            finally:
                is_playing_event.clear()

    async def send_vobiz_audio(text: str, lang: str):
        """Enqueues a sentence for sequential TTS playback."""
        await audio_queue.put((text, lang))

    # Start the sequential audio sender
    _audio_sender_task = asyncio.create_task(vobiz_audio_sender())

    async def process_caller_audio(raw_pcm: bytes):
        nonlocal active_turn_task, stream_id, call_id, caller_num
        if not raw_pcm:
            return

        # Resample entire contiguous turn cleanly if mu-law (8kHz)
        if is_mulaw:
            dur = len(raw_pcm) / 16000.0  # 8000 samples/sec * 2 bytes/sample
            if dur < 0.35:
                return
            try:
                import audioop
                pcm_16k, _ = audioop.ratecv(raw_pcm, 2, 1, 8000, 16000, None)
            except Exception:
                pcm_16k = raw_pcm
        else:
            dur = len(raw_pcm) / 32000.0  # 16000 samples/sec * 2 bytes/sample
            if dur < 0.35:
                return
            pcm_16k = raw_pcm

        # Package into 16kHz mono WAV
        out_bio = io.BytesIO()
        with wave.open(out_bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm_16k)
        wav_data = out_bio.getvalue()

        transcript, detected_lang = await asyncio.to_thread(stt.transcribe, wav_data, None)
        if not transcript.strip() or is_noise_hallucination(transcript, dur, 0.04):
            print(f"🔇 [vobiz-ws] Discarded noise artifact: \"{transcript.strip()}\"")
            return

        try:
            # Dynamically determine the active turn language for telephony voice synthesis
            turn_lang, bcp47, _ = llm.detect_turn_language(transcript, fallback_lang=llm.current_lang, stt_lang=detected_lang)
            print(f"🎤 [vobiz-ws] Caller [{bcp47}]: {transcript}")
            if call_id:
                db.log_call_query(call_id, transcript, lang=bcp47)

            is_hangup = is_hangup_intent(transcript)

            buffer = ""
            full_agent_reply = ""
            async for piece in llm.areply_stream(transcript, stt_lang=detected_lang):
                buffer += piece
                full_agent_reply += piece
                ready_sentences, buffer = split_ready_sentences(buffer)
                for sentence in ready_sentences:
                    if sentence.strip():
                        await send_vobiz_audio(sentence.strip(), bcp47)

            if buffer.strip():
                full_agent_reply += buffer
                await send_vobiz_audio(buffer.strip(), bcp47)

            should_hangup = is_hangup or is_agent_farewell(full_agent_reply)

            if should_hangup:
                farewell_origin = "agent farewell: 'Thank you for your time. Have a great day!'" if is_agent_farewell(full_agent_reply) else "caller hangup intent"
                print(f"📞 [vobiz-ws] Hangup condition met ({farewell_origin}). Waiting for farewell audio playback to finish...")
                # Wait for sequential audio queue to empty and currently playing chunk to finish
                while not audio_queue.empty() or is_playing_event.is_set():
                    await asyncio.sleep(0.2)
                await asyncio.sleep(1.0)  # Telephone network buffer
                print(f"📞 [vobiz-ws] Farewell delivered to caller. Disconnecting call stream {stream_id}...")
                await websocket.send_text(json.dumps({
                    "event": "stop",
                    "streamId": stream_id,
                }))
                await fire_hangup_url(call_id=call_id, caller_num=caller_num, reason="caller_hangup_requested")
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
                start_obj = data.get("start", {}) if isinstance(data.get("start"), dict) else {}
                stream_obj = data.get("stream", {}) if isinstance(data.get("stream"), dict) else {}
                stream_id = (
                    data.get("streamId")
                    or data.get("stream_id")
                    or data.get("streamSid")
                    or start_obj.get("streamId")
                    or start_obj.get("stream_id")
                    or start_obj.get("streamSid")
                    or stream_obj.get("streamId")
                    or stream_obj.get("stream_id")
                    or ""
                )
                call_id = (
                    data.get("callId")
                    or data.get("call_id")
                    or data.get("callSid")
                    or data.get("CallUUID")
                    or start_obj.get("callId")
                    or start_obj.get("call_id")
                    or start_obj.get("callSid")
                    or ("CALL-TEL-" + uuid.uuid4().hex[:6].upper())
                )
                caller_num = (
                    data.get("caller")
                    or data.get("from")
                    or data.get("From")
                    or start_obj.get("from")
                    or start_obj.get("caller")
                    or "Telephony"
                )
                media_meta = data.get("media", {}) or start_obj.get("media", {})
                fmt_meta = (media_meta.get("contentType") or "").lower()
                rate_meta = media_meta.get("sampleRate") or 8000
                if "l16" in fmt_meta or "pcm" in fmt_meta or rate_meta == 16000:
                    is_mulaw = False
                else:
                    is_mulaw = True
                db.log_call_start(call_id, caller_number=caller_num, language="gu-IN")
                print(f"🚀 [vobiz-ws] Phone stream started: streamId={stream_id or '(none)'}, callId={call_id}, format={'mu-law 8kHz' if is_mulaw else 'PCM16 16kHz'}")
                # Greet caller with opening welcome message in Gujarati over the phone
                await send_vobiz_audio(WELCOME_MESSAGE, "gu-IN")
                last_user_activity = asyncio.get_running_loop().time()
                silence_prompt_active = False
                if inactivity_task is None or inactivity_task.done():
                    inactivity_task = asyncio.create_task(telephony_inactivity_monitor())

            elif event == "media":
                if not stream_id:
                    stream_id = data.get("streamId") or data.get("stream_id") or data.get("streamSid") or ""
                media = data.get("media", {})
                b64_chunk = media.get("payload", "")
                if not b64_chunk:
                    continue

                chunk_bytes = base64.b64decode(b64_chunk)
                if is_mulaw:
                    try:
                        import audioop
                        pcm_chunk = audioop.ulaw2lin(chunk_bytes, 2)
                    except Exception:
                        from utils.audio_utils import _MULAW_DECODE_TABLE
                        import struct
                        pcm_chunk = struct.pack(f"<{len(chunk_bytes)}h", *[_MULAW_DECODE_TABLE[b] for b in chunk_bytes])
                else:
                    pcm_chunk = chunk_bytes

                pcm = np.frombuffer(pcm_chunk, dtype=np.int16)
                if len(pcm) == 0:
                    continue

                rms = float(np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2)))

                # Barge-in: if caller interrupts while agent is speaking, clear audio
                if is_playing_event.is_set():  # Fix #20: use asyncio.Event
                    if rms > 0.045:
                        consecutive_barge += 1
                        if consecutive_barge >= 2:
                            print("🛑 [vobiz-ws] Barge-in confirmed from caller. Clearing playback queue.")
                            if active_turn_task and not active_turn_task.done():
                                active_turn_task.cancel()
                            is_playing_event.clear()   # Fix #20: clear event instead of setting bool
                            # Drain the audio queue to discard pending sentences
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except Exception:
                                    break
                            clear_msg = {"event": "clearAudio"}
                            if stream_id:
                                clear_msg["streamId"] = stream_id
                            await websocket.send_text(json.dumps(clear_msg))
                            audio_chunks = [pcm_chunk]
                            pre_speech_buffer.clear()
                            is_speech_active = True
                            silence_prompt_active = False
                            last_speech_time = asyncio.get_running_loop().time()
                            last_user_activity = last_speech_time
                    else:
                        consecutive_barge = 0
                    continue

                consecutive_barge = 0

                if rms > vad_threshold:
                    consecutive_speech += 1
                    if consecutive_speech >= 2:
                        if not is_speech_active:
                            is_speech_active = True
                            silence_prompt_active = False
                            last_user_activity = asyncio.get_running_loop().time()
                            # Prepend pre-speech buffer so initial consonants (like "H" in "Hello") are not clipped
                            audio_chunks.extend(list(pre_speech_buffer))
                            pre_speech_buffer.clear()
                        audio_chunks.append(pcm_chunk)
                        last_speech_time = asyncio.get_running_loop().time()
                        last_user_activity = last_speech_time
                    else:
                        pre_speech_buffer.append(pcm_chunk)
                else:
                    consecutive_speech = 0
                    if is_speech_active:
                        audio_chunks.append(pcm_chunk)
                        # Check 0.40s silence commit for ultra-low turn-taking latency
                        now = asyncio.get_running_loop().time()
                        if now - last_speech_time >= 0.40:
                            is_speech_active = False
                            silence_prompt_active = False
                            last_user_activity = now
                            total_pcm = b"".join(audio_chunks)
                            audio_chunks = []
                            pre_speech_buffer.clear()
                            if active_turn_task and not active_turn_task.done():
                                active_turn_task.cancel()
                            active_turn_task = asyncio.create_task(process_caller_audio(total_pcm))
                    else:
                        pre_speech_buffer.append(pcm_chunk)

            elif event in ("playedStream", "clearedAudio"):
                is_playing_event.clear()

            elif event == "stop":
                print(f"👋 [vobiz-ws] Stream stopped: {stream_id}")
                break

    except WebSocketDisconnect:
        print(f"👋 [vobiz-ws] Telephony client disconnected ({client_host})")
    except Exception as e:
        print(f"⚠️ [vobiz-ws] Telephony stream error: {e}")
    finally:
        if inactivity_task and not inactivity_task.done():
            inactivity_task.cancel()
        if call_id:
            db.log_call_end(call_id)
            try:
                mongo_store = get_shared_mongo_store()
                turns_formatted = []
                for h in llm.history:
                    turns_formatted.append({
                        "role": h.get("role", "user"),
                        "text": h.get("content", ""),
                    })
                mongo_store.save_call_transcript(
                    call_id=call_id,
                    caller_number=caller_num,
                    turns=turns_formatted,
                    language=getattr(llm, "current_lang", "gu"),
                    status="completed",
                )
            except Exception as me:
                print(f"[vobiz-ws] Mongo transcript save notice: {me}")
            try:
                from intelligence.post_call import analyze_and_record_call
                asyncio.create_task(analyze_and_record_call(call_id, llm.history.copy(), db))
            except Exception as pe:
                print(f"[vobiz-ws] Telephony post-call notice: {pe}")


def create_app():
    return app
