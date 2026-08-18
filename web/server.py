"""
FastAPI Unified Web & WebSocket Gateway Server.
Serves the Glassmorphic Web UI, PDF RAG Upload APIs, DB Explorer APIs, and Real-time Voice WebSocket.
"""

import base64
import json
import os
import re
import traceback
from typing import Any
import requests

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from db.database import Database
from llm import LLM
from rag import RAGStore
from stt import STT
from tts import TTS

SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def split_ready_sentences(buffer: str):
    parts = SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    *complete, remainder = parts
    return complete, remainder


# Initialize application
app = FastAPI(
    title="University Voice Agent Gateway",
    description="Multilingual Voice Agent & Document Intelligence Portal",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    # 1. Ollama status
    ollama_status = "unavailable"
    active_model = config.LLM_MODEL
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            models = [m.get("name") for m in r.json().get("models", [])]
            if any(config.LLM_MODEL in m for m in models):
                ollama_status = "ready"
            else:
                ollama_status = f"connected (model {config.LLM_MODEL} not loaded)"
    except Exception:
        ollama_status = "offline"

    # 2. Sarvam status
    sarvam_configured = bool(config.SARVAM_API_KEY and len(config.SARVAM_API_KEY) > 10)

    # 3. Database status
    db_mode = "SQLite (Local Fallback)" if db.use_sqlite else "PostgreSQL (Production)"

    # 4. RAG store metrics
    doc_count = len(rag.list_documents())
    chunk_count = rag.collection.count()

    return {
        "llm": {
            "status": ollama_status,
            "model": active_model,
            "provider": "Ollama (Local)",
        },
        "sarvam_ai": {
            "configured": sarvam_configured,
            "stt_model": config.STT_MODEL,
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
# WebSocket: Real-time Voice & Text Call Gateway
# -----------------------------------------------------------------------------

class WebVoiceSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.stt = STT()
        self.llm = LLM()
        self.tts = TTS()
        self.language_code = config.DEFAULT_LANGUAGE
        self._hook_llm_tools()

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

        # 2. Query LLM & Stream TTS Chunks
        buffer = ""
        full_agent_reply = ""

        for piece in self.llm.reply_stream(user_text):
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
                        audio_chunk = self.tts.synthesize(
                            sentence.strip(), language_code=detected_lang
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
                audio_chunk = self.tts.synthesize(
                    buffer.strip(), language_code=detected_lang
                )
                if audio_chunk:
                    await self.ws.send_bytes(audio_chunk)
            except Exception as err:
                print(f"[web-ws] TTS buffer error: {err}")

        # Signal completion
        await self.ws.send_json({
            "event": "agent_done",
            "full_text": full_agent_reply.strip(),
            "language": detected_lang,
        })

    async def process_audio_payload(self, audio_bytes: bytes):
        """Transcribes input audio bytes and runs pipeline."""
        if len(audio_bytes) < 200:
            return

        try:
            transcript, detected_lang = self.stt.transcribe(
                audio_bytes, language_code=self.language_code
            )
        except Exception as e:
            print(f"[web-ws] STT error: {e}")
            await self.ws.send_json({"event": "error", "message": f"STT failed: {str(e)}"})
            return

        if not transcript.strip():
            await self.ws.send_json({"event": "empty_transcript"})
            return

        print(f"🎙️ [web-ws] User [{detected_lang}]: {transcript}")

        await self.ws.send_json({
            "event": "user_transcript",
            "text": transcript,
            "language": detected_lang,
        })

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
                await session.process_audio_payload(audio_bytes)

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
                    elif event == "text_query":
                        # Dual text chat input support
                        text = payload.get("text", "")
                        lang = payload.get("language_code", session.language_code)
                        await websocket.send_json({
                            "event": "user_transcript",
                            "text": text,
                            "language": lang,
                        })
                        await session.process_user_query(text, lang)

                    elif event == "interrupt":
                        # Client signal to interrupt playback
                        print(f"🛑 [web-ws] Barge-in interruption from {client_host}")

                    elif event == "ping":
                        await websocket.send_json({"event": "pong"})
                except Exception as e:
                    print(f"[web-ws] JSON message parse error: {e}")

    except WebSocketDisconnect:
        print(f"👋 [web-ws] Client disconnected ({client_host})")
    except Exception as e:
        print(f"⚠️ [web-ws] Connection error: {e}")


def create_app():
    return app
