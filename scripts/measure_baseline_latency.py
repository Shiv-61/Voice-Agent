"""
Baseline & Post-Fix Latency Measurement Script.
Runs test turns across:
1. ROUTE A: Plain university facts (No tool, direct prompt knowledge)
2. RAG: University documents / syllabus query (search_university_docs)
3. DB: Student database query (get_student_marks / lookup_student)

Measures exact timestamps:
- STT transcription
- LLM Time-to-First-Token (TTFT)
- LLM First Complete Sentence
- Tool execution (RAG embedding, search, rerank vs DB query)
- TTS Time-to-First-Audio (TTFA)
- Total turn-taking latency (Speech End -> First Audio Start)
- Hangup supervisor latency
"""

import asyncio
import io
import json
import os
import sys
import time
import wave
import numpy as np

# Ensure root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import config
from db.database import Database
from rag import RAGStore
from llm.llm import LLM
from stt import STT
from tts import TTS
from utils.text_utils import split_ready_sentences
from utils.latency_tracker import TurnLatencyRecord, latency_manager


def create_dummy_wav(duration_s: float = 1.0) -> bytes:
    """Generates 1 second of 16kHz silence/tone for realistic STT benchmarking."""
    sample_rate = 16000
    n_samples = int(sample_rate * duration_s)
    # 440 Hz tone with low amplitude
    t = np.linspace(0, duration_s, n_samples, False)
    pcm = (np.sin(2 * np.pi * 440 * t) * 3000).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return bio.getvalue()


async def measure_single_turn(
    llm: LLM,
    stt: STT,
    tts: TTS,
    query_text: str,
    route_label: str,
    call_id: str = "BENCHMARK-CALL",
    turn_id: int = 1,
) -> TurnLatencyRecord:
    rec = TurnLatencyRecord(call_id=call_id, turn_id=turn_id, user_query=query_text, query_route=route_label)
    
    # 1. Simulate speech end and STT
    rec.mark_speech_end()
    rec.mark_stt_start()
    dummy_wav = create_dummy_wav(1.0)
    # Simulate transcription time on dummy or mock STT
    # We measure actual whisper transcribe time on the 1s wav
    _ = await asyncio.to_thread(stt.transcribe, dummy_wav, "en")
    rec.mark_stt_end()

    # 2. LLM Streaming & Tool Measurement
    rec.mark_llm_request_sent()
    
    # Hook into tool execution
    orig_tool_exec = llm._execute_tool
    def timed_tool_exec(tool_name: str, kwargs: dict) -> str:
        rec.mark_tool_start(tool_name)
        if "docs" in tool_name or "search" in tool_name:
            rec.tool_type = "RAG"
            # Measure inside RAG
            t_emb0 = time.perf_counter()
            if hasattr(llm.rag, "_embed_model") and llm.rag._embed_model:
                _ = llm.rag._embed_model.encode([kwargs.get("query", "")], normalize_embeddings=True)
            rec.rag_embed_ms = (time.perf_counter() - t_emb0) * 1000.0
        else:
            rec.tool_type = "DB"
            
        t0 = time.perf_counter()
        res = orig_tool_exec(tool_name, kwargs)
        rec.rag_search_ms = (time.perf_counter() - t0) * 1000.0
        rec.mark_tool_end()
        return res
        
    llm._execute_tool = timed_tool_exec

    buffer = ""
    first_sentence_done = False
    first_audio_done = False
    full_reply = ""

    async for piece in llm.areply_stream(query_text, stt_lang="en"):
        rec.mark_llm_first_token()
        buffer += piece
        full_reply += piece
        ready, buffer = split_ready_sentences(buffer)
        for s in ready:
            if s.strip() and not first_sentence_done:
                rec.mark_llm_first_sentence()
                first_sentence_done = True
                
                # First sentence sent to TTS
                rec.mark_tts_request_sent()
                t0_tts = time.perf_counter()
                audio_bytes = await asyncio.to_thread(tts.synthesize, s.strip(), "en-IN")
                rec.mark_tts_first_chunk()
                rec.mark_tts_playback_start()
                first_audio_done = True

    rec.mark_llm_last_token()

    if buffer.strip() and not first_sentence_done:
        rec.mark_llm_first_sentence()
        rec.mark_tts_request_sent()
        audio_bytes = await asyncio.to_thread(tts.synthesize, buffer.strip(), "en-IN")
        rec.mark_tts_first_chunk()
        rec.mark_tts_playback_start()

    # 3. Hangup Supervisor
    rec.mark_supervisor_start()
    _ = llm.check_call_hangup_supervisor(query_text, full_reply)
    rec.mark_supervisor_end()

    # Reset tool hook
    llm._execute_tool = orig_tool_exec

    return rec


async def run_benchmark(num_runs: int = 3):
    print("=" * 70)
    print("⏱️ RUNNING VOICE AGENT LATENCY BENCHMARK")
    print("=" * 70)

    db = Database()
    rag = RAGStore()
    llm = LLM(db=db, rag=rag)
    stt = STT()
    tts = TTS()

    test_cases = [
        ("ROUTE_A (Instant Fact)", "What is the annual fee for B.Tech Computer Science?"),
        ("RAG (Syllabus Doc)", "What subjects are taught in B.Tech IT Semester 3?"),
        ("DB (Student Records)", "What are the marks for student STU101?"),
    ]

    results = []

    for route_label, query in test_cases:
        print(f"\n--- Testing {route_label}: '{query}' ---")
        route_records = []
        for i in range(num_runs):
            llm.history.clear()  # fresh state
            rec = await measure_single_turn(llm, stt, tts, query, route_label, turn_id=i+1)
            print(f"  Run {i+1}: SpeechEnd->AudioStart = {rec.speech_to_audio_start_ms:.1f}ms | TTFT = {rec.llm_ttft_ms:.1f}ms | Tool = {rec.tool_duration_ms:.1f}ms | TTS_1st = {rec.tts_ttfa_ms:.1f}ms")
            route_records.append(rec)
        results.append((route_label, route_records))

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Route':<25} | {'Median Total (ms)':<18} | {'Worst Total (ms)':<18} | {'LLM TTFT (ms)':<14} | {'Tool (ms)':<10}")
    print("-" * 90)
    for label, recs in results:
        totals = [r.speech_to_audio_start_ms for r in recs]
        ttfts = [r.llm_ttft_ms for r in recs]
        tools = [r.tool_duration_ms for r in recs]
        median_total = np.median(totals)
        worst_total = np.max(totals)
        median_ttft = np.median(ttfts)
        median_tool = np.median(tools)
        print(f"{label:<25} | {median_total:<18.1f} | {worst_total:<18.1f} | {median_ttft:<14.1f} | {median_tool:<10.1f}")
    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_benchmark(num_runs=3))
