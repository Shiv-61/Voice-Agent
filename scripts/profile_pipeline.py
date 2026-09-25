"""
Pipeline Latency Profiler.
Measures isolated latency of each component:
1. Database Query
2. ChromaDB Vector Search (RAG)
3. OpenRouter LLM Call (TTFT & total time)
4. Sarvam TTS API
"""

import asyncio
import time
import json
import os
import sys

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import httpx
import config
from db.database import Database
from rag import RAGStore
from tts import TTS

async def profile_database():
    print("\n--- 1. Profiling Database Query ---")
    db = Database()
    t0 = time.perf_counter()
    res = db.get_admission_info("CSE")
    dt = (time.perf_counter() - t0) * 1000
    print(f"Database query took: {dt:.2f}ms (Result: {len(res)} rows)")

async def profile_rag():
    print("\n--- 2. Profiling ChromaDB RAG Vector Store ---")
    rag = RAGStore()
    t0 = time.perf_counter()
    matches = rag.query_documents("eligibility for B.Tech CSE", n_results=2)
    dt = (time.perf_counter() - t0) * 1000
    print(f"ChromaDB search took: {dt:.2f}ms (Matches: {len(matches)})")

async def profile_tts():
    print("\n--- 3. Profiling Sarvam TTS API ---")
    if not config.SARVAM_API_KEY:
        print("SARVAM_API_KEY not set, skipping TTS profile")
        return
    tts = TTS()
    text = "For B.Tech Computer Science at DDU IT, the annual fee is 2.5 Lakh rupees."
    t0 = time.perf_counter()
    audio = await asyncio.to_thread(tts.synthesize, text, "en-IN")
    dt = (time.perf_counter() - t0) * 1000
    print(f"Sarvam TTS synthesis took: {dt:.2f}ms (Audio bytes: {len(audio)})")

async def profile_llm(model_name: str):
    print(f"\n--- 4. Profiling LLM: '{model_name}' ---")
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a fast voice assistant. Answer in 1 short spoken sentence."},
            {"role": "user", "content": "What is the fee for B.Tech Computer Science?"}
        ],
        "temperature": 0.3,
        "max_tokens": 150,
        "stream": True,
    }

    t0 = time.perf_counter()
    ttft = None
    tokens = []
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("POST", config.OPENROUTER_URL, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if line.startswith("data:") and line != "data: [DONE]":
                        try:
                            d = json.loads(line[5:].strip())
                            delta = d["choices"][0]["delta"].get("content", "")
                            if delta:
                                if ttft is None:
                                    ttft = (time.perf_counter() - t0) * 1000
                                tokens.append(delta)
                        except Exception:
                            pass
        total_time = (time.perf_counter() - t0) * 1000
        reply = "".join(tokens).strip()
        print(f"Model: {model_name}")
        print(f"⚡ TTFT (Time-to-first-token): {ttft:.1f}ms" if ttft else "No token received")
        print(f"⏱️ Total Generation Time: {total_time:.1f}ms")
        print(f"Words generated: {len(tokens)} chunks")
        print(f"Response preview: \"{reply[:80]}...\"")
    except Exception as e:
        print(f"Error testing {model_name}: {e}")

async def main():
    await profile_database()
    await profile_rag()
    await profile_tts()
    # Test current model
    await profile_llm(config.LLM_MODEL)
    # Test fast alternatives
    await profile_llm("meta-llama/llama-3.3-70b-instruct")
    await profile_llm("google/gemini-2.0-flash-001")
    await profile_llm("qwen/qwen-2.5-72b-instruct")

if __name__ == "__main__":
    asyncio.run(main())
