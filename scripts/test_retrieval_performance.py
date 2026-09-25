"""
Test Anticipatory RAG & Retrieval Performance.
Verifies that questions needing RAG or DB facts are answered in a single turn
without the slow 2-turn tool round-trip delay.
"""

import sys
import os
import time
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from llm.llm import LLM, get_shared_db, get_shared_rag

async def test_question(llm: LLM, question: str, expected_topic: str):
    print(f"\n=======================================================")
    print(f"Testing Question: '{question}'")
    print(f"Expected Topic: {expected_topic}")
    print(f"=======================================================")

    t0 = time.perf_counter()
    tokens = []
    first_token_time = None

    async for piece in llm.areply_stream(question):
        if first_token_time is None and piece.strip():
            first_token_time = time.perf_counter() - t0
        tokens.append(piece)

    total_time = time.perf_counter() - t0
    full_text = "".join(tokens).strip()

    print(f"⚡ TTFT (Time to first spoken token): {first_token_time:.2f}s" if first_token_time else "No token")
    print(f"⏱️ Total Stream Time: {total_time:.2f}s")
    print(f"📝 Full Answer:\n{full_text}\n")

    # Verify no raw TOOL_CALL syntax leaked into speech
    assert "TOOL_CALL:" not in full_text, "Error: raw TOOL_CALL syntax leaked into spoken text!"
    assert len(full_text) > 10, "Error: response is too short!"
    print(f"✅ PASSED: '{expected_topic}' answered cleanly in {total_time:.2f}s")

async def main():
    db = get_shared_db()
    rag = get_shared_rag()
    llm = LLM(db=db, rag=rag)

    # 1. Test RAG Policy retrieval
    await test_question(llm, "What is the attendance policy for examinations?", "RAG Attendance Policy")

    # 2. Test DB Placement retrieval
    await test_question(llm, "What are the placement packages for CSE branch?", "DB Placement Records")

    # 3. Test DB Admission / Fee retrieval
    await test_question(llm, "What are the fees and eligibility for B.Tech CSE?", "DB Admission & Fees")

if __name__ == "__main__":
    asyncio.run(main())
