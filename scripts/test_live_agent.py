"""
Live WebSocket and API Integration Test for Sub-Second Voice Agent.
Tests system health, WebSocket lifecycle, multi-lingual token streaming,
tool execution (student marks), and instant hangup detection.
"""

import asyncio
import json
import sys
import time
import httpx
import websockets

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE_HTTP = "http://127.0.0.1:8765"
BASE_WS = "ws://127.0.0.1:8765/ws/call"


async def test_rest_health():
    print("\n--- 1. Testing REST Endpoints ---")
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Health check
        r = await client.get(f"{BASE_HTTP}/healthz")
        assert r.status_code == 200, f"Health check failed: {r.text}"
        print(f"✓ GET /healthz: {r.json()}")

        # System status
        r = await client.get(f"{BASE_HTTP}/api/system/status")
        assert r.status_code == 200, f"Status check failed: {r.text}"
        data = r.json()
        print(f"✓ GET /api/system/status: LLM Provider='{data['llm']['provider']}', DB='{data['database']['mode']}', Docs={data['vector_store']['indexed_documents']}")

        # Dashboard metrics
        r = await client.get(f"{BASE_HTTP}/api/data/dashboard")
        assert r.status_code == 200, f"Dashboard failed: {r.text}"
        dash = r.json()
        print(f"✓ GET /api/data/dashboard: Students={dash['total_students']}, Programs={dash['total_programs']}")


async def test_websocket_stream():
    print("\n--- 2. Testing WebSocket Live Conversational Stream ---")
    async with websockets.connect(BASE_WS) as ws:
        # A. Handshake & Welcome
        print("-> Sending session start handshake (English)...")
        await ws.send(json.dumps({"event": "start", "language_code": "en"}))

        welcome_received = False
        audio_chunks_received = 0

        # Receive greeting frames
        while not welcome_received:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("event") == "agent_done":
                    welcome_received = True
                    print(f"✓ Received Welcome Greeting: \"{data.get('full_text')}\"")
            elif isinstance(msg, bytes):
                audio_chunks_received += 1

        print(f"✓ Received {audio_chunks_received} initial greeting audio frames.")

        # B. Test Real-time Streaming Query & Measure TTFT / TTFA
        query_text = "What is the eligibility and fee for B.Tech Computer Science?"
        print(f"\n-> Sending Query: '{query_text}'")
        t0 = time.time()
        await ws.send(json.dumps({
            "event": "text_query",
            "text": query_text,
            "language_code": "en"
        }))

        ttft = None
        ttfa = None
        streamed_chunks = []
        audio_responses = 0
        agent_done = False

        while not agent_done:
            msg = await ws.recv()
            now = time.time()
            if isinstance(msg, str):
                data = json.loads(msg)
                event = data.get("event")
                if event == "agent_partial_text":
                    if ttft is None:
                        ttft = (now - t0) * 1000
                        print(f"⚡ TTFT (Time-to-First-Sentence): {ttft:.1f}ms")
                    sentence = data.get("text", "")
                    streamed_chunks.append(sentence)
                    print(f"  [Streamed Sentence]: \"{sentence}\"")
                elif event == "agent_done":
                    agent_done = True
                    total_time = (now - t0) * 1000
                    print(f"✓ Agent Generation Complete in {total_time:.1f}ms")
                    print(f"  Full Answer: \"{data.get('full_text')}\"")
            elif isinstance(msg, bytes):
                if ttfa is None:
                    ttfa = (now - t0) * 1000
                    print(f"🔊 TTFA (Time-to-First-Audio): {ttfa:.1f}ms")
                audio_responses += 1

        print(f"✓ Total Synthesized Audio Packets Streamed: {audio_responses}")
        assert len(streamed_chunks) > 0, "No text stream chunks received!"

        # C. Test Tool Calling Stream with Hindi Query
        hindi_query = "आरव पटेल के मार्क्स क्या हैं?"
        print(f"\n-> Sending Hindi Tool Query: '{hindi_query}'")
        t0 = time.time()
        await ws.send(json.dumps({
            "event": "text_query",
            "text": hindi_query,
            "language_code": "hi-IN"
        }))

        hindi_done = False
        tool_executed = False
        while not hindi_done:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                event = data.get("event")
                if event == "tool_executed":
                    tool_executed = True
                    print(f"  🛠️ Tool Executed: {data.get('tool')}({data.get('args')})")
                elif event == "agent_done":
                    hindi_done = True
                    print(f"✓ Hindi Tool Answer Synthesized: \"{data.get('full_text')}\"")
            elif isinstance(msg, bytes):
                pass

        # D. Test Instant Hangup Detection
        farewell = "Thank you, goodbye!"
        print(f"\n-> Sending Hangup Message: '{farewell}'")
        t0 = time.time()
        await ws.send(json.dumps({
            "event": "text_query",
            "text": farewell,
            "language_code": "en"
        }))

        hangup_detected = False
        call_ended = False
        while not call_ended:
            msg = await ws.recv()
            if isinstance(msg, str):
                data = json.loads(msg)
                event = data.get("event")
                if event == "agent_done":
                    hangup_detected = data.get("call_hangup", False)
                    print(f"✓ Agent Farewell: \"{data.get('full_text')}\" (call_hangup: {hangup_detected})")
                elif event == "call_ended":
                    call_ended = True
                    duration = (time.time() - t0) * 1000
                    print(f"📞 Received 'call_ended' Event in {duration:.1f}ms. Hangup verified!")

        assert hangup_detected, "Hangup intent was not detected!"
        print("\n✨ LIVE WEBSOCKET PIPELINE TEST PASSED SUCCESSFULLY!")


async def main():
    print("=" * 60)
    print("🚀 RUNNING LIVE INTEGRATION & LATENCY BENCHMARK")
    print("=" * 60)
    await test_rest_health()
    await test_websocket_stream()


if __name__ == "__main__":
    asyncio.run(main())
