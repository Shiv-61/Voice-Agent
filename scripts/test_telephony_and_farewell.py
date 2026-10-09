#!/usr/bin/env python3
"""
Test suite for:
1. Primary Answer URL (XML response)
2. Hangup URL (POST webhook notification & DB logging)
3. Agent farewell detection across Gujarati, Hindi, and English
4. Adding new data to RAG and verifying retrieval
"""

import os
import sys
import xml.etree.ElementTree as ET

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from web.server import app, rag, db
from utils.text_utils import is_agent_farewell, is_hangup_intent


def test_farewell_detection():
    print("=" * 60)
    print("TEST 1: Agent Farewell Detection ('Thank you for your time. Have a great day!')")
    print("=" * 60)

    test_cases = [
        # English
        ("Thank you for your time. Have a great day!", True, "English full"),
        ("Thank you for your time, have a good day!", True, "English good day"),
        ("Thank you for your time.", True, "English thank you for your time"),
        ("Have a great day ahead!", True, "English have a great day"),
        # Hindi
        ("आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!", True, "Hindi full"),
        ("समय के लिए धन्यवाद।", True, "Hindi samay ke liye dhanyawad"),
        ("आपका दिन शुभ रहे!", True, "Hindi din shubh rahe"),
        ("डीडीयू में आपका स्वागत है। आपका दिन शुभ हो!", True, "Hindi embedded"),
        # Gujarati
        ("તમારા સમય માટે આભાર. તમારો દિવસ શુભ રહે!", True, "Gujarati full"),
        ("તમારા સમય માટે આભાર!", True, "Gujarati samay maate aabhar"),
        ("તમારો દિવસ શુભ રહે!", True, "Gujarati divas shubh rahe"),
        ("ડીડીયુ આઈટી તરફથી આભાર. તમારો દિવસ સારો રહે!", True, "Gujarati embedded"),
        # Negative tests
        ("B.Tech IT ની વાર્ષિક ફી ૨.૫ લાખ રૂપિયા છે.", False, "Gujarati regular fact"),
        ("DDU placement rate is 96.5 percent.", False, "English regular fact"),
        ("कंप्यूटर साइंस की कटऑफ 85% है।", False, "Hindi regular fact"),
    ]

    all_passed = True
    for phrase, expected, desc in test_cases:
        res = is_agent_farewell(phrase)
        status = "✅ PASS" if res == expected else "❌ FAIL"
        if res != expected:
            all_passed = False
        print(f"  [{status}] {desc}: '{phrase[:45]}...' -> {res} (expected {expected})")

    assert all_passed, "Some farewell detection test cases failed!"
    print("🎉 All farewell detection tests passed!\n")


def test_primary_answer_endpoint():
    print("=" * 60)
    print("TEST 2: Primary Answer URL (POST -> XML Call Instructions)")
    print("=" * 60)

    client = TestClient(app)

    endpoints = ["/api/vobiz/answer", "/api/telephony/answer", "/answer", "/primary-answer"]
    for ep in endpoints:
        # Test POST with standard telephony payload
        payload = {
            "CallUUID": "TEST-CALL-UUID-1234",
            "From": "+919876543210",
            "To": "+912682520502",
            "Direction": "inbound",
            "CallStatus": "ringing",
        }
        resp = client.post(ep, data=payload)
        assert resp.status_code == 200, f"Endpoint {ep} returned status {resp.status_code}"
        assert "application/xml" in resp.headers.get("content-type", ""), f"Wrong content-type for {ep}"

        # Verify XML structure
        root = ET.fromstring(resp.text)
        assert root.tag == "Response", f"Root tag is not Response: {root.tag}"
        stream_el = root.find("Stream")
        assert stream_el is not None, "Missing <Stream> tag in XML"
        assert stream_el.attrib.get("bidirectional") == "true", "bidirectional attribute missing or not true"
        assert stream_el.attrib.get("keepCallAlive") == "true", "keepCallAlive attribute missing or not true"
        assert "ws/vobiz" in stream_el.text, f"WebSocket URL missing from Stream tag: {stream_el.text}"

        print(f"  ✅ [PASS] {ep} returned valid XML:")
        print(f"     {resp.text.strip()[:100]}...")

    print("🎉 Primary Answer URL endpoints verified!\n")


def test_hangup_endpoint():
    print("=" * 60)
    print("TEST 3: Hangup URL (POST -> Notified When Call Ends)")
    print("=" * 60)

    client = TestClient(app)

    endpoints = ["/api/vobiz/hangup", "/api/telephony/hangup", "/hangup"]
    for ep in endpoints:
        call_uuid = f"CALL-TEST-{os.urandom(3).hex().upper()}"
        # Seed call start in DB first
        db.log_call_start(call_uuid, caller_number="+919998887776", language="gu-IN")

        payload = {
            "CallUUID": call_uuid,
            "From": "+919998887776",
            "To": "+912682520502",
            "Duration": "45",
            "HangupCause": "normal_clearing",
            "CallStatus": "completed",
        }
        resp = client.post(ep, data=payload)
        assert resp.status_code == 200, f"Endpoint {ep} returned status {resp.status_code}"
        data = resp.json()
        assert data.get("status") == "success", f"Failed status in response: {data}"
        assert data.get("call_id") == call_uuid, f"Mismatched call_id: {data}"
        assert data.get("duration") == 45, f"Mismatched duration: {data}"

        print(f"  ✅ [PASS] {ep} acknowledged hangup for {call_uuid}:")
        print(f"     {data}")

    print("🎉 Hangup URL endpoints verified!\n")


def test_rag_new_data_ingestion():
    print("=" * 60)
    print("TEST 4: Adding New Data to RAG & Querying Knowledge Base")
    print("=" * 60)

    client = TestClient(app)

    # 1. Test Text Ingestion via API
    custom_title = "DDU_Special_Scholarship_2026.txt"
    custom_text = """
    DDU Special IT Merit Scholarship 2026:
    Under the new 2026 policy, students securing above 92% in 10+2 PCM are eligible
    for a 50% tuition fee waiver in B.Tech Information Technology.
    The application for this merit scholarship must be submitted before 15 August 2026.
    Contact scholarship cell at scholarship@ddu.ac.in.
    """

    resp = client.post("/api/documents/text", json={
        "title": custom_title,
        "text": custom_text,
        "category": "admissions_and_campus",
    })
    assert resp.status_code == 200, f"Text upload failed: {resp.text}"
    upload_res = resp.json()
    assert upload_res.get("success") is True, "Upload returned success=False"
    print(f"  ✅ [PASS] Ingested new data into RAG: {upload_res['data']['filename']} ({upload_res['data']['total_chunks']} chunks)")

    # 2. Test Semantic Search on New Data
    search_query = "What is the scholarship for students with 92% in 10+2?"
    search_resp = client.post("/api/documents/search", json={"query": search_query, "n_results": 2})
    assert search_resp.status_code == 200
    search_data = search_resp.json()
    assert len(search_data.get("results", [])) > 0, "No results returned for new data search!"

    top_result = search_data["results"][0]
    print(f"  ✅ [PASS] Semantic Search Query: \"{search_query}\"")
    print(f"     Top Match Score: {top_result.get('similarity_score')}")
    print(f"     Snippet: \"{top_result.get('text')[:120]}...\"")
    found_in_results = any("92%" in r.get("text", "") for r in search_data.get("results", []))
    print(f"     Found in results: {found_in_results}")
    assert found_in_results, f"Expected 92% scholarship chunk to be retrieved among top results: {search_data['results']}"

    # Cleanup test document from vector store
    doc_id = upload_res["data"]["doc_id"]
    del_resp = client.delete(f"/api/documents/{doc_id}")
    print(f"  🧹 Cleaned up temporary test document ({doc_id})")

    print("🎉 RAG new data ingestion and retrieval verified!\n")


if __name__ == "__main__":
    test_farewell_detection()
    test_primary_answer_endpoint()
    test_hangup_endpoint()
    test_rag_new_data_ingestion()
    print("=" * 60)
    print("🌟 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
