"""
Verification test suite for Voice Agent & RAG Knowledge Base.
Tests Database, RAGStore, LLM tool dispatcher, and FastAPI endpoints.
"""

import os
import sys

# Ensure UTF-8 stdout encoding on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db.database import Database
from rag import RAGStore
from llm import LLM
from web.server import app


def test_database():
    print("\n--- 1. Testing Database Layer ---")
    db = Database()
    student = db.lookup_student("Aarav Patel")
    assert student is not None, "Failed to lookup student"
    print(f"✓ Student Lookup: {student['name']} ({student['student_id']}) - {student['department_name']}")

    marks = db.get_student_marks(student["student_id"])
    assert len(marks) > 0, "Failed to fetch student marks"
    print(f"✓ Student Marks: {len(marks)} subjects recorded.")

    attendance = db.get_student_attendance(student["student_id"])
    assert len(attendance) > 0, "Failed to fetch student attendance"
    print(f"✓ Student Attendance: {len(attendance)} subject records.")

    placements = db.get_placement_stats("CSE")
    assert len(placements) > 0, "Failed to fetch placement stats"
    print(f"✓ Placement Stats: Highest Package ₹{placements[0]['highest_package_lpa']} LPA.")

    admissions = db.get_admission_info("CSE")
    assert len(admissions) > 0, "Failed to fetch admission info"
    print(f"✓ Admission Info: {admissions[0]['program']} -> {admissions[0]['fee_per_year']}.")


def test_rag():
    print("\n--- 2. Testing RAG Knowledge Base ---")
    rag = RAGStore()
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_university_policy.pdf")
    assert os.path.exists(pdf_path), f"Missing PDF at {pdf_path}"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    result = rag.ingest_pdf(pdf_bytes, "sample_university_policy.pdf")
    assert result["total_chunks"] > 0, "No chunks were extracted from PDF"
    print(f"✓ PDF Ingested: {result['filename']} -> {result['total_chunks']} chunks indexed.")

    # Test Semantic Queries
    queries = [
        "What is the hostel curfew timing?",
        "What are the scholarship criteria for JEE Main?",
        "What is the minimum attendance requirement?",
    ]

    for q in queries:
        matches = rag.query_documents(q, n_results=2)
        assert len(matches) > 0, f"No matches found for query '{q}'"
        print(f"✓ Query '{q}' -> Top Match ({int(matches[0]['similarity_score']*100)}%): {matches[0]['text'][:80]}...")


def test_llm_tool_dispatch():
    print("\n--- 3. Testing LLM Tool Execution Dispatcher ---")
    llm = LLM()

    # Test DB Tool Call
    db_res = llm._execute_tool("lookup_student", {"identifier": "Dev Shah"})
    assert "Dev Shah" in db_res, f"Unexpected DB tool result: {db_res}"
    print(f"✓ LLM DB Tool Dispatch ('lookup_student'): {db_res}")

    # Test RAG Tool Call
    rag_res = llm._execute_tool("search_university_docs", {"query": "scholarships for toppers"})
    assert "retrieved_context" in rag_res, f"Unexpected RAG tool result: {rag_res}"
    print(f"✓ LLM RAG Tool Dispatch ('search_university_docs'): Context successfully retrieved.")


def test_fastapi_endpoints():
    print("\n--- 4. Testing FastAPI REST Endpoints ---")
    client = TestClient(app)

    # Health & System Status
    r = client.get("/api/system/status")
    assert r.status_code == 200, f"Status API failed: {r.text}"
    print(f"✓ GET /api/system/status -> DB: {r.json()['database']['mode']}, Vector Store: {r.json()['vector_store']['indexed_documents']} docs.")

    # Dashboard
    r = client.get("/api/data/dashboard")
    assert r.status_code == 200, f"Dashboard API failed: {r.text}"
    print(f"✓ GET /api/data/dashboard -> Students: {r.json()['total_students']}, Programs: {r.json()['total_programs']}.")

    # Document List & Chunks
    r = client.get("/api/documents")
    assert r.status_code == 200, f"Documents list API failed: {r.text}"
    docs = r.json()["documents"]
    print(f"✓ GET /api/documents -> {len(docs)} documents found in collection.")
    if docs:
        doc_id = docs[0]["doc_id"]
        r_chunks = client.get(f"/api/documents/{doc_id}/chunks")
        assert r_chunks.status_code == 200, f"Chunks API failed: {r_chunks.text}"
        print(f"✓ GET /api/documents/{doc_id}/chunks -> {len(r_chunks.json()['chunks'])} chunks retrieved.")

    # Document Search
    r = client.post("/api/documents/search", json={"query": "attendance", "n_results": 2})
    assert r.status_code == 200, f"Search API failed: {r.text}"
    print(f"✓ POST /api/documents/search -> {len(r.json()['results'])} matching chunks returned.")

    # Add Student Record CRUD
    new_student = {
        "student_id": "STU104",
        "name": "Priya Sharma",
        "department_id": "CSE",
        "semester": 6,
        "parent_phone": "+919876543299",
        "marks": [{"subject": "Cloud Computing", "marks_obtained": 96, "max_marks": 100, "grade": "A+"}],
        "attendance": [{"subject": "Cloud Computing", "total_classes": 40, "classes_attended": 39}],
    }
    r_add = client.post("/api/data/students", json=new_student)
    assert r_add.status_code == 200, f"Add student API failed: {r_add.text}"
    print(f"✓ POST /api/data/students -> Successfully added {new_student['name']} ({new_student['student_id']}).")

    # Verify newly added student via DB lookup
    db = Database()
    s_verify = db.lookup_student("Priya Sharma")
    assert s_verify is not None and s_verify["student_id"] == "STU104", "Failed to lookup newly created student"
    print(f"✓ Verified New Student in DB: {s_verify['name']} (Sem {s_verify['semester']} {s_verify['department_name']})")


if __name__ == "__main__":
    print("==================================================")
    print(" 🚀 RUNNING VOICE AGENT & RAG VERIFICATION SUITE  ")
    print("==================================================")
    test_database()
    test_rag()
    test_llm_tool_dispatch()
    test_fastapi_endpoints()
    print("\n==================================================")
    print(" ✨ ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!  ")
    print("==================================================")
