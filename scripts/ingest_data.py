#!/usr/bin/env python3
"""
RAG Ingestion Utility Script.
Easily ingest new documents (PDF, TXT, MD) or raw text into the ChromaDB Knowledge Base
and verify semantic search retrieval.

Usage:
    # Ingest a PDF file
    python scripts/ingest_data.py path/to/document.pdf

    # Ingest a text or markdown file
    python scripts/ingest_data.py path/to/policy.txt --category campus

    # Ingest raw text snippet directly
    python scripts/ingest_data.py --text "DDU Hostel curfew for final year students is extended to 11:00 PM." --title "hostel_curfew_update.txt"
"""

import argparse
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import RAGStore


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents or text into the University Voice Agent RAG Vector Store"
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Path to a .pdf, .txt, or .md file to ingest",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Raw text content to ingest directly into RAG",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="",
        help="Title / filename identifier for the ingested content",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="general_campus",
        choices=["college_curriculum", "student_records", "admissions_and_campus", "general_campus"],
        help="Domain category for the document (default: general_campus)",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default="",
        help="Optional test query to verify retrieval after ingestion",
    )
    args = parser.parse_args()

    rag = RAGStore()

    if args.file_path:
        path = os.path.abspath(args.file_path)
        if not os.path.exists(path):
            print(f"❌ Error: File not found at '{path}'")
            sys.exit(1)

        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        print(f"📄 Processing '{filename}' ({os.path.getsize(path)} bytes)...")

        if ext == ".pdf":
            result = rag.ingest_pdf(path, filename)
        elif ext in (".txt", ".md"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            result = rag.ingest_text(content, filename, category=args.category)
        else:
            print(f"❌ Error: Unsupported file format '{ext}'. Supported formats: .pdf, .txt, .md")
            sys.exit(1)

        print("\n✅ Ingestion Complete!")
        print(f"   • Document ID : {result.get('doc_id')}")
        print(f"   • Filename    : {result.get('filename')}")
        print(f"   • Category    : {result.get('category')}")
        print(f"   • Chunks      : {result.get('total_chunks')}")
        print(f"   • Status      : {result.get('status')}")

    elif args.text.strip():
        title = args.title.strip() or "custom_snippet.txt"
        print(f"📝 Ingesting text snippet '{title}' ({len(args.text)} chars)...")
        result = rag.ingest_text(args.text.strip(), title, category=args.category)
        print("\n✅ Ingestion Complete!")
        print(f"   • Document ID : {result.get('doc_id')}")
        print(f"   • Title       : {result.get('filename')}")
        print(f"   • Category    : {result.get('category')}")
        print(f"   • Chunks      : {result.get('total_chunks')}")

    else:
        print("❌ Error: Must provide either a file path or --text content.")
        parser.print_help()
        sys.exit(1)

    # Optional test verification query
    test_q = args.test_query or (args.text[:50] if args.text else "")
    if test_q:
        print(f"\n🔍 Testing semantic search with query: \"{test_q}\"")
        matches = rag.query_documents(test_q, n_results=2)
        if matches:
            for i, m in enumerate(matches, 1):
                score = m.get("similarity_score", 0)
                snip = m.get("text", "")[:150].replace("\n", " ")
                fn = m.get("metadata", {}).get("filename", "")
                print(f"   {i}. [{fn}] (score: {score}): \"{snip}...\"")
        else:
            print("   (No matching chunks found above threshold)")

    print(f"\n🎉 Active Knowledge Base now contains {rag.collection.count()} indexed chunks.")
    print("Voice agent will immediately answer questions using this new data!")


if __name__ == "__main__":
    main()
