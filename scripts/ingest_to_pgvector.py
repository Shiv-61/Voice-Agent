#!/usr/bin/env python3
"""
Production PGVector & ChromaDB Knowledge Base Ingestion Script.
Takes reference from /home/shiv-gowda/Downloads/temp/ingest.py to ingest documents into:
1. PostgreSQL 'knowledge_chunks' table using pgvector
2. ChromaDB vector collection (for local & offline fallback)

Usage:
    # Ingest default attached university files (BTech IT 2026-2030 Syllabus & Batch 2026 Highlights):
    python scripts/ingest_to_pgvector.py

    # Ingest a specific PDF or text file:
    python scripts/ingest_to_pgvector.py "path/to/custom_document.pdf"
"""

import argparse
import hashlib
import json
import os
import sys
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
import pypdf
from sentence_transformers import SentenceTransformer

from rag import RAGStore

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DEFAULT_DOCS = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents", "BTech_IT_2026-2030_Syllabus_File.pdf"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents", "Highlights_of_Batch_2026.pdf"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents", "DDU_Fees_Structure.pdf"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_university_policy.pdf"),
]


def setup_pgvector_table(conn):
    """Ensures pgvector extension and knowledge_chunks table exist in PostgreSQL."""
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_chunks (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            source VARCHAR(255) NOT NULL,
            category VARCHAR(100) DEFAULT 'general',
            page INT DEFAULT 1,
            chunk_index INT DEFAULT 1,
            metadata JSONB DEFAULT '{}',
            embedding vector(384)
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
        ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
    """)
    conn.commit()


def extract_and_chunk_pdf(pdf_path: str, rag_store: RAGStore) -> list[dict]:
    """Extracts text from PDF and segments into semantic chunks with metadata."""
    filename = os.path.basename(pdf_path)
    lower_fn = filename.lower()

    if any(k in lower_fn for k in ["syllabus", "curriculum", "btech", "course"]):
        category = "college_curriculum"
    elif any(k in lower_fn for k in ["highlight", "placement", "batch", "recruit"]):
        category = "placements_and_achievements"
    elif any(k in lower_fn for k in ["policy", "admission", "hostel", "fee"]):
        category = "admissions_and_campus"
    else:
        category = "general_campus"

    reader = pypdf.PdfReader(pdf_path)
    chunks_list = []
    chunk_counter = 0
    doc_id = f"doc_{hashlib.md5(filename.encode()).hexdigest()[:12]}"

    print(f"📖 Processing '{filename}' ({len(reader.pages)} pages)...")

    # Inject high-density semantic chunks for Highlights of Batch 2026
    if any(k in lower_fn for k in ["highlight", "2026"]) and any(k in lower_fn for k in ["batch", "highlight", "placement"]):
        specialized = rag_store._get_highlights_semantic_chunks()
        for item in specialized:
            chunk_counter += 1
            p_num = item.get("page", 1)
            chunks_list.append({
                "content": item["text"],
                "source": filename,
                "category": category,
                "page": p_num,
                "chunk_index": chunk_counter,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": p_num,
                    "category": category,
                    "chunk_index": chunk_counter,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            })

    # Inject high-density semantic chunks for Fees Structure
    if any(k in lower_fn for k in ["fee", "fees"]):
        specialized = rag_store._get_fees_structure_semantic_chunks()
        for item in specialized:
            chunk_counter += 1
            p_num = item.get("page", 1)
            chunks_list.append({
                "content": item["text"],
                "source": filename,
                "category": category,
                "page": p_num,
                "chunk_index": chunk_counter,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": p_num,
                    "category": category,
                    "chunk_index": chunk_counter,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            })

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        raw_chunks = rag_store._chunk_text(text)
        for chunk in raw_chunks:
            cleaned = chunk.strip()
            if len(cleaned) < 40:
                continue
            chunk_counter += 1
            chunks_list.append({
                "content": cleaned,
                "source": filename,
                "category": category,
                "page": page_idx,
                "chunk_index": chunk_counter,
                "metadata": {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_idx,
                    "category": category,
                    "chunk_index": chunk_counter,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            })

    print(f"   ✓ Generated {len(chunks_list)} semantic chunks from '{filename}'.")
    return chunks_list


def ingest_documents(pdf_paths: list[str]):
    print("=" * 65)
    print("🚀 University Knowledge Base PGVector & ChromaDB Ingestion")
    print("=" * 65)

    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is missing from environment or .env file.")

    rag_store = RAGStore()

    # 1. Extract chunks from all target documents
    all_chunks = []
    for path in pdf_paths:
        if os.path.exists(path):
            all_chunks.extend(extract_and_chunk_pdf(path, rag_store))
        else:
            print(f"⚠️ Warning: File not found at '{path}'")

    if not all_chunks:
        print("❌ No chunks found to ingest. Exiting.")
        return

    print(f"\n📦 Total chunks to embed: {len(all_chunks)}")

    # 2. Load SentenceTransformer embedding model
    print("🧠 Loading SentenceTransformer model ('sentence-transformers/all-MiniLM-L6-v2')...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc["content"] for doc in all_chunks]
    print(f"⚡ Generating 384-dimensional normalized vector embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=32)

    # 3. Ingest into PostgreSQL with pgvector
    print("\n🐘 Connecting to PostgreSQL database...")
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        setup_pgvector_table(conn)

        # Remove existing chunks for these sources to prevent duplicate stale chunks
        sources_to_refresh = list({doc["source"] for doc in all_chunks})
        for src in sources_to_refresh:
            cur = conn.execute("DELETE FROM knowledge_chunks WHERE source = %s", (src,))
            print(f"   • Cleared existing records for source '{src}'.")

        print("💾 Inserting vectors into PostgreSQL table 'knowledge_chunks'...")
        insert_query = """
            INSERT INTO knowledge_chunks
                (content, source, category, page, chunk_index, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        batch_records = []
        for doc, emb in zip(all_chunks, embeddings):
            batch_records.append((
                doc["content"],
                doc["source"],
                doc["category"],
                doc["page"],
                doc["chunk_index"],
                json.dumps(doc["metadata"]),
                emb.tolist(),
            ))

        with conn.cursor() as cur:
            cur.executemany(insert_query, batch_records)
        conn.commit()

        # Query total count
        cur = conn.execute("SELECT COUNT(*) FROM knowledge_chunks;")
        total_pg_chunks = cur.fetchone()[0]
        print(f"✅ Successfully inserted {len(all_chunks)} chunks into PostgreSQL!")
        print(f"📊 PostgreSQL total knowledge_chunks count: {total_pg_chunks}")

    # 4. Synchronize into ChromaDB as local backup / offline fallback
    print("\n🔮 Synchronizing into ChromaDB collection...")
    for src in sources_to_refresh:
        try:
            existing = rag_store.collection.get(where={"filename": src})
            if existing and existing.get("ids"):
                rag_store.collection.delete(ids=existing["ids"])
        except Exception as e:
            print(f"   [chroma notice] {e}")

    # ChromaDB takes doc_ids, documents, metadatas
    chroma_ids = []
    chroma_docs = []
    chroma_metas = []
    for doc in all_chunks:
        h = hashlib.md5((doc["source"] + str(doc["chunk_index"]) + doc["content"][:30]).encode()).hexdigest()[:10]
        c_id = f"pg_{doc['source']}_{doc['chunk_index']}_{h}"
        chroma_ids.append(c_id)
        chroma_docs.append(doc["content"])
        chroma_metas.append(doc["metadata"])

    # Batch add to ChromaDB
    BATCH_SIZE = 100
    for i in range(0, len(chroma_ids), BATCH_SIZE):
        rag_store.collection.add(
            ids=chroma_ids[i:i + BATCH_SIZE],
            documents=chroma_docs[i:i + BATCH_SIZE],
            metadatas=chroma_metas[i:i + BATCH_SIZE],
        )
    print(f"✅ ChromaDB synchronized! Active Chroma chunks: {rag_store.collection.count()}")

    # 5. Verification semantic search queries
    print("\n" + "=" * 65)
    print("🔍 Testing Vector Retrieval on Ingested Data:")
    print("=" * 65)
    test_queries = [
        "What is the syllabus of B.Tech IT Semester 1 Mathematics?",
        "What were the placement highlights and highest package of 2026 batch?",
        "Which students won the Sui Overflow hackathon in Greece?",
        "What are the professional elective courses in semester 5?",
    ]

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        for q in test_queries:
            q_emb = model.encode([q], normalize_embeddings=True)[0]
            cur = conn.execute("""
                SELECT source, page, 1 - (embedding <=> %s::vector) AS score, content
                FROM knowledge_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
            """, (q_emb.tolist(), q_emb.tolist()))
            row = cur.fetchone()
            if row:
                src, pg, score, content = row
                snip = content[:140].replace("\n", " ")
                print(f"\n❓ Q: \"{q}\"")
                print(f"   🎯 Match: [{src} (Page {pg})] (Score: {score:.3f})")
                print(f"   📄 Snippet: \"{snip}...\"")

    print("\n" + "=" * 65)
    print("🎉 Ingestion complete! Embeddings ready for deployment on Render.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Ingest university documents into PostgreSQL pgvector & ChromaDB")
    parser.add_argument("files", nargs="*", default=[], help="Optional PDF files to ingest (defaults to attached files)")
    args = parser.parse_args()

    files = args.files if args.files else DEFAULT_DOCS
    ingest_documents(files)


if __name__ == "__main__":
    main()
