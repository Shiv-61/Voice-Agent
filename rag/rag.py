"""
RAG (Retrieval-Augmented Generation) Vector Store.
Ingests, chunks, embeds, and queries university documents (PDFs) using ChromaDB & PyPDF.
"""

import hashlib
import io
import os
import re
import time
from typing import Any

import pypdf
import chromadb
from chromadb.config import Settings


class RAGStore:
    def __init__(self, persist_dir: str | None = None):
        if persist_dir is None:
            persist_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "db", "chroma_db")
        os.makedirs(persist_dir, exist_ok=True)

        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="university_documents",
            metadata={"description": "University admission brochures, rules, policies, and FAQs"},
        )
        print(f"[rag] Vector Store initialized at '{persist_dir}'. Active chunks: {self.collection.count()}")

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
        """Splits text into overlapping chunks respecting sentence boundaries."""
        # Clean extra whitespaces
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        sentences = re.split(r"(?<=[.!?।\n])\s+", cleaned_text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current_len + len(sentence) > chunk_size and current_chunk:
                chunk_str = " ".join(current_chunk)
                chunks.append(chunk_str)
                # Keep overlap from the end of current_chunk
                overlap_tokens = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    if overlap_len + len(s) <= overlap:
                        overlap_tokens.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current_chunk = overlap_tokens
                current_len = overlap_len

            current_chunk.append(sentence)
            current_len += len(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def ingest_pdf(self, file_source: str | bytes, filename: str) -> dict[str, Any]:
        """
        Extracts text from a PDF file or bytes, chunks it, and indexes it into ChromaDB.
        """
        if isinstance(file_source, bytes):
            reader = pypdf.PdfReader(io.BytesIO(file_source))
        else:
            reader = pypdf.PdfReader(file_source)

        total_pages = len(reader.pages)
        doc_id = hashlib.md5(f"{filename}_{time.time()}".encode("utf-8")).hexdigest()[:12]
        upload_time = time.strftime("%Y-%m-%d %H:%M:%S")

        all_chunks = []
        all_ids = []
        all_metadatas = []

        chunk_counter = 0
        for page_idx, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                continue

            page_chunks = self._chunk_text(page_text)
            for chunk in page_chunks:
                chunk_counter += 1
                chunk_id = f"{doc_id}_p{page_idx}_c{chunk_counter}"
                all_ids.append(chunk_id)
                all_chunks.append(chunk)
                all_metadatas.append({
                    "doc_id": doc_id,
                    "filename": filename,
                    "page": page_idx,
                    "chunk_index": chunk_counter,
                    "upload_time": upload_time,
                })

        if all_chunks:
            self.collection.add(
                ids=all_ids,
                documents=all_chunks,
                metadatas=all_metadatas,
            )
            print(f"[rag] Successfully ingested '{filename}': {len(all_chunks)} chunks across {total_pages} pages.")

        return {
            "doc_id": doc_id,
            "filename": filename,
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "upload_time": upload_time,
            "status": "indexed" if all_chunks else "empty",
        }

    def query_documents(self, query: str, n_results: int = 3) -> list[dict[str, Any]]:
        """
        Queries the vector store for the most relevant document chunks.
        """
        if not query.strip() or self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )

        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": round(max(0.0, 1.0 - (dist if dist is not None else 0.0)), 3),
                })

        return formatted_results

    def list_documents(self) -> list[dict[str, Any]]:
        """
        Lists all unique documents indexed in the vector store with summary metrics.
        """
        total = self.collection.count()
        if total == 0:
            return []

        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])

        docs_map: dict[str, dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue

            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "Unknown Document"),
                    "upload_time": meta.get("upload_time", "N/A"),
                    "total_chunks": 0,
                    "max_page": 0,
                }
            docs_map[doc_id]["total_chunks"] += 1
            docs_map[doc_id]["max_page"] = max(docs_map[doc_id]["max_page"], meta.get("page", 1))

        return list(docs_map.values())

    def get_document_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        """Retrieves all indexed chunks, pages, and metadata for a specific document."""
        try:
            data = self.collection.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
            docs = data.get("documents", [])
            metas = data.get("metadatas", [])
            ids = data.get("ids", [])

            chunks = []
            for chunk_id, doc, meta in zip(ids, docs, metas):
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": doc,
                    "page": meta.get("page", 1),
                    "chunk_index": meta.get("chunk_index", 1),
                    "filename": meta.get("filename", "Unknown Document"),
                })
            # Sort chunks by page and index
            chunks.sort(key=lambda c: (c["page"], c["chunk_index"]))
            return chunks
        except Exception as e:
            print(f"[rag] Failed to get chunks for '{doc_id}': {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """Deletes all chunks associated with a specific document ID."""
        try:
            self.collection.delete(where={"doc_id": doc_id})
            print(f"[rag] Deleted document ID '{doc_id}' from vector store.")
            return True
        except Exception as e:
            print(f"[rag] Failed to delete document '{doc_id}': {e}")
            return False
