"""
RAG (Retrieval-Augmented Generation) Vector Store.
Ingests, chunks, embeds, and queries university documents (PDFs) using ChromaDB & PyPDF.
"""

import hashlib
import io
import json
import os
import re
import time
from typing import Any

import pypdf
import chromadb
from chromadb.config import Settings

import config


class RAGStore:
    def __init__(self, persist_dir: str | None = None):
        if persist_dir is None:
            persist_dir = config.CHROMA_PERSIST_DIR
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
        self.has_pgvector = False
        self._embed_model = None
        self._check_pgvector()
        self._auto_seed_sample_pdf()

    def _check_pgvector(self):
        """Checks if PostgreSQL pgvector knowledge_chunks table is active and populated."""
        if not getattr(config, "DATABASE_URL", None) or "postgres" not in config.DATABASE_URL.lower():
            return
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            with psycopg.connect(config.DATABASE_URL) as conn:
                register_vector(conn)
                cur = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name='knowledge_chunks';")
                if cur.fetchone()[0] > 0:
                    cur2 = conn.execute("SELECT COUNT(*) FROM knowledge_chunks;")
                    cnt = cur2.fetchone()[0]
                    if cnt > 0:
                        self.has_pgvector = True
                        print(f"[rag] Active PGVector Knowledge Base connected! Total chunks: {cnt}")
        except Exception as e:
            print(f"[rag] PGVector connection notice: {e}")

    def _auto_seed_sample_pdf(self):
        """Auto-seeds default university policy document if missing from vector store."""
        try:
            indexed_names = [d.get("filename") for d in self.list_documents()]
            if "sample_university_policy.pdf" not in indexed_names:
                sample_pdf = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "sample_university_policy.pdf",
                )
                if os.path.exists(sample_pdf):
                    with open(sample_pdf, "rb") as f:
                        content = f.read()
                    self.ingest_pdf(content, "sample_university_policy.pdf")
                    print("[rag] Auto-seeded sample university policy document into vector store.")
        except Exception as e:
            print(f"[rag] Auto-seed notice: {e}")


    def _split_sentences(self, text: str) -> list[str]:
        """Splits text into sentences while protecting decimals (2.5) and abbreviations (Dr., B.Tech., Th., Int.)."""
        abbrev = {
            "dr", "prof", "mr", "mrs", "ms", "rs", "b.tech", "m.tech", "ph.d",
            "dept", "univ", "no", "vs", "lpa", "th", "int", "tw", "prac",
            "lect", "tut", "sess", "ext", "sem", "cr",
        }
        pattern = re.compile(r"([.!?।\n]+)(\s+)")
        pos = 0
        sentences = []
        for m in pattern.finditer(text):
            punct = m.group(1)
            punct_idx = m.start(1)
            end_idx = m.end()
            if "." in punct:
                prefix = text[:punct_idx]
                if prefix and prefix[-1].isdigit():
                    continue
                words = prefix.split()
                if words:
                    last_w = re.sub(r"^[^\w]+|[^\w.]+$", "", words[-1].lower())
                    if last_w in abbrev or (len(last_w) == 1 and last_w.isalpha()):
                        continue
            s = text[pos:punct_idx + len(punct)].strip()
            if s:
                sentences.append(s)
            pos = end_idx
        rem = text[pos:].strip()
        if rem:
            sentences.append(rem)
        return sentences

    def _chunk_sentences(self, text: str, chunk_size: int = 850, overlap: int = 120) -> list[str]:
        """Splits text into overlapping chunks respecting sentence boundaries."""
        sentences = self._split_sentences(text)
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

    def _chunk_text(self, text: str, chunk_size: int = 850, overlap: int = 120) -> list[str]:
        """Splits text into overlapping chunks respecting sentence and curriculum section boundaries."""
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return []

        # If page contains distinct Semester tables (e.g. course structure pages 2-5)
        if re.search(r"Semester\s*[–\-]\s*[IVX\d]+", cleaned_text, flags=re.I):
            splits = re.split(r"(?=(?:B\.?\s*Tech\s+)?Semester\s*[–\-]\s*[IVX\d]+)", cleaned_text, flags=re.I)
            sem_chunks = []
            prefix = ""
            for s in splits:
                s = s.strip()
                if not s:
                    continue
                if not re.search(r"Semester\s*[–\-]\s*[IVX\d]+", s, flags=re.I):
                    prefix = s + " "
                    continue
                candidate = (prefix + s).strip()
                if len(candidate) <= 1200:
                    sem_chunks.append(candidate)
                else:
                    sem_chunks.extend(self._chunk_sentences(candidate, chunk_size, overlap))
            if sem_chunks:
                return sem_chunks

        return self._chunk_sentences(cleaned_text, chunk_size, overlap)

    def ingest_pdf(self, file_source: str | bytes, filename: str) -> dict[str, Any]:
        """
        Extracts text from a PDF file or bytes, chunks it, and indexes it into ChromaDB.
        """
        if isinstance(file_source, bytes):
            raw_bytes = file_source
            reader = pypdf.PdfReader(io.BytesIO(file_source))
        else:
            with open(file_source, "rb") as f:
                raw_bytes = f.read()
            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))

        total_pages = len(reader.pages)
        content_hash = hashlib.md5(raw_bytes).hexdigest()[:12]
        doc_id = f"doc_{content_hash}"
        upload_time = time.strftime("%Y-%m-%d %H:%M:%S")

        lower_fn = filename.lower()
        if any(k in lower_fn for k in ["curriculum", "syllabus", "course", "semester", "credit", "scheme", "regulation", "branch", "subject"]):
            category = "college_curriculum"
        elif any(k in lower_fn for k in ["student", "mark", "grade", "attendance", "result", "transcript", "batch", "stu", "roll"]):
            category = "student_records"
        elif any(k in lower_fn for k in ["admission", "fee", "hostel", "placement"]):
            category = "admissions_and_campus"
        else:
            category = "general_campus"

        # Content-based classification fallback if filename is generic (e.g. DDU_Doc.pdf)
        if category == "general_campus" and reader.pages:
            try:
                first_page_text = (reader.pages[0].extract_text() or "").lower()[:800]
                if any(k in first_page_text for k in ["syllabus", "credit", "semester", "subject", "curriculum", "course structure", "scheme"]):
                    category = "college_curriculum"
                elif any(k in first_page_text for k in ["attendance", "marks", "grade", "result", "roll no", "enrollment", "student record"]):
                    category = "student_records"
                elif any(k in first_page_text for k in ["admission", "fee", "hostel", "placement", "eligibility", "package"]):
                    category = "admissions_and_campus"
            except Exception as cat_err:
                print(f"[rag] Content-based categorization notice: {cat_err}")

        # Fix #17: Delete old chunks if same FILENAME was previously indexed
        # (content hash may differ if file was updated/re-saved)
        try:
            existing_by_name = self.collection.get(where={"filename": filename})
            if existing_by_name and existing_by_name.get("ids"):
                old_ids_to_delete = existing_by_name["ids"]
                # If the doc_id matches exactly (same content), skip re-ingestion
                existing_doc_ids = {m.get("doc_id") for m in (existing_by_name.get("metadatas") or []) if m}
                if doc_id in existing_doc_ids and len(existing_doc_ids) == 1:
                    print(f"[rag] Document '{filename}' (doc_id={doc_id}) is already indexed with {len(old_ids_to_delete)} chunks.")
                    return {
                        "doc_id": doc_id,
                        "filename": filename,
                        "category": category,
                        "total_pages": total_pages,
                        "total_chunks": len(old_ids_to_delete),
                        "status": "already_indexed",
                    }
                # Content has changed — remove stale chunks so they don't pollute retrieval
                print(f"[rag] Removing {len(old_ids_to_delete)} stale chunks for '{filename}' before re-indexing.")
                self.collection.delete(ids=old_ids_to_delete)
        except Exception as dedup_err:
            print(f"[rag] Dedup check notice: {dedup_err}")

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
                    "category": category,
                })

        if all_chunks:
            self.collection.add(
                ids=all_ids,
                documents=all_chunks,
                metadatas=all_metadatas,
            )
            print(f"[rag] Successfully ingested '{filename}' [{category}]: {len(all_chunks)} chunks across {total_pages} pages.")
            self._sync_chunks_to_pgvector(all_chunks, filename, category, doc_id, all_metadatas)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "total_pages": total_pages,
            "total_chunks": len(all_chunks),
            "upload_time": upload_time,
            "status": "indexed" if all_chunks else "empty",
        }

    def _sync_chunks_to_pgvector(
        self, chunks: list[str], filename: str, category: str, doc_id: str, metadatas: list[dict]
    ):
        """Synchronizes ingested chunks and embeddings to PostgreSQL pgvector table 'knowledge_chunks'."""
        if not self.has_pgvector:
            return
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            if self._embed_model is None:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            embs = self._embed_model.encode(chunks, normalize_embeddings=True)
            with psycopg.connect(config.DATABASE_URL) as conn:
                register_vector(conn)
                conn.execute("DELETE FROM knowledge_chunks WHERE source = %s OR metadata->>'doc_id' = %s;", (filename, doc_id))
                insert_query = """
                    INSERT INTO knowledge_chunks
                        (content, source, category, page, chunk_index, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                """
                records = []
                for chunk, meta, emb in zip(chunks, metadatas, embs):
                    records.append((
                        chunk,
                        filename,
                        category,
                        meta.get("page", 1),
                        meta.get("chunk_index", 1),
                        json.dumps(meta),
                        emb.tolist(),
                    ))
                with conn.cursor() as cur:
                    cur.executemany(insert_query, records)
                conn.commit()
                print(f"[rag] Synchronized {len(records)} chunks to PGVector for '{filename}'.")
        except Exception as e:
            print(f"[rag] PGVector sync notice: {e}")

    def ingest_text(
        self, text: str, filename: str, category: str = "general_campus"
    ) -> dict[str, Any]:
        """
        Chunks and indexes raw text or markdown information into ChromaDB.
        Enables adding FAQs, policies, fee updates, or new guidelines without needing a PDF.
        """
        cleaned_text = re.sub(r"\s+", " ", text).strip()
        if not cleaned_text:
            return {"error": "Text is empty", "status": "failed", "total_chunks": 0}

        content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()[:12]
        doc_id = f"doc_{content_hash}"
        upload_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Automatic category detection if general_campus
        lower_fn = (filename + " " + cleaned_text[:200]).lower()
        if category == "general_campus":
            if any(k in lower_fn for k in ["curriculum", "syllabus", "course", "semester", "credit", "scheme", "regulation", "subject"]):
                category = "college_curriculum"
            elif any(k in lower_fn for k in ["student", "mark", "grade", "attendance", "result", "transcript", "batch", "stu", "roll"]):
                category = "student_records"
            elif any(k in lower_fn for k in ["admission", "fee", "hostel", "placement"]):
                category = "admissions_and_campus"

        # Remove previous chunks with identical filename/title
        try:
            existing = self.collection.get(where={"filename": filename})
            if existing and existing.get("ids"):
                old_ids = existing["ids"]
                print(f"[rag] Removing {len(old_ids)} stale chunks for text '{filename}' before re-indexing.")
                self.collection.delete(ids=old_ids)
        except Exception as dedup_err:
            print(f"[rag] Text dedup notice: {dedup_err}")

        chunks = self._chunk_text(cleaned_text)
        all_chunks = []
        all_ids = []
        all_metadatas = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}_c{idx}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadatas.append({
                "doc_id": doc_id,
                "filename": filename,
                "page": 1,
                "chunk_index": idx,
                "upload_time": upload_time,
                "category": category,
            })

        if all_chunks:
            self.collection.add(
                ids=all_ids,
                documents=all_chunks,
                metadatas=all_metadatas,
            )
            print(f"[rag] Successfully ingested text '{filename}' [{category}]: {len(all_chunks)} chunks.")
            self._sync_chunks_to_pgvector(all_chunks, filename, category, doc_id, all_metadatas)

        return {
            "doc_id": doc_id,
            "filename": filename,
            "category": category,
            "total_pages": 1,
            "total_chunks": len(all_chunks),
            "upload_time": upload_time,
            "status": "indexed" if all_chunks else "empty",
        }

    def _query_pgvector(
        self, query: str, n_results: int = 4, min_similarity: float = 0.40
    ) -> list[dict[str, Any]]:
        """Queries PostgreSQL knowledge_chunks using pgvector cosine distance."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
            if self._embed_model is None:
                from sentence_transformers import SentenceTransformer
                self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            q_emb = self._embed_model.encode([query], normalize_embeddings=True)[0]

            with psycopg.connect(config.DATABASE_URL) as conn:
                register_vector(conn)
                cur = conn.execute("""
                    SELECT content, source, category, page, metadata,
                           1 - (embedding <=> %s::vector) AS similarity_score
                    FROM knowledge_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (q_emb.tolist(), q_emb.tolist(), n_results * 2))
                rows = cur.fetchall()

            results = []
            for row in rows:
                content, source, category, page, meta, score = row
                if score < min_similarity:
                    continue
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                meta = meta or {}
                meta.update({"filename": source, "page": page, "category": category})
                results.append({
                    "text": content,
                    "metadata": meta,
                    "similarity_score": round(float(score), 3),
                })
            return results[:n_results]
        except Exception as e:
            print(f"[rag] PGVector query notice: {e}")
            return []

    def query_documents(
        self, query: str, n_results: int = 4, min_similarity: float = 0.50
    ) -> list[dict[str, Any]]:
        """
        Queries the vector store with hybrid semantic similarity and domain keyword boosting.
        Checks active PostgreSQL pgvector collection first, falling back to ChromaDB.
        """
        if not query.strip():
            return []

        # 1. Prefer PostgreSQL pgvector if active (for persistent cloud Render deployment)
        if self.has_pgvector:
            pg_res = self._query_pgvector(query, n_results=n_results, min_similarity=min_similarity)
            if pg_res:
                return pg_res

        if self.collection.count() == 0:
            return []

        # Query a candidate pool to allow intelligent hybrid re-ranking
        candidate_k = min(max(n_results * 4, 16), self.collection.count())
        results = self.collection.query(
            query_texts=[query],
            n_results=candidate_k,
        )

        # Detect if query targets a specific academic semester
        num_to_roman = {
            "1": ["i", "1", "one", "first", "એક", "૧", "વન", "પહેલું", "પહેલા", "પ્રથમ"],
            "2": ["ii", "2", "two", "second", "બે", "૨", "ટુ", "ટૂ", "બીજું", "બીજા"],
            "3": ["iii", "3", "three", "third", "ત્રણ", "૩", "થ્રી", "ત્રીજું", "ત્રીજા"],
            "4": ["iv", "4", "four", "fourth", "ચાર", "૪", "ફોર", "ચોથું", "ચોથા"],
            "5": ["v", "5", "five", "fifth", "પાંચ", "૫", "ફાઈવ"],
            "6": ["vi", "6", "six", "sixth", "છ", "૬", "સિક્સ"],
            "7": ["vii", "7", "seven", "seventh", "સાત", "૭", "સેવન"],
            "8": ["viii", "8", "eight", "eighth", "આઠ", "૮", "એઈટ"],
        }
        target_sem = None
        q_lower = query.lower()
        for s_num, aliases in num_to_roman.items():
            for a in aliases:
                if re.search(r"\b(?:sem|semester)\s*[–\-]*(?:ester)?\s*" + re.escape(a) + r"\b", q_lower) or \
                   re.search(r"\b" + re.escape(a) + r"\s*(?:sem|semester)\b", q_lower) or \
                   re.search(r"સેમેસ્ટર\s*" + re.escape(a), q_lower):
                    target_sem = s_num
                    break
            if target_sem:
                break

        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                if not doc or len(doc.strip()) < 60:
                    continue  # Filter out trivial stub chunks

                d = float(dist) if dist is not None else 0.0
                similarity = round(max(0.0, min(1.0, 1.0 / (1.0 + max(0.0, d)))), 3)

                # Target semester boost / penalty
                if target_sem:
                    doc_lower = doc.lower()
                    target_aliases = num_to_roman[target_sem]
                    has_target = any(
                        re.search(r"(?:semester|sem)\s*[–\-]?\s*" + re.escape(a) + r"\b", doc_lower)
                        for a in target_aliases
                    )
                    if has_target:
                        similarity = min(1.0, similarity + 0.20)
                    else:
                        # Conflicting semester check
                        has_conflict = False
                        for other_num, other_aliases in num_to_roman.items():
                            if other_num != target_sem:
                                if any(re.search(r"(?:semester|sem)\s*[–\-]?\s*" + re.escape(a) + r"\b", doc_lower) for a in other_aliases):
                                    has_conflict = True
                                    break
                        if has_conflict:
                            similarity = max(0.0, similarity - 0.15)

                if similarity < min_similarity:
                    continue

                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity_score": round(similarity, 3),
                })

        # Sort by boosted similarity score descending
        formatted_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return formatted_results[:n_results]

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
                    "category": meta.get("category", "general_campus"),
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
            alt_id = doc_id[4:] if doc_id.startswith("doc_") else f"doc_{doc_id}"
            try:
                self.collection.delete(where={"doc_id": alt_id})
            except Exception:
                pass

            if self.has_pgvector:
                try:
                    import psycopg
                    with psycopg.connect(config.DATABASE_URL) as conn:
                        conn.execute("DELETE FROM knowledge_chunks WHERE metadata->>'doc_id' = %s OR metadata->>'doc_id' = %s OR source = %s;", (doc_id, alt_id, doc_id))
                        conn.commit()
                except Exception as pe:
                    print(f"[rag] PGVector delete notice: {pe}")

            print(f"[rag] Deleted document ID '{doc_id}' from vector store.")
            return True
        except Exception as e:
            print(f"[rag] Failed to delete document '{doc_id}': {e}")
            return False
