"""
Database layer for University Admission & Student Info Agent.
Supports PostgreSQL (via psycopg2) with fallback to SQLite for local development.
"""

import datetime
import json
import os
import re
import sqlite3
import config

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


INDIC_NAME_MAP = {
    # Gujarati
    "આરવ": "Aarav", "પટેલ": "Patel",
    "રાજ": "Raj", "મહેતા": "Mehta",
    "રિયા": "Riya", "શર્મા": "Sharma",
    "દેવ": "Dev", "શાહ": "Shah",
    "પ્રિયા": "Priya",
    # Hindi
    "आरव": "Aarav", "पटेल": "Patel",
    "राज": "Raj", "मेहता": "Mehta",
    "रिया": "Riya", "शर्मा": "Sharma",
    "देव": "Dev", "शाह": "Shah",
    "प्रिया": "Priya",
}


def transliterate_student_name(identifier: str) -> str:
    """Translates common Hindi and Gujarati student names to Latin script, stripping genitive suffixes."""
    tokens = identifier.strip().split()
    translated = []
    for t in tokens:
        clean_t = re.sub(r'[\?\.\,\!\:\;]+$', '', t)
        base_t = re.sub(r'(ની|ના|નો|નું|ને|જી|जी|ભાઈ|બહેન|બેન)$', '', clean_t)
        translated.append(INDIC_NAME_MAP.get(clean_t, INDIC_NAME_MAP.get(base_t, t)))
    return " ".join(translated)


import threading

class Database:
    def __init__(self):
        self.use_sqlite = False
        self.conn = None
        self._lock = threading.Lock()

        if PSYCOPG2_AVAILABLE and config.DATABASE_URL:
            try:
                self.conn = psycopg2.connect(config.DATABASE_URL)
                self.conn.autocommit = True
                print("[db] Connected to PostgreSQL database.")
                self._init_postgres_schema()
                self.close_stale_calls()
                return
            except Exception as e:
                err_str = str(e)
                if 'database "university_agent" does not exist' in err_str:
                    try:
                        from urllib.parse import urlparse, urlunparse
                        parsed = urlparse(config.DATABASE_URL)
                        maint_url = urlunparse(parsed._replace(path="/postgres"))
                        temp_conn = psycopg2.connect(maint_url)
                        temp_conn.autocommit = True
                        with temp_conn.cursor() as cur:
                            cur.execute("CREATE DATABASE university_agent;")
                        temp_conn.close()
                        self.conn = psycopg2.connect(config.DATABASE_URL)
                        self.conn.autocommit = True
                        print("[db] Created and connected to PostgreSQL 'university_agent' database.")
                        self._init_postgres_schema()
                        self.close_stale_calls()
                        return
                    except Exception as ce:
                        print(f"[db] PostgreSQL auto-creation notice: {ce}")
                print(f"[db] PostgreSQL connection failed ({e}). Falling back to SQLite.")

        # Fallback to local SQLite DB
        self.use_sqlite = True
        sqlite_db_path = os.path.join(os.path.dirname(__file__), "university.db")
        self.conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                cur = self.conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL;")
                cur.execute("PRAGMA synchronous=NORMAL;")
                self.conn.commit()
            except Exception as pe:
                print(f"[db] WAL pragma notice: {pe}")
            self._init_sqlite_schema()
            self.close_stale_calls()
        print("[db] Connected to SQLite database (WAL mode enabled).")

    def _init_postgres_schema(self):
        """Initializes PostgreSQL schema and seed data if empty."""
        schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_file):
            try:
                with open(schema_file, "r", encoding="utf-8") as f:
                    sql_script = f.read()
                cursor = self.conn.cursor()
                cursor.execute(sql_script)
                self._migrate_call_logs_columns()
                print("[db] PostgreSQL schema and seed data verified.")
            except Exception as err:
                print(f"[db] PostgreSQL schema init notice: {err}")

    def _init_sqlite_schema(self):
        """Initializes local SQLite schema and populates seed data if empty."""
        schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_file):
            with open(schema_file, "r", encoding="utf-8") as f:
                sql_script = f.read()
            # Clean up Postgres specific syntax for SQLite compatibility
            sql_script = sql_script.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            sql_script = sql_script.replace("ON CONFLICT DO NOTHING", "")
            sql_script = sql_script.replace("INSERT INTO", "INSERT OR IGNORE INTO")
            cursor = self.conn.cursor()
            # Always ensure tables exist; seed rows only on a fresh (empty) DB.
            if "-- Seed Data" in sql_script:
                schema_part, seed_part = sql_script.split("-- Seed Data", 1)
            else:
                schema_part, seed_part = sql_script, ""
            cursor.executescript(schema_part)
            student_count = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
            if seed_part and student_count == 0:
                cursor.executescript(seed_part)
            self.conn.commit()
            self._migrate_call_logs_columns()

    def _migrate_call_logs_columns(self):
        """Ensures modern college CRM columns exist in call_logs (Fix #16, #7)."""
        new_cols = [
            ("intent", "VARCHAR(100)"),
            ("lead_status", "VARCHAR(50)"),
            ("sentiment", "VARCHAR(20)"),
            ("disposition", "VARCHAR(100)"),   # Fix #16
            ("follow_up_action", "TEXT"),         # Fix #16
            ("total_turns", "INT DEFAULT 0"),     # Fix #7
            ("summary", "TEXT"),
        ]
        cursor = self.conn.cursor()
        for col, col_type in new_cols:
            try:
                if self.use_sqlite:
                    cursor.execute(f"ALTER TABLE call_logs ADD COLUMN {col} {col_type}")
                else:
                    cursor.execute(f"ALTER TABLE call_logs ADD COLUMN IF NOT EXISTS {col} {col_type}")
                self.conn.commit()
            except Exception:
                if not self.use_sqlite:
                    try:
                        self.conn.rollback()
                    except Exception:
                        pass

    def _execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute query thread-safely and return list of dictionaries."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                if self.use_sqlite:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                else:
                    cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    cursor.execute(query, params)
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
            except Exception as err:
                print(f"[db] Query error: {err}")
                return []
    # ------------------------------------------------------------------
    # Domain Queries (Tools for Agent)
    # ------------------------------------------------------------------

    def lookup_student(self, identifier: str) -> dict | None:
        """Lookup student by ID or Name (supports English, Hindi, and Gujarati scripts)."""
        clean_id = transliterate_student_name(identifier)
        query = """
            SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            WHERE LOWER(s.student_id) = LOWER(?) OR LOWER(s.name) LIKE LOWER(?) OR LOWER(s.name) LIKE LOWER(?)
            LIMIT 1
        """ if self.use_sqlite else """
            SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            WHERE LOWER(s.student_id) = LOWER(%s) OR LOWER(s.name) LIKE LOWER(%s) OR LOWER(s.name) LIKE LOWER(%s)
            LIMIT 1
        """
        pattern_orig = f"%{identifier}%"
        pattern_clean = f"%{clean_id}%"
        results = self._execute_query(query, (clean_id, pattern_clean, pattern_orig))
        return results[0] if results else None

    def get_student_marks(self, student_id: str) -> list[dict]:
        """Get marks of a student by student_id."""
        query = """
            SELECT subject, marks_obtained, max_marks, grade
            FROM marks
            WHERE LOWER(student_id) = LOWER(?)
        """ if self.use_sqlite else """
            SELECT subject, marks_obtained, max_marks, grade
            FROM marks
            WHERE LOWER(student_id) = LOWER(%s)
        """
        return self._execute_query(query, (student_id,))

    def get_student_attendance(self, student_id: str) -> list[dict]:
        """Get attendance details of a student by student_id."""
        query = """
            SELECT subject, total_classes, classes_attended, attendance_percentage
            FROM attendance
            WHERE LOWER(student_id) = LOWER(?)
        """ if self.use_sqlite else """
            SELECT subject, total_classes, classes_attended, attendance_percentage
            FROM attendance
            WHERE LOWER(student_id) = LOWER(%s)
        """
        return self._execute_query(query, (student_id,))

    def get_placement_stats(self, department: str = "") -> list[dict]:
        """Get placement statistics, optionally filtered by department."""
        if department:
            query = """
                SELECT p.year, d.department_name, p.highest_package_lpa, p.average_package_lpa, p.placement_rate_pct, p.top_recruiters
                FROM placement_stats p
                JOIN departments d ON p.department_id = d.department_id
                WHERE LOWER(d.department_name) LIKE LOWER(?) OR LOWER(d.department_id) = LOWER(?)
            """ if self.use_sqlite else """
                SELECT p.year, d.department_name, p.highest_package_lpa, p.average_package_lpa, p.placement_rate_pct, p.top_recruiters
                FROM placement_stats p
                JOIN departments d ON p.department_id = d.department_id
                WHERE LOWER(d.department_name) LIKE LOWER(%s) OR LOWER(d.department_id) = LOWER(%s)
            """
            pattern = f"%{department}%"
            return self._execute_query(query, (pattern, department))
        else:
            query = """
                SELECT p.year, d.department_name, p.highest_package_lpa, p.average_package_lpa, p.placement_rate_pct, p.top_recruiters
                FROM placement_stats p
                JOIN departments d ON p.department_id = d.department_id
            """
            return self._execute_query(query)

    def get_admission_info(self, program: str = "") -> list[dict]:
        """Get admission details, eligibility, fees, deadlines."""
        if program:
            query = """
                SELECT program, eligibility, fee_per_year, last_date_to_apply
                FROM admission_info
                WHERE LOWER(program) LIKE LOWER(?)
            """ if self.use_sqlite else """
                SELECT program, eligibility, fee_per_year, last_date_to_apply
                FROM admission_info
                WHERE LOWER(program) LIKE LOWER(%s)
            """
            return self._execute_query(query, (f"%{program}%",))
        else:
            query = "SELECT program, eligibility, fee_per_year, last_date_to_apply FROM admission_info"
            return self._execute_query(query)

    def get_departments(self) -> list[dict]:
        """Get all department records."""
        return self._execute_query("SELECT department_id, department_name FROM departments ORDER BY department_id")

    def get_all_student_identifiers(self) -> list[dict]:
        """Returns student_id and name for all enrolled students for fast anticipatory lookup."""
        return self._execute_query("SELECT student_id, name FROM students ORDER BY student_id")

    def add_student(
        self,
        student_id: str,
        name: str,
        department_id: str,
        semester: int,
        parent_phone: str = "",
        marks_list: list[dict] | None = None,
        attendance_list: list[dict] | None = None,
    ) -> bool:
        """Inserts a new student and associated marks/attendance into the database thread-safely."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                # 1. Insert student
                insert_student_sql = """
                    INSERT OR IGNORE INTO students (student_id, name, department_id, semester, parent_phone)
                    VALUES (?, ?, ?, ?, ?)
                """ if self.use_sqlite else """
                    INSERT INTO students (student_id, name, department_id, semester, parent_phone)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (student_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        department_id = EXCLUDED.department_id,
                        semester = EXCLUDED.semester,
                        parent_phone = EXCLUDED.parent_phone
                """
                cursor.execute(insert_student_sql, (student_id, name, department_id, semester, parent_phone))

                # Replace (not append) this student's marks/attendance so
                # re-saving a student never stacks up duplicate rows.
                del_ph = "?" if self.use_sqlite else "%s"
                cursor.execute(f"DELETE FROM marks WHERE student_id = {del_ph}", (student_id,))
                cursor.execute(f"DELETE FROM attendance WHERE student_id = {del_ph}", (student_id,))

                # 2. Insert marks
                if marks_list:
                    for m in marks_list:
                        subject = m.get("subject", "General Subject")
                        obtained = int(m.get("marks_obtained", 85))
                        max_m = int(m.get("max_marks", 100))
                        grade = m.get("grade", "A")

                        insert_marks_sql = """
                            INSERT INTO marks (student_id, subject, marks_obtained, max_marks, grade)
                            VALUES (?, ?, ?, ?, ?)
                        """ if self.use_sqlite else """
                            INSERT INTO marks (student_id, subject, marks_obtained, max_marks, grade)
                            VALUES (%s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_marks_sql, (student_id, subject, obtained, max_m, grade))

                # 3. Insert attendance
                if attendance_list:
                    for a in attendance_list:
                        subject = a.get("subject", "General Subject")
                        total = int(a.get("total_classes", 40))
                        attended = int(a.get("classes_attended", 36))
                        pct = round((attended / total) * 100, 2) if total > 0 else 90.0

                        insert_att_sql = """
                            INSERT INTO attendance (student_id, subject, total_classes, classes_attended, attendance_percentage)
                            VALUES (?, ?, ?, ?, ?)
                        """ if self.use_sqlite else """
                            INSERT INTO attendance (student_id, subject, total_classes, classes_attended, attendance_percentage)
                            VALUES (%s, %s, %s, %s, %s)
                        """
                        cursor.execute(insert_att_sql, (student_id, subject, total, attended, pct))

                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] Add student error: {e}")
                return False

    # ------------------------------------------------------------------
    # Call History Logging
    # ------------------------------------------------------------------

    def log_call_start(self, call_id: str, caller_number: str = "Web", language: str = "en-IN") -> bool:
        """Opens a new call log entry when a voice call begins."""
        started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        query = """
            INSERT INTO call_logs (call_id, caller_number, language, started_at, status)
            VALUES (?, ?, ?, ?, 'ongoing')
        """ if self.use_sqlite else """
            INSERT INTO call_logs (call_id, caller_number, language, started_at, status)
            VALUES (%s, %s, %s, %s, 'ongoing')
        """
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(query, (call_id, caller_number, language, started_at))
                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] log_call_start error: {e}")
                return False

    def log_call_query(self, call_id: str, query_text: str, lang: str = "gu-IN") -> bool:
        """Appends a caller query to the in-memory buffer (Fix #12: no per-turn DB write).

        Note: actual DB flush happens in flush_queries_bulk() at call end.
        This method is kept for backward compatibility; it is now a no-op that
        returns True so callers don't change.
        """
        # Buffering is done in WebVoiceSession; this method is retained for
        # telephony (vobiz) callers that pass through here.
        if not call_id or not (query_text or "").strip():
            return False
        ph = "?" if self.use_sqlite else "%s"
        with self._lock:
            try:
                cursor = self.conn.cursor()
                if self.use_sqlite:
                    cursor.execute(f"SELECT queries_json FROM call_logs WHERE call_id = {ph}", (call_id,))
                    rows = [dict(r) for r in cursor.fetchall()]
                else:
                    cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    cursor.execute(f"SELECT queries_json FROM call_logs WHERE call_id = {ph}", (call_id,))
                    rows = [dict(r) for r in cursor.fetchall()]
                if not rows:
                    return False
                try:
                    queries = json.loads(rows[0].get("queries_json") or "[]")
                except Exception:
                    queries = []
                queries.append({
                    "text": query_text.strip(),
                    "lang": lang,
                    "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })
                cursor = self.conn.cursor()
                cursor.execute(
                    f"UPDATE call_logs SET queries_json = {ph}, total_turns = total_turns + 1 WHERE call_id = {ph}",
                    (json.dumps(queries), call_id),
                )
                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] log_call_query error: {e}")
                return False

    def flush_queries_bulk(self, call_id: str, turns: list[dict]) -> bool:
        """Fix #12: Bulk-writes buffered turn entries at call end — single DB write instead of N.

        Each turn dict: {"text": str, "lang": str, "ts": str, "role": str}
        """
        if not call_id or not turns:
            return False
        ph = "?" if self.use_sqlite else "%s"
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    f"UPDATE call_logs SET queries_json = {ph}, total_turns = {ph} WHERE call_id = {ph}",
                    (json.dumps(turns), len([t for t in turns if t.get("role") == "user"]), call_id),
                )
                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] flush_queries_bulk error: {e}")
                return False

    def log_call_end(self, call_id: str) -> bool:
        """Closes a call log entry, stamping end time and duration."""
        if not call_id:
            return False
        ph = "?" if self.use_sqlite else "%s"
        with self._lock:
            try:
                cursor = self.conn.cursor()
                if self.use_sqlite:
                    cursor.execute(f"SELECT started_at FROM call_logs WHERE call_id = {ph}", (call_id,))
                    rows = [dict(r) for r in cursor.fetchall()]
                else:
                    cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                    cursor.execute(f"SELECT started_at FROM call_logs WHERE call_id = {ph}", (call_id,))
                    rows = [dict(r) for r in cursor.fetchall()]

                if not rows:
                    return False
                try:
                    started_str = rows[0]["started_at"]
                    if started_str.endswith("Z"):
                        started_str = started_str[:-1] + "+00:00"
                    started = datetime.datetime.fromisoformat(started_str)
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=datetime.timezone.utc)
                    now_utc = datetime.datetime.now(datetime.timezone.utc)
                    duration = max(0, int((now_utc - started).total_seconds()))
                except Exception:
                    duration = 0
                ended_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                cursor = self.conn.cursor()
                cursor.execute(
                    f"UPDATE call_logs SET ended_at = {ph}, duration_seconds = {ph}, status = 'completed' WHERE call_id = {ph}",
                    (ended_at, duration, call_id),
                )
                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] log_call_end error: {e}")
                return False

    def close_stale_calls(self) -> None:
        """Marks calls left open (e.g. by a server restart) as interrupted."""
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    "UPDATE call_logs SET ended_at = started_at, duration_seconds = 0, status = 'interrupted' WHERE ended_at IS NULL"
                )
                self.conn.commit()
            except Exception as e:
                print(f"[db] close_stale_calls notice: {e}")

    def update_call_analytics(
        self,
        call_id: str,
        intent: str,
        lead_status: str,
        sentiment: str,
        summary: str,
        disposition: str = "",      # Fix #16: now persisted
        follow_up_action: str = "", # Fix #16: now persisted
    ) -> bool:
        """Updates call log entry with AI-extracted CRM disposition thread-safely."""
        if not call_id:
            return False
        ph = "?" if self.use_sqlite else "%s"
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    f"UPDATE call_logs SET intent={ph}, lead_status={ph}, sentiment={ph}, "
                    f"summary={ph}, disposition={ph}, follow_up_action={ph} WHERE call_id={ph}",
                    (intent, lead_status, sentiment, summary, disposition, follow_up_action, call_id),
                )
                self.conn.commit()
                return True
            except Exception as e:
                print(f"[db] update_call_analytics error: {e}")
                return False

    def get_call_history(self, limit: int = 50) -> list[dict]:
        """Returns call log entries, most recent calls first."""
        ph = "?" if self.use_sqlite else "%s"
        return self._execute_query(
            f"SELECT call_id, caller_number, language, started_at, ended_at, duration_seconds, "
            f"total_turns, queries_json, status, intent, lead_status, sentiment, "
            f"disposition, follow_up_action, summary "
            f"FROM call_logs ORDER BY started_at DESC LIMIT {ph}",
            (limit,),
        )
