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
    # Gujarati First Names
    "આરવ": "Aarav", "રિયા": "Riya", "અર્જુન": "Arjun", "પ્રિયા": "Priya",
    "રાહુલ": "Rahul", "સ્નેહા": "Sneha", "કરણ": "Karan", "અનન્યા": "Ananya",
    "વિવેક": "Vivek", "નેહા": "Neha", "આદિત્ય": "Aditya", "ઇશા": "Isha", "ઈશા": "Isha",
    "રોહન": "Rohan", "કાવ્યા": "Kavya", "માનવ": "Manav", "પૂજા": "Pooja",
    "ધ્રુવ": "Dhruv", "મીરા": "Meera", "યશ": "Yash", "સિમરન": "Simran",
    # Gujarati Surnames
    "પટેલ": "Patel", "શાહ": "Shah", "મહેતા": "Mehta", "દેસાઈ": "Desai",
    "જોશી": "Joshi", "કુમાર": "Kumar",
    # Hindi First Names
    "आरव": "Aarav", "रिया": "Riya", "अर्जुन": "Arjun", "प्रिया": "Priya",
    "राहुल": "Rahul", "स्नेहा": "Sneha", "करण": "Karan", "अनन्या": "Ananya",
    "विवेक": "Vivek", "नेहा": "Neha", "आदित्य": "Aditya", "ईशा": "Isha", "इशा": "Isha",
    "रोहन": "Rohan", "काव्या": "Kavya", "मानव": "Manav", "पूजा": "Pooja",
    "ध्रुव": "Dhruv", "मीरा": "Meera", "यश": "Yash", "सिमरन": "Simran",
    # Hindi Surnames
    "पटेल": "Patel", "शाह": "Shah", "मेहता": "Mehta", "देसाई": "Desai",
    "जोशी": "Joshi", "कुमार": "Kumar",
    # Additional common names
    "રાજ": "Raj", "દેવ": "Dev", "શર્મા": "Sharma",
    "राज": "Raj", "देव": "Dev", "शर्मा": "Sharma",
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
        self._lock = threading.RLock()

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
        """Initializes PostgreSQL schema, compatibility columns, and core tables if missing."""
        try:
            cursor = self.conn.cursor()

            # 1. Detect existing columns on students table and ensure backward-compatible aliases
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'students';
            """)
            existing_cols = {row[0] for row in cursor.fetchall()}
            if existing_cols:
                if "student_name" in existing_cols and "name" not in existing_cols:
                    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS name VARCHAR(100);")
                if "course" in existing_cols and "department_id" not in existing_cols:
                    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS department_id VARCHAR(50);")
                if "batch" in existing_cols and "semester" not in existing_cols:
                    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS semester INT DEFAULT 4;")
                if "mobile_number" in existing_cols and "parent_phone" not in existing_cols:
                    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS parent_phone VARCHAR(50);")
                if "student_id" not in existing_cols:
                    cursor.execute("ALTER TABLE students ADD COLUMN IF NOT EXISTS student_id VARCHAR(50);")

                cursor.execute("""
                    UPDATE students SET
                        student_id = COALESCE(student_id, 'STU' || id::text),
                        name = COALESCE(name, student_name),
                        department_id = COALESCE(department_id, CASE 
                            WHEN course ILIKE '%computer%' THEN 'CSE'
                            WHEN course ILIKE '%information%' OR course ILIKE '%it%' THEN 'IT'
                            WHEN course ILIKE '%electronic%' OR course ILIKE '%ece%' THEN 'ECE'
                            WHEN course ILIKE '%mech%' THEN 'MECH'
                            WHEN course ILIKE '%civil%' THEN 'CIVIL'
                            WHEN course ILIKE '%elect%' THEN 'ELECT'
                            ELSE course END),
                        semester = COALESCE(semester, 4),
                        parent_phone = COALESCE(parent_phone, mobile_number);
                """)
                cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id);")

            # 2. Ensure core application tables exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS departments (
                    department_id VARCHAR(50) PRIMARY KEY,
                    department_name VARCHAR(100) NOT NULL
                );
                INSERT INTO departments (department_id, department_name) VALUES
                ('CSE', 'Computer Science & Engineering'),
                ('IT', 'Information Technology'),
                ('ECE', 'Electronics & Communication Engineering'),
                ('MECH', 'Mechanical Engineering'),
                ('CIVIL', 'Civil Engineering'),
                ('ELECT', 'Electrical Engineering')
                ON CONFLICT (department_id) DO NOTHING;
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admission_info (
                    id SERIAL PRIMARY KEY,
                    program VARCHAR(150) UNIQUE NOT NULL,
                    eligibility TEXT NOT NULL,
                    fee_per_year VARCHAR(255) NOT NULL,
                    last_date_to_apply VARCHAR(100) NOT NULL
                );
                INSERT INTO admission_info (program, eligibility, fee_per_year, last_date_to_apply) VALUES
                ('B.Tech 1st Year (Admission Year 2023-24)', '10+2 with PCM (min 60% aggregate) + GUJCET / JEE Main score', '₹1,66,950 / year', '31st July 2026'),
                ('B.Tech 2nd Year (Admission Year 2022-23)', 'Completed 1st Year B.Tech / D2D Diploma to Degree', '₹1,52,000 / year', '31st July 2026'),
                ('B.Tech 3rd Year (Admission Year 2021-22)', 'Completed 2nd Year B.Tech', '₹1,52,000 / year', '31st July 2026'),
                ('B.Tech 4th Year (Admission Year 2020-21)', 'Completed 3rd Year B.Tech', '₹1,52,000 / year', '31st July 2026'),
                ('M.Tech 1st Year (Admission Year 2023-24)', 'B.Tech/B.E. in relevant discipline + valid GATE score', '₹55,125 / year', '15th August 2026'),
                ('M.Tech 2nd Year (Admission Year 2022-23)', 'Completed 1st Year M.Tech', '₹52,500 / year', '15th August 2026'),
                ('B.Tech Information Technology (IT)', '10+2 with PCM (min 60% aggregate) + GUJCET / JEE Main', '₹1,66,950 / year (1st yr), ₹1,52,000 / year (2nd-4th yr)', '31st July 2026'),
                ('B.Tech Computer Science (CSE)', '10+2 with PCM (min 60% aggregate) + JEE Main score', '₹1,66,950 / year (1st yr), ₹1,52,000 / year (2nd-4th yr)', '31st July 2026')
                ON CONFLICT (program) DO UPDATE SET
                    eligibility = EXCLUDED.eligibility,
                    fee_per_year = EXCLUDED.fee_per_year,
                    last_date_to_apply = EXCLUDED.last_date_to_apply;
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS placement_stats (
                    id SERIAL PRIMARY KEY,
                    year INT NOT NULL,
                    department_id VARCHAR(50),
                    highest_package_lpa NUMERIC(5, 2),
                    average_package_lpa NUMERIC(5, 2),
                    placement_rate_pct NUMERIC(5, 2),
                    top_recruiters TEXT
                );
                INSERT INTO placement_stats (year, department_id, highest_package_lpa, average_package_lpa, placement_rate_pct, top_recruiters) VALUES
                (2025, 'CSE',  45.00, 12.50, 96.50, 'Google, Microsoft, Amazon, TCS, Infosys'),
                (2025, 'IT',   40.00, 11.80, 95.00, 'Amazon, Oracle, Infosys, Wipro'),
                (2025, 'ECE',  28.00,  9.20, 91.00, 'Qualcomm, Intel, Samsung, L&T'),
                (2025, 'MECH', 18.00,  7.50, 85.00, 'Tata Motors, L&T, Mahindra, Bosch')
                ON CONFLICT DO NOTHING;
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS call_logs (
                    id SERIAL PRIMARY KEY,
                    call_id VARCHAR(50) NOT NULL UNIQUE,
                    caller_number VARCHAR(64) DEFAULT 'Web',
                    language VARCHAR(20) DEFAULT 'gu-IN',
                    started_at VARCHAR(64) NOT NULL,
                    ended_at VARCHAR(64),
                    duration_seconds INT DEFAULT 0,
                    total_turns INT DEFAULT 0,
                    queries_json TEXT DEFAULT '[]',
                    status VARCHAR(30) DEFAULT 'ongoing',
                    intent VARCHAR(100),
                    lead_status VARCHAR(50),
                    sentiment VARCHAR(20),
                    disposition VARCHAR(100),
                    follow_up_action TEXT,
                    summary TEXT
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS marks (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(50),
                    subject VARCHAR(100) NOT NULL,
                    marks_obtained INT NOT NULL,
                    max_marks INT NOT NULL DEFAULT 100,
                    grade VARCHAR(20)
                );
                CREATE TABLE IF NOT EXISTS attendance (
                    id SERIAL PRIMARY KEY,
                    student_id VARCHAR(50),
                    subject VARCHAR(100) NOT NULL,
                    total_classes INT NOT NULL,
                    classes_attended INT NOT NULL,
                    attendance_percentage NUMERIC(5, 2)
                );
            """)

            self._migrate_call_logs_columns()
            print("[db] PostgreSQL schema and compatibility verified.")
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
            SELECT COALESCE(s.student_id, 'STU' || s.id::text) AS student_id,
                   s.id,
                   COALESCE(s.name, s.student_name) AS name,
                   COALESCE(s.student_name, s.name) AS student_name,
                   COALESCE(d.department_name, s.course, s.department_id, 'General') AS department_name,
                   COALESCE(s.course, d.department_name) AS course,
                   COALESCE(s.semester, 4) AS semester,
                   COALESCE(s.batch, '') AS batch,
                   COALESCE(s.parent_phone, s.mobile_number, '') AS parent_phone,
                   COALESCE(s.mobile_number, s.parent_phone, '') AS mobile_number,
                   s.cpi,
                   s.attendance_percentage
            FROM students s
            LEFT JOIN departments d ON s.department_id = d.department_id
            WHERE LOWER(COALESCE(s.student_id, '')) = LOWER(%s)
               OR s.id::text = %s
               OR LOWER(COALESCE(s.name, s.student_name, '')) LIKE LOWER(%s)
               OR LOWER(COALESCE(s.student_name, s.name, '')) LIKE LOWER(%s)
            LIMIT 1
        """
        pattern_orig = f"%{identifier}%"
        pattern_clean = f"%{clean_id}%"
        if self.use_sqlite:
            results = self._execute_query(query, (clean_id, pattern_clean, pattern_orig))
        else:
            raw_num = clean_id.replace("STU", "").replace("stu", "").strip()
            results = self._execute_query(query, (clean_id, raw_num, pattern_clean, pattern_orig))
        if results:
            r = dict(results[0])
            r["student_id"] = str(r.get("student_id") or r.get("id") or "")
            r["name"] = r.get("name") or r.get("student_name") or ""
            r["student_name"] = r["name"]
            r["department_name"] = r.get("department_name") or r.get("course") or ""
            r["course"] = r["department_name"]
            if r.get("cpi") is not None:
                r["cpi"] = float(r["cpi"])
            if r.get("attendance_percentage") is not None:
                r["attendance_percentage"] = float(r["attendance_percentage"])
            return r
        return None

    def get_student_marks(self, student_id: str) -> list[dict]:
        """Get marks of a student by student_id."""
        query = """
            SELECT subject, marks_obtained, max_marks, grade
            FROM marks
            WHERE LOWER(student_id) = LOWER(?)
        """ if self.use_sqlite else """
            SELECT subject, marks_obtained, max_marks, grade
            FROM marks
            WHERE LOWER(student_id) = LOWER(%s) OR student_id = %s
        """
        raw_id = str(student_id).replace("STU", "").replace("stu", "").strip()
        marks = self._execute_query(query, (student_id,) if self.use_sqlite else (student_id, raw_id))
        if marks:
            return marks

        # Fallback to student record CPI if marks table has no rows
        stu = self.lookup_student(student_id)
        if stu and stu.get("cpi") is not None:
            cpi_val = float(stu["cpi"])
            return [{
                "subject": "Cumulative Performance Index (CPI)",
                "marks_obtained": int(round(cpi_val * 10)),
                "max_marks": 100,
                "grade": f"{cpi_val:.2f} CPI",
            }]
        return []

    def get_student_attendance(self, student_id: str) -> list[dict]:
        """Get attendance details of a student by student_id."""
        query = """
            SELECT subject, total_classes, classes_attended, attendance_percentage
            FROM attendance
            WHERE LOWER(student_id) = LOWER(?)
        """ if self.use_sqlite else """
            SELECT subject, total_classes, classes_attended, attendance_percentage
            FROM attendance
            WHERE LOWER(student_id) = LOWER(%s) OR student_id = %s
        """
        raw_id = str(student_id).replace("STU", "").replace("stu", "").strip()
        att = self._execute_query(query, (student_id,) if self.use_sqlite else (student_id, raw_id))
        if att:
            return att

        # Fallback to student record attendance_percentage
        stu = self.lookup_student(student_id)
        if stu and stu.get("attendance_percentage") is not None:
            pct_val = float(stu["attendance_percentage"])
            return [{
                "subject": "Overall Attendance",
                "total_classes": 100,
                "classes_attended": int(round(pct_val)),
                "attendance_percentage": pct_val,
            }]
        return []

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
        query = """
            SELECT student_id, name FROM students ORDER BY student_id
        """ if self.use_sqlite else """
            SELECT COALESCE(student_id, 'STU' || id::text) AS student_id,
                   COALESCE(name, student_name) AS name
            FROM students
            ORDER BY id
        """
        return self._execute_query(query)

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
            ON CONFLICT (call_id) DO NOTHING
        """ if self.use_sqlite else """
            INSERT INTO call_logs (call_id, caller_number, language, started_at, status)
            VALUES (%s, %s, %s, %s, 'ongoing')
            ON CONFLICT (call_id) DO NOTHING
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

    def get_call_queries(self, call_id: str) -> list[dict]:
        """Returns the list of query/turn dictionaries recorded for a given call_id."""
        if not call_id:
            return []
        try:
            with self._lock:
                cursor = self.conn.cursor()
                ph = "?" if self.use_sqlite else "%s"
                cursor.execute(f"SELECT queries_json FROM call_logs WHERE call_id = {ph}", (call_id,))
                row = cursor.fetchone()
                if row and row[0]:
                    raw = row[0]
                    if isinstance(raw, str):
                        return json.loads(raw)
                    elif isinstance(raw, list):
                        return raw
        except Exception as e:
            print(f"[db] get_call_queries notice: {e}")
        return []

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
