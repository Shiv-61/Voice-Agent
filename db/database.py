"""
Database layer for University Admission & Student Info Agent.
Supports PostgreSQL (via psycopg2) with fallback to SQLite for local development.
"""

import os
import sqlite3
import config

try:
    import psycopg2
    import psycopg2.extras
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False


class Database:
    def __init__(self):
        self.use_sqlite = False
        self.conn = None

        if PSYCOPG2_AVAILABLE:
            try:
                self.conn = psycopg2.connect(config.DATABASE_URL)
                self.conn.autocommit = True
                print("[db] Connected to PostgreSQL database.")
                return
            except Exception as e:
                print(f"[db] PostgreSQL connection failed ({e}). Falling back to SQLite.")

        # Fallback to local SQLite DB
        self.use_sqlite = True
        sqlite_db_path = os.path.join(os.path.dirname(__file__), "university.db")
        self.conn = sqlite3.connect(sqlite_db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_sqlite_schema()
        print("[db] Connected to SQLite database.")

    def _init_sqlite_schema(self):
        """Initializes local SQLite schema and populates seed data if empty."""
        schema_file = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_file):
            with open(schema_file, "r", encoding="utf-8") as f:
                sql_script = f.read()
            # Clean up Postgres specific syntax for SQLite compatibility
            sql_script = sql_script.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
            sql_script = sql_script.replace("ON CONFLICT DO NOTHING", "-- ON CONFLICT IGNORED")
            cursor = self.conn.cursor()
            cursor.executescript(sql_script)
            self.conn.commit()

    def _execute_query(self, query: str, params: tuple = ()) -> list[dict]:
        """Execute query and return list of dictionaries."""
        cursor = self.conn.cursor()
        try:
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
        """Lookup student by ID or Name."""
        query = """
            SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            WHERE LOWER(s.student_id) = LOWER(?) OR LOWER(s.name) LIKE LOWER(?)
            LIMIT 1
        """ if self.use_sqlite else """
            SELECT s.student_id, s.name, d.department_name, s.semester, s.parent_phone
            FROM students s
            JOIN departments d ON s.department_id = d.department_id
            WHERE LOWER(s.student_id) = LOWER(%s) OR LOWER(s.name) LIKE LOWER(%s)
            LIMIT 1
        """
        pattern = f"%{identifier}%"
        results = self._execute_query(query, (identifier, pattern))
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
