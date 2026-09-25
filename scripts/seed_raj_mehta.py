import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from db.database import Database

db = Database()
cur = db.conn.cursor()

# Insert IT department
cur.execute("INSERT OR IGNORE INTO departments (department_id, department_name) VALUES ('IT', 'Information Technology')")

# Insert Raj Mehta
cur.execute("INSERT OR IGNORE INTO students (student_id, name, department_id, semester, parent_phone) VALUES ('STU105', 'Raj Mehta', 'IT', 6, '+919876543299')")

# Insert marks
cur.execute("""
INSERT OR IGNORE INTO marks (student_id, subject, marks_obtained, max_marks, grade) VALUES 
('STU105', 'Cloud Computing', 92, 100, 'A+'),
('STU105', 'Artificial Intelligence', 95, 100, 'A+'),
('STU105', 'Software Engineering', 89, 100, 'A')
""")

# Insert attendance
cur.execute("""
INSERT OR IGNORE INTO attendance (student_id, subject, total_classes, classes_attended, attendance_percentage) VALUES 
('STU105', 'Cloud Computing', 40, 38, 95.0),
('STU105', 'Artificial Intelligence', 42, 40, 95.24),
('STU105', 'Software Engineering', 38, 35, 92.11)
""")

db.conn.commit()

print("✓ Raj Mehta seeded successfully!")
print("English lookup:", db.lookup_student('Raj Mehta'))
print("Gujarati lookup:", db.lookup_student('રાજ મહેતા'))
print("Hindi lookup:", db.lookup_student('राज मेहता'))
print("Aarav Patel (Gujarati):", db.lookup_student('આરવ પટેલ'))
