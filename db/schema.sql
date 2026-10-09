-- University Database Schema & Seed Data

CREATE TABLE IF NOT EXISTS departments (
    department_id VARCHAR(10) PRIMARY KEY,
    department_name VARCHAR(100) NOT NULL
);

CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id VARCHAR(10) REFERENCES departments(department_id),
    semester INT NOT NULL,
    parent_phone VARCHAR(15)
);

CREATE TABLE IF NOT EXISTS marks (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(20) REFERENCES students(student_id),
    subject VARCHAR(100) NOT NULL,
    marks_obtained INT NOT NULL,
    max_marks INT NOT NULL DEFAULT 100,
    grade VARCHAR(5),
    CONSTRAINT unique_student_subject_marks UNIQUE (student_id, subject)
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(20) REFERENCES students(student_id),
    subject VARCHAR(100) NOT NULL,
    total_classes INT NOT NULL,
    classes_attended INT NOT NULL,
    attendance_percentage NUMERIC(5, 2),
    CONSTRAINT unique_student_subject_att UNIQUE (student_id, subject)
);

CREATE TABLE IF NOT EXISTS placement_stats (
    id SERIAL PRIMARY KEY,
    year INT NOT NULL,
    department_id VARCHAR(10) REFERENCES departments(department_id),
    highest_package_lpa NUMERIC(5, 2),
    average_package_lpa NUMERIC(5, 2),
    placement_rate_pct NUMERIC(5, 2),
    top_recruiters TEXT
);

CREATE TABLE IF NOT EXISTS admission_info (
    id SERIAL PRIMARY KEY,
    program VARCHAR(100) NOT NULL,
    eligibility TEXT NOT NULL,
    fee_per_year VARCHAR(50) NOT NULL,
    last_date_to_apply VARCHAR(50) NOT NULL
);

-- Fix #16: Added disposition and follow_up_action columns
-- Fix #7:  queries_json now stores JSON objects with timestamps
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

-- RAG Knowledge Base Vector Store for PostgreSQL & pgvector (Render Cloud & Local)
CREATE EXTENSION IF NOT EXISTS vector;

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

CREATE INDEX IF NOT EXISTS knowledge_chunks_embedding_idx
ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- Seed Data

INSERT INTO departments (department_id, department_name) VALUES
('CSE', 'Computer Science & Engineering'),
('ECE', 'Electronics & Communication Engineering'),
('MECH', 'Mechanical Engineering')
ON CONFLICT DO NOTHING;

-- Fix #21: Expanded from 3 → 12 students covering all departments, semesters,
-- low-attendance cases, and failing marks for realistic demo scenarios.
INSERT INTO students (student_id, name, department_id, semester, parent_phone) VALUES
('STU101', 'Aarav Patel',    'CSE',  4, '+919876543210'),
('STU102', 'Riya Sharma',    'CSE',  6, '+919876543211'),
('STU103', 'Dev Shah',       'ECE',  4, '+919876543212'),
('STU104', 'Priya Mehta',    'CSE',  2, '+919876543213'),
('STU105', 'Raj Desai',      'MECH', 6, '+919876543214'),
('STU106', 'Ananya Joshi',   'ECE',  2, '+919876543215'),
('STU107', 'Krish Patel',    'CSE',  8, '+919876543216'),
('STU108', 'Dhruv Trivedi',  'MECH', 4, '+919876543217'),
('STU109', 'Sneha Verma',    'CSE',  6, '+919876543218'),
('STU110', 'Mihir Rana',     'ECE',  8, '+919876543219'),
('STU111', 'Diya Kapoor',    'CSE',  4, '+919876543220'),
('STU112', 'Arjun Nair',     'MECH', 2, '+919876543221')
ON CONFLICT DO NOTHING;

INSERT INTO marks (student_id, subject, marks_obtained, max_marks, grade) VALUES
-- STU101
('STU101', 'Data Structures & Algorithms', 88, 100, 'A'),
('STU101', 'Database Management Systems',  92, 100, 'A+'),
('STU101', 'Operating Systems',            79, 100, 'B+'),
-- STU102
('STU102', 'Artificial Intelligence',      95, 100, 'A+'),
('STU102', 'Computer Networks',            86, 100, 'A'),
('STU102', 'Software Engineering',         91, 100, 'A+'),
-- STU103
('STU103', 'Digital Signal Processing',    74, 100, 'B'),
('STU103', 'VLSI Design',                  68, 100, 'B-'),
-- STU104
('STU104', 'Engineering Mathematics',      55, 100, 'C'),
('STU104', 'Programming Fundamentals',     72, 100, 'B'),
-- STU105
('STU105', 'Thermodynamics',               83, 100, 'A'),
('STU105', 'Machine Design',               76, 100, 'B+'),
('STU105', 'Manufacturing Processes',      90, 100, 'A+'),
-- STU106
('STU106', 'Basic Electronics',            81, 100, 'A'),
('STU106', 'Engineering Drawing',          69, 100, 'B-'),
-- STU107
('STU107', 'Compiler Design',              94, 100, 'A+'),
('STU107', 'Cloud Computing',              88, 100, 'A'),
('STU107', 'Cryptography',                 92, 100, 'A+'),
-- STU108 — failing grade scenario
('STU108', 'Fluid Mechanics',              41, 100, 'F'),
('STU108', 'Engineering Mechanics',        58, 100, 'C'),
-- STU109
('STU109', 'Machine Learning',             97, 100, 'A+'),
('STU109', 'Big Data Analytics',           89, 100, 'A'),
-- STU110
('STU110', 'Embedded Systems',             85, 100, 'A'),
('STU110', 'Microprocessors',              78, 100, 'B+'),
-- STU111 — borderline scenario
('STU111', 'Computer Architecture',        62, 100, 'C+'),
('STU111', 'Theory of Computation',        58, 100, 'C'),
-- STU112
('STU112', 'Engineering Physics',          77, 100, 'B+'),
('STU112', 'Workshop Practice',            85, 100, 'A')
ON CONFLICT DO NOTHING;

INSERT INTO attendance (student_id, subject, total_classes, classes_attended, attendance_percentage) VALUES
-- STU101 — good attendance
('STU101', 'Data Structures & Algorithms', 40, 36, 90.00),
('STU101', 'Database Management Systems',  40, 38, 95.00),
('STU101', 'Operating Systems',            40, 32, 80.00),
-- STU102 — excellent
('STU102', 'Artificial Intelligence',      45, 43, 95.55),
('STU102', 'Computer Networks',            45, 41, 91.11),
-- STU103
('STU103', 'Digital Signal Processing',    40, 30, 75.00),
('STU103', 'VLSI Design',                  40, 28, 70.00),
-- STU104 — BELOW 75% threshold scenario
('STU104', 'Engineering Mathematics',      50, 34, 68.00),
('STU104', 'Programming Fundamentals',     50, 40, 80.00),
-- STU105
('STU105', 'Thermodynamics',               42, 38, 90.47),
('STU105', 'Machine Design',               42, 33, 78.57),
-- STU106
('STU106', 'Basic Electronics',            44, 35, 79.54),
-- STU107 — high achiever
('STU107', 'Compiler Design',              40, 40, 100.00),
('STU107', 'Cloud Computing',              40, 38, 95.00),
-- STU108 — critically low attendance
('STU108', 'Fluid Mechanics',              50, 32, 64.00),
('STU108', 'Engineering Mechanics',        50, 36, 72.00),
-- STU109
('STU109', 'Machine Learning',             46, 45, 97.82),
-- STU110
('STU110', 'Embedded Systems',             40, 37, 92.50),
-- STU111 — borderline
('STU111', 'Computer Architecture',        44, 34, 77.27),
('STU111', 'Theory of Computation',        44, 33, 75.00),
-- STU112
('STU112', 'Engineering Physics',          40, 35, 87.50)
ON CONFLICT DO NOTHING;

INSERT INTO placement_stats (year, department_id, highest_package_lpa, average_package_lpa, placement_rate_pct, top_recruiters) VALUES
(2025, 'CSE',  45.00, 12.50, 96.50, 'Google, Microsoft, Amazon, TCS, Infosys'),
(2025, 'ECE',  28.00,  9.20, 91.00, 'Qualcomm, Intel, Samsung, L&T'),
(2025, 'MECH', 18.00,  7.50, 85.00, 'Tata Motors, L&T, Mahindra, Bosch')
ON CONFLICT DO NOTHING;

INSERT INTO admission_info (program, eligibility, fee_per_year, last_date_to_apply) VALUES
('B.Tech Computer Science (CSE)', '10+2 with Physics, Chem, Math (min 60% aggregate) + JEE Main score', '₹2,50,000 / year', '31st July 2026'),
('B.Tech Electronics (ECE)',      '10+2 with PCM (min 55% aggregate)', '₹2,10,000 / year', '31st July 2026'),
('M.Tech Artificial Intelligence','B.Tech/B.E. in relevant field + GATE score', '₹1,80,000 / year', '15th August 2026')
ON CONFLICT DO NOTHING;
