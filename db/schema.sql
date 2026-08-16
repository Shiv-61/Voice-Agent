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
    grade VARCHAR(5)
);

CREATE TABLE IF NOT EXISTS attendance (
    id SERIAL PRIMARY KEY,
    student_id VARCHAR(20) REFERENCES students(student_id),
    subject VARCHAR(100) NOT NULL,
    total_classes INT NOT NULL,
    classes_attended INT NOT NULL,
    attendance_percentage NUMERIC(5, 2)
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

-- Seed Data
INSERT INTO departments (department_id, department_name) VALUES
('CSE', 'Computer Science & Engineering'),
('ECE', 'Electronics & Communication Engineering'),
('MECH', 'Mechanical Engineering')
ON CONFLICT DO NOTHING;

INSERT INTO students (student_id, name, department_id, semester, parent_phone) VALUES
('STU101', 'Aarav Patel', 'CSE', 4, '+919876543210'),
('STU102', 'Riya Sharma', 'CSE', 6, '+919876543211'),
('STU103', 'Dev Shah', 'ECE', 4, '+919876543212')
ON CONFLICT DO NOTHING;

INSERT INTO marks (student_id, subject, marks_obtained, max_marks, grade) VALUES
('STU101', 'Data Structures & Algorithms', 88, 100, 'A'),
('STU101', 'Database Management Systems', 92, 100, 'A+'),
('STU101', 'Operating Systems', 79, 100, 'B+'),
('STU102', 'Artificial Intelligence', 95, 100, 'A+'),
('STU102', 'Computer Networks', 86, 100, 'A'),
('STU103', 'Digital Signal Processing', 74, 100, 'B')
ON CONFLICT DO NOTHING;

INSERT INTO attendance (student_id, subject, total_classes, classes_attended, attendance_percentage) VALUES
('STU101', 'Data Structures & Algorithms', 40, 36, 90.00),
('STU101', 'Database Management Systems', 40, 38, 95.00),
('STU101', 'Operating Systems', 40, 32, 80.00),
('STU102', 'Artificial Intelligence', 45, 43, 95.55),
('STU103', 'Digital Signal Processing', 40, 30, 75.00)
ON CONFLICT DO NOTHING;

INSERT INTO placement_stats (year, department_id, highest_package_lpa, average_package_lpa, placement_rate_pct, top_recruiters) VALUES
(2025, 'CSE', 45.00, 12.50, 96.50, 'Google, Microsoft, Amazon, TCS, Infosys'),
(2025, 'ECE', 28.00, 9.20, 91.00, 'Qualcomm, Intel, Samsung, L&T'),
(2025, 'MECH', 18.00, 7.50, 85.00, 'Tata Motors, L&T, Mahindra, Bosch')
ON CONFLICT DO NOTHING;

INSERT INTO admission_info (program, eligibility, fee_per_year, last_date_to_apply) VALUES
('B.Tech Computer Science (CSE)', '10+2 with Physics, Chem, Math (min 60% aggregate) + JEE Main score', '₹2,50,000 / year', '31st July 2026'),
('B.Tech Electronics (ECE)', '10+2 with PCM (min 55% aggregate)', '₹2,10,000 / year', '31st July 2026'),
('M.Tech Artificial Intelligence', 'B.Tech/B.E. in relevant field + GATE score', '₹1,80,000 / year', '15th August 2026')
ON CONFLICT DO NOTHING;
