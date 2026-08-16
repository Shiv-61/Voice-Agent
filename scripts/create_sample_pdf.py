"""
Minimal pure-Python PDF generator producing valid PDF 1.4 documents without third-party dependencies.
"""

def generate_pdf(text_pages: list[str], filename: str):
    """Generates a valid PDF 1.4 file with standard Helvetica font."""
    objects = []
    
    # 1: Catalog, 2: Pages, then Page objects and Content objects
    num_pages = len(text_pages)
    
    # Objects layout:
    # 1: Catalog
    # 2: Pages
    # 3: Font
    # For each page i (0-indexed):
    #   Page object: 4 + 2*i
    #   Content object: 4 + 2*i + 1
    
    font_obj_num = 3
    page_obj_nums = [4 + 2 * i for i in range(num_pages)]
    content_obj_nums = [4 + 2 * i + 1 for i in range(num_pages)]
    
    # Object 1: Catalog
    catalog = f"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    
    # Object 2: Pages
    kids_str = " ".join([f"{num} 0 R" for num in page_obj_nums])
    pages_obj = f"2 0 obj\n<< /Type /Pages /Kids [{kids_str}] /Count {num_pages} >>\nendobj\n"
    
    # Object 3: Font
    font_obj = f"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    
    body_parts = [catalog, pages_obj, font_obj]
    
    for i, page_text in enumerate(text_pages):
        page_num = page_obj_nums[i]
        content_num = content_obj_nums[i]
        
        # Build stream content
        stream_lines = ["BT", "/F1 11 Tf", "50 740 Td", "14 TL"]
        lines = page_text.split("\n")
        for line in lines:
            # Escape parenthesis
            escaped_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream_lines.append(f"({escaped_line}) '")
        stream_lines.append("ET")
        
        stream_data = "\n".join(stream_lines)
        stream_len = len(stream_data.encode("utf-8"))
        
        page_def = (
            f"{page_num} 0 obj\n"
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_num} 0 R >>\n"
            f"endobj\n"
        )
        
        content_def = (
            f"{content_num} 0 obj\n"
            f"<< /Length {stream_len} >>\n"
            f"stream\n"
            f"{stream_data}\n"
            f"endstream\n"
            f"endobj\n"
        )
        
        body_parts.append(page_def)
        body_parts.append(content_def)
    
    # Assemble PDF
    header = "%PDF-1.4\n"
    offsets = [0]
    current_pos = len(header.encode("utf-8"))
    
    for part in body_parts:
        offsets.append(current_pos)
        current_pos += len(part.encode("utf-8"))
    
    total_objects = len(body_parts) + 1
    xref = f"xref\n0 {total_objects}\n0000000000 65535 f \n"
    for off in offsets[1:]:
        xref += f"{off:010d} 00000 n \n"
    
    trailer = (
        f"trailer\n"
        f"<< /Size {total_objects} /Root 1 0 R >>\n"
        f"startxref\n"
        f"{current_pos}\n"
        f"%%EOF\n"
    )
    
    pdf_bytes = header.encode("utf-8") + "".join(body_parts).encode("utf-8") + xref.encode("utf-8") + trailer.encode("utf-8")
    with open(filename, "wb") as f:
        f.write(pdf_bytes)
    print(f"[pdf-generator] Successfully generated binary PDF '{filename}' ({len(pdf_bytes)} bytes)")


if __name__ == "__main__":
    pages = [
        (
            "Apex Institute of Technology & Science - University Prospectus 2026\n\n"
            "1. Merit-Based Scholarships:\n"
            "- Students with JEE Main percentile above 95 receive a 50% tuition fee waiver for all four years.\n"
            "- State Board Toppers (Top 10 rank holders) receive a 100% full tuition scholarship including hostel accommodation.\n"
            "- Need-Based Financial Aid: Families with annual income below INR 3 Lakhs can apply for the Vidya Samman Grant covering up to 75% of academic expenses.\n\n"
            "2. Admission Counseling & Documents Required:\n"
            "- Class 10th and 12th original mark sheets.\n"
            "- Transfer Certificate (TC) and Migration Certificate.\n"
            "- Valid Government ID Proof (Aadhaar Card / Passport).\n"
            "- Category / Caste certificate if applying under reserved quotas."
        ),
        (
            "Campus Life, Hostel Code of Conduct & Facilities\n\n"
            "1. Hostel Timings & Curfew Policy:\n"
            "- All student hostels (Aryabhata Boys Hostel and Gargi Girls Hostel) have an evening in-time curfew of 9:30 PM on weekdays.\n"
            "- On Saturdays and Sundays, curfew is extended up to 10:30 PM with prior warden permission.\n"
            "- Night-out passes require an online request submitted via the Student ERP at least 24 hours in advance, verified by parent SMS OTP.\n\n"
            "2. Library & Central Computing Facilities:\n"
            "- Central Library is open 24x7 during end-semester examination periods and from 8:00 AM to 11:00 PM on regular days.\n"
            "- High-speed 1 Gbps Wi-Fi is available across all academic blocks, hostels, and cafeteria zones.\n\n"
            "3. Anti-Ragging Policy:\n"
            "- Zero-tolerance policy against any form of ragging or harassment. Helpline: 1800-180-5522."
        ),
        (
            "Academic Regulations & Evaluation Scheme\n\n"
            "1. Attendance Requirement:\n"
            "- Minimum mandatory attendance is 75% per subject to appear in End-Semester Examinations.\n"
            "- Medical condonation of up to 10% is permitted upon submission of government hospital certificates within 7 days of absence.\n\n"
            "2. Grading System & SGPA/CGPA Calculation:\n"
            "- Grade A+: 90 to 100 marks (Grade Point 10)\n"
            "- Grade A: 80 to 89 marks (Grade Point 9)\n"
            "- Grade B+: 70 to 79 marks (Grade Point 8)\n"
            "- Grade B: 60 to 69 marks (Grade Point 7)\n"
            "- Passing minimum is 40 marks in each subject (Grade C).\n\n"
            "3. Placement Eligibility Criteria:\n"
            "- Minimum CGPA of 6.5 with no active backlogs is mandatory to participate in campus placement drives starting from Semester 7."
        )
    ]
    generate_pdf(pages, "sample_university_policy.pdf")
