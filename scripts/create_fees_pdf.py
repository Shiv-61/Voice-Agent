#!/usr/bin/env python3
"""
Generates the official DDU Annual Fees Structure document in both binary PDF and text formats.
"""

import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.create_sample_pdf import generate_pdf


def main():
    target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "documents")
    os.makedirs(target_dir, exist_ok=True)

    pdf_path = os.path.join(target_dir, "DDU_Fees_Structure.pdf")
    txt_path = os.path.join(target_dir, "DDU_Fees_Structure.txt")

    pages = [
        (
            "Dharmsinh Desai University (DDU), Faculty of Technology\n"
            "Official Annual Fee Structure for B.Tech & M.Tech Programs\n\n"
            "1. Comprehensive Annual Fee Table:\n"
            "Course / Program                                   B Tech         M Tech\n"
            "-------------------------------------------------------------------------\n"
            "1st year (Admission Year 2023-24)                  1,66,950       55,125\n"
            "2nd year (Admission Year 2022-23)                  1,52,000       52,500\n"
            "3rd year (Admission Year 2021-22)                  1,52,000       -\n"
            "4th year (Admission Year 2020-21)                  1,52,000       -\n\n"
            "2. B.Tech Tuition Fee Breakdown (All Engineering Branches):\n"
            "- First Year (Admission Year 2023-24): Rs. 1,66,950 per annum\n"
            "  (One Lakh Sixty-Six Thousand Nine Hundred Fifty Rupees)\n"
            "- Second Year (Admission Year 2022-23): Rs. 1,52,000 per annum\n"
            "  (One Lakh Fifty-Two Thousand Rupees)\n"
            "- Third Year (Admission Year 2021-22): Rs. 1,52,000 per annum\n"
            "  (One Lakh Fifty-Two Thousand Rupees)\n"
            "- Fourth Year (Admission Year 2020-21): Rs. 1,52,000 per annum\n"
            "  (One Lakh Fifty-Two Thousand Rupees)\n\n"
            "Applicable Branches: Information Technology (IT), Computer Science (CSE),\n"
            "Electronics & Communication (ECE), Mechanical Engineering (MECH)."
        ),
        (
            "Dharmsinh Desai University (DDU), Faculty of Technology\n"
            "M.Tech Tuition Fee Breakdown & Payment Guidelines\n\n"
            "1. M.Tech Program Fee Structure:\n"
            "- First Year M.Tech (Admission Year 2023-24): Rs. 55,125 per annum\n"
            "  (Fifty-Five Thousand One Hundred Twenty-Five Rupees)\n"
            "- Second Year M.Tech (Admission Year 2022-23): Rs. 52,500 per annum\n"
            "  (Fifty-Two Thousand Five Hundred Rupees)\n"
            "- Third & Fourth Year: Not Applicable (-) as M.Tech is a 2-year postgraduate program.\n\n"
            "2. Admission & Payment Rules:\n"
            "- Fees are payable on an annual basis prior to semester commencement.\n"
            "- Mode of payment: Online via Student Portal (Netbanking / UPI / Debit / Credit Cards)\n"
            "  or offline via Bank Challan at designated bank counters.\n"
            "- Application deadline for academic admission: 31 July 2026.\n"
            "- For queries contact University Accounts Office: 0268-2520502 or info@ddu.ac.in."
        )
    ]

    generate_pdf(pages, pdf_path)

    full_text = "\n\n".join(pages)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"[create_fees] Successfully wrote text format '{txt_path}' ({len(full_text)} chars)")


if __name__ == "__main__":
    main()
