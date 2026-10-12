"""
Comprehensive Test Suite for Multilingual Semester Parser & Semester RAG Retrieval.
Tests:
1. Unit tests for semester and number parser:
   - Digits: 1..8, 'sem 3', 'semester 4'
   - English words & ordinals: 'first', 'second', 'third', '1st', '2nd', '3rd', etc.
   - Roman numerals: 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'
   - Gujarati digits, words, ordinals, and transliterated: '૧'..'૮', 'ત્રણ', 'ત્રીજું', 'સેમેસ્ટર ત્રણ', 'સેમ ૩'
   - Hindi digits, words, ordinals, and transliterated: '१'..'८', 'तीन', 'तीसरे', 'तीसरा सेमेस्टर', 'सेम 3'
2. Regression test set for Semesters 1 to 8:
   - In English, Hindi, and Gujarati phrasing
   - Asserts that 100% of returned chunks have the requested semester!
3. Clarification test:
   - When caller asks for syllabus/semester without specifying a number, returns clarification prompt.
"""

import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.semester_parser import (
    extract_semester,
    is_semester_query,
    normalize_rag_query,
    get_semester_clarification_phrase,
)
from rag import RAGStore


def test_parser_english_digits_and_ordinals():
    cases = [
        ("What is the syllabus for semester 1?", 1),
        ("Tell me about sem 2 subjects", 2),
        ("B.Tech IT 3rd semester subjects", 3),
        ("fourth sem syllabus", 4),
        ("5th semester credits", 5),
        ("subjects in sem-6", 6),
        ("seventh semester courses", 7),
        ("semester 8 industrial internship", 8),
        ("sem 1st", 1),
        ("second sem", 2),
        ("third semester", 3),
        ("eighth sem", 8),
    ]
    for text, expected in cases:
        assert extract_semester(text) == expected, f"Failed for '{text}': got {extract_semester(text)}, expected {expected}"


def test_parser_roman_numerals():
    cases = [
        ("semester I subjects", 1),
        ("sem II syllabus", 2),
        ("B.Tech Sem III", 3),
        ("Semester IV curriculum", 4),
        ("sem V courses", 5),
        ("Semester VI scheme", 6),
        ("sem VII syllabus", 7),
        ("Semester VIII internship", 8),
    ]
    for text, expected in cases:
        assert extract_semester(text) == expected, f"Failed for Roman numeral '{text}': got {extract_semester(text)}, expected {expected}"


def test_parser_gujarati_phrasing():
    cases = [
        ("સેમેસ્ટર ૧ નો સિલેબસ શું છે?", 1),
        ("સેમ ૨ માં કયા વિષયો આવે છે?", 2),
        ("સેમેસ્ટર ત્રણ નો સિલેબસ આપો", 3),
        ("ત્રીજું સેમેસ્ટર વિષયો", 3),
        ("ત્રીજા સેમેસ્ટર ના વિષયો", 3),
        ("ચોથા સેમેસ્ટર નો અભ્યાસક્રમ", 4),
        ("પાંચમું સેમેસ્ટર ક્રેડિટ", 5),
        ("છઠ્ઠા સેમેસ્ટર નો સિલેબસ", 6),
        ("સાતમું સેમેસ્ટર", 7),
        ("આઠમું સેમેસ્ટર ઇન્ટર્નશિપ", 8),
        ("સેમ ૩", 3),
        ("સેમેસ્ટર ૪", 4),
    ]
    for text, expected in cases:
        assert extract_semester(text) == expected, f"Failed for Gujarati '{text}': got {extract_semester(text)}, expected {expected}"


def test_parser_hindi_phrasing():
    cases = [
        ("सेमेस्टर १ का सिलेबस क्या है?", 1),
        ("दूसरे सेमेस्टर के विषय", 2),
        ("तीसरे सेमेस्टर का सिलेबस बताइए", 3),
        ("तीसरा सेमेस्टर विषय", 3),
        ("चौथे सेमेस्टर का पाठ्यक्रम", 4),
        ("पांचवें सेमेस्टर के क्रेडिट", 5),
        ("छठे सेमेस्टर का सिलेबस", 6),
        ("सातवें सेमेस्टर के विषय", 7),
        ("आठवां सेमेस्टर इंटर्नशिप", 8),
        ("सेम 3 का सिलेबस", 3),
        ("सेमेस्टर ४", 4),
    ]
    for text, expected in cases:
        assert extract_semester(text) == expected, f"Failed for Hindi '{text}': got {extract_semester(text)}, expected {expected}"


def test_clarification_when_no_semester_specified():
    unspecified = [
        "What is the semester syllabus?",
        "સેમેસ્ટર નો સિલેબસ શું છે?",
        "सेमेस्टर का सिलेबस बताइए",
        "Which subjects are taught in the curriculum?",
    ]
    for q in unspecified:
        assert is_semester_query(q) is True
        assert extract_semester(q) is None

    rag = RAGStore()
    res = rag.query_documents("What is the semester syllabus?")
    assert len(res) > 0
    assert res[0].get("metadata", {}).get("clarification") is True
    assert "clarification_prompt" in res[0].get("metadata", {})


def test_regression_semesters_1_to_8_retrieval():
    """
    Asserts that for Semesters 1 to 8, in English, Hindi, and Gujarati phrasing,
    EVERY returned chunk strictly belongs to the requested semester!
    """
    rag = RAGStore()

    sem_queries = {
        1: [
            "What subjects are in B.Tech IT Semester 1?",
            "સેમેસ્ટર ૧ નો સિલેબસ શું છે?",
            "पहले सेमेस्टर के विषय बताइए",
        ],
        2: [
            "What subjects are in Semester 2?",
            "બીજા સેમેસ્ટર ના વિષયો",
            "दूसरे सेमेस्टर का सिलेबस",
        ],
        3: [
            "What subjects are in B.Tech IT Semester 3?",
            "ત્રીજા સેમેસ્ટર નો સિલેબસ શું છે?",
            "तीसरे सेमेस्टर के विषय",
        ],
        4: [
            "What subjects are in Semester 4?",
            "ચોથા સેમેસ્ટર ના વિષયો",
            "चौथे सेमेस्टर का सिलेबस",
        ],
        5: [
            "What subjects are in Semester 5?",
            "પાંચમા સેમેસ્ટર ના વિષયો",
            "पांचवें सेमेस्टर का सिलेबस",
        ],
        6: [
            "What subjects are in Semester 6?",
            "છઠ્ઠા સેમેસ્ટર ના વિષયો",
            "छठे सेमेस्टर का सिलेबस",
        ],
        7: [
            "What subjects are in Semester 7?",
            "સાતમા સેમેસ્ટર ના વિષયો",
            "सातवें सेमेस्टर का सिलेबस",
        ],
        8: [
            "What is in Semester 8?",
            "આઠમા સેમેસ્ટર નો સિલેબસ",
            "आठवें सेमेस्टर का पाठ्यक्रम",
        ],
    }

    for target_sem, queries in sem_queries.items():
        for q in queries:
            results = rag.query_documents(q, n_results=3)
            assert len(results) > 0, f"No results returned for query: '{q}'"
            for idx, r in enumerate(results):
                sem = r.get("metadata", {}).get("semester")
                assert sem == target_sem, (
                    f"Wrong semester returned for query '{q}'! Expected {target_sem}, got {sem} in chunk: {r['text'][:80]}"
                )


if __name__ == "__main__":
    print("\n🚀 Running Multilingual Semester Parser & RAG Regression Tests...")
    test_parser_english_digits_and_ordinals()
    print("✅ English digits and ordinals passed!")
    test_parser_roman_numerals()
    print("✅ Roman numerals passed!")
    test_parser_gujarati_phrasing()
    print("✅ Gujarati phrasing passed!")
    test_parser_hindi_phrasing()
    print("✅ Hindi phrasing passed!")
    test_clarification_when_no_semester_specified()
    print("✅ Clarification for unspecified semester passed!")
    test_regression_semesters_1_to_8_retrieval()
    print("✅ All 8 Semesters regression across EN/HI/GU passed (100% accurate)!")
    print("\n🎉 ALL SEMESTER RAG TESTS PASSED PERFECTLY!\n")
