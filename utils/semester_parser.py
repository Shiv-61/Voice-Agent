"""
Multilingual Semester & Topic Parser, Query Normalizer, and Filter Builder.
Supports:
- Digits: 1..8, 'sem 3', 'semester 4'
- English words & ordinals: 'first', 'second', 'third', '1st', '2nd', '3rd', etc.
- Roman numerals: 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII'
- Gujarati digits, words, ordinals, transliterated:
  - '૧', '૨', '૩'..
  - 'એક', 'બે', 'ત્રણ', 'ચાર', 'પાંચ', 'છ', 'સાત', 'આઠ'
  - 'પહેલું', 'પહેલા', 'બીજું', 'બીજા', 'ત્રીજું', 'ત્રીજા', 'ચોથું', 'પાંચમું', 'છઠ્ઠું', 'સાતમું', 'આઠમું'
  - 'સેમેસ્ટર ત્રણ', 'સેમ ૩', 'ત્રીજા સેમેસ્ટર'
- Hindi digits, words, ordinals, transliterated:
  - '१', '२', '३'..
  - 'एक', 'दो', 'तीन', 'चार', 'पांच', 'पाँच', 'छह', 'सात', 'आठ'
  - 'पहला', 'पहले', 'दूसरा', 'दूसरे', 'तीसरा', 'तीसरे', 'चौथा', 'चौथे', 'पांचवां', 'पांचवे', 'छठा', 'सातवां', 'आठवां'
  - 'तीसरे सेमेस्टर', 'सेमेस्टर ३', 'सेम 3'
"""

import re
from typing import Optional, Tuple, Dict, Any


_SEMESTER_ALIASES: Dict[int, list[str]] = {
    1: [
        "1", "1st", "first", "one", "i",
        "૧", "એક", "પહેલું", "પહેલા", "પ્રથમ", "વન",
        "१", "एक", "पहला", "पहले", "पहली", "प्रथम",
    ],
    2: [
        "2", "2nd", "second", "two", "ii",
        "૨", "બે", "બીજું", "બીજા", "દ્વિતીય", "ટુ", "ટૂ",
        "२", "दो", "दूसरा", "दूसरे", "दूसरी", "द्वितीय",
    ],
    3: [
        "3", "3rd", "third", "three", "iii",
        "૩", "ત્રણ", "ત્રીજું", "ત્રીજા", "તૃતીય", "થ્રી",
        "३", "तीन", "तीसरा", "तीसरे", "तीसरी", "तृतीय",
    ],
    4: [
        "4", "4th", "fourth", "four", "iv",
        "૪", "ચાર", "ચોથું", "ચોથા", "ચતુર્થ", "ફોર",
        "४", "चार", "चौथा", "चौथे", "चौथी", "चतुर्थ",
    ],
    5: [
        "5", "5th", "fifth", "five", "v",
        "૫", "પાંચ", "પાંચમું", "પાંચમા", "પાંચમે", "ફાઈવ", "ફાઇવ",
        "५", "पांच", "पाँच", "पांचवां", "पांचवे", "पांचवें", "पाँचवाँ", "पाँचवे", "पाँचवें",
    ],
    6: [
        "6", "6th", "sixth", "six", "vi",
        "૬", "છ", "છઠ્ઠું", "છઠ્ઠા", "સિક્સ",
        "६", "छह", "छठा", "छठे", "छठी",
    ],
    7: [
        "7", "7th", "seventh", "seven", "vii",
        "૭", "સાત", "સાતમું", "સાતમા", "સેવન",
        "७", "सात", "सातवां", "सातवें", "सातवे", "सातवीं",
    ],
    8: [
        "8", "8th", "eighth", "eight", "viii",
        "૮", "આઠ", "આઠમું", "આઠમા", "એઈટ",
        "८", "आठ", "आठवां", "आठवें", "आठवे", "आठवीं",
    ],
}

_SEMESTER_KEYWORDS = [
    "semester", "sem", "semesters", "syllabus", "curriculum", "subject", "subjects", "course", "courses",
    "સેમેસ્ટર", "સેમ", "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "વિષયો",
    "सेमेस्टर", "सेम", "सिलेबस", "पाठ्यक्रम", "विषय",
]

_CLARIFICATION_PROMPTS = {
    "gu": "તમે કયા સેમેસ્ટર (૧ થી ૮) નો સિલેબસ જાણવા માંગો છો? કૃપા કરીને સેમેસ્ટર નંબર જણાવો.",
    "hi": "आप किस सेमेस्टर (1 से 8) का सिलेबस जानना चाहते हैं? कृपया सेमेस्टर नंबर बताएं।",
    "en": "Which semester (1 to 8) syllabus are you looking for? Please specify the semester number.",
}


# Indic/Latin aware boundary patterns because Devanagari/Gujarati combining vowel signs
# and anusvara (e.g. ं, ँ, ે, ો) are not categorized as \w in standard regex \b.
_B_START = r"(?<![a-zA-Z0-9\u0900-\u097F\u0A80-\u0AFF])"
_B_END = r"(?![a-zA-Z0-9\u0900-\u097F\u0A80-\u0AFF])"


def is_semester_query(text: str) -> bool:
    """Returns True if the user text asks about a semester syllabus, curriculum, or subjects."""
    if not text:
        return False
    lower = text.lower()
    return any(k in lower for k in _SEMESTER_KEYWORDS)


def extract_semester(text: str) -> Optional[int]:
    """
    Extracts the academic semester number (1 to 8) from user or tool query text.
    Handles digits, Roman numerals, words, ordinals in English, Gujarati, and Hindi.
    """
    if not text:
        return None
    lower = text.lower()

    # 1. Direct patterns like 'sem 3', 'semester 3', 'sem-3', 'sem3', 'semester-3'
    digit_match = re.search(rf"{_B_START}(?:sem|semester|સેમ|સેમેસ્ટર|सेम|सेमेस्टर)\s*[-–]?\s*([1-8૧-૮१-८]){_B_END}", lower)
    if digit_match:
        ch = digit_match.group(1)
        # Convert Gujarati or Hindi digit to int
        if ch in "12345678":
            return int(ch)
        if ch in "૧૨૩૪૫૬૭૮":
            return "૧૨૩૪૫૬૭૮".index(ch) + 1
        if ch in "१२३४५६७८":
            return "१२३४५६७८".index(ch) + 1

    # Inverted patterns: '3rd sem', '3 sem', '૩ સેમેસ્ટર', '3rd semester'
    inv_digit_match = re.search(rf"{_B_START}([1-8૧-૮१-८])\s*(?:st|nd|rd|th)?\s*(?:sem|semester|સેમ|સેમેસ્ટર|सेम|सेमेस्टर){_B_END}", lower)
    if inv_digit_match:
        ch = inv_digit_match.group(1)
        if ch in "12345678":
            return int(ch)
        if ch in "૧૨૩૪૫૬૭૮":
            return "૧૨૩૪૫૬૭૮".index(ch) + 1
        if ch in "१२३४५६७८":
            return "१२३४५६૭८".index(ch) + 1

    # 2. Check full alias list associated with semester context or ordinals
    for sem_num, aliases in _SEMESTER_ALIASES.items():
        for alias in aliases:
            # Pattern: alias followed by sem / semester
            # e.g., 'third sem', 'ત્રીજું સેમેસ્ટર', 'तीसरे सेमेस्टर'
            pat_before = rf"{_B_START}{re.escape(alias)}\s*(?:st|nd|rd|th)?\s*(?:sem|semester|સેમ|સેમેસ્ટર|સેમેસ્‍ટર|सेम|सेमेस्टर){_B_END}"
            # Pattern: sem / semester followed by alias
            # e.g., 'semester third', 'sem iii', 'સેમેસ્ટર ત્રણ', 'सेमेस्टर तीन'
            pat_after = rf"{_B_START}(?:sem|semester|સેમ|સેમેસ્ટર|सेम|सेमेस्टर)\s*[-–]?\s*(?:ester)?\s*{re.escape(alias)}{_B_END}"
            
            if re.search(pat_before, lower) or re.search(pat_after, lower):
                return sem_num

    # 3. If query mentions semester/syllabus keywords, check standalone ordinal or number word
    if is_semester_query(text):
        for sem_num, aliases in _SEMESTER_ALIASES.items():
            for alias in aliases:
                # Disallow common false-positive 1-letter words unless Roman or digit
                if len(alias) == 1 and alias not in "12345678૧૨૩૪૫૬૭૮१२३४५६७८":
                    continue
                if re.search(rf"{_B_START}{re.escape(alias)}{_B_END}", lower):
                    return sem_num

    return None


def extract_batch_year(text: str) -> Optional[int]:
    """Extracts batch year (e.g. 2026, 2025, 2024) from placement or highlight queries."""
    if not text:
        return None
    m = re.search(r"\b(202[0-9])\b", text)
    if m:
        return int(m.group(1))
    return None


def normalize_rag_query(query: str, caller_lang: str = "en") -> Tuple[str, Dict[str, Any]]:
    """
    Rewrites caller's query into compact canonical English search text and metadata filter.
    Returns (normalized_query, metadata_filter).
    """
    clean_q = query.strip()
    sem = extract_semester(clean_q)
    batch_year = extract_batch_year(clean_q)

    metadata_filter: Dict[str, Any] = {}

    if sem is not None:
        metadata_filter["semester"] = sem
        metadata_filter["doc_type"] = "syllabus"
        norm_text = f"DDU B.Tech IT Semester {sem} syllabus course structure subjects credits"
        return norm_text, metadata_filter

    if batch_year is not None:
        metadata_filter["batch_year"] = batch_year
        norm_text = f"DDU IT Batch {batch_year} placements highest average package offers hackathons"
        return norm_text, metadata_filter

    # General query: keep clean text
    return clean_q, metadata_filter


def get_semester_clarification_phrase(lang: str = "gu") -> str:
    """Returns the prompt instructing caller to specify which semester (1 to 8)."""
    lang_code = lang.split("-")[0].lower() if lang else "gu"
    return _CLARIFICATION_PROMPTS.get(lang_code, _CLARIFICATION_PROMPTS["gu"])
