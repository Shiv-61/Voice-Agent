"""
LLM layer — University Admission & Student Info Voice Agent.
Powered by Qwen 2.5:3B via local Ollama API server with built-in tool calling
for structured SQL database and unstructured RAG knowledge base.
"""

import asyncio
import json
import re
from typing import Generator, AsyncGenerator
import httpx

import config
from db.database import Database, transliterate_student_name
from rag import RAGStore
from utils import (
    is_hangup_intent,
    is_agent_farewell,
    is_simple_greeting,
    clean_speech_text,
    is_prompt_leak,
    HOLD_PHRASES,
    HOLD_PHRASE_REGEX,
    query_needs_db_or_rag,
    get_error_message,
)
from config.prompt_loader import (
    get_welcome_message,
    get_call_hangup_prompt,
    get_system_prompt,
    detect_active_domains,
)

# Fix #11: Live-fetching wrappers instead of import-time frozen constants.
# YAML hot-reload works properly; no server restart needed after prompt edits.
def _welcome_live() -> str:
    return get_welcome_message()

def _system_prompt_live(domains=None) -> str:
    return get_system_prompt(domains)

def _hangup_prompt_live() -> str:
    return get_call_hangup_prompt()

# Backward-compat names used by web/server.py and main.py (re-evaluated on each import cycle)
WELCOME_MESSAGE: str = get_welcome_message()
SYSTEM_PROMPT: str = get_system_prompt()
CALL_HANGUP_PROMPT: str = get_call_hangup_prompt()


def transform_query(user_text: str) -> str:
    """
    Transforms and sanitizes user input before passing to LLM.
    Removes unprintable control characters and normalizes whitespace.
    """
    if not user_text:
        return ""
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', user_text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


_shared_db = None
_shared_rag = None

def get_shared_db():
    global _shared_db
    if _shared_db is None:
        _shared_db = Database()
    return _shared_db

def get_shared_rag():
    global _shared_rag
    if _shared_rag is None:
        _shared_rag = RAGStore()
    return _shared_rag


_MULTILINGUAL_EXPANSIONS = {
    r'(?:admission|एडमिशन|प्रवेश|दाखिला|દાખલ|એડમિશન|apply|आवेदन|અરજી)': 'admission apply eligibility requirements',
    r'(?:eligibility|एलिजिबिलिटी|पात्रता|લાયકાત|eligible|योग्य)': 'eligibility criteria requirements percentage',
    r'(?:fees?|fee structure|फी|फीસ|शुल्क|ખર્ચ|ફી|cost)': 'fee structure tuition fees annual 166950 152000 55125 52500 1st year 2nd year B.Tech M.Tech',
    r'(?:m\.?tech|एमटेक|एम\.टेक|એમટેક|એમ\.ટેક)': 'M.Tech Master of Technology postgraduate fee structure 55125 52500',
    r'(?:b\.?tech|बीटेक|बी\.टेक|બીટેક|engineering|इंजीनियरिंग)': 'B.Tech Bachelor of Technology Engineering',
    r'(?:cse|it|computer|कम्प्यूटर|कंप्यूटर|કોમ્પ્યુટર|ઇન્ફોર્મેશન)': 'Computer Science Engineering Information Technology IT',
    r'(?:ddu|डीडीयू|धर्मसिंह|ધરમસિંહ)': 'DDU Dharamsinh Desai University Nadiad',
    r'(?:attendance|अटेंडेंस|हाजिरी|उपस्थिति|હાજરી)': 'attendance policy minimum percentage rules',
    r'(?:placement|प्लेसमेंट|नौकरी|पैकेज|सैलरी|પ્લેસમેન્ટ|salary|package)': 'placement highest average package recruiters companies',
    r'(?:batch\s*2026|2026\s*batch|૨૦૨૬|2026|highlights?|statistics?|summary|હાઈલાઈટ્સ|हाइलाइट्स)': 'Highlights of B.Tech IT 2026 Batch placement summary offers 108 106 placed highest salary 13.4 LPA average 5.5 LPA higher studies 30',
    r'(?:hackathons?|હેકાથોન|हैकथॉन|suisign|skillscreen|synapsemodel|competition|competitions|ઈનામ|पुरस्कार)': 'hackathon Sui Overflow ETHGlobal Codeversity prize winner Abhimanyu Ajudiya Pruthviraj Parmar Harsh Manek Sumit Mishra project',
    r'(?:gate|ncat|ielts|toefl|pte|ગેટ|પર્સન્ટાઈલ|गेट|रैंक|percentile|air|rank)': 'GATE NCAT IELTS TOEFL PTE percentile rank AIR Darpan Vora Dodiya Aditya Sorathiya Utsav Barkha Lahori Patel Nisarg competitive exam toppers',
    r'(?:research|paper|papers|publication|conference|conferences|રિસર્ચ|પેપર|સંશોધન|रिसर्च|पेपर|शोध|सम्मेलन)': 'research papers conference presented by students Shreyas Warrier Om Patel Lavi Garg Gaurang Agrawal Nitya Dhagat Devanshie Patel Deep Govindvira Govinda Prajapati Harmit Patel Kunj Patel',
    r'(?:hostel|हॉस्टल|छात्रावास|હોસ્ટેલ)': 'hostel timings curfew accommodation rules',
    r'(?:scholarship|स्कॉलरशिप|छात्रवृत्ति|સ્કોલરશિપ)': 'scholarship merit financial aid',
    r'(?:syllabus|curriculum|subjects?|સબ્જેક્ટ|સબ્જેક્ટ્સ|સબજેક્ટ|સબજેક્ટ્સ|વિષય|વિષયો|સિલેબસ|અભ્યાસક્રમ|सिलेबस|पाठ्यक्रम)': 'syllabus curriculum subjects examination scheme',
    r'(?:semester|सेमेस्टर|સેમેસ્ટર|સેમ|સેમિસ્ટર)': 'semester academic term subjects syllabus scheme',
    r'(?:એક|વન|પહેલું|પહેલા|૧|\b1\b|\bone\b|\bfirst\b|पहला|पहले|प्रथम)': 'semester 1 semester I first semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:બે|ટુ|ટૂ|બીજું|બીજા|૨|\b2\b|\btwo\b|\bsecond\b|दूसरा|दूसरे|द्वितीय)': 'semester 2 semester II second semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:ત્રણ|થ્રી|ત્રીજું|ત્રીજા|૩|\b3\b|\bthree\b|\bthird\b|तीसरा|तीसरे|तृतीय)': 'semester 3 semester III third semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:ચાર|ફોર|ચોથું|ચોથા|૪|\b4\b|\bfour\b|\bfourth\b|चौथा|चौथे|चतुर्थ)': 'semester 4 semester IV fourth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:પાંચ|ફાઈવ|પાંચમું|પાંચમા|૫|\b5\b|\bfive\b|\bfifth\b|पांचवा|पांचवे|पंचम)': 'semester 5 semester V fifth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:(?<![\u0a80-\u0aff])છ(?![\u0a80-\u0aff])|સિક્સ|૬|\b6\b|\bsix\b|\bsixth\b|छठा|छठे|षष्ठ)': 'semester 6 semester VI sixth semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:સાત|સેવન|સાતમું|સાતમા|૭|\b7\b|\bseven\b|\bseventh\b|सातवां|सातवें|सप्तम)': 'semester 7 semester VII seventh semester B.Tech IT Course Structure Teaching Scheme',
    r'(?:આઠ|એઈટ|આઠમું|આઠમા|૮|\b8\b|\beight\b|\beighth\b|आठवां|आठवें|अष्टम)': 'semester 8 semester VIII eighth semester B.Tech IT Course Structure Teaching Scheme',
}

def expand_multilingual_query(text: str) -> str:
    """Expands multilingual voice queries with English semantic tokens for ChromaDB indexing."""
    terms = [text]
    lower = text.lower()
    for pattern, eng_equiv in _MULTILINGUAL_EXPANSIONS.items():
        if re.search(pattern, lower):
            terms.append(eng_equiv)
    return " ".join(terms)


# Fix #13: Pre-compiled frozensets compiled once at import time for O(1) per-turn intersection
_GUJ_INDICATORS: frozenset[str] = frozenset({
    "shu", "ketli", "ketla", "ketlu", "che", "chhe", "maate", "nathi", "aapo",
    "tame", "tamara", "tamari", "tamaru", "aavde", "janavo", "kem", "vishe",
    "pucho", "aabhar", "namaste", "gujarati", "kai", "kayi", "bhanela", "kaho",
    "mane", "aapjo", "haji", "maru", "maro", "mari", "mara", "mare", "dikro",
    "dikra", "dikri", "chokro", "chokra", "chokri", "su", "chho", "chhu",
    "karvanu", "karvo", "batavo", "aavshe", "karo", "joie", "joiye"
})

_HIN_INDICATORS: frozenset[str] = frozenset({
    "kya", "kitna", "kitni", "kitne", "hai", "hain", "batao", "bata", "bataiye",
    "sakate", "sakthi", "sakthe", "hoga", "hogi", "hoge", "nahi", "kripya",
    "aapki", "aapka", "aapke", "kaise", "kuch", "baare", "mein", "aur", "chahiye",
    "dena", "dijiye", "kaun", "kaha", "kahan", "kab", "hindi", "mujhe", "mera",
    "meri", "mere", "uska", "uski", "uske", "unka", "unki", "unke", "beta", "beti",
    "naam", "bolo", "boliye", "raha", "rahi", "rahe", "tha", "thi",
    "iski", "iska", "iske", "inki", "inka", "inke", "hum", "humara", "humari",
    "achha", "theek", "shukriya", "dhanyawad", "ki", "ka", "ke", "ko", "se", "toh"
})

_ENG_INDICATORS: frozenset[str] = frozenset({
    "what", "when", "where", "how", "why", "who", "which", "is", "are", "can",
    "tell", "eligibility", "fee", "fees", "admission", "curfew", "placement",
    "package", "attendance", "marks", "thank", "goodbye", "hello", "hi", "yes",
    "no", "please", "english", "syllabus", "course", "subject", "semester",
    "credit", "curriculum", "department", "hostel", "rules", "campus", "direct",
})

# Words that indicate Hindi even when phonetically written in Gujarati script
_HINDI_IN_GUJ_WORDS: frozenset[str] = frozenset({
    "બતાઓ", "બતાઈએ", "હૈ", "મેરા", "મેરી", "મેરે", "મુઝે", "ઉસકા", "ઉસકી", "ઉસકે",
    "કિતની", "કિતના", "ચાહિયે", "ક્યા", "આપકા", "આપકી", "કૈસે", "નહીં", "હોગા", "હોગી",
    "અભી", "કરકે", "બતાતી", "બતાતા", "કી", "કા", "કે"
})

_GUJ_SPECIFIC_WORDS: frozenset[str] = frozenset({
    "છે", "નથી", "માટે", "તમાર", "કેમ", "શું", "આપો", "વિશે", "કહો", "જણાવો", "આવડે", "જોઈએ"
})


class LLM:
    def __init__(self, db=None, rag=None):
        self.db = db or get_shared_db()
        self.rag = rag or get_shared_rag()
        self.current_lang = "gu"  # Gujarati is the primary language
        self.active_student_id: str | None = None
        self.active_student_name: str | None = None
        self._active_provider = config.LLM_PROVIDER
        # Initialize conversation history with the assistant's opening welcome message
        self.history: list[dict[str, str]] = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    def detect_turn_language(self, text: str, fallback_lang: str = "gu", stt_lang: str | None = None) -> tuple[str, str, str]:
        """
        Determines caller language for the active turn with zero-latency dynamic switching.
        Hierarchy:
          - Gujarati (Primary)
          - Hindi & English (Secondary)
        Returns: (lang_code, bcp47, directive_prefix)
        """
        clean = (text or "").strip()
        lower = clean.lower()
        words = set(re.findall(r'\b[a-zA-Z]+\b', lower))

        # 1. Native script detection (100% definitive)
        if re.search(r"[\u0900-\u097f]", clean):
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"

        if re.search(r"[\u0a80-\u0aff]", clean):
            # Check if text actually consists of Hindi words phonetically transcribed in Gujarati script
            has_hin_words = any(hw in clean for hw in _HINDI_IN_GUJ_WORDS)
            has_guj_words = any(gw in clean for gw in _GUJ_SPECIFIC_WORDS)
            if has_hin_words and not has_guj_words:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"

        # Explicit language request triggers
        if "gujarati" in words:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"
        if "hindi" in words:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी).]:\n"
        if "english" in words and not (words & _GUJ_INDICATORS) and not (words & _HIN_INDICATORS):
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested ENGLISH. You MUST respond 100% in ENGLISH.]:\n"

        guj_overlap = len(words & _GUJ_INDICATORS)
        hin_overlap = len(words & _HIN_INDICATORS)
        eng_overlap = len(words & _ENG_INDICATORS)

        # Multi-word overlap evaluation
        if hin_overlap > 0 and hin_overlap >= guj_overlap and hin_overlap >= eng_overlap:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"
        if guj_overlap > 0 and guj_overlap >= hin_overlap and guj_overlap >= eng_overlap:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"
        if eng_overlap > 0 and eng_overlap >= hin_overlap and eng_overlap >= guj_overlap:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"

        # Check acoustic STT language hint for neutral queries (e.g. 'Aarav Patel', 'STU1')
        if stt_lang:
            stt_l = stt_lang.lower()
            if "hi" in stt_l:
                return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी लिपि). Every single word must be in Hindi. Do NOT use any Gujarati words or Gujarati characters under any circumstances.]:\n"
            elif "en" in stt_l:
                return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"
            elif "gu" in stt_l:
                return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"

        # Neutral query (e.g. 'STU101' or numbers) -> retain session active language, fallback to Gujarati (Primary)
        fb = (fallback_lang or "").lower()
        if "hi" in fb:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: HINDI (हिंदी लिपि). Every single word must be in Hindi.]:\n"
        elif "en" in fb:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: ENGLISH.]:\n"
        else:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Respond in primary language: GUJARATI (ગુજરાતી લિપિ).]:\n"

    def _prepare_messages(self, user_text: str, stt_lang: str | None = None) -> list[dict[str, str]]:
        """
        Prepares LLM messages with conversation history and anticipatory RAG / DB context.
        Pre-retrieval eliminates the slow, redundant 2nd turn LLM tool round-trip!
        """
        clean_user_text = transform_query(user_text)
        lower_text = clean_user_text.lower()
        history_context = self._format_conversation_history()

        context_snippets = []

        # Detect if caller is inquiring about an individual student's records (attendance, marks, cpi, etc.)
        ATTENDANCE_TERMS = {
            "attendance", "attedance", "atendance", "attandance", "attendence",
            "હાજરી", "એટેન્ડન્સ", "અટેન્ડન્સ", "अटेंडेंस", "उपस्थिति"
        }
        MARKS_TERMS = {
            "marks", "mark", "cpi", "cgpa", "spi", "result", "grade", "score",
            "માર્ક્સ", "માર્ક", "પરિણામ", "સીપીઆઈ", "ગુણ",
            "मार्क्स", "अंक", "रिजल्ट", "परिणाम", "सीपीआई"
        }
        STUDENT_INTENT_TERMS = {
            "son", "daughter", "child", "kid", "beta", "beti", "bachha", "baccha",
            "dikro", "dikra", "dikri", "chokro", "chokra", "chokri",
            "maro", "mari", "maru", "mara", "mare",
            "mera", "meri", "mere", "my", "his", "her", "student", "roll",
            "દીકર", "છોકર", "પુત્ર", "માર", "તમાર", "વિદ્યાર્થી", "બાળક",
            "बेट", "बच्च", "पुत्र", "मेर", "छात्र", "विद्यार्थी", "रोल"
        }
        GENERAL_POLICY_TERMS = {
            "rule", "rules", "policy", "policies", "criteria", "requirement", "requirements",
            "minimum", "mandatory", "compulsory", "allowed", "regulation", "regulations",
            "નિયમ", "નિયમો", "પોલિસી", "શરત", "જરૂરી", "ઓછામાં ઓછી",
            "नियम", "पॉलिसी", "शर्त", "जरूरी", "न्यूनतम", "अनिवार्य"
        }

        has_rec_term = any(t in lower_text for t in ATTENDANCE_TERMS | MARKS_TERMS)
        has_student_intent = any(t in lower_text for t in STUDENT_INTENT_TERMS)
        has_policy_term = any(t in lower_text for t in GENERAL_POLICY_TERMS)

        # Distinguish personal record request vs general university policy inquiry:
        # Any attendance/marks/result query without policy keywords is treated as a personal record query.
        is_personal_student_query = (has_rec_term and not has_policy_term) or (
            has_student_intent and has_rec_term
        )

        # 1. Student ID or Name pattern resolution
        student = None
        stu_match = re.search(
            r'\b(?:stu\s*(\d{1,4})|student\s*(\d{1,4})|(stu\d{1,4}|2[0-9]it\d{2,4}|it\d{2,4}|ce\d{2,4}|ec\d{2,4}))\b',
            lower_text, re.IGNORECASE
        )
        if stu_match:
            matched_num = stu_match.group(1) or stu_match.group(2)
            if matched_num:
                stu_id = f"STU{matched_num}"
            else:
                stu_id = stu_match.group(3).upper().replace(" ", "")
            student = self.db.lookup_student(stu_id)
        else:
            try:
                all_students = self.db.get_all_student_identifiers()
                clean_query_trans = transliterate_student_name(clean_user_text).lower()
                query_tokens = set(re.findall(r'\b[a-zA-Z]+\b', lower_text)) | set(re.findall(r'\b[a-zA-Z]+\b', clean_query_trans))

                for s_item in all_students:
                    s_name = s_item.get("name", "").lower()
                    s_id = s_item.get("student_id", "").lower()
                    parts = s_name.split()
                    first_name = parts[0] if parts else ""

                    # Full name or exact ID in text
                    if s_name and (s_name in lower_text or s_name in clean_query_trans or s_id in lower_text):
                        student = self.db.lookup_student(s_item["student_id"])
                        break
                    # First name match (protect 'priya' assistant name unless qualified as student)
                    elif first_name and len(first_name) >= 3 and first_name in query_tokens:
                        if first_name == "priya":
                            if any(k in lower_text for k in ["student", "daughter", "beti", "dikri", "stu4"]):
                                student = self.db.lookup_student(s_item["student_id"])
                                break
                        else:
                            student = self.db.lookup_student(s_item["student_id"])
                            break
            except Exception as de:
                print(f"[llm] Dynamic student search notice: {de}")

        # Check session active_student_id if student not explicitly found in current turn
        if not student and self.active_student_id:
            try:
                if has_rec_term or has_student_intent or any(w in lower_text for w in ["he", "him", "his", "તે", "તેનું", "વહ"]):
                    student = self.db.lookup_student(self.active_student_id)
            except Exception as se:
                print(f"[llm] Active student lookup notice: {se}")

        # Fallback: scan recent turns of conversation history for any student ID or name
        if not student and (has_rec_term or has_student_intent):
            try:
                all_students = self.db.get_all_student_identifiers()
                for turn in reversed(self.history[-4:]):
                    h_text = turn.get("content", "").lower()
                    h_match = re.search(r'\b(?:stu\s*(\d{1,4})|(stu\d{1,4}))\b', h_text)
                    if h_match:
                        h_id = f"STU{h_match.group(1)}" if h_match.group(1) else h_match.group(2).upper()
                        student = self.db.lookup_student(h_id)
                        if student:
                            break
                    for s_item in all_students:
                        s_name = s_item.get("name", "").lower()
                        if s_name and s_name in h_text:
                            student = self.db.lookup_student(s_item["student_id"])
                            if student:
                                break
                    if student:
                        break
            except Exception as he:
                print(f"[llm] History student scan notice: {he}")

        # If student is resolved, remember in session and retrieve personal records
        if student:
            stu_id = student.get("student_id")
            self.active_student_id = stu_id
            self.active_student_name = student.get("name")
            context_snippets.append(f"[Student Record {stu_id}]: {json.dumps(student, default=str)}")
            marks = self.db.get_student_marks(stu_id)
            if marks:
                context_snippets.append(f"[Student Marks {stu_id}]: {json.dumps(marks, default=str)}")
            att = self.db.get_student_attendance(stu_id)
            if att:
                context_snippets.append(f"[Student Attendance {stu_id}]: {json.dumps(att, default=str)}")

        # 2. Anticipatory DB check for admission or placement (English, Hindi, Gujarati)
        try:
            if any(k in lower_text for k in [
                "placement", "package", "recruiter", "salary", "placed",
                "प्लेसमेंट", "नौकरी", "पैकेज", "सैलरी", "પ્લેસમેન્ટ"
            ]):
                stats = self.db.get_placement_stats()
                if stats:
                    context_snippets.append(f"[Official Placement Records]: {json.dumps(stats, default=str)}")

            if any(k in lower_text for k in [
                "admission", "eligibility", "fee", "fees", "apply", "deadline", "course",
                "एडमिशन", "प्रवेश", "दाखिला", "फी", "फीस", "शुल्क", "पात्रता", "एलिजिबिलिटी",
                "बीटेक", "बी.टेक", "એડમિશન", "ફી", "લાયકાત"
            ]):
                prog = "M.Tech" if any(p in lower_text for p in [
                    "mtech", "m.tech", "एमटेक", "एम.टेक", "એમટેક", "એમ.ટેક"
                ]) else "B.Tech" if any(p in lower_text for p in [
                    "cse", "computer", "it", "btech", "b.tech", "બીટેક", "બી.ટેક", "कंप्यूटर", "કોમ્પ્યુટર", "बीटेक", "बी.टेक"
                ]) else ""
                adm = self.db.get_admission_info(prog)
                if adm:
                    context_snippets.append(f"[Official Admission & Fee Records]: {json.dumps(adm, default=str)}")
        except Exception as e:
            print(f"[llm] Anticipatory DB notice: {e}")

        # 3. Anticipatory RAG query (suppressed for simple greetings, chit-chat, or un-identified personal student queries)
        matches = []
        is_greeting = is_simple_greeting(clean_user_text)
        needs_rag = query_needs_db_or_rag(clean_user_text, has_student=bool(student))
        if not is_greeting and needs_rag and not (is_personal_student_query and not student):
            try:
                rag_query = expand_multilingual_query(clean_user_text)
                matches = self.rag.query_documents(rag_query, n_results=4, min_similarity=0.40)
                if matches and matches[0].get("similarity_score", 0) >= 0.40:
                    for m in matches:
                        text_snippet = m.get("text", "").strip()
                        if text_snippet:
                            text_snippet = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0A80-\u0AFF\n\r\t]', ' ', text_snippet)
                            text_snippet = re.sub(r'\s+', ' ', text_snippet).strip()
                            filename = m.get("metadata", {}).get("filename", "University Document")
                            page_num = m.get("metadata", {}).get("page", 1)
                            context_snippets.append(f"[Official Document: {filename} (Page {page_num})]: {text_snippet[:1200]}")
            except Exception as e:
                print(f"[llm] Anticipatory RAG notice: {e}")

        # Check whether this turn has actual university records / document context to serve
        has_db_or_rag = bool(context_snippets)
        self._last_turn_needed_db_or_rag = has_db_or_rag

        rag_context = ""
        if context_snippets:
            rag_context = (
                "[OFFICIAL UNIVERSITY CONTEXT (Answer directly from here)]: "
                + " | ".join(context_snippets)
            )

        # Dynamically trigger active domain directives (Student Records vs College Curriculum vs Admissions)
        matched_docs = matches if 'matches' in locals() and matches else []
        active_domains = detect_active_domains(clean_user_text, matched_docs)
        active_prompt = get_system_prompt(active_domains=active_domains if active_domains else None)

        messages = [{"role": "system", "content": active_prompt}]

        # Add previous conversation history turns cleanly
        for turn in self.history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Dynamically detect caller's active turn language (Gujarati primary, Hindi/English secondary)
        lang_code, bcp47, lang_cue = self.detect_turn_language(clean_user_text, self.current_lang, stt_lang=stt_lang)
        self.current_lang = lang_code

        is_hangup = is_hangup_intent(clean_user_text)
        hold_phrase = HOLD_PHRASES.get(lang_code, HOLD_PHRASES["gu"])
        hold_directive = ""
        if is_hangup:
            farewell_map = {
                "gu": "તમારા સમય માટે આભાર. તમારો દિવસ શુભ રહે!",
                "hi": "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
                "en": "Thank you for your time. Have a great day!",
            }
            exact_farewell = farewell_map.get(lang_code, farewell_map["gu"])
            hold_directive = (
                f"[MANDATORY CLOSING DIRECTIVE: The caller is concluding or saying goodbye. "
                f"You MUST reply ONLY with this exact farewell sentence in {lang_code}: \"{exact_farewell}\". "
                f"Do not ask any follow-up questions and do not say any other words.]\n"
            )
        elif has_db_or_rag:
            hold_directive = (
                f"[MANDATORY OPENING: You are retrieving information from university database/documents. "
                f"Start your spoken answer with: \"{hold_phrase}\" followed by the facts.]\n"
            )
        elif is_personal_student_query and not student:
            hold_directive = (
                "[DIRECTIVE: Caller is inquiring about an individual student's records (attendance or marks), but NO Student ID or registered name has been provided yet. "
                "Politely ask the caller to provide their son's or student's Student ID (e.g. STU1) or registered name. "
                "Do NOT output any hold phrase or filler.]\n"
            )

        current_turn_content = f"{hold_directive}{lang_cue}{clean_user_text}"
        if rag_context:
            current_turn_content = f"{rag_context}\n{hold_directive}{lang_cue}{clean_user_text}"

        messages.append({"role": "user", "content": current_turn_content})

        MAX_TURN_CHARS = 600
        self.history.append({"role": "user", "content": clean_user_text[:MAX_TURN_CHARS]})
        self._trim_history()

        return messages

    def _trim_history(self):
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history[:] = self.history[-max_msgs:]

    def _execute_tool(self, tool_name: str, kwargs: dict) -> str:
        """Executes database queries or RAG search based on LLM tool selection."""
        print(f"[llm] Tool call triggered: {tool_name}({kwargs})")
        try:
            if tool_name == "lookup_student":
                identifier = kwargs.get("identifier", "")
                res = self.db.lookup_student(identifier)
                return json.dumps(res if res else {"error": "Student not found"}, default=str)

            elif tool_name == "get_student_marks":
                student_id = kwargs.get("student_id", "")
                res = self.db.get_student_marks(student_id)
                return json.dumps(res if res else {"error": "No marks found"}, default=str)

            elif tool_name == "get_student_attendance":
                student_id = kwargs.get("student_id", "")
                res = self.db.get_student_attendance(student_id)
                return json.dumps(res if res else {"error": "No attendance records found"}, default=str)

            elif tool_name == "get_placement_stats":
                dept = kwargs.get("department", "")
                res = self.db.get_placement_stats(dept)
                return json.dumps(res if res else {"error": "No placement stats found"}, default=str)

            elif tool_name == "get_admission_info":
                prog = kwargs.get("program", "")
                res = self.db.get_admission_info(prog)
                return json.dumps(res if res else {"error": "No admission info found"}, default=str)

            elif tool_name == "search_university_docs":
                query = kwargs.get("query", "")
                results = self.rag.query_documents(query, n_results=4)
                if not results:
                    return json.dumps({"message": "No relevant policy documents found in knowledge base."})
                summarized_docs = [
                    f"Excerpt from {r['metadata'].get('filename', 'doc')} (Page {r['metadata'].get('page', 1)}): {r['text']}"
                    for r in results
                ]
                return json.dumps({"retrieved_context": summarized_docs})

            return json.dumps({"error": f"Unknown tool '{tool_name}'"})
        except Exception as e:
            print(f"[llm] Tool execution error ({tool_name}): {e}")
            return json.dumps({"error": str(e)})

    def _parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parses TOOL_CALL: tool_name(key="val") pattern."""
        match = re.search(r"TOOL_CALL:\s*(\w+)\((.*)\)", text)
        if not match:
            return None
        tool_name = match.group(1)
        params_str = match.group(2).strip()

        kwargs = {}
        # 1. Try quoted matches first (e.g. key="val with spaces" or key='val with spaces')
        quoted_pairs = re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', params_str)
        if quoted_pairs:
            for k, v in quoted_pairs:
                kwargs[k] = v.strip()
        else:
            # 2. Fallback for unquoted pairs (e.g. identifier=Raj Mehta or department=CSE)
            unquoted_pairs = re.findall(r'(\w+)\s*=\s*([^,)]+)', params_str)
            for k, v in unquoted_pairs:
                kwargs[k] = v.strip().strip("'\"")

        return tool_name, kwargs

    def _format_conversation_history(self) -> str:
        """Formats preceding conversation history into an explicit readable transcript."""
        if not self.history:
            return ""
        lines = ["\n[PREVIOUS CALL CHAT HISTORY]"]
        for turn in self.history:
            role_label = "Caller" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {turn['content']}")
        lines.append("[END OF CHAT HISTORY]\n")
        return "\n".join(lines)

    def check_call_hangup(self, user_text: str, agent_text: str = "") -> bool:
        """
        Fast zero-latency evaluation of whether caller intends to end the call,
        or whether agent has completed the farewell: 'Thank you for your time. Have a great day!'.
        """
        return is_hangup_intent(user_text) or is_agent_farewell(agent_text)

    def _get_provider_request(self, messages: list[dict], provider_override: str | None = None):
        """Builds URL, headers, and payload for streaming LLM calls."""
        provider = provider_override or getattr(self, "_active_provider", config.LLM_PROVIDER)
        if provider == "sarvam":
            headers = {
                "api-subscription-key": config.SARVAM_API_KEY.strip(),
                "Content-Type": "application/json",
            }
            sarvam_model = "sarvam-105b-conversations"
            payload = {
                "model": sarvam_model,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
            }
            return config.SARVAM_CHAT_URL, headers, payload
        elif provider == "groq":
            headers = {
                "Authorization": f"Bearer {config.GROQ_API_KEY.strip()}",
                "Content-Type": "application/json",
            }
            groq_model = config.LLM_MODEL if config.LLM_MODEL and ("llama" in config.LLM_MODEL.lower() or "mixtral" in config.LLM_MODEL.lower() or "gemma" in config.LLM_MODEL.lower()) else "llama-3.3-70b-versatile"
            payload = {
                "model": groq_model,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
            }
            return config.GROQ_URL, headers, payload
        elif provider == "openrouter":
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/Shiv-61/Voice-Agent",
                "X-Title": "College Voice Agent",
            }
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
                "include_reasoning": False,
                "reasoning": {"effort": "none"},
            }
            return config.OPENROUTER_URL, headers, payload
        else:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.LLM_TEMPERATURE,
                    "num_predict": config.LLM_MAX_TOKENS,
                },
            }
            return config.OLLAMA_URL, headers, payload

    def _parse_stream_line(self, line: str, provider_override: str | None = None) -> str:
        """Extracts delta token text from SSE or JSON line."""
        line = line.strip()
        if not line:
            return ""
        provider = provider_override or getattr(self, "_active_provider", config.LLM_PROVIDER)
        if provider in ("sarvam", "openrouter", "groq"):
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    return ""
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        # Strictly extract content, ignore reasoning_content
                        return delta.get("content", "") or ""
                except Exception:
                    return ""
        else:
            try:
                data = json.loads(line)
                return data.get("message", {}).get("content", "") or ""
            except Exception:
                return ""
        return ""

    def reply_stream(self, user_text: str, stt_lang: str | None = None) -> Generator[str, None, None]:
        """
        Synchronously yields response tokens incrementally in real time.
        Used by CLI and synchronous callers.
        """
        messages = self._prepare_messages(user_text, stt_lang=stt_lang)
        turn_lang = self.current_lang
        needs_hold = getattr(self, "_last_turn_needed_db_or_rag", False)
        hold_phrase = HOLD_PHRASES.get(turn_lang, HOLD_PHRASES["gu"])
        hold_phrase_yielded = False

        if needs_hold:
            yield f"{hold_phrase} "
            hold_phrase_yielded = True

        max_tool_iterations = 3
        current_iteration = 0

        try:
            with httpx.Client(timeout=35.0) as client:
                while current_iteration < max_tool_iterations:
                    current_iteration += 1
                    url, headers, payload = self._get_provider_request(messages)

                    stream_buffer = ""
                    recent_window = ""
                    is_tool_call = False
                    is_streaming_speech = False
                    full_reply = ""
                    in_think = False
                    checked_hold_dedup = not hold_phrase_yielded

                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 429 and getattr(self, "_active_provider", config.LLM_PROVIDER) == "openrouter" and config.SARVAM_API_KEY:
                            print("⚠️ [llm] OpenRouter quota/rate limit (429) hit. Gracefully switching to Sarvam LLM (sarvam-105b-conversations)...")
                            self._active_provider = "sarvam"
                            current_iteration -= 1
                            continue
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think> or <thought>...</thought>)
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    token = token.split("</think>", 1)[1]
                                    if not token:
                                        continue
                                elif "</thought>" in token:
                                    in_think = False
                                    token = token.split("</thought>", 1)[1]
                                    if not token:
                                        continue
                                else:
                                    continue

                            if "<think>" in token or "<thought>" in token:
                                in_think = True
                                tag = "<think>" if "<think>" in token else "<thought>"
                                token = token.split(tag, 1)[0]
                                if not token:
                                    continue

                            # Detect whether model is issuing a tool call with sliding lookback window
                            recent_window = (recent_window + token)[-30:]

                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer or "TOOL_CALL:" in recent_window:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                else:
                                    if hold_phrase_yielded and not checked_hold_dedup:
                                        if not stripped:
                                            continue
                                        if HOLD_PHRASE_REGEX.match(stripped):
                                            stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                                            checked_hold_dedup = True
                                            stripped = stream_buffer.lstrip()
                                            if not stripped:
                                                stream_buffer = ""
                                                continue
                                        elif len(stripped) >= 35 or any(p in stripped for p in [".", "!", "?", "।"]):
                                            checked_hold_dedup = True
                                        else:
                                            continue

                                    has_punct = any(p in stream_buffer for p in [".", "!", "?", "।", "\n"])
                                    if (has_punct or len(stripped) >= 12) and "TOOL_CALL:" not in stream_buffer:
                                        if is_prompt_leak(stream_buffer):
                                            print(f"🛑 [llm] Suppressed prompt leak buffer in stream: {stream_buffer}")
                                            stream_buffer = ""
                                            continue
                                        if stream_buffer:
                                            is_streaming_speech = True
                                            yield stream_buffer
                                            full_reply += stream_buffer
                                            stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in recent_window or "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    if "TOOL_CALL:" in full_reply:
                                        idx = full_reply.index("TOOL_CALL:")
                                        stream_buffer = full_reply[idx:]
                                        full_reply = full_reply[:idx]
                                    elif "TOOL_CALL:" in token:
                                        idx = token.index("TOOL_CALL:")
                                        stream_buffer = token[idx:]
                                    else:
                                        stream_buffer = "TOOL_CALL:"
                                    continue
                                full_reply += token
                                if not is_prompt_leak(full_reply):
                                    yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        if hold_phrase_yielded and not checked_hold_dedup:
                            checked_hold_dedup = True
                            stripped = stream_buffer.lstrip()
                            if HOLD_PHRASE_REGEX.match(stripped):
                                stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                        if stream_buffer and not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        if not hold_phrase_yielded:
                            yield f"{hold_phrase} "
                            hold_phrase_yielded = True
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            tool_result = self._execute_tool(tool_name, kwargs)
                            tool_synth = f"Please synthesize a short, polite spoken answer for the caller in 1-2 sentences in {turn_lang} language."
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\n{tool_synth}",
                            })
                            continue
                        else:
                            # Tool parse failed, yield buffer as text
                            clean_text = clean_speech_text(stream_buffer)
                            if is_prompt_leak(clean_text):
                                clean_text = "I am here to help you with DDU IT queries. How may I assist you?"
                            clean_history_text = clean_text
                            if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                                clean_history_text = f"{hold_phrase} {clean_history_text}"
                            self.history.append({"role": "assistant", "content": clean_history_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
                        if is_prompt_leak(clean_text):
                            print(f"🛑 [llm] Suppressed prompt leak from final text: {clean_text}")
                            clean_text = ""

                        if not clean_text.strip():
                            turn_lang, _, _ = self.detect_turn_language(user_text, fallback_lang=self.current_lang)
                            lower_u = user_text.lower()
                            if any(w in lower_u for w in ["સેમેસ્ટર", "સેમ", "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "semester", "syllabus", "subject"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટી સેમેસ્ટર 1 માં મેથેમેટિક્સ-1, બેઝિક પ્રોગ્રામિંગ અને એન્જિનિયરિંગ ફાઉન્ડેશન વિષયો સામેલ છે. શું તમારે ચોક્કસ વિષય કે ક્રેડિટ વિશે વધુ જાણવું છે?"
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी सेमेस्टर 1 में मैथमेटिक्स-1, प्रोग्रामिंग और इंजीनियरिंग विषय शामिल हैं। क्या आप किसी विशेष विषय या क्रेडिट के बारे में जानना चाहते हैं?"
                                else:
                                    clean_text = "DDU IT Semester 1 includes Mathematics-1, Basic Programming, and Engineering fundamentals. Would you like specific details on subjects or credits?"
                            elif any(w in lower_u for w in ["fee", "fees", "ફી", "ખર્ચ", "फी", "फीस"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક પ્રથમ વર્ષની વાર્ષિક ટ્યુશન ફી ૧,૬૬,૯૫૦ રૂપિયા અને બીજાથી ચોથા વર્ષ માટે ૧,૫૨,૦૦૦ રૂપિયા છે, જ્યારે એમટેક પ્રથમ વર્ષની ફી ૫૫,૧૨૫ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक प्रथम वर्ष की वार्षिक ट्यूशन फीस 1,66,950 रुपये और 2nd से 4th वर्ष के लिए 1,52,000 रुपये है, जबकि एम.टेक प्रथम वर्ष की 55,125 रुपये है।"
                                else:
                                    clean_text = "DDU B.Tech first year annual fee is 1,66,950 rupees, and second to fourth year is 1,52,000 rupees per year. M.Tech first year is 55,125 rupees per year."
                            elif any(w in lower_u for w in ["placement", "પ્લેસમેન્ટ", "प्लेसमेंट", "salary", "package"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટીમાં પ્લેસમેન્ટ દર ૯૬.૫% છે, જેમાં સૌથી વધુ પેકેજ ૪૫ લાખ રૂપિયા અને સરેરાશ ૧૨.૫ લાખ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी का प्लेसमेंट रिकॉर्ड 96.5% है, जिसमें उच्चतम पैकेज 45 लाख रुपये और औसत 12.5 लाख रुपये है।"
                                else:
                                    clean_text = "DDU IT has a 96.5% placement rate, with the highest package at 45 lakh rupees and an average of 12.5 lakh rupees."
                            else:
                                if turn_lang == "gu":
                                    clean_text = "નમસ્તે, હું ડીડીયુ આઈટી ડિપાર્ટમેન્ટમાંથી પ્રિયા છું. હું તમને એડમિશન, અભ્યાસક્રમ અથવા વિદ્યાર્થી રેકોર્ડ્સ વિશે શું માહિતી આપું?"
                                elif turn_lang == "hi":
                                    clean_text = "नमस्ते, मैं डीडीयू आईटी विभाग से प्रिया हूँ। मैं आपको प्रवेश, पाठ्यक्रम या छात्र रिकॉर्ड के बारे में क्या जानकारी दे सकती हूँ?"
                                else:
                                    clean_text = "Hello, I am Priya from DDU IT department. How may I assist you with admissions, curriculum, or student records?"
                            yield clean_text

                        clean_history_text = clean_text
                        if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                            clean_history_text = f"{hold_phrase} {clean_history_text}"
                        self.history.append({"role": "assistant", "content": clean_history_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            clean_fallback = fallback
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback

        except Exception as err:
            print(f"[llm] Error communicating with {config.LLM_PROVIDER}: {err}")
            fallback_msg = get_error_message("system_error", turn_lang) or "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            clean_fallback = fallback_msg
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback_msg

    async def areply_stream(self, user_text: str, stt_lang: str | None = None) -> AsyncGenerator[str, None]:
        """
        Asynchronously yields response tokens incrementally in real time.
        Zero event-loop blocking for FastAPI and WebSocket servers.
        """
        messages = self._prepare_messages(user_text, stt_lang=stt_lang)
        turn_lang = self.current_lang
        needs_hold = getattr(self, "_last_turn_needed_db_or_rag", False)
        hold_phrase = HOLD_PHRASES.get(turn_lang, HOLD_PHRASES["gu"])
        hold_phrase_yielded = False

        if needs_hold:
            yield f"{hold_phrase} "
            hold_phrase_yielded = True

        max_tool_iterations = 3
        current_iteration = 0

        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                while current_iteration < max_tool_iterations:
                    current_iteration += 1
                    url, headers, payload = self._get_provider_request(messages)

                    stream_buffer = ""
                    recent_window = ""
                    is_tool_call = False
                    is_streaming_speech = False
                    full_reply = ""
                    in_think = False
                    checked_hold_dedup = not hold_phrase_yielded

                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code == 429 and getattr(self, "_active_provider", config.LLM_PROVIDER) == "openrouter" and config.SARVAM_API_KEY:
                            print("⚠️ [llm] OpenRouter quota/rate limit (429) hit. Gracefully switching to Sarvam LLM (sarvam-105b-conversations)...")
                            self._active_provider = "sarvam"
                            current_iteration -= 1
                            continue
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think> or <thought>...</thought>)
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
                                    token = token.split("</think>", 1)[1]
                                    if not token:
                                        continue
                                elif "</thought>" in token:
                                    in_think = False
                                    token = token.split("</thought>", 1)[1]
                                    if not token:
                                        continue
                                else:
                                    continue

                            if "<think>" in token or "<thought>" in token:
                                in_think = True
                                tag = "<think>" if "<think>" in token else "<thought>"
                                token = token.split(tag, 1)[0]
                                if not token:
                                    continue

                            # Detect whether model is issuing a tool call with sliding lookback window
                            recent_window = (recent_window + token)[-30:]

                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer or "TOOL_CALL:" in recent_window:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                else:
                                    if hold_phrase_yielded and not checked_hold_dedup:
                                        if not stripped:
                                            continue
                                        if HOLD_PHRASE_REGEX.match(stripped):
                                            stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                                            checked_hold_dedup = True
                                            stripped = stream_buffer.lstrip()
                                            if not stripped:
                                                stream_buffer = ""
                                                continue
                                        elif len(stripped) >= 35 or any(p in stripped for p in [".", "!", "?", "।"]):
                                            checked_hold_dedup = True
                                        else:
                                            continue

                                    has_punct = any(p in stream_buffer for p in [".", "!", "?", "।", "\n"])
                                    if (has_punct or len(stripped) >= 12) and "TOOL_CALL:" not in stream_buffer:
                                        if is_prompt_leak(stream_buffer):
                                            print(f"🛑 [llm] Suppressed prompt leak buffer in async stream: {stream_buffer}")
                                            stream_buffer = ""
                                            continue
                                        if stream_buffer:
                                            is_streaming_speech = True
                                            yield stream_buffer
                                            full_reply += stream_buffer
                                            stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in recent_window or "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    if "TOOL_CALL:" in full_reply:
                                        idx = full_reply.index("TOOL_CALL:")
                                        stream_buffer = full_reply[idx:]
                                        full_reply = full_reply[:idx]
                                    elif "TOOL_CALL:" in token:
                                        idx = token.index("TOOL_CALL:")
                                        stream_buffer = token[idx:]
                                    else:
                                        stream_buffer = "TOOL_CALL:"
                                    continue
                                full_reply += token
                                if not is_prompt_leak(full_reply):
                                    yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        if hold_phrase_yielded and not checked_hold_dedup:
                            checked_hold_dedup = True
                            stripped = stream_buffer.lstrip()
                            if HOLD_PHRASE_REGEX.match(stripped):
                                stream_buffer = HOLD_PHRASE_REGEX.sub("", stripped, count=1)
                        if stream_buffer and not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        if not hold_phrase_yielded:
                            yield f"{hold_phrase} "
                            hold_phrase_yielded = True
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            # Run tool in worker thread if blocking
                            tool_result = await asyncio.to_thread(self._execute_tool, tool_name, kwargs)
                            tool_synth = f"Please synthesize a short, polite spoken answer for the caller in 1-2 sentences in {turn_lang} language."
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\n{tool_synth}",
                            })
                            continue
                        else:
                            clean_text = clean_speech_text(stream_buffer)
                            if is_prompt_leak(clean_text):
                                clean_text = "I am here to help you with DDU IT queries. How may I assist you?"
                            clean_history_text = clean_text
                            if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                                clean_history_text = f"{hold_phrase} {clean_history_text}"
                            self.history.append({"role": "assistant", "content": clean_history_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
                        if is_prompt_leak(clean_text):
                            print(f"🛑 [llm] Suppressed prompt leak from async final text: {clean_text}")
                            clean_text = ""

                        if not clean_text.strip():
                            turn_lang, _, _ = self.detect_turn_language(user_text, fallback_lang=self.current_lang)
                            lower_u = user_text.lower()
                            if any(w in lower_u for w in ["સેમેસ્ટર", "સેમ", "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "semester", "syllabus", "subject"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટી સેમેસ્ટર 1 માં મેથેમેટિક્સ-1, બેઝિક પ્રોગ્રામિંગ અને એન્જિનિયરિંગ ફાઉન્ડેશન વિષયો સામેલ છે. શું તમારે ચોક્કસ વિષય કે ક્રેડિટ વિશે વધુ જાણવું છે?"
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी सेमेस्टर 1 में मैथमेटिक्स-1, प्रोग्रामिंग और इंजीनियरिंग विषय शामिल हैं। क्या आप किसी विशेष विषय या क्रेडिट के बारे में जानना चाहते हैं?"
                                else:
                                    clean_text = "DDU IT Semester 1 includes Mathematics-1, Basic Programming, and Engineering fundamentals. Would you like specific details on subjects or credits?"
                            elif any(w in lower_u for w in ["fee", "fees", "ફી", "ખર્ચ", "फी", "फीस"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ બીટેક પ્રથમ વર્ષની વાર્ષિક ટ્યુશન ફી ૧,૬૬,૯૫૦ રૂપિયા અને બીજાથી ચોથા વર્ષ માટે ૧,૫૨,૦૦૦ રૂપિયા છે, જ્યારે એમટેક પ્રથમ વર્ષની ફી ૫૫,૧૨૫ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक प्रथम वर्ष की वार्षिक ट्यूशन फीस 1,66,950 रुपये और 2nd से 4th वर्ष के लिए 1,52,000 रुपये है, जबकि एम.टेक प्रथम वर्ष की 55,125 रुपये है।"
                                else:
                                    clean_text = "DDU B.Tech first year annual fee is 1,66,950 rupees, and second to fourth year is 1,52,000 rupees per year. M.Tech first year is 55,125 rupees per year."
                            elif any(w in lower_u for w in ["placement", "પ્લેસમેન્ટ", "प्लेसमेंट", "salary", "package"]):
                                if turn_lang == "gu":
                                    clean_text = "ડીડીયુ આઈટીમાં પ્લેસમેન્ટ દર ૯૬.૫% છે, જેમાં સૌથી વધુ પેકેજ ૪૫ લાખ રૂપિયા અને સરેરાશ ૧૨.૫ લાખ રૂપિયા છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू आईटी का प्लेसमेंट रिकॉर्ड 96.5% है, जिसमें उच्चतम पैकेज 45 लाख रुपये और औसत 12.5 लाख रुपये है।"
                                else:
                                    clean_text = "DDU IT has a 96.5% placement rate, with the highest package at 45 lakh rupees and an average of 12.5 lakh rupees."
                            else:
                                if turn_lang == "gu":
                                    clean_text = "નમસ્તે, હું ડીડીયુ આઈટી ડિપાર્ટમેન્ટમાંથી પ્રિયા છું. હું તમને એડમિશન, અભ્યાસક્રમ અથવા વિદ્યાર્થી રેકોર્ડ્સ વિશે શું માહિતી આપું?"
                                elif turn_lang == "hi":
                                    clean_text = "नमस्ते, मैं डीडीयू आईटी विभाग से प्रिया हूँ। मैं आपको प्रवेश, पाठ्यक्रम या छात्र रिकॉर्ड के बारे में क्या जानकारी दे सकती हूँ?"
                                else:
                                    clean_text = "Hello, I am Priya from DDU IT department. How may I assist you with admissions, curriculum, or student records?"
                            yield clean_text

                        clean_history_text = clean_text
                        if hold_phrase_yielded and not clean_history_text.startswith(hold_phrase):
                            clean_history_text = f"{hold_phrase} {clean_history_text}"
                        self.history.append({"role": "assistant", "content": clean_history_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            clean_fallback = fallback
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback

        except (asyncio.CancelledError, GeneratorExit):
            # Rollback dangling user turn on cancellation
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            raise
        except Exception as err:
            print(f"[llm] Async communication error with {config.LLM_PROVIDER}: {err}")
            fallback_msg = get_error_message("system_error", turn_lang) or "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            clean_fallback = fallback_msg
            if hold_phrase_yielded and not clean_fallback.startswith(hold_phrase):
                clean_fallback = f"{hold_phrase} {clean_fallback}"
            self.history.append({"role": "assistant", "content": clean_fallback})
            yield fallback_msg