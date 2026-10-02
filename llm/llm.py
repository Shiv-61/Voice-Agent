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
from utils import is_hangup_intent, clean_speech_text, is_prompt_leak
from config.prompt_loader import (
    get_welcome_message,
    get_call_hangup_prompt,
    get_system_prompt,
    detect_active_domains,
    get_fallback,
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
    r'(?:fees?|फी|फीस|शुल्क|ખર્ચ|ફી|cost)': 'fee structure tuition fees annual',
    r'(?:b\.?tech|बीटेक|बी\.टेक|બીટેક|engineering|इंजीनियरिंग)': 'B.Tech Bachelor of Technology Engineering',
    r'(?:cse|it|computer|कम्प्यूटर|कंप्यूटर|કોમ્પ્યુટર|ઇન્ફોર્મેશન)': 'Computer Science Engineering Information Technology IT',
    r'(?:ddu|डीडीयू|धर्मसिंह|ધરમસિંહ)': 'DDU Dharamsinh Desai University Nadiad',
    r'(?:attendance|अटेंडेंस|हाजिरी|उपस्थिति|હાજરી)': 'attendance policy minimum percentage rules',
    r'(?:placement|प्लेसमेंट|नौकरी|पैकेज|सैलरी|પ્લેસમેન્ટ|salary|package)': 'placement highest average package recruiters companies',
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
    "shu", "ketli", "ketla", "che", "chhe", "maate", "nathi", "aapo", "tame",
    "tamara", "tamari", "tamaru", "aavde", "janavo", "kem", "vishe", "pucho",
    "aabhar", "namaste", "gujarati", "kai", "kayi", "bhanela", "kaho", "mane",
    "aapjo", "haji", "maru", "maro", "mari",
})
_HIN_INDICATORS: frozenset[str] = frozenset({
    "kya", "kitna", "kitni", "hai", "hain", "batao", "bata", "sakate", "sakthi",
    "sakthe", "hoga", "nahi", "kripya", "aapki", "aapka", "kaise", "kuch",
    "baare", "mein", "aur", "chahiye", "dena", "kaun", "kaha", "kahan", "kab",
    "hindi", "bataiye", "dijiye", "mujhe", "mera", "meri", "mere",
})
_ENG_INDICATORS: frozenset[str] = frozenset({
    "what", "when", "where", "how", "why", "who", "which", "is", "are", "can",
    "tell", "eligibility", "fee", "fees", "admission", "curfew", "placement",
    "package", "attendance", "marks", "thank", "goodbye", "hello", "hi", "yes",
    "no", "please", "english", "syllabus", "course", "subject", "semester",
    "credit", "curriculum", "department", "hostel", "rules", "campus", "direct",
})


class LLM:
    def __init__(self, db=None, rag=None):
        self.db = db or get_shared_db()
        self.rag = rag or get_shared_rag()
        self.current_lang = "gu"  # Gujarati is the primary language
        # Initialize conversation history with the assistant's opening welcome message
        self.history: list[dict[str, str]] = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    def detect_turn_language(self, text: str, fallback_lang: str = "gu") -> tuple[str, str, str]:
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
        if re.search(r"[\u0a80-\u0aff]", clean):
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી). Do NOT use Hindi or English.]:\n"
        if re.search(r"[\u0900-\u097f]", clean):
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी). Do NOT use Gujarati or English.]:\n"

        # Fix #13: use pre-compiled frozensets (module-level) for O(1) word lookup per language
        guj_overlap = len(words & _GUJ_INDICATORS)
        hin_overlap = len(words & _HIN_INDICATORS)
        eng_overlap = len(words & _ENG_INDICATORS)

        # Explicit language request triggers
        if "gujarati" in words:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"
        if "hindi" in words:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी).]:\n"
        if "english" in words and guj_overlap == 0 and hin_overlap == 0:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller requested ENGLISH. You MUST respond 100% in ENGLISH.]:\n"

        # High-confidence distinctive vocabulary
        if any(w in words for w in ["chhe", "nathi", "tamari", "kem", "maate", "aavde", "aapjo"]):
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"
        if any(w in words for w in ["batao", "kripya", "aapka", "aapki", "kaise", "chahiye", "bataiye"]):
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी).]:\n"

        # Multi-word overlap evaluation
        if guj_overlap >= 2 and guj_overlap >= hin_overlap:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in GUJARATI (Primary). You MUST respond 100% in GUJARATI using native Gujarati script (ગુજરાતી).]:\n"
        if hin_overlap >= 2:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in HINDI. You MUST respond 100% in HINDI using native Devanagari script (हिंदी).]:\n"
        if eng_overlap >= 1 and guj_overlap == 0 and hin_overlap == 0:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Caller asked in ENGLISH. You MUST respond 100% in ENGLISH.]:\n"

        # Neutral query (e.g. 'STU101' or numbers) -> retain session active language, fallback to Gujarati (Primary)
        fb = (fallback_lang or "").lower()
        if "hi" in fb:
            return "hi", "hi-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: HINDI (हिंदी लिपि).]:\n"
        elif "en" in fb:
            return "en", "en-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Continue in active conversation language: ENGLISH.]:\n"
        else:
            return "gu", "gu-IN", "[DYNAMIC LANGUAGE DIRECTIVE: Respond in primary language: GUJARATI (ગુજરાતી લિપિ).]:\n"

    def _prepare_messages(self, user_text: str) -> list[dict[str, str]]:
        """
        Prepares LLM messages with conversation history and anticipatory RAG / DB context.
        Pre-retrieval eliminates the slow, redundant 2nd turn LLM tool round-trip!
        """
        clean_user_text = transform_query(user_text)
        history_context = self._format_conversation_history()

        context_snippets = []

        # 1. Anticipatory RAG query with multilingual semantic expansion
        try:
            rag_query = expand_multilingual_query(clean_user_text)
            matches = self.rag.query_documents(rag_query, n_results=4)
            if matches and matches[0].get("similarity_score", 0) >= 0.50:
                for m in matches:
                    text_snippet = m.get("text", "").strip()
                    if text_snippet:
                        # Strip non-printable / private unicode symbols (e.g. font-icon glyphs)
                        text_snippet = re.sub(r'[^\x20-\x7E\u0900-\u097F\u0A80-\u0AFF\n\r\t]', ' ', text_snippet)
                        text_snippet = re.sub(r'\s+', ' ', text_snippet).strip()
                        filename = m.get("metadata", {}).get("filename", "University Document")
                        page_num = m.get("metadata", {}).get("page", 1)
                        context_snippets.append(f"[Official Document: {filename} (Page {page_num})]: {text_snippet[:1200]}")
        except Exception as e:
            print(f"[llm] Anticipatory RAG notice: {e}")

        # 2. Anticipatory DB check for admission or placement (English, Hindi, Gujarati)
        try:
            lower_text = clean_user_text.lower()
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
                prog = "CSE" if any(p in lower_text for p in [
                    "cse", "computer", "it", "btech", "b.tech", "बीटेक", "बी.टेक", "कंप्यूटर", "કોમ્પ્યુટર", "બીટેક"
                ]) else "ECE" if any(p in lower_text for p in ["ece", "electronics", "ईसीई", "इलेक्ट्रॉनिक्स"]) else ""
                adm = self.db.get_admission_info(prog)
                if adm:
                    context_snippets.append(f"[Official Admission & Fee Records]: {json.dumps(adm, default=str)}")

            # Student ID or Name pattern (supports English, Hindi, and Gujarati)
            student = None
            stu_match = re.search(r'\b(stu\d{2,4}|2[0-9]it\d{2,4}|it\d{2,4}|ce\d{2,4}|ec\d{2,4})\b', lower_text, re.IGNORECASE)
            if stu_match:
                stu_id = stu_match.group(1).upper()
                student = self.db.lookup_student(stu_id)
            else:
                try:
                    all_students = self.db.get_all_student_identifiers()
                    clean_query_trans = transliterate_student_name(clean_user_text).lower()
                    for s_item in all_students:
                        s_name = s_item.get("name", "").lower()
                        s_id = s_item.get("student_id", "").lower()
                        if s_name and (s_name in lower_text or s_name in clean_query_trans or s_id in lower_text):
                            student = self.db.lookup_student(s_item["student_id"])
                            break
                except Exception as de:
                    print(f"[llm] Dynamic student search notice: {de}")

            if student:
                stu_id = student.get("student_id")
                context_snippets.append(f"[Student Record {stu_id}]: {json.dumps(student, default=str)}")
                marks = self.db.get_student_marks(stu_id)
                if marks:
                    context_snippets.append(f"[Student Marks {stu_id}]: {json.dumps(marks, default=str)}")
                att = self.db.get_student_attendance(stu_id)
                if att:
                    context_snippets.append(f"[Student Attendance {stu_id}]: {json.dumps(att, default=str)}")
        except Exception as e:
            print(f"[llm] Anticipatory DB notice: {e}")

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
        lang_code, bcp47, lang_cue = self.detect_turn_language(clean_user_text, self.current_lang)
        self.current_lang = lang_code

        current_turn_content = f"{lang_cue}{clean_user_text}"
        if rag_context:
            current_turn_content = f"{rag_context}\n{lang_cue}{clean_user_text}"

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

    def check_call_hangup(self, user_text: str) -> bool:
        """
        Fast zero-latency evaluation of whether caller intends to end the call.
        Replaces slow blocking LLM round-trips with instant multi-lingual intent matching.
        """
        return is_hangup_intent(user_text)

    def _get_provider_request(self, messages: list[dict]):
        """Builds URL, headers, and payload for streaming LLM calls."""
        if config.LLM_PROVIDER == "sarvam":
            headers = {
                "api-subscription-key": config.SARVAM_API_KEY.strip(),
                "Content-Type": "application/json",
            }
            sarvam_model = config.LLM_MODEL if config.LLM_MODEL and "sarvam" in config.LLM_MODEL else "sarvam-105b-conversations"
            payload = {
                "model": sarvam_model,
                "messages": messages,
                "temperature": config.LLM_TEMPERATURE,
                "max_tokens": config.LLM_MAX_TOKENS,
                "stream": True,
            }
            return config.SARVAM_CHAT_URL, headers, payload
        elif config.LLM_PROVIDER == "groq":
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
        elif config.LLM_PROVIDER == "openrouter":
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

    def _parse_stream_line(self, line: str) -> str:
        """Extracts delta token text from SSE or JSON line."""
        line = line.strip()
        if not line:
            return ""
        if config.LLM_PROVIDER in ("sarvam", "openrouter", "groq"):
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

    def reply_stream(self, user_text: str) -> Generator[str, None, None]:
        """
        Synchronously yields response tokens incrementally in real time.
        Used by CLI and synchronous callers.
        """
        messages = self._prepare_messages(user_text)

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

                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think>)
                            if "<think>" in token:
                                in_think = True
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
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
                                elif len(stripped) >= 12 and "TOOL_CALL:" not in stream_buffer:
                                    if is_prompt_leak(stream_buffer):
                                        print(f"🛑 [llm] Suppressed prompt leak buffer in stream: {stream_buffer}")
                                        stream_buffer = ""
                                        continue
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
                        if not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            tool_result = self._execute_tool(tool_name, kwargs)
                            tool_synth = get_fallback("tool_synthesis_suffix", "Please synthesize a short, polite spoken answer for the caller in 1-2 sentences.")
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
                            self.history.append({"role": "assistant", "content": clean_text})
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
                                    clean_text = "ડીડીયુ બીટેક આઈટીની વાર્ષિક ટ્યુશન ફી ૨.૫ લાખ રૂપિયા છે અને અરજી કરવાની છેલ્લી તારીખ ૩૧ જુલાઈ ૨૦૨૬ છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी की वार्षिक ट्यूशन फीस 2.5 लाख रुपये है और आवेदन की अंतिम तिथि 31 जुलाई 2026 है।"
                                else:
                                    clean_text = "The annual tuition fee for DDU B.Tech IT is 2.5 lakh rupees, and the application deadline is 31 July 2026."
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

                        self.history.append({"role": "assistant", "content": clean_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            self.history.append({"role": "assistant", "content": fallback})
            yield fallback

        except Exception as err:
            print(f"[llm] Error communicating with {config.LLM_PROVIDER}: {err}")
            fallback_msg = "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            self.history.append({"role": "assistant", "content": fallback_msg})
            yield fallback_msg

    async def areply_stream(self, user_text: str) -> AsyncGenerator[str, None]:
        """
        Asynchronously yields response tokens incrementally in real time.
        Zero event-loop blocking for FastAPI and WebSocket servers.
        """
        messages = self._prepare_messages(user_text)

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

                    async with client.stream("POST", url, headers=headers, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            token = self._parse_stream_line(line)
                            if not token:
                                continue

                            # Filter thinking reasoning tokens (<think>...</think>)
                            if "<think>" in token:
                                in_think = True
                            if in_think:
                                if "</think>" in token:
                                    in_think = False
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
                                elif len(stripped) >= 12 and "TOOL_CALL:" not in stream_buffer:
                                    if is_prompt_leak(stream_buffer):
                                        print(f"🛑 [llm] Suppressed prompt leak buffer in async stream: {stream_buffer}")
                                        stream_buffer = ""
                                        continue
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
                        if not is_prompt_leak(stream_buffer):
                            full_reply += stream_buffer
                            yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            # Run tool in worker thread if blocking
                            tool_result = await asyncio.to_thread(self._execute_tool, tool_name, kwargs)
                            tool_synth = get_fallback("tool_synthesis_suffix", "Please synthesize a short, polite spoken answer for the caller in 1-2 sentences.")
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
                            self.history.append({"role": "assistant", "content": clean_text})
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
                                    clean_text = "ડીડીયુ બીટેક આઈટીની વાર્ષિક ટ્યુશન ફી ૨.૫ લાખ રૂપિયા છે અને અરજી કરવાની છેલ્લી તારીખ ૩૧ જુલાઈ ૨૦૨૬ છે."
                                elif turn_lang == "hi":
                                    clean_text = "डीडीयू बी.टेक आईटी की वार्षिक ट्यूशन फीस 2.5 लाख रुपये है और आवेदन की अंतिम तिथि 31 जुलाई 2026 है।"
                                else:
                                    clean_text = "The annual tuition fee for DDU B.Tech IT is 2.5 lakh rupees, and the application deadline is 31 July 2026."
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

                        self.history.append({"role": "assistant", "content": clean_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            self.history.append({"role": "assistant", "content": fallback})
            yield fallback

        except (asyncio.CancelledError, GeneratorExit):
            # Rollback dangling user turn on cancellation
            if self.history and self.history[-1].get("role") == "user":
                self.history.pop()
            raise
        except Exception as err:
            print(f"[llm] Async communication error with {config.LLM_PROVIDER}: {err}")
            fallback_msg = get_fallback("system_error", "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly.")
            self.history.append({"role": "assistant", "content": fallback_msg})
            yield fallback_msg