"""
Text processing and conversational boundary utilities for real-time voice streaming.
"""

import re


# ---------------------------------------------------------------------------
# Shared language utilities (Fix #18 — eliminate TTS/STT duplication)
# ---------------------------------------------------------------------------

def normalize_lang(lang: str | None, default: str = "en-IN") -> str:
    """Normalises a BCP-47 language code to the exact format Sarvam expects."""
    if not lang or lang in ("unknown", ""):
        return default
    if lang in ("en", "en-IN"):
        return "en-IN"
    if lang in ("hi", "hi-IN"):
        return "hi-IN"
    if lang in ("gu", "gu-IN"):
        return "gu-IN"
    if len(lang) == 2:
        return f"{lang}-IN"
    return lang


# Multilingual error/notice messages (Fix #22)
_ERRORS: dict[str, dict[str, str]] = {
    "stt_error": {
        "gu-IN": "માફ કરશો, હું તમારો અવાજ સ્પષ્ટ ન સાંભળ્યો. કૃપા કરીને ફરીથી બોલો.",
        "hi-IN": "माफ़ कीजिए, आपकी आवाज़ स्पष्ट नहीं सुनाई दी। कृपया दोबारा बोलें।",
        "en-IN": "Sorry, I could not hear that clearly. Please speak again.",
    },
    "tts_error": {
        "gu-IN": "માફ કરશો, ઓડિઓ સ્ટ્રીમ કરવામાં સમસ્યા થઈ. ફરીથી પ્રયાસ કરો.",
        "hi-IN": "माफ़ कीजिए, ऑडियो में समस्या आई। कृपया दोबारा प्रयास करें।",
        "en-IN": "Sorry, there was an audio error. Please try again.",
    },
    "system_error": {
        "gu-IN": "માફ કરશો, સિસ્ટમ સાથે કનેક્ટ થવામાં મુશ્કેલી. થોડી વાર પછી પ્રયાસ કરો.",
        "hi-IN": "माफ़ कीजिए, सिस्टम से कनेक्ट करने में समस्या। थोड़ी देर बाद प्रयास करें।",
        "en-IN": "Sorry, trouble connecting to the university system. Please try again shortly.",
    },
}


def get_error_message(key: str, lang: str | None = None) -> str:
    """Returns a caller-language-appropriate error message."""
    norm = normalize_lang(lang, "en-IN")
    bucket = _ERRORS.get(key, _ERRORS["system_error"])
    return bucket.get(norm, bucket.get("en-IN", ""))


# Common honorifics and technical abbreviations across academic voice dialogues
ABBREVIATIONS = {
    "dr", "prof", "mr", "mrs", "ms", "sr", "jr", "vs",
    "i.e", "e.g", "dept", "univ", "govt", "b.tech", "m.tech",
    "ph.d", "bba", "mba", "bca", "mca", "b.sc", "m.sc"
}

# Fast-path hangup patterns across English, Hindi, and Gujarati (spoken by caller)
FAST_HANGUP_PATTERNS = [
    # English
    r"\b(bye|goodbye|bye[\s-]bye|good[\s-]bye)\b",
    r"\b(hang[\s-]?up|disconnect|cut the call|end the call|end call)\b",
    r"\b(that['’]?s all|that is all|nothing else|no more questions)\b",
    r"\b(have a (good|great|nice) day|see you later|talk to you later)\b",
    r"\b(thank you,?\s+bye|thanks,?\s+bye)\b",
    r"\b(thank you for your time)\b",
    # Hindi
    r"(अलविदा|बाय|बाय\s*बाय)",
    r"(फोन\s*रख\s*दो|कॉल\s*कट\s*कर\s*दो|कॉल\s*काट\s*दो)",
    r"(बस\s*इतना\s*ही|धन्यवाद[,\s]*बस|और\s*कुछ\s*नहीं|कोई\s*सवाल\s*नहीं)",
    r"(समय\s*के\s*लिए\s*धन्यवाद|दिन\s*शुभ\s*हो)",
    # Gujarati
    r"(આવજો|બાય|બાય\s*બાય)",
    r"(ફોન\s*મૂકી\s*દો|કૉલ\s*કટ\s*કરો)",
    r"(બસ\s*આટલું\s*જ|આભાર[,\s]*બસ|કંઈ\s*નથી\s*પૂછવું|કોઈ\s*પ્રશ્ન\s*નથી)",
    r"(સમય\s*માટે\s*આભાર|દિવસ\s*શુભ\s*રહે)",
]

COMPILED_HANGUP_REGEX = [re.compile(p, re.IGNORECASE) for p in FAST_HANGUP_PATTERNS]

# Patterns detecting the exact agent closing farewell:
# "Thank you for your time. Have a great day!" across English, Hindi, and Gujarati
AGENT_FAREWELL_PATTERNS = [
    # English
    r"\bthank\s+you\s+for\s+your\s+time\b",
    r"\bhave\s+a\s+(great|good|nice|wonderful)\s+day\b",
    # Hindi (Devanagari)
    r"(?:आपके\s*)?समय\s*(?:के\s*लिए|देने\s*के\s*लिए)\s*धन्यवाद",
    r"(?:आपका\s*)?दिन\s*(?:शुभ|अच्छा|मंगलमय)\s*(?:हो|रहे|बने)",
    # Gujarati (Gujarati script)
    r"(?:તમારા\s*)?સમય\s*(?:માટે|આપવા\s*બદલ)\s*આભાર",
    r"(?:તમારો\s*)?દિવસ\s*(?:શુભ|સારો)\s*રહે",
    # Romanized / Transliterated
    r"\baapke\s+samay\s+ke\s+liye\s+dhanyawad\b",
    r"\baapka\s+din\s+shubh\b",
    r"\btamara\s+samay\s+maate\s+aabhar\b",
    r"\btamaro\s+divas\s+shubh\b",
]

COMPILED_AGENT_FAREWELL_REGEX = [re.compile(p, re.IGNORECASE) for p in AGENT_FAREWELL_PATTERNS]


def split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Splits text into ready sentences while preserving numbers with decimals
    (e.g., '8.5 CGPA', '45.0 LPA') and abbreviations ('Dr.', 'Prof.', 'B.Tech.').

    Returns (list_of_complete_sentences, remainder_buffer).
    """
    if not buffer:
        return [], ""

    # Match sentence punctuation (. ! ? । \n) followed by whitespace
    pattern = re.compile(r"([.!?।\n]+)(\s+)")
    pos = 0
    complete = []

    for m in pattern.finditer(buffer):
        punct = m.group(1)
        end_idx = m.end()
        punct_idx = m.start(1)

        # If '.', check if preceded by digit (e.g. 8.5) or abbreviation
        if "." in punct:
            prefix = buffer[:punct_idx]
            # Check if immediately preceded by a digit
            if prefix and prefix[-1].isdigit():
                continue
            # Check last word
            words = prefix.split()
            if words:
                last_word = words[-1].lower()
                last_word = re.sub(r"^[^\w]+|[^\w.]+$", "", last_word)
                if last_word in ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                    continue

        sentence = buffer[pos:punct_idx + len(punct)].strip()
        if sentence:
            complete.append(sentence)
        pos = end_idx

    remainder = buffer[pos:]
    return complete, remainder


def is_hangup_intent(user_text: str) -> bool:
    """
    Fast sub-millisecond evaluation of whether caller intends to conclude/hang up.
    Replaces slow, blocking LLM calls for hangup evaluation.
    """
    if not user_text:
        return False

    clean_text = user_text.strip().lower()
    for regex in COMPILED_HANGUP_REGEX:
        if regex.search(clean_text):
            return True

    return False


def is_agent_farewell(agent_text: str) -> bool:
    """
    Fast sub-millisecond evaluation of whether the agent has uttered the closing farewell:
    "Thank you for your time. Have a great day!" (in English, Hindi, or Gujarati).
    """
    if not agent_text:
        return False

    clean_text = agent_text.strip()
    for regex in COMPILED_AGENT_FAREWELL_REGEX:
        if regex.search(clean_text):
            return True

    return False


def clean_speech_text(text: str) -> str:
    """
    Strips markdown formatting, bold/italics markers, hashes, URLs, and code blocks
    so the speech synthesizer produces pristine audio.
    """
    if not text:
        return ""

    # Remove thinking tags
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove code fences and inline code
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"`.*?`", "", cleaned)
    # Remove markdown headers and list bullets
    cleaned = re.sub(r"^[#*+\-\s]+", "", cleaned, flags=re.MULTILINE)
    # Remove formatting characters (*, _, #, ~, [])
    cleaned = re.sub(r"[*_#~]", "", cleaned)
    cleaned = re.sub(r"\[.*?\]", "", cleaned)
    # Remove third-person LLM meta-reasoning prefixes (e.g. "The user is asking...", "I should answer...")
    cleaned = re.sub(r"^(?:The user is (?:asking|inquiring|looking|requesting|wondering)[^.]*\.\s*)+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:I should (?:answer|provide|tell|respond)[^.]*\.\s*)+", "", cleaned, flags=re.IGNORECASE)
    # Strip spoken preambles like "Sure, let me check that for you" or "Let me check"
    cleaned = re.sub(
        r"^(?:(?:sure|okay|ok|yes),?\s*)?(?:let me check(?:\s+that)?(?:\s+for you)?|i will check(?:\s+that)?(?:\s+for you)?|just a moment(?:\s+please)?)[.,!]*\s*",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"^(?:(?:હા|જી|હાજી),?\s*)?(?:હું હમણાં જ (?:વિગતો જોઈ લઉં છું|ચેક કરું છું)|તપાસીને જણાવું છું)[.,!।]*\s*",
        "", cleaned, flags=re.IGNORECASE
    )
    cleaned = re.sub(
        r"^(?:(?:जी|हाँ|हां),?\s*)?(?:मैं अभी (?:चेक करके बताती हूँ|देखती हूँ|जाँच करती हूँ)|पता करके बताती हूँ)[.,!।]*\s*",
        "", cleaned, flags=re.IGNORECASE
    )
    # Normalize excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Standard hold phrases across Gujarati, Hindi, and English for DB & RAG lookups
HOLD_PHRASES: dict[str, str] = {
    "gu": "કૃપા કરીને લાઇન પર રહો.",
    "hi": "कृपया लाइन पर बने रहें।",
    "en": "Please stay on the line.",
}

# Regex to detect and strip duplicate hold phrases produced by LLMs
HOLD_PHRASE_REGEX = re.compile(
    r"^(?:(?:હા|જી|હાજી|हाँ|जी|sure|ok|okay),?\s*)?"
    r"(?:(?:કૃપા\s*કરીને|કૃપા\s*કરી)\s*(?:લાઇન|લાઈન)\s*પર\s*(?:રહો|બને\s*રહો)|"
    r"कृपया\s*लाइन\s*पर\s*(?:बने\s*रहें|रहें)|"
    r"please\s*(?:stay|hold)\s*(?:on\s*)?(?:the\s*)?line)"
    r"(?:[.,!|।]*\s*(?:હું\s*(?:હમણાં\s*જ\s*)?(?:વિગતો|માહિતી)\s*(?:જોઈ\s*લઉં\s*છું|તપાસું\s*છું)|"
    r"मैं\s*(?:अभी\s*)?(?:जानकारी|डिटेल्स|रिकॉर्ड)\s*(?:देख\s*रही\s*हूँ|चेक\s*करती\s*हूँ)|"
    r"(?:let\s*me\s*check|let\s*me\s*find)\s*(?:that|the\s*details)?(?:\s*for\s*you)?))?[.,!।]*\s*",
    re.IGNORECASE
)


def query_needs_db_or_rag(text: str, has_student: bool = False) -> bool:
    """
    Detects if caller's query requires looking up records in the database or RAG document store.
    Personal student queries only return True if a student identifier is present or resolved.
    """
    if not text:
        return False
    lower = text.lower()

    # 1. Direct Student ID match (always True)
    has_student_id = bool(re.search(r'\b(?:stu\s*\d+|student\s*\d+|2[0-9]it\d+|it\d+|ce\d+|ec\d+)\b', lower))
    if has_student_id:
        return True

    # Distinguish personal student queries from general policy inquiries
    personal_indicators = [
        "son", "daughter", "child", "kid", "beta", "beti", "bachha", "baccha",
        "dikro", "dikra", "dikri", "chokro", "chokra", "chokri",
        "maro", "mari", "maru", "mara", "mare",
        "mera", "meri", "mere", "my", "his", "her", "student", "roll",
        "દીકર", "છોકર", "પુત્ર", "માર", "તમાર", "વિદ્યાર્થી", "બાળક",
        "बेट", "बच्च", "पुत्र", "मेर", "छात्र", "विद्यार्थी", "रोल"
    ]
    policy_indicators = [
        "rule", "rules", "policy", "policies", "criteria", "requirement", "requirements",
        "minimum", "mandatory", "compulsory", "allowed", "regulation", "regulations",
        "નિયમ", "નિયમો", "પોલિસી", "શરત", "જરૂરી", "ઓછામાં ઓછી",
        "નિયમો", "નિયમ",
        "नियम", "पॉलिसी", "शर्त", "जरूरी", "न्यूनतम", "अनिवार्य"
    ]
    student_keywords = [
        "attendance", "attedance", "atendance", "attandance", "attendence",
        "marks", "cpi", "cgpa", "spi", "result", "grade", "score", "roll number",
        "roll no", "student", "batch", "passing year", "mobile number", "contact number",
        "એટેન્ડન્સ", "હાજરી", "માર્ક્સ", "ગુણ", "સીપીઆઈ", "વિદ્યાર્થી", "રોલ નંબર", "પરિણામ",
        "અટેન્ડન્સ", "માર્ક",
        "अटेंडेंस", "उपस्थिति", "मार्क्स", "अंक", "सीपीआई", "छात्र", "रोल नंबर", "रिजल्ट",
        "परिणाम", "छात्र रिकॉर्ड",
    ]

    has_student_kw = any(k in lower for k in student_keywords)
    has_personal_ind = any(k in lower for k in personal_indicators)
    has_policy_ind = any(k in lower for k in policy_indicators)

    # Personal student queries without an identified student do NOT need DB yet
    if has_student_kw:
        if has_student:
            return True
        if has_personal_ind and not has_policy_ind:
            return False
        if not has_policy_ind and any(k in lower for k in ["attendance", "attedance", "marks", "result", "હાજરી", "માર્ક્સ", "अटेंडेंस", "मार्क्स"]):
            return False
        if has_policy_ind:
            return True

    # 2. Admission & Fees queries
    adm_keywords = [
        "admission", "eligibility", "fee", "fees", "tuition", "deadline", "last date",
        "cutoff", "cut-off", "seat", "seats", "intake", "apply", "process",
        "એડમિશન", "દાખલો", "પ્રવેશ", "ફી", "લાયકાત", "છેલ્લી તારીખ", "બેઠક",
        "एडमिशन", "प्रवेश", "दाखिला", "फीस", "फी", "शुल्क", "पात्रता", "कटऑफ",
        "अंतिम तिथि", "सीट",
    ]
    if any(k in lower for k in adm_keywords):
        return True

    # 3. Placement records & Batch 2026 highlights
    placement_keywords = [
        "placement", "placements", "package", "salary", "recruiter", "recruiters",
        "highest package", "average package", "placed", "company", "companies",
        "batch 2026", "2026 batch", "offers", "higher studies", "lpa",
        "પ્લેસમેન્ટ", "પેકેજ", "નોકરી", "કંપની", "ઓફર", "હાઈલાઈટ્સ", "૨૦૨૬", "સૌથી વધુ", "સરેરાશ",
        "प्लेसमेंट", "पैकेज", "सैलरी", "नौकरी", "कंपनी", "कंपनियां", "ऑफर", "हाइलाइट्स", "2026", "उच्चतम", "औसत",
    ]
    if any(k in lower for k in placement_keywords):
        return True

    # 4. Student Achievements, Hackathons, Competitive Exams & Research Papers
    achievement_keywords = [
        "hackathon", "hackathons", "sui overflow", "ethglobal", "codeversity",
        "suisign", "synapsemodel", "skillscreen", "prize", "prizes", "award", "awards",
        "competition", "competitions", "achievement", "achievements",
        "gate", "ncat", "ielts", "toefl", "pte", "percentile", "air", "rank",
        "research", "paper", "papers", "publication", "publications", "conference", "conferences",
        "journal", "darpan vora", "abhimanyu", "pruthviraj", "harsh manek",
        "shreyas warrier", "om patel",
        "હેકાથોન", "હરીફાઈ", "પુરસ્કાર", "ઇનામ", "સિદ્ધિ", "ગેટ", "ટોફલ", "પર્સન્ટાઈલ",
        "રિસર્ચ", "પેપર", "સંશોધન", "કોન્ફરન્સ",
        "हैकथॉन", "प्रतियोगिता", "पुरस्कार", "उपलब्धि", "गेट", "टोफेल", "रैंक", "पर्सेंटाइल",
        "रिसर्च", "पेपर", "शोध", "सम्मेलन",
    ]
    if any(k in lower for k in achievement_keywords):
        return True

    # 5. Curriculum, Syllabus, Hostel, Policy documents (RAG)
    rag_keywords = [
        "syllabus", "curriculum", "subject", "subjects", "semester", "credit", "credits",
        "course structure", "teaching scheme", "exam", "examination",
        "hostel", "mess", "curfew", "rules", "rule", "regulations", "scholarship",
        "financial aid", "ragging", "anti-ragging", "faculty", "hod", "library",
        "campus timing",
        "સિલેબસ", "અભ્યાસક્રમ", "વિષય", "વિષયો", "સેમેસ્ટર", "ક્રેડિટ", "પરીક્ષા",
        "હોસ્ટેલ", "મેસ", "નિયમ", "નિયમો", "સ્કોલરશિપ", "છાત્રાલય", "પુસ્તકાલય",
        "सिलेबस", "पाठ्यक्रम", "विषय", "विषयों", "सेमेस्टर", "क्रेडिट", "परीक्षा",
        "हॉस्टल", "मेस", "नियम", "स्कॉलरशिप", "छात्रवृत्ति", "छात्रावास", "पुस्तकालय",
    ]
    if any(k in lower for k in rag_keywords):
        return True

    return False


STANDALONE_FILLER_PATTERNS = [
    r"^(?:sure,?\s*)?let me check(?:\s+that)?(?:\s+for you)?[.!?]*$",
    r"^(?:sure,?\s*)?let me look(?:\s+that)?(?:\s+up)?[.!?]*$",
    r"^(?:sure,?\s*)?just a moment(?:\s+please)?[.!?]*$",
    r"^(?:sure,?\s*)?i will check(?:\s+that)?(?:\s+for you)?[.!?]*$",
    r"^(?:હા,?\s*)?હું હમણાં જ વિગતો જોઈ લઉં છું[.!?।]*$",
    r"^(?:જી,?\s*)?હું હમણાં જ (?:વિગતો જોઈ લઉં છું|ચેક કરું છું)[.!?।]*$",
    r"^(?:जी,?\s*)?मैं अभी चेक करके बताती हूँ[.!?।]*$",
]
COMPILED_FILLER_REGEX = [re.compile(p, re.IGNORECASE) for p in STANDALONE_FILLER_PATTERNS]


def is_filler_phrase(text: str) -> bool:
    """Detects standalone filler / preamble phrases that shouldn't be spoken."""
    if not text:
        return False
    clean = text.strip()
    # Explicitly protect mandatory hold phrases
    return any(r.match(clean) for r in COMPILED_FILLER_REGEX)


GREETING_PATTERNS = {
    "hello", "hi", "hey", "halo", "namaste", "namaskar", "namaskaram",
    "kem cho", "kem chho", "kemcho", "kemchho", "good morning", "good afternoon",
    "good evening", "good day", "kaise ho", "kya haal hai", "kya haal",
    "નમસ્તે", "નમસ્કાર", "કેમ છો", "કેમછો", "હેલો", "હાય", "સુપ્રભાત",
    "नमस्ते", "नमस्कार", "हेलो", "हाय", "सुप्रभात", "कैसे हो", "क्या हाल है", "प्रणाम"
}


def is_simple_greeting(text: str) -> bool:
    """
    Returns True if the utterance is solely or primarily a conversational greeting
    without asking any factual, academic, or institutional question.
    """
    if not text:
        return False
    clean = re.sub(r'[.,!?।;:\-_"\'(){}\[\]<>/\\#*&~`+=]', ' ', text.strip().lower())
    words = clean.split()
    if not words:
        return False
    joined = " ".join(words)
    if joined in GREETING_PATTERNS or clean.strip() in GREETING_PATTERNS:
        return True
    if len(words) <= 3 and any(g in joined for g in ["hello", "hi", "hey", "namaste", "kem cho", "કેમ છો", "नमस्ते"]):
        question_words = {
            "fee", "fees", "admission", "marks", "result", "cpi", "syllabus", "hostel",
            "placement", "package", "branch", "cutoff", "eligibility",
            "ફી", "પ્રવેશ", "માર્ક્સ", "હાજરી", "फीસ", "दाखिला", "नंबर", "कटऑफ"
        }
        if not (set(words) & question_words):
            return True
    return False



PROMPT_LEAK_PATTERNS = [
    r"you are a (?:polite|helpful|professional|university|virtual|voice|ai)",
    r"university admission & student desk assistant",
    r"call initiation & welcome message",
    r"welcome message flow",
    r"strict safety & anti-abuse",
    r"strict unaware",
    r"voice call style & conversational rules",
    r"10 to 15 words per sentence",
    r"ai identity transparency",
    r"natural spoken pronunciation",
    r"exact language mirroring",
    r"context continuity",
    r"direct context utilization",
    r"verified official university context",
    r"end official context",
    r"here'?s a thinking process",
    r"thinking process",
    r"the user is asking",
    r"caller's current question",
    r"caller's question",
    r"preceding conversation",
    r"system prompt",
    r"available tools:",
    r"tool calling:",
    r"tool_result",
]

COMPILED_PROMPT_LEAK_REGEX = [re.compile(p, re.IGNORECASE) for p in PROMPT_LEAK_PATTERNS]


def is_prompt_leak(text: str) -> bool:
    """Detects whether text contains internal prompt instructions or meta-commentary."""
    if not text:
        return False
    lower = text.strip().lower()
    return any(r.search(lower) for r in COMPILED_PROMPT_LEAK_REGEX)


COMMON_NOISE_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks", "thanks.", "thank you very much.",
    "धन्यवाद", "धन्यवाद।", "शुक्रिया", "આભાર", "આભાર.",
    "ha", "haan", "haa", "hum", "hmm", "uh", "um", "ah", "oh", "you", "the",
    "bye", "okay", "yes", "no", "ok",
    "हा", "हाँ", "हूँ", "हम्म", "હું", "હા", "ના",
}


def is_noise_hallucination(text: str, audio_dur: float = 1.0, audio_rms: float = 0.05) -> bool:
    """
    Detects whether an STT transcript is an artifact or hallucination produced from
    ambient room noise, breathing, mouse clicks, or silence.
    """
    if not text:
        return True
    cleaned = text.strip()
    # 1. Punctuation only, empty, or completely devoid of English, Hindi, or Gujarati characters
    if not re.search(r"[a-zA-Z0-9\u0900-\u097F\u0A80-\u0AFF]", cleaned):
        return True

    # 1b. Discard transcripts containing non-target Indic scripts (Odia, Telugu, Kannada, etc.)
    if re.search(r"[\u0980-\u0a7f\u0b00-\u0d7f]", cleaned):
        return True

    # 2. Very short audio or low RMS that produced common single-word silence hallucinations
    lower = cleaned.lower().strip(".,!?। ")
    if (audio_dur < 1.4 or audio_rms < 0.030) and lower in COMMON_NOISE_HALLUCINATIONS:
        return True

    # 3. Repeated hesitation or breath sounds (e.g., 'uhhh', 'hmmm', 'ahhh')
    if re.fullmatch(r"(?:[uhmaoe]|hm)+", lower):
        return True

    return False

