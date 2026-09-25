"""
Text processing and conversational boundary utilities for real-time voice streaming.
"""

import re

# Common honorifics and technical abbreviations across academic voice dialogues
ABBREVIATIONS = {
    "dr", "prof", "mr", "mrs", "ms", "sr", "jr", "vs",
    "i.e", "e.g", "dept", "univ", "govt", "b.tech", "m.tech",
    "ph.d", "bba", "mba", "bca", "mca", "b.sc", "m.sc"
}

# Fast-path hangup patterns across English, Hindi, and Gujarati
FAST_HANGUP_PATTERNS = [
    # English
    r"\b(bye|goodbye|bye[\s-]bye|good[\s-]bye)\b",
    r"\b(hang[\s-]?up|disconnect|cut the call|end the call|end call)\b",
    r"\b(that['’]?s all|that is all|nothing else|no more questions)\b",
    r"\b(have a (good|great|nice) day|see you later|talk to you later)\b",
    r"\b(thank you,?\s+bye|thanks,?\s+bye)\b",
    # Hindi
    r"(अलविदा|बाय|बाय\s*बाय)",
    r"(फोन\s*रख\s*दो|कॉल\s*कट\s*कर\s*दो|कॉल\s*काट\s*दो)",
    r"(बस\s*इतना\s*ही|धन्यवाद[,\s]*बस|और\s*कुछ\s*नहीं|कोई\s*सवाल\s*नहीं)",
    # Gujarati
    r"(આવજો|બાય|બાય\s*બાય)",
    r"(ફોન\s*મૂકી\s*દો|કૉલ\s*કટ\s*કરો)",
    r"(બસ\s*આટલું\s*જ|આભાર[,\s]*બસ|કંઈ\s*નથી\s*પૂછવું|કોઈ\s*પ્રશ્ન\s*નથી)",
]

COMPILED_HANGUP_REGEX = [re.compile(p, re.IGNORECASE) for p in FAST_HANGUP_PATTERNS]


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
    # Normalize excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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

