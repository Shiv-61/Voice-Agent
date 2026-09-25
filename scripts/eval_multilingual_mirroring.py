"""
Deterministic Multilingual Mirroring Validator for College Voice Agent.
Inspired by voice-ai-agent-builder's multilingual_eval:
Uses fast, free Unicode script block analysis (NO LLM needed) to audit
that the agent mirrors caller language (English, Hindi, Gujarati) with 100% precision.
"""

import sys
import re

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Unicode ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
GUJARATI_RANGE = (0x0A80, 0x0AFF)
LATIN_RANGE = (0x0041, 0x007A)

# Permitted English loanwords across Indian colleges (do not penalize if spoken in Hindi/Gujarati)
COLLEGE_LOANWORDS = {
    "btech", "b.tech", "cse", "it", "ece", "ddu", "jee", "percent", "lpa",
    "semester", "fees", "admission", "curfew", "hostel", "package", "stu101",
    "cgpa", "exam", "ai", "attendance", "marks", "rank", "gujcet", "acpc"
}

def detect_dominant_script(text: str) -> str:
    """Classifies text into 'hindi', 'gujarati', or 'english' via Unicode script density."""
    clean_text = text.lower()
    for word in COLLEGE_LOANWORDS:
        clean_text = clean_text.replace(word, "")

    counts = {"hindi": 0, "gujarati": 0, "english": 0}

    for ch in clean_text:
        code = ord(ch)
        if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
            counts["hindi"] += 1
        elif GUJARATI_RANGE[0] <= code <= GUJARATI_RANGE[1]:
            counts["gujarati"] += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            counts["english"] += 1

    # If non-Latin scripts are present, pick the highest non-Latin
    if counts["hindi"] > 0 or counts["gujarati"] > 0:
        return "hindi" if counts["hindi"] >= counts["gujarati"] else "gujarati"
    elif counts["english"] > 0:
        return "english"
    return "unknown"


def audit_turn_mirroring(caller_text: str, agent_text: str, expected_lang: str) -> dict:
    """Audits whether agent correctly mirrored caller language in this turn."""
    detected_agent_script = detect_dominant_script(agent_text)
    match = (detected_agent_script == expected_lang)

    return {
        "expected_lang": expected_lang,
        "detected_script": detected_agent_script,
        "is_correct": match,
        "caller_snippet": caller_text[:60],
        "agent_snippet": agent_text[:60],
    }


def main():
    print("=============================================================")
    print("🎓 DETERMINISTIC MULTILINGUAL SCRIPT AUDITOR (COLLEGE DESK)")
    print("=============================================================\n")

    test_pairs = [
        ("What is the fee for B.Tech CSE?", "The fee for B.Tech Computer Science is 2.5 lakh rupees per year.", "english"),
        ("बीटेक में एडमिशन के लिए क्या पात्रता है?", "बीटेक में एडमिशन के लिए 10+2 में फिजिक्स, केमिस्ट्री और मैथ्स में 60 प्रतिशत अंक चाहिए।", "hindi"),
        ("ડીડીયુમાં પ્લેસમેન્ટ કેવું છે?", "ડીડીયુ આઈટી બ્રાન્ચમાં હાઈએસ્ટ પેકેજ 45 લાખ અને એવરેજ 12.5 લાખ રૂપિયા છે.", "gujarati"),
        ("Can I get hostel accommodation?", "Hostel facilities are available for both boys and girls with night curfew at 9:30 PM.", "english"),
        ("क्या अटेंडेंस 75% अनिवार्य है?", "हाँ, परीक्षा में बैठने के लिए कम से कम 75 प्रतिशत हाजिरी अनिवार्य है।", "hindi"),
    ]

    passed = 0
    for idx, (caller, agent, expected) in enumerate(test_pairs, start=1):
        res = audit_turn_mirroring(caller, agent, expected)
        status = "✅ PASS" if res["is_correct"] else "❌ FAIL"
        if res["is_correct"]:
            passed += 1
        print(f"Turn {idx}: {status}")
        print(f"  Caller: \"{res['caller_snippet']}\"")
        print(f"  Agent:  \"{res['agent_snippet']}\"")
        print(f"  Expected: {expected} | Detected: {res['detected_script']}\n")

    accuracy = (passed / len(test_pairs)) * 100
    print(f"Result: {passed}/{len(test_pairs)} passed ({accuracy:.1f}% Mirroring Accuracy)")
    assert passed == len(test_pairs), "Multilingual script auditor failed on test pairs!"
    print("✨ ALL DETERMINISTIC MULTILINGUAL CHECKS PASSED!\n")

if __name__ == "__main__":
    main()
