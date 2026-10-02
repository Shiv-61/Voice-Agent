"""
Test Suite: Dynamic In-Call Language Switching & Gujarati Primary Priority.
Verifies that:
1. Welcome message defaults to Gujarati (primary).
2. Question in Gujarati -> 100% Gujarati response.
3. Question in Hindi -> 100% Hindi response.
4. Question in English -> 100% English response.
5. In-call switching from Gujarati -> Hindi -> English -> Gujarati works seamlessly.
"""

import sys
import os

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.llm import LLM
from config.prompt_loader import get_welcome_message, load_prompts
from scripts.eval_multilingual_mirroring import detect_dominant_script


def test_yaml_multilingual_policy():
    print("\n--- 1. Testing YAML Multilingual Hierarchy & Policy ---")
    cfg = load_prompts()
    lp = cfg.get("language_policy", {})
    assert "gujarati" in lp.get("primary", "").lower(), f"Expected Gujarati as primary, got {lp.get('primary')}"
    secondaries = [s.lower() for s in lp.get("secondary", [])]
    assert any("hindi" in s for s in secondaries), "Hindi missing from secondary languages"
    assert any("english" in s for s in secondaries), "English missing from secondary languages"

    welcome = get_welcome_message()
    assert "નમસ્તે" in welcome, f"Welcome message should be in Gujarati, got: {welcome}"
    print(f"✓ Primary Language: {lp.get('primary')}")
    print(f"✓ Secondary Languages: {lp.get('secondary')}")
    print(f"✓ Welcome Message: \"{welcome}\"")


def test_dynamic_switching_detection():
    print("\n--- 2. Testing Dynamic Turn Language Detection ---")
    llm = LLM()

    # Turn 1: Gujarati (Primary)
    t1 = "ડીડીયુમાં એડમિશન માટે કેટલી ફી છે?"
    lang, bcp47, cue = llm.detect_turn_language(t1, "gu")
    assert lang == "gu" and bcp47 == "gu-IN", f"Expected gu, got {lang}"
    assert "GUJARATI" in cue
    print(f"✓ Turn 1 (Gujarati): '{t1}' -> detected: {lang} ({bcp47})")

    # Turn 2: Switch to Hindi
    t2 = "और हॉस्टल का क्या नियम है? कर्फ्यू का समय क्या है?"
    lang, bcp47, cue = llm.detect_turn_language(t2, "gu")
    assert lang == "hi" and bcp47 == "hi-IN", f"Expected hi, got {lang}"
    assert "HINDI" in cue
    print(f"✓ Turn 2 (Hindi Switch): '{t2}' -> detected: {lang} ({bcp47})")

    # Turn 3: Switch to English
    t3 = "What about the highest package and top recruiters?"
    lang, bcp47, cue = llm.detect_turn_language(t3, "hi")
    assert lang == "en" and bcp47 == "en-IN", f"Expected en, got {lang}"
    assert "ENGLISH" in cue
    print(f"✓ Turn 3 (English Switch): '{t3}' -> detected: {lang} ({bcp47})")

    # Turn 4: Switch back to Gujarati
    t4 = "આભાર, બસ આટલું જ પૂછવું હતું."
    lang, bcp47, cue = llm.detect_turn_language(t4, "en")
    assert lang == "gu" and bcp47 == "gu-IN", f"Expected gu, got {lang}"
    assert "GUJARATI" in cue
    print(f"✓ Turn 4 (Gujarati Switch Back): '{t4}' -> detected: {lang} ({bcp47})")

    # Turn 5: Neutral ID input maintains previous language
    t5 = "STU101"
    lang, bcp47, cue = llm.detect_turn_language(t5, "gu")
    assert lang == "gu" and bcp47 == "gu-IN"
    print(f"✓ Turn 5 (Neutral ID): '{t5}' -> maintained active language: {lang} ({bcp47})")


def test_live_llm_dynamic_switching_generation():
    print("\n--- 3. Testing Live LLM Multilingual Generation ---")
    llm = LLM()

    # Step A: Ask in Hindi
    hindi_q = "बीटेक कंप्यूटर साइंस की फीस कितनी है?"
    print(f"-> Caller asks in Hindi: '{hindi_q}'")
    reply_hi = "".join(list(llm.reply_stream(hindi_q)))
    script_hi = detect_dominant_script(reply_hi)
    print(f"<- Assistant reply: \"{reply_hi}\"")
    print(f"   Dominant Script: {script_hi.upper()}")
    assert script_hi == "hindi", f"Expected Hindi script, got {script_hi}"

    # Step B: Caller switches in same call to English
    eng_q = "What is the curfew time for the college hostel?"
    print(f"\n-> Caller switches in-call to English: '{eng_q}'")
    reply_en = "".join(list(llm.reply_stream(eng_q)))
    script_en = detect_dominant_script(reply_en)
    print(f"<- Assistant reply: \"{reply_en}\"")
    print(f"   Dominant Script: {script_en.upper()}")
    assert script_en == "english", f"Expected English script, got {script_en}"

    # Step C: Caller switches in same call to Gujarati
    guj_q = "ડીડીયુ આઈટીમાં હાઈએસ્ટ પેકેજ કેટલું મળ્યું છે?"
    print(f"\n-> Caller switches in-call to Gujarati: '{guj_q}'")
    reply_gu = "".join(list(llm.reply_stream(guj_q)))
    script_gu = detect_dominant_script(reply_gu)
    print(f"<- Assistant reply: \"{reply_gu}\"")
    print(f"   Dominant Script: {script_gu.upper()}")
    assert script_gu == "gujarati", f"Expected Gujarati script, got {script_gu}"


if __name__ == "__main__":
    print("=" * 65)
    print("🧪 RUNNING DYNAMIC IN-CALL LANGUAGE SWITCHING TEST SUITE")
    print("=" * 65)
    test_yaml_multilingual_policy()
    test_dynamic_switching_detection()
    test_live_llm_dynamic_switching_generation()
    print("\n🎉 ALL DYNAMIC IN-CALL LANGUAGE SWITCHING TESTS PASSED 100%!\n")
