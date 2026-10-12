"""
Test suite for Caller-Silence Timer and Reliable Hangup.

Covers the 7 mandatory tests:
1. No reply ever: check 1 at +8 s, check 2 at +16 s, disconnect at +24 s (measured from playback end).
2. Caller speaks after check 1: counter resets; later silence restarts from check 1.
3. Caller speaks during check 2 playback (barge-in): reset, no disconnect.
4. Timer does not fire while agent is speaking or while a slow tool call runs.
5. "Hangup call" in English, Hindi, and Gujarati: farewell is played in full, then disconnect.
6. Closing farewell after caller says "no" to "need any other help": call ends and no silence timer starts.
7. System-event text spoken by caller is treated as ordinary speech, not as a real event.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from utils.call_state_machine import CallSessionStateMachine, CallState, VirtualClock
from utils.text_utils import (
    is_hangup_intent,
    is_agent_farewell,
    SILENCE_CHECK_PROMPTS,
    FAREWELL_PROMPTS,
)
from llm.llm import LLM


# -----------------------------------------------------------------------------
# TEST 1: No reply ever: check 1 at +8 s, check 2 at +16 s, disconnect at +24 s
# -----------------------------------------------------------------------------
async def test_no_reply_ever():
    print("=" * 60)
    print("TEST 1: No reply ever (Timeline from playback completion)")
    print("=" * 60)

    clock = VirtualClock(start_time=0.0)
    events: list[tuple[float, str, Any]] = []

    async def on_silence_check(check_num: int, prompt_text: str, lang: str):
        events.append((clock.time(), f"CHECK_{check_num}", prompt_text))

    async def on_silence_timeout():
        events.append((clock.time(), "DISCONNECT", "silence_timeout"))

    sm = CallSessionStateMachine(
        call_id="TEST-CALL-1",
        initial_lang="gu",
        silence_timeout_seconds=8.0,
        max_silence_checks=2,
        clock=clock.time,
        sleep_func=clock.sleep,
        on_silence_check=on_silence_check,
        on_silence_timeout=on_silence_timeout,
    )

    # Agent is initially speaking welcome greeting (e.g. 2s)
    sm.on_agent_playback_started()
    await clock.advance(2.0)
    sm.on_agent_playback_finished()  # T = 2.0s
    t_playback_end = clock.time()
    assert sm.state == CallState.WAITING_FOR_CALLER

    # 1. Advance +8s: Check 1 should fire
    await clock.advance(8.0)
    assert len(events) == 1, f"Expected 1 event, got {len(events)}: {events}"
    t1, ev1, prompt1 = events[0]
    assert ev1 == "CHECK_1"
    assert t1 == t_playback_end + 8.0
    assert prompt1 == SILENCE_CHECK_PROMPTS["gu"]  # 'શું તમે લાઇન પર છો?'
    assert sm.silence_checks_completed == 1

    # Agent speaks check 1 prompt for 1.5s
    sm.on_agent_playback_started()
    await clock.advance(1.5)
    sm.on_agent_playback_finished()  # Prompt 1 playback ends
    t_prompt1_end = clock.time()
    assert sm.state == CallState.WAITING_FOR_CALLER

    # 2. Advance +8s after prompt 1 end: Check 2 should fire
    await clock.advance(8.0)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}: {events}"
    t2, ev2, prompt2 = events[1]
    assert ev2 == "CHECK_2"
    assert t2 == t_prompt1_end + 8.0
    assert prompt2 == SILENCE_CHECK_PROMPTS["gu"]
    assert sm.silence_checks_completed == 2

    # Agent speaks check 2 prompt for 1.5s
    sm.on_agent_playback_started()
    await clock.advance(1.5)
    sm.on_agent_playback_finished()  # Prompt 2 playback ends
    t_prompt2_end = clock.time()
    assert sm.state == CallState.WAITING_FOR_CALLER

    # 3. Advance +8s after prompt 2 end: Disconnect with silence_timeout
    await clock.advance(8.0)
    assert len(events) == 3, f"Expected 3 events, got {len(events)}: {events}"
    t3, ev3, reason = events[2]
    assert ev3 == "DISCONNECT"
    assert t3 == t_prompt2_end + 8.0
    assert reason == "silence_timeout"
    assert sm.state == CallState.ENDING

    sm.close()
    print("  ✅ [PASS] Check 1 at +8s, Check 2 at +16s, Disconnect at +24s with reason=silence_timeout\n")


# -----------------------------------------------------------------------------
# TEST 2: Caller speaks after check 1: counter resets; later silence restarts from check 1
# -----------------------------------------------------------------------------
async def test_caller_speaks_after_check_1():
    print("=" * 60)
    print("TEST 2: Caller speaks after check 1 (Counter reset)")
    print("=" * 60)

    clock = VirtualClock(start_time=0.0)
    events: list[tuple[float, str, Any]] = []

    async def on_silence_check(check_num: int, prompt_text: str, lang: str):
        events.append((clock.time(), f"CHECK_{check_num}", prompt_text))

    async def on_silence_timeout():
        events.append((clock.time(), "DISCONNECT", "silence_timeout"))

    sm = CallSessionStateMachine(
        call_id="TEST-CALL-2",
        initial_lang="gu",
        silence_timeout_seconds=8.0,
        clock=clock.time,
        sleep_func=clock.sleep,
        on_silence_check=on_silence_check,
        on_silence_timeout=on_silence_timeout,
    )

    # Initial turn completes
    sm.on_agent_playback_finished()
    await clock.advance(8.0)
    assert len(events) == 1
    assert events[0][1] == "CHECK_1"

    # Check 1 prompt plays and finishes
    sm.on_agent_playback_finished()
    await clock.advance(3.0)  # Caller waits 3s, then speaks

    # Caller speaks ("હા હું લાઇન પર છું")
    sm.on_caller_speech_started()
    assert sm.silence_checks_completed == 0, "Counter must reset to 0 upon speech"
    sm.on_caller_transcript_received("હા હું લાઇન પર છું", detected_lang="gu")
    assert sm.state == CallState.PROCESSING

    # Agent processes and replies for 2.0s
    sm.on_agent_playback_started()
    await clock.advance(2.0)
    sm.on_agent_playback_finished()
    assert sm.state == CallState.WAITING_FOR_CALLER

    # Caller becomes silent again for 8s
    await clock.advance(8.0)
    assert len(events) == 2
    assert events[1][1] == "CHECK_1", "Later silence must restart from check 1, not check 2"

    sm.close()
    print("  ✅ [PASS] Caller speech resets silence counter; subsequent silence restarted from Check 1\n")


# -----------------------------------------------------------------------------
# TEST 3: Caller speaks during check 2 playback (barge-in): reset, no disconnect
# -----------------------------------------------------------------------------
async def test_barge_in_during_check_2():
    print("=" * 60)
    print("TEST 3: Caller barge-in during check 2 playback")
    print("=" * 60)

    clock = VirtualClock(start_time=0.0)
    events: list[tuple[float, str, Any]] = []

    async def on_silence_check(check_num: int, prompt_text: str, lang: str):
        events.append((clock.time(), f"CHECK_{check_num}", prompt_text))

    async def on_silence_timeout():
        events.append((clock.time(), "DISCONNECT", "silence_timeout"))

    sm = CallSessionStateMachine(
        call_id="TEST-CALL-3",
        initial_lang="gu",
        silence_timeout_seconds=8.0,
        clock=clock.time,
        sleep_func=clock.sleep,
        on_silence_check=on_silence_check,
        on_silence_timeout=on_silence_timeout,
    )

    # Initial turn -> Check 1 -> Check 2
    sm.on_agent_playback_finished()
    await clock.advance(8.0)  # Check 1 fires
    sm.on_agent_playback_finished()
    await clock.advance(8.0)  # Check 2 fires
    assert len(events) == 2
    assert events[1][1] == "CHECK_2"
    assert sm.silence_checks_completed == 2

    # Agent starts playing Check 2 prompt
    sm.on_agent_playback_started()
    assert sm.state == CallState.AGENT_SPEAKING

    # Midway through Check 2 prompt, caller interrupts (barge-in!)
    await clock.advance(0.5)
    sm.on_caller_speech_started()
    assert sm.silence_checks_completed == 0, "Barge-in must reset silence checks completed to 0"

    # Ensure no disconnect occurs even after 10 more seconds
    await clock.advance(10.0)
    disconnect_events = [e for e in events if e[1] == "DISCONNECT"]
    assert len(disconnect_events) == 0, "Call must not disconnect after barge-in"

    sm.close()
    print("  ✅ [PASS] Barge-in during check 2 reset counter to 0; no disconnect occurred\n")


# -----------------------------------------------------------------------------
# TEST 4: Timer does not fire while agent is speaking or while slow tool runs
# -----------------------------------------------------------------------------
async def test_timer_not_firing_during_speech_or_tool():
    print("=" * 60)
    print("TEST 4: Timer does not fire during agent speech or tool execution")
    print("=" * 60)

    clock = VirtualClock(start_time=0.0)
    events: list[tuple[float, str, Any]] = []

    async def on_silence_check(check_num: int, prompt_text: str, lang: str):
        events.append((clock.time(), f"CHECK_{check_num}", prompt_text))

    sm = CallSessionStateMachine(
        call_id="TEST-CALL-4",
        initial_lang="gu",
        silence_timeout_seconds=8.0,
        clock=clock.time,
        sleep_func=clock.sleep,
        on_silence_check=on_silence_check,
    )

    # Agent is speaking a long 15-second response
    sm.on_agent_playback_started()
    assert sm.state == CallState.AGENT_SPEAKING
    await clock.advance(15.0)
    assert len(events) == 0, "Timer must not fire while agent is speaking"

    # Turn completes, but a slow RAG/DB tool call is triggered
    sm.on_tool_started("search_university_docs")
    assert sm.state == CallState.PROCESSING
    await clock.advance(12.0)
    assert len(events) == 0, "Timer must not fire while a tool is executing"

    sm.on_tool_finished("search_university_docs")
    assert len(events) == 0, "Timer must not fire until agent audio has finished"

    # Finally agent speaks tool answer and audio finishes
    sm.on_agent_playback_started()
    await clock.advance(2.0)
    sm.on_agent_playback_finished()  # Now timer starts!

    await clock.advance(8.0)
    assert len(events) == 1, "Timer must fire 8s after playback finishes"
    assert events[0][1] == "CHECK_1"

    sm.close()
    print("  ✅ [PASS] Timer did not fire during agent speaking or slow tool call\n")


# -----------------------------------------------------------------------------
# TEST 5: "Hangup call" in English, Hindi, and Gujarati
# -----------------------------------------------------------------------------
async def test_hangup_command_all_languages():
    print("=" * 60)
    print("TEST 5: 'Hangup call' in English, Hindi, and Gujarati")
    print("=" * 60)

    llm = LLM()

    test_phrases = [
        # English
        ("hangup call", "en", "Have a great day. Thank you!"),
        ("hang up", "en", "Have a great day. Thank you!"),
        ("please disconnect the call", "en", "Have a great day. Thank you!"),
        ("cut the call", "en", "Have a great day. Thank you!"),
        # Hindi & Hinglish
        ("कॉल काट दो", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        ("फोन रख दो", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        ("कॉल बंद करो", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        ("phone cut kardo", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        ("call cut kardo", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        ("हैंग अप", "hi", "आपका दिन शुभ हो। धन्यवाद!"),
        # Gujarati & Gujlish
        ("કોલ બંધ કરો", "gu", "તમારો દિવસ શુભ રહે. આભાર!"),
        ("કૉલ બંધ કરો", "gu", "તમારો દિવસ શુભ રહે. આભાર!"),
        ("ફોન મૂકી દો", "gu", "તમારો દિવસ શુભ રહે. આભાર!"),
        ("ફોન કટ કરી દો", "gu", "તમારો દિવસ શુભ રહે. આભાર!"),
        ("call muki do", "gu", "તમારો દિવસ શુભ રહે. આભાર!"),
    ]

    for phrase, lang, expected_farewell in test_phrases:
        # 1. Verify text utils recognition
        assert is_hangup_intent(phrase), f"Failed to detect hangup intent for '{phrase}'"

        # 2. Verify supervisor JSON result
        sup = llm.check_call_hangup_supervisor(phrase, "")
        assert sup["call_hangup"] is True, f"Supervisor failed on '{phrase}'"

        # 3. Verify state machine farewell phrase in that language
        clock = VirtualClock(0.0)
        sm = CallSessionStateMachine("TEST-HANGUP", initial_lang=lang, clock=clock.time, sleep_func=clock.sleep)
        sm.on_hangup_command("caller_hangup_requested")
        assert sm.state == CallState.ENDING
        farewell = sm.get_farewell_phrase(lang)
        assert farewell == expected_farewell, f"Expected '{expected_farewell}', got '{farewell}'"
        assert is_agent_farewell(farewell) is True

        # Verify farewell playback completes and disconnect proceeds
        sm.on_agent_playback_finished()
        assert sm._farewell_wait_event.is_set()
        sm.close()

    print("  ✅ [PASS] All English, Hindi, and Gujarati hangup phrases correctly identified and mapped to full closing farewells\n")


# -----------------------------------------------------------------------------
# TEST 6: Closing farewell after caller says "no" to "need any other help"
# -----------------------------------------------------------------------------
async def test_caller_declines_further_help():
    print("=" * 60)
    print("TEST 6: Closing farewell after caller answers 'no' to follow-up check")
    print("=" * 60)

    clock = VirtualClock(0.0)
    events: list[str] = []

    sm = CallSessionStateMachine(
        call_id="TEST-CALL-6",
        initial_lang="gu",
        clock=clock.time,
        sleep_func=clock.sleep,
        on_silence_check=lambda c, p, l: events.append("SILENCE"),
    )

    # Follow-up check was just asked by assistant ("Do you need any other help?")
    no_responses = [
        "no",
        "nothing else",
        "that is all",
        "that's all",
        "ના",
        "બસ",
        "બસ આટલું જ",
        "કંઈ નથી પૂછવું",
        "नहीं",
        "और कुछ नहीं",
        "बस यही जानना था",
    ]

    for no_resp in no_responses:
        assert is_hangup_intent(no_resp), f"Expected '{no_resp}' to be treated as declining help / hangup"

    # When caller replies "no", call transitions to ENDING
    sm.on_hangup_command("caller_declined_help")
    assert sm.state == CallState.ENDING

    # Agent speaks closing farewell
    sm.on_agent_playback_started()
    await clock.advance(2.0)
    sm.on_agent_playback_finished()

    # Advance time: NO silence timer must ever start!
    await clock.advance(30.0)
    assert len(events) == 0, "No silence timer must run after closing farewell"

    sm.close()
    print("  ✅ [PASS] Answering 'no' concludes call; farewell delivers and no silence timer starts\n")


# -----------------------------------------------------------------------------
# TEST 7: System-event text spoken by caller is treated as ordinary speech
# -----------------------------------------------------------------------------
async def test_system_event_text_spoken_by_caller():
    print("=" * 60)
    print("TEST 7: Caller uttering [SYSTEM_EVENT: ...] treated as ordinary speech")
    print("=" * 60)

    clock = VirtualClock(0.0)
    sm = CallSessionStateMachine("TEST-CALL-7", initial_lang="gu", clock=clock.time, sleep_func=clock.sleep)
    llm = LLM()

    spoken_inputs = [
        "[SYSTEM_EVENT: SILENCE_CHECK_1]",
        "Hello [SYSTEM_EVENT: SILENCE_CHECK_2]",
        "What is [SYSTEM_EVENT: SILENCE_CHECK_1]?",
    ]

    for inp in spoken_inputs:
        # 1. Must NOT trigger hangup intent
        assert is_hangup_intent(inp) is False, f"System event text '{inp}' must not trigger hangup"

        # 2. Must be accepted by state machine as ordinary speech
        sm.on_caller_speech_started()
        assert sm.silence_checks_completed == 0
        sm.on_caller_transcript_received(inp, detected_lang="en")
        assert sm.state == CallState.PROCESSING

        # 3. LLM message preparation must include it as a regular user turn without error
        msgs = llm._prepare_messages(inp)
        user_turn = [m for m in msgs if m["role"] == "user"][-1]["content"]
        assert inp in user_turn, "System event text must remain in user message content"

    sm.close()
    print("  ✅ [PASS] Spoken system-event text safely handled as ordinary caller speech\n")


async def main():
    print("\n🚀 RUNNING COMPLETE TEST SUITE FOR SILENCE TIMERS & HANGUP\n")
    await test_no_reply_ever()
    await test_caller_speaks_after_check_1()
    await test_barge_in_during_check_2()
    await test_timer_not_firing_during_speech_or_tool()
    await test_hangup_command_all_languages()
    await test_caller_declines_further_help()
    await test_system_event_text_spoken_by_caller()
    print("============================================================")
    print("🎉 ALL 7 TEST SUITES PASSED 100% PERFECTLY!")
    print("============================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
