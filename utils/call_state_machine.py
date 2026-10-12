"""
Call Session State Machine for Voice Agent.
Implements explicit state management, caller-silence timers, language locking,
and reliable hangup orchestration.
"""

from __future__ import annotations

import asyncio
from enum import Enum
import logging
import time
from typing import Awaitable, Callable, Optional, Any

import config
from utils.text_utils import SILENCE_CHECK_PROMPTS, FAREWELL_PROMPTS

logger = logging.getLogger("voice_agent.state_machine")


class CallState(str, Enum):
    AGENT_SPEAKING = "AGENT_SPEAKING"
    PROCESSING = "PROCESSING"
    WAITING_FOR_CALLER = "WAITING_FOR_CALLER"
    ENDING = "ENDING"


class VirtualClock:
    """
    Virtual clock for testing that simulates time advancement without real delays.
    Allows exact verification of silence timeouts (+8s, +16s, +24s) in unit tests.
    """

    def __init__(self, start_time: float = 0.0):
        self._current_time = float(start_time)
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def time(self) -> float:
        return self._current_time

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        wake_time = self._current_time + delay
        self._sleepers.append((wake_time, fut))
        self._sleepers.sort(key=lambda x: x[0])
        try:
            await fut
        except asyncio.CancelledError:
            self._sleepers = [(t, f) for t, f in self._sleepers if f is not fut]
            raise

    async def advance(self, seconds: float) -> None:
        """Advances virtual time and resolves any sleepers that are due."""
        # Yield to allow any newly spawned asyncio.create_task to enter sleep()
        await asyncio.sleep(0)
        self._current_time += seconds
        ready: list[asyncio.Future[None]] = []
        remaining: list[tuple[float, asyncio.Future[None]]] = []
        for wake_time, fut in self._sleepers:
            if wake_time <= self._current_time and not fut.done():
                ready.append(fut)
            else:
                remaining.append((wake_time, fut))
        self._sleepers = remaining
        for fut in ready:
            if not fut.done():
                fut.set_result(None)
        # Yield to allow awoken tasks to execute their callbacks
        await asyncio.sleep(0)




class CallSessionStateMachine:
    """
    Explicit state machine governing call turn-taking and silence timeouts.
    States:
      - AGENT_SPEAKING: TTS audio playback in progress. Timer is stopped.
      - PROCESSING: STT, LLM streaming, or RAG/DB tool call running. Timer is stopped.
      - WAITING_FOR_CALLER: Agent audio has fully finished playing. 8s silence timer is active.
      - ENDING: Disconnection in progress or closing farewell audio playing. No timer.
    """

    def __init__(
        self,
        call_id: str,
        initial_lang: str = "gu",
        silence_timeout_seconds: float = float(config.SILENCE_TIMEOUT_SECONDS),
        max_silence_checks: int = config.MAX_SILENCE_CHECKS,
        max_farewell_wait_seconds: float = float(config.MAX_FAREWELL_WAIT_SECONDS),
        clock: Optional[Callable[[], float]] = None,
        sleep_func: Optional[Callable[[float], Awaitable[None]]] = None,
        on_silence_check: Optional[Callable[[int, str, str], Awaitable[None]]] = None,
        on_silence_timeout: Optional[Callable[[], Awaitable[None]]] = None,
        on_hangup: Optional[Callable[[str, str], Awaitable[None]]] = None,
    ):
        self.call_id = call_id
        self.locked_language = initial_lang or "gu"
        self.has_caller_spoken = False
        self.state = CallState.AGENT_SPEAKING

        self.silence_timeout_seconds = silence_timeout_seconds
        self.max_silence_checks = max_silence_checks
        self.max_farewell_wait_seconds = max_farewell_wait_seconds

        self._clock = clock if clock is not None else time.time
        self._sleep_func = sleep_func if sleep_func is not None else asyncio.sleep

        self.on_silence_check = on_silence_check
        self.on_silence_timeout = on_silence_timeout
        self.on_hangup = on_hangup

        self.silence_checks_completed = 0
        self._timer_task: Optional[asyncio.Task[None]] = None
        self._disconnect_reason: Optional[str] = None
        self._farewell_wait_event = asyncio.Event()

        self._log_event("CALL_INITIALIZED", f"initial_lang={self.locked_language}, timeout={self.silence_timeout_seconds}s")

    def _now(self) -> float:
        return float(self._clock())

    def _log_event(self, event_name: str, details: str = "") -> None:
        t = self._now()
        msg = f"[{self.call_id}] [{t:.3f}s] {event_name}: {details}"
        print(msg)
        logger.info(msg)

    def _cancel_timer(self, reason: str = "") -> None:
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            self._timer_task = None
            self._log_event("TIMER_CANCEL", reason)

    def get_silence_phrase(self, lang: Optional[str] = None) -> str:
        """Returns the exact silence-check phrase in the locked language."""
        target_lang = lang or self.locked_language
        return SILENCE_CHECK_PROMPTS.get(target_lang, SILENCE_CHECK_PROMPTS["gu"])

    def get_farewell_phrase(self, lang: Optional[str] = None) -> str:
        """Returns the closing farewell in the locked language."""
        target_lang = lang or self.locked_language
        return FAREWELL_PROMPTS.get(target_lang, FAREWELL_PROMPTS["gu"])

    # -------------------------------------------------------------------------
    # State Transitions & Event Hooks
    # -------------------------------------------------------------------------

    def on_agent_playback_started(self) -> None:
        """Called when agent audio playback starts playing to caller."""
        self._cancel_timer("Agent speaking")
        if self.state != CallState.ENDING:
            self.state = CallState.AGENT_SPEAKING

    def on_agent_playback_finished(self) -> None:
        """
        Called strictly when agent audio playback has FINISHED playing to caller.
        T = moment agent audio has finished.
        """
        if self.state == CallState.ENDING:
            self._log_event("FAREWELL_DELIVERED", "Farewell audio finished playing")
            self._farewell_wait_event.set()
            return

        self.state = CallState.WAITING_FOR_CALLER
        self._start_silence_timer()

    def _start_silence_timer(self) -> None:
        self._cancel_timer("Restarting timer")
        if self.state != CallState.WAITING_FOR_CALLER:
            return

        self._log_event(
            "TIMER_START",
            f"Waiting for caller ({self.silence_timeout_seconds}s timeout, checks_done={self.silence_checks_completed})",
        )
        self._timer_task = asyncio.create_task(self._run_silence_timer())

    async def _run_silence_timer(self) -> None:
        try:
            await self._sleep_func(self.silence_timeout_seconds)
            await self._on_silence_timer_fired()
        except asyncio.CancelledError:
            pass

    async def _on_silence_timer_fired(self) -> None:
        if self.state != CallState.WAITING_FOR_CALLER:
            return

        if self.silence_checks_completed == 0:
            self.silence_checks_completed = 1
            prompt_text = self.get_silence_phrase()
            self._log_event("SILENCE_CHECK_1", f"Prompting caller: '{prompt_text}' ({self.locked_language})")
            self.state = CallState.AGENT_SPEAKING
            if self.on_silence_check:
                await self.on_silence_check(1, prompt_text, self.locked_language)

        elif self.silence_checks_completed == 1:
            self.silence_checks_completed = 2
            prompt_text = self.get_silence_phrase()
            self._log_event("SILENCE_CHECK_2", f"Prompting caller: '{prompt_text}' ({self.locked_language})")
            self.state = CallState.AGENT_SPEAKING
            if self.on_silence_check:
                await self.on_silence_check(2, prompt_text, self.locked_language)

        else:
            # Silence timeout expired after Check 2: disconnect call (no further agent speech)
            self.state = CallState.ENDING
            self._disconnect_reason = "silence_timeout"
            self._log_event("SILENCE_HANGUP", "Disconnecting call after silence checks with reason=silence_timeout")
            if self.on_silence_timeout:
                await self.on_silence_timeout()

    def on_caller_speech_started(self) -> None:
        """
        Called when VAD detects speech energy or barge-in begins.
        Immediately cancels the silence timer and resets the silence check counter to 0.
        """
        is_barge_in = (self.state == CallState.AGENT_SPEAKING)
        self._cancel_timer("Caller speech started")
        self.silence_checks_completed = 0
        if is_barge_in:
            self._log_event("BARGE_IN", "Caller interrupted agent playback; silence counter reset to 0")
        else:
            self._log_event("CALLER_SPEECH", "Speech energy detected; silence counter reset to 0")

    def on_caller_transcript_received(self, user_text: str, detected_lang: Optional[str] = None) -> None:
        """
        Called when a non-empty user transcript is finalized.
        Enforces sticky language locking and transitions state to PROCESSING.
        """
        self._cancel_timer("Processing caller transcript")
        self.silence_checks_completed = 0

        # Sticky Language Lock: lock to first caller reply language unless explicit change requested
        clean_text = user_text.strip().lower()
        if not self.has_caller_spoken:
            self.has_caller_spoken = True
            if detected_lang:
                self.locked_language = detected_lang
        else:
            # Explicit language change requested
            if any(w in clean_text for w in ["english", "speak in english", "અંગ્રેજી"]):
                self.locked_language = "en"
            elif any(w in clean_text for w in ["hindi", "speak in hindi", "हिन्दी", "હિન્દી"]):
                self.locked_language = "hi"
            elif any(w in clean_text for w in ["gujarati", "speak in gujarati", "ગુજરાતી"]):
                self.locked_language = "gu"

        self.state = CallState.PROCESSING
        self._log_event("PROCESSING_START", f"query='{user_text[:50]}' locked_lang={self.locked_language}")

    def on_tool_started(self, tool_name: str) -> None:
        """Called when a RAG search or database lookup starts."""
        self._cancel_timer(f"Tool {tool_name} running")
        self.state = CallState.PROCESSING
        self._log_event("TOOL_START", tool_name)

    def on_tool_finished(self, tool_name: str) -> None:
        """Called when a RAG search or database lookup finishes."""
        self._log_event("TOOL_FINISH", tool_name)

    def on_hangup_command(self, reason: str = "caller_hangup_requested") -> None:
        """Called when the caller requests hangup or declining farewell is triggered."""
        self._cancel_timer(f"Hangup requested: {reason}")
        self.state = CallState.ENDING
        self._disconnect_reason = reason
        self._log_event("COMMAND_HANGUP", f"reason={reason}")

    async def wait_for_farewell_completion(self, max_wait_seconds: Optional[float] = None) -> None:
        """Waits up to max_wait_seconds for farewell playback to complete before dropping line."""
        timeout = max_wait_seconds if max_wait_seconds is not None else self.max_farewell_wait_seconds
        try:
            await asyncio.wait_for(self._farewell_wait_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self._log_event("FAREWELL_TIMEOUT", f"Safety wait of {timeout}s elapsed; dropping line now")

    def close(self) -> None:
        """Cleans up timers upon session termination."""
        self._cancel_timer("Call session closed")
        self.state = CallState.ENDING
        self._farewell_wait_event.set()
