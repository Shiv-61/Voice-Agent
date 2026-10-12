"""
Turn-by-turn Latency Tracker and Profiling Instrumentation.
Records exact timing across all pipeline stages:
- VAD speech end & STT final transcript
- LLM request sent, first token (TTFT), first sentence complete, last token
- Tool call execution (RAG vs DB; inside RAG: embedding, vector search, rerank)
- TTS request sent, first audio chunk, playback start, playback end
- Hangup supervisor request start / end

Outputs a formatted per-turn summary line:
total (caller stops talking -> agent audio starts) + stage durations.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TurnLatencyRecord:
    call_id: str
    turn_id: int
    user_query: str = ""
    query_route: str = "ROUTE_A"  # ROUTE_A, RAG, DB
    
    # Timestamps (using time.perf_counter())
    speech_end_ts: float = 0.0
    stt_start_ts: float = 0.0
    stt_end_ts: float = 0.0
    
    llm_req_start_ts: float = 0.0
    llm_first_token_ts: float = 0.0
    llm_first_sentence_ts: float = 0.0
    llm_last_token_ts: float = 0.0
    
    tool_type: str = ""  # "RAG" or "DB"
    tool_name: str = ""
    tool_start_ts: float = 0.0
    tool_end_ts: float = 0.0
    rag_embed_ms: float = 0.0
    rag_search_ms: float = 0.0
    rag_rerank_ms: float = 0.0
    
    tts_req_start_ts: float = 0.0
    tts_first_chunk_ts: float = 0.0
    tts_playback_start_ts: float = 0.0
    tts_playback_end_ts: float = 0.0
    
    supervisor_start_ts: float = 0.0
    supervisor_end_ts: float = 0.0

    def mark_speech_end(self):
        self.speech_end_ts = time.perf_counter()

    def mark_stt_start(self):
        self.stt_start_ts = time.perf_counter()

    def mark_stt_end(self):
        self.stt_end_ts = time.perf_counter()

    def mark_llm_request_sent(self):
        self.llm_req_start_ts = time.perf_counter()

    def mark_llm_first_token(self):
        if self.llm_first_token_ts == 0.0:
            self.llm_first_token_ts = time.perf_counter()

    def mark_llm_first_sentence(self):
        if self.llm_first_sentence_ts == 0.0:
            self.llm_first_sentence_ts = time.perf_counter()

    def mark_llm_last_token(self):
        self.llm_last_token_ts = time.perf_counter()

    def mark_tool_start(self, tool_name: str, tool_type: str = ""):
        self.tool_start_ts = time.perf_counter()
        self.tool_name = tool_name
        self.tool_type = tool_type or ("RAG" if "docs" in tool_name or "search" in tool_name else "DB")
        self.query_route = self.tool_type

    def mark_tool_end(self):
        self.tool_end_ts = time.perf_counter()

    def mark_tts_request_sent(self):
        if self.tts_req_start_ts == 0.0:
            self.tts_req_start_ts = time.perf_counter()

    def mark_tts_first_chunk(self):
        if self.tts_first_chunk_ts == 0.0:
            self.tts_first_chunk_ts = time.perf_counter()

    def mark_tts_playback_start(self):
        if self.tts_playback_start_ts == 0.0:
            self.tts_playback_start_ts = time.perf_counter()

    def mark_tts_playback_end(self):
        self.tts_playback_end_ts = time.perf_counter()

    def mark_supervisor_start(self):
        self.supervisor_start_ts = time.perf_counter()

    def mark_supervisor_end(self):
        self.supervisor_end_ts = time.perf_counter()

    # Derived Latency Durations (in milliseconds)
    @property
    def stt_duration_ms(self) -> float:
        if self.stt_end_ts > self.stt_start_ts > 0:
            return (self.stt_end_ts - self.stt_start_ts) * 1000.0
        return 0.0

    @property
    def llm_ttft_ms(self) -> float:
        if self.llm_first_token_ts > self.llm_req_start_ts > 0:
            return (self.llm_first_token_ts - self.llm_req_start_ts) * 1000.0
        return 0.0

    @property
    def llm_first_sentence_ms(self) -> float:
        if self.llm_first_sentence_ts > self.llm_req_start_ts > 0:
            return (self.llm_first_sentence_ts - self.llm_req_start_ts) * 1000.0
        return 0.0

    @property
    def llm_total_gen_ms(self) -> float:
        if self.llm_last_token_ts > self.llm_req_start_ts > 0:
            return (self.llm_last_token_ts - self.llm_req_start_ts) * 1000.0
        return 0.0

    @property
    def tool_duration_ms(self) -> float:
        if self.tool_end_ts > self.tool_start_ts > 0:
            return (self.tool_end_ts - self.tool_start_ts) * 1000.0
        return 0.0

    @property
    def tts_ttfa_ms(self) -> float:
        if self.tts_first_chunk_ts > self.tts_req_start_ts > 0:
            return (self.tts_first_chunk_ts - self.tts_req_start_ts) * 1000.0
        return 0.0

    @property
    def supervisor_ms(self) -> float:
        if self.supervisor_end_ts > self.supervisor_start_ts > 0:
            return (self.supervisor_end_ts - self.supervisor_start_ts) * 1000.0
        return 0.0

    @property
    def speech_to_audio_start_ms(self) -> float:
        """Total time from caller stops talking (speech_end) to agent audio starts."""
        ref_start = self.speech_end_ts or self.stt_start_ts or self.llm_req_start_ts
        ref_audio = self.tts_playback_start_ts or self.tts_first_chunk_ts
        if ref_audio > ref_start > 0:
            return (ref_audio - ref_start) * 1000.0
        return 0.0

    def summary_line(self) -> str:
        s2a = self.speech_to_audio_start_ms
        stt = self.stt_duration_ms
        ttft = self.llm_ttft_ms
        f_sent = self.llm_first_sentence_ms
        tool_d = self.tool_duration_ms
        tts_1st = self.tts_ttfa_ms
        sup = self.supervisor_ms

        tool_str = ""
        if self.tool_name:
            if self.tool_type == "RAG":
                tool_str = f" | Tool(RAG): {tool_d:.1f}ms [emb:{self.rag_embed_ms:.1f}ms, search:{self.rag_search_ms:.1f}ms, rerank:{self.rag_rerank_ms:.1f}ms]"
            else:
                tool_str = f" | Tool(DB-{self.tool_name}): {tool_d:.1f}ms"

        return (
            f"⏱️ [LATENCY] call_id={self.call_id} turn={self.turn_id} route={self.query_route} | "
            f"SpeechEnd->AudioStart: {s2a:.1f}ms | "
            f"STT: {stt:.1f}ms | "
            f"LLM_TTFT: {ttft:.1f}ms | "
            f"FirstSent: {f_sent:.1f}ms{tool_str} | "
            f"TTS_1st: {tts_1st:.1f}ms | "
            f"Sup: {sup:.1f}ms"
        )


class LatencyTrackerManager:
    """Manages active turn latency records across calls."""
    def __init__(self):
        self._records: dict[str, TurnLatencyRecord] = {}
        self._turn_counters: dict[str, int] = {}

    def start_turn(self, call_id: str, speech_end_ts: Optional[float] = None) -> TurnLatencyRecord:
        cnt = self._turn_counters.get(call_id, 0) + 1
        self._turn_counters[call_id] = cnt
        rec = TurnLatencyRecord(call_id=call_id, turn_id=cnt)
        if speech_end_ts:
            rec.speech_end_ts = speech_end_ts
        else:
            rec.mark_speech_end()
        self._records[call_id] = rec
        return rec

    def get_current_turn(self, call_id: str) -> Optional[TurnLatencyRecord]:
        return self._records.get(call_id)

    def finish_turn(self, call_id: str) -> Optional[TurnLatencyRecord]:
        rec = self._records.get(call_id)
        if rec:
            print(rec.summary_line())
        return rec


# Global latency manager
latency_manager = LatencyTrackerManager()
