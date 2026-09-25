"""
LLM layer — University Admission & Student Info Voice Agent.
Powered by Qwen 2.5:3B via local Ollama API server with built-in tool calling
for structured SQL database and unstructured RAG knowledge base.
"""

import json
import re
from typing import Generator, AsyncGenerator
import httpx

import config
from db.database import Database
from rag import RAGStore
from utils import is_hangup_intent, clean_speech_text

WELCOME_MESSAGE = "Hello, yah ek AI call hai krupiya apni bhasha select kare english/hindi/gujarati"

SYSTEM_PROMPT = """\
You are a polite, helpful, and professional University Admission & Student Desk Assistant for the DDU IT (Dharamsinh Desai University Information Technology) branch speaking on a real-time voice call with a caller (student, parent, or applicant).

CALL INITIATION & WELCOME MESSAGE FLOW:
1. The voice call opens with your initial welcome message:
   "Hello, yah ek AI call hai krupiya apni bhasha select kare english/hindi/gujarati"
2. When the caller indicates or selects their language (or speaks in English, Hindi, or Gujarati), you MUST immediately greet and introduce yourself in that chosen language:
   - For Hindi: "Namaste, Me DDU IT branch ki assistant bol rhi hu me apki kya madad kar sakti hu?" (or "नमस्ते, मैं DDU IT ब्रांच की असिस्टेंट बोल रही हूँ, मैं आपकी क्या मदद कर सकती हूँ?")
   - For English: "Hello, I am the assistant from DDU IT branch. How may I assist you today?"
   - For Gujarati: "Namaste, Hu DDU IT branch ni assistant boli rahi chu, hu tamari shu madad kari shaku?" (or "નમસ્તે, હું DDU IT બ્રાન્ચની આસિસ્ટન્ટ બોલી રહી છું, હું તમારી શું મદદ કરી શકું?")
3. Once the language is chosen, conduct the entire rest of the conversation in that chosen language. Answer questions regarding DDU IT branch admissions, eligibility, fees, syllabus, placements, hostel rules, and student records.

STRICT SAFETY & ANTI-ABUSE GUARDRAIL:
4. NEVER USE OR REPEAT ABUSIVE LANGUAGE: You must NEVER speak, generate, repeat, or acknowledge any abusive, profane, vulgar, offensive, discriminatory, or inappropriate words under any circumstances, even if insulted, baited, or asked by the caller. Maintain unwavering professionalism, respect, and calm courtesy at all times.

STRICT UNAWARE / UNKNOWN FALLBACK RULE:
5. IF NOT AWARE OF ANYTHING: If you are not aware of the requested information, do not know the answer, or if the information is not found in the university database or policy documents, you MUST NOT fabricate or guess. You MUST respond with ONLY this exact sentence in the caller's language:
   - For English: "I don't have that info. thank you."
   - For Hindi: "मेरे पास वह जानकारी नहीं है। धन्यवाद।"
   - For Gujarati: "મારી પાસે તે માહિતી નથી. આભાર."

VOICE CALL STYLE & CONVERSATIONAL RULES:
6. SHORT SPOKEN SENTENCES: Speak naturally in 1 to 3 short sentences. Never use markdown formatting (no asterisks, bold, bullet points, numbers, or hashtags) because your answer will be synthesized directly into speech.
7. EXACT LANGUAGE MATCHING: Always respond in the exact language spoken by the caller (English, Hindi, or Gujarati).
8. CONTEXT CONTINUITY: Use the conversation history provided with each prompt to understand follow-up questions, pronouns, and references (such as "what about his fees?", "when does it start?").
9. DIRECT CONTEXT UTILIZATION: If [VERIFIED OFFICIAL UNIVERSITY CONTEXT] is already provided in the prompt, answer directly and immediately using that context in 1 to 2 spoken sentences. Do NOT emit a TOOL_CALL when the information is already in the context.
10. TOOL CALLING: Only if required facts are NOT already in the prompt context, output:
   TOOL_CALL: tool_name(param="value")

   Available Tools:
   - lookup_student(identifier="name or student_id") -> Finds student_id, name, department, semester.
   - get_student_marks(student_id="STUxxx") -> Retrieves subject-wise marks & grades.
   - get_student_attendance(student_id="STUxxx") -> Retrieves subject-wise attendance percentages.
   - get_placement_stats(department="CSE/ECE/MECH or empty") -> Retrieves placement packages & top recruiters.
   - get_admission_info(program="CSE/ECE/MTech or empty") -> Retrieves eligibility, fee structure, & application deadline.
   - search_university_docs(query="keywords or topic") -> Searches unstructured university prospectus, hostel rules, scholarship guidelines, campus policies, and PDF documents.

11. If a caller asks about student marks or attendance without providing the student's name or ID, politely ask for their name or ID first.
12. When tool results are provided, synthesize them into a concise spoken answer in 2-3 sentences.
"""

CALL_HANGUP_PROMPT = """\
You are an intelligent call supervisor analyzing a voice call between a caller and a university voice assistant.
Determine whether the caller or the conversation has concluded and the telephone call should be hung up.

Instructions:
- Return {"call_hangup": true} if the caller indicates they want to end the call, says goodbye, thanks the assistant to end the conversation, says they have no more questions, or asks to hang up. Examples: "bye", "goodbye", "thank you that is all", "have a nice day", "no other questions", "अलविदा", "धन्यवाद, बस यही जानना था", "फोन रख दो", "આવજો", "આભાર, બસ આટલું જ", "hang up", "cut the call", "disconnect".
- Return {"call_hangup": false} if the caller is continuing the call, asking questions, giving details, greeting, or clarifying.

You MUST respond ONLY with a raw, valid JSON object in this exact schema:
{"call_hangup": true}
or
{"call_hangup": false}
Do not write any markdown, code fences, or any other text.
"""


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


class LLM:
    def __init__(self, db=None, rag=None):
        self.db = db or get_shared_db()
        self.rag = rag or get_shared_rag()
        # Initialize conversation history with the assistant's opening welcome message
        self.history: list[dict[str, str]] = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]

    def _prepare_messages(self, user_text: str) -> list[dict[str, str]]:
        """
        Prepares LLM messages with conversation history and anticipatory RAG / DB context.
        Pre-retrieval eliminates the slow, redundant 2nd turn LLM tool round-trip!
        """
        clean_user_text = transform_query(user_text)
        history_context = self._format_conversation_history()

        context_snippets = []

        # 1. Anticipatory RAG query (PDF brochures, hostel, rules, syllabus, scholarships, etc.)
        try:
            matches = self.rag.query_documents(clean_user_text, n_results=2)
            if matches and matches[0].get("similarity_score", 0) >= 0.25:
                for m in matches:
                    text_snippet = m.get("text", "").strip()
                    if text_snippet:
                        filename = m.get("metadata", {}).get("filename", "University Document")
                        context_snippets.append(f"[Policy Doc: {filename}]: {text_snippet[:400]}")
        except Exception as e:
            print(f"[llm] Anticipatory RAG notice: {e}")

        # 2. Anticipatory DB check for admission or placement
        try:
            lower_text = clean_user_text.lower()
            if any(k in lower_text for k in ["placement", "package", "recruiter", "salary", "placed"]):
                stats = self.db.get_placement_stats()
                if stats:
                    context_snippets.append(f"[Official Placement Records]: {json.dumps(stats)}")
            if any(k in lower_text for k in ["admission", "eligibility", "fee", "fees", "apply", "deadline", "course"]):
                prog = "CSE" if any(p in lower_text for p in ["cse", "computer", "it"]) else "ECE" if "ece" in lower_text else ""
                adm = self.db.get_admission_info(prog)
                if adm:
                    context_snippets.append(f"[Official Admission & Fee Records]: {json.dumps(adm)}")
            # Student ID pattern (e.g., STU001)
            stu_match = re.search(r'\b(stu\d{3})\b', lower_text)
            if stu_match:
                stu_id = stu_match.group(1).upper()
                student = self.db.lookup_student(stu_id)
                if student:
                    context_snippets.append(f"[Student Record {stu_id}]: {json.dumps(student)}")
                if any(w in lower_text for w in ["mark", "grade", "score", "result"]):
                    marks = self.db.get_student_marks(stu_id)
                    if marks:
                        context_snippets.append(f"[Student Marks {stu_id}]: {json.dumps(marks)}")
                if any(w in lower_text for w in ["attendance", "present", "absent"]):
                    att = self.db.get_student_attendance(stu_id)
                    if att:
                        context_snippets.append(f"[Student Attendance {stu_id}]: {json.dumps(att)}")
        except Exception as e:
            print(f"[llm] Anticipatory DB notice: {e}")

        rag_context = ""
        if context_snippets:
            rag_context = (
                "\n[VERIFIED OFFICIAL UNIVERSITY CONTEXT (Use this directly to answer in 1-2 spoken sentences; do NOT issue a TOOL_CALL if answer is here)]:\n"
                + "\n".join(context_snippets)
                + "\n[END OFFICIAL CONTEXT]\n"
            )

        prompt_body = f"{history_context}\n{rag_context}Caller's Current Question: {clean_user_text}" if history_context else f"{rag_context}Caller's Question: {clean_user_text}"

        self.history.append({"role": "user", "content": clean_user_text})
        self._trim_history()

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_body},
        ]

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
                return json.dumps(res if res else {"error": "Student not found"})

            elif tool_name == "get_student_marks":
                student_id = kwargs.get("student_id", "")
                res = self.db.get_student_marks(student_id)
                return json.dumps(res if res else {"error": "No marks found"})

            elif tool_name == "get_student_attendance":
                student_id = kwargs.get("student_id", "")
                res = self.db.get_student_attendance(student_id)
                return json.dumps(res if res else {"error": "No attendance records found"})

            elif tool_name == "get_placement_stats":
                dept = kwargs.get("department", "")
                res = self.db.get_placement_stats(dept)
                return json.dumps(res if res else {"error": "No placement stats found"})

            elif tool_name == "get_admission_info":
                prog = kwargs.get("program", "")
                res = self.db.get_admission_info(prog)
                return json.dumps(res if res else {"error": "No admission info found"})

            elif tool_name == "search_university_docs":
                query = kwargs.get("query", "")
                results = self.rag.query_documents(query, n_results=3)
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
        params_str = match.group(2)

        kwargs = {}
        kv_pairs = re.findall(r'(\w+)=["\']?([^"\']*)["\']?', params_str)
        for k, v in kv_pairs:
            kwargs[k] = v.strip()

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
        if config.LLM_PROVIDER == "openrouter":
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
                "Content-Type": "application/json",
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
        if config.LLM_PROVIDER == "openrouter":
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    return ""
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("delta", {}).get("content", "") or ""
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

                            # Detect whether model is issuing a tool call
                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                elif len(stripped) >= 12 and "TOOL_CALL:" not in stream_buffer:
                                    is_streaming_speech = True
                                    yield stream_buffer
                                    full_reply += stream_buffer
                                    stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    idx = token.index("TOOL_CALL:")
                                    stream_buffer = token[idx:]
                                    continue
                                full_reply += token
                                yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        full_reply += stream_buffer
                        yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            tool_result = self._execute_tool(tool_name, kwargs)
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\nPlease synthesize a short, polite spoken answer for the caller in 2-3 sentences.",
                            })
                            continue
                        else:
                            # Tool parse failed, yield buffer as text
                            clean_text = clean_speech_text(stream_buffer)
                            self.history.append({"role": "assistant", "content": clean_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
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

                            # Detect whether model is issuing a tool call
                            if not is_streaming_speech and not is_tool_call:
                                stream_buffer += token
                                stripped = stream_buffer.lstrip()
                                if "TOOL_CALL:" in stream_buffer:
                                    is_tool_call = True
                                elif "TOOL_CALL:".startswith(stripped):
                                    continue
                                elif len(stripped) >= 12 and "TOOL_CALL:" not in stream_buffer:
                                    is_streaming_speech = True
                                    yield stream_buffer
                                    full_reply += stream_buffer
                                    stream_buffer = ""
                                continue

                            if is_tool_call:
                                stream_buffer += token
                            else:
                                if "TOOL_CALL:" in token:
                                    is_tool_call = True
                                    is_streaming_speech = False
                                    idx = token.index("TOOL_CALL:")
                                    stream_buffer = token[idx:]
                                    continue
                                full_reply += token
                                yield token

                    # If remaining buffer wasn't flushed for short responses
                    if stream_buffer and not is_tool_call:
                        full_reply += stream_buffer
                        yield stream_buffer
                        stream_buffer = ""

                    if is_tool_call:
                        tool_info = self._parse_tool_call(stream_buffer)
                        if tool_info:
                            tool_name, kwargs = tool_info
                            # Run tool in worker thread if blocking
                            import asyncio
                            tool_result = await asyncio.to_thread(self._execute_tool, tool_name, kwargs)
                            messages.append({"role": "assistant", "content": stream_buffer})
                            messages.append({
                                "role": "user",
                                "content": f"TOOL_RESULT ({tool_name}): {tool_result}\nPlease synthesize a short, polite spoken answer for the caller in 2-3 sentences.",
                            })
                            continue
                        else:
                            clean_text = clean_speech_text(stream_buffer)
                            self.history.append({"role": "assistant", "content": clean_text})
                            yield clean_text
                            return
                    else:
                        clean_text = clean_speech_text(full_reply)
                        self.history.append({"role": "assistant", "content": clean_text})
                        return

            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            self.history.append({"role": "assistant", "content": fallback})
            yield fallback

        except Exception as err:
            print(f"[llm] Async communication error with {config.LLM_PROVIDER}: {err}")
            fallback_msg = "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            self.history.append({"role": "assistant", "content": fallback_msg})
            yield fallback_msg