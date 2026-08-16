"""
LLM layer — University Admission & Student Info Voice Agent.
Powered by Qwen 2.5:3B via local Ollama API server with built-in tool calling
for structured SQL database and unstructured RAG knowledge base.
"""

import json
import re
from typing import Generator
import requests

import config
from db.database import Database
from rag import RAGStore

SYSTEM_PROMPT = """\
You are a polite, helpful, and clear University Admission & Student Desk Assistant speaking on a real-time voice call with a student's parent or prospective student.

Key Responsibilities & Rules:
1. Answer queries regarding Admission process, Eligibility, Fees, Deadlines, Campus Placements, Hostel/Campus Rules, Scholarships, and specific Student Marks or Attendance.
2. Voice Call Conversational Style: Use short sentences, natural phrasing, and a polite, helpful tone. Never use markdown formatting (no asterisks, bold text, bullet points, numbered lists, or hashtags) because your response will be read aloud by Text-To-Speech.
3. Language: Respond in the exact language spoken by the user (English, Hindi, or Gujarati).
4. Tool Calling: If you need information from the university database or uploaded policy documents to answer, call a tool using this exact format:
   TOOL_CALL: tool_name(param="value")

   Available Tools:
   - lookup_student(identifier="name or student_id") -> Finds student_id, name, department, semester.
   - get_student_marks(student_id="STUxxx") -> Retrieves subject-wise marks & grades.
   - get_student_attendance(student_id="STUxxx") -> Retrieves subject-wise attendance percentages.
   - get_placement_stats(department="CSE/ECE/MECH or empty") -> Retrieves placement packages & top recruiters.
   - get_admission_info(program="CSE/ECE/MTech or empty") -> Retrieves eligibility, fee structure, & application deadline.
   - search_university_docs(query="keywords or topic") -> Searches unstructured university prospectus, hostel rules, scholarship guidelines, campus policies, and PDF documents.

5. If a parent asks for student marks/attendance without providing student name or ID, politely ask for the student's name or Student ID first.
6. When tool results are provided to you, summarize them concisely in 2-3 spoken sentences.
"""


class LLM:
    def __init__(self):
        self.db = Database()
        self.rag = RAGStore()
        self.history: list[dict[str, str]] = []

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

    def reply_stream(self, user_text: str) -> Generator[str, None, None]:
        """
        Yields response text incrementally. Handles multi-turn tool execution transparently.
        """
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        }

        max_tool_iterations = 3
        current_iteration = 0

        try:
            while current_iteration < max_tool_iterations:
                current_iteration += 1
                resp = requests.post(config.OLLAMA_URL, json=payload, timeout=12)
                resp.raise_for_status()
                data = resp.json()
                reply_content = data.get("message", {}).get("content", "")

                tool_info = self._parse_tool_call(reply_content)
                if tool_info:
                    tool_name, kwargs = tool_info
                    tool_result = self._execute_tool(tool_name, kwargs)

                    messages.append({"role": "assistant", "content": reply_content})
                    messages.append({
                        "role": "user",
                        "content": f"TOOL_RESULT ({tool_name}): {tool_result}\nPlease synthesize a short, polite spoken answer for the caller in 2-3 sentences.",
                    })
                    payload["messages"] = messages
                    continue
                else:
                    # Final text response without tools
                    clean_reply = re.sub(r'[*_#`~]', '', reply_content).strip()
                    self.history.append({"role": "assistant", "content": clean_reply})
                    yield clean_reply
                    return

            # Fallback if tool iterations exceeded
            fallback = "I have fetched the information. How else may I assist you with university admissions?"
            self.history.append({"role": "assistant", "content": fallback})
            yield fallback

        except Exception as err:
            print(f"[llm] Error communicating with Ollama: {err}")
            fallback_msg = "I am sorry, I am having trouble accessing the university system at this moment. Please try again shortly."
            yield fallback_msg
            self.history.append({"role": "assistant", "content": fallback_msg})