"""
LLM layer — University Admission & Student Info Voice Agent.
Powered by Qwen 2.5:3B via local Ollama API server with built-in tool calling.
"""

import json
import re
import requests

import config
from db.database import Database

SYSTEM_PROMPT = """\
You are a polite, helpful, and clear University Admission & Student Desk Assistant speaking on a voice call with a student's parent or prospective student.

Key Responsibilities & Rules:
1. Answer queries regarding Admission process, Eligibility, Fees, Deadlines, Campus Placements, Achievements, and specific Student Marks or Attendance.
2. Voice Call Conversational Style: Use short sentences, natural phrasing, and polite tone. Never use markdown formatting (no asterisks, bold text, bullet points, or numbered lists) because your response will be read aloud by Text-To-Speech.
3. Language: Respond in the exact language spoken by the user (English, Hindi, or Gujarati).
4. Tool Calling: If you need information from the university database to answer, call a tool by using the exact format:
   TOOL_CALL: tool_name(param="value")

   Available Tools:
   - lookup_student(identifier="name or student_id") -> Finds student_id, name, department, semester.
   - get_student_marks(student_id="STUxxx") -> Retrieves subject-wise marks & grades.
   - get_student_attendance(student_id="STUxxx") -> Retrieves subject-wise attendance percentages.
   - get_placement_stats(department="CSE/ECE/MECH or empty") -> Retrieves placement packages & top recruiters.
   - get_admission_info(program="CSE/ECE/MTech or empty") -> Retrieves eligibility, fee structure, & application deadline.

5. If a parent asks for student marks/attendance without giving student name or ID, politely ask for the student's name or Student ID first.
6. When tool results are provided to you, summarize them concisely in 2-3 sentences.
"""

class LLM:
    def __init__(self):
        self.db = Database()
        self.history = []  # list of {"role": ..., "content": ...}

    def _trim_history(self):
        max_msgs = config.MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history[:] = self.history[-max_msgs:]

    def _execute_tool(self, tool_name: str, kwargs: dict) -> str:
        """Executes database queries based on LLM tool selection."""
        print(f"[llm] Tool call triggered: {tool_name}({kwargs})")
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

        return "Unknown tool"

    def _parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        """Parses TOOL_CALL: tool_name(key="val") pattern."""
        match = re.search(r"TOOL_CALL:\s*(\w+)\((.*)\)", text)
        if not match:
            return None
        tool_name = match.group(1)
        params_str = match.group(2)
        
        # Parse kwargs
        kwargs = {}
        kv_pairs = re.findall(r'(\w+)=["\']?([^"\']+)["\']?', params_str)
        for k, v in kv_pairs:
            kwargs[k] = v.strip()

        return tool_name, kwargs

    def reply_stream(self, user_text: str):
        """
        Yields response text incrementally. Handles tool execution transparently.
        """
        self.history.append({"role": "user", "content": user_text})
        self._trim_history()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": False,  # non-streaming first turn to catch tool calls cleanly
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.LLM_MAX_TOKENS,
            },
        }

        try:
            resp = requests.post(config.OLLAMA_URL, json=payload, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            initial_reply = data.get("message", {}).get("content", "")

            # Check if LLM requested a tool call
            tool_info = self._parse_tool_call(initial_reply)
            if tool_info:
                tool_name, kwargs = tool_info
                tool_result = self._execute_tool(tool_name, kwargs)
                
                # Append tool execution turn to messages
                messages.append({"role": "assistant", "content": initial_reply})
                messages.append({
                    "role": "user",
                    "content": f"TOOL_RESULT ({tool_name}): {tool_result}\nPlease summarize this in 2 short, natural spoken sentences for the caller."
                })

                # Call LLM again to synthesize final response
                payload["messages"] = messages
                payload["stream"] = True

                full_reply = ""
                with requests.post(config.OLLAMA_URL, json=payload, stream=True) as s_resp:
                    s_resp.raise_for_status()
                    for line in s_resp.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        piece = chunk.get("message", {}).get("content", "")
                        if piece:
                            full_reply += piece
                            yield piece
                        if chunk.get("done"):
                            break

                # Clean markdown characters for audio output
                clean_reply = re.sub(r'[*_#`~]', '', full_reply)
                self.history.append({"role": "assistant", "content": clean_reply})
                return

            # If no tool call, stream directly or yield the reply
            clean_reply = re.sub(r'[*_#`~]', '', initial_reply)
            self.history.append({"role": "assistant", "content": clean_reply})
            yield clean_reply

        except Exception as err:
            print(f"[llm] Error calling Ollama: {err}")
            fallback_msg = "I am sorry, I am having trouble accessing the university system right now."
            yield fallback_msg
            self.history.append({"role": "assistant", "content": fallback_msg})