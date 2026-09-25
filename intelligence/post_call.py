"""
College Voice Agent — Asynchronous Post-Call Intelligence & Lead Analytics.
Inspired by enterprise voice-ai-agent-builder architecture:
Decouples heavy CRM disposition, intent categorization, lead scoring,
and call summarization from the live conversational loop.
"""

import json
import re
import httpx
import config

POST_CALL_SYSTEM_PROMPT = """\
You are an expert College Admissions & Student Affairs CRM Auditor analyzing a concluded voice call transcript between a caller (student, parent, or applicant) and the DDU IT University AI Voice Assistant.

Your task is to analyze the conversation and extract structured college CRM intelligence.

Return ONLY a raw, valid JSON object matching this schema:
{
  "intent": "<One of: 'Admission & Eligibility', 'Fee Structure & Scholarships', 'Hostel & Campus Rules', 'Student Records & Attendance', 'Placement Statistics', 'General College Inquiry'>",
  "lead_status": "<One of: 'Hot Lead (Intends to Apply)', 'Prospective Applicant', 'Current Student', 'Parent Inquiry', 'Resolved / Info Provided'>",
  "sentiment": "<One of: 'Positive', 'Neutral', 'Frustrated'>",
  "summary": "<A 1-sentence concise factual summary of caller inquiry and college response>"
}

Do not include markdown code fences, backticks, or any conversational text. Output only raw JSON.
"""

def _heuristic_fallback(transcript_text: str) -> dict:
    """Fast deterministic fallback if LLM extraction fails."""
    lower = transcript_text.lower()

    # Intent
    if any(k in lower for k in ["fee", "फीस", "फी", "cost", "scholarship", "छात्रवृत्ति"]):
        intent = "Fee Structure & Scholarships"
    elif any(k in lower for k in ["admission", "eligibility", "apply", "एडमिशन", "प्रवेश", "btech", "बीटेक"]):
        intent = "Admission & Eligibility"
    elif any(k in lower for k in ["placement", "package", "salary", "recruiter", "प्लेसमेंट"]):
        intent = "Placement Statistics"
    elif any(k in lower for k in ["attendance", "mark", "grade", "stu", "रिजल्ट", "हाजिरी"]):
        intent = "Student Records & Attendance"
    elif any(k in lower for k in ["hostel", "curfew", "room", "हॉस्टल"]):
        intent = "Hostel & Campus Rules"
    else:
        intent = "General College Inquiry"

    # Lead Status
    if any(k in lower for k in ["apply", "admission", "एडमिशन", "form"]):
        lead_status = "Hot Lead (Intends to Apply)"
    elif any(k in lower for k in ["stu", "student", "roll", "attendance"]):
        lead_status = "Current Student"
    else:
        lead_status = "Prospective Applicant"

    # Sentiment
    if any(k in lower for k in ["thank", "धन्यवाद", "great", "good", "helpful", "आभार"]):
        sentiment = "Positive"
    elif any(k in lower for k in ["bad", "wrong", "angry", "bekaar", "gussa"]):
        sentiment = "Frustrated"
    else:
        sentiment = "Neutral"

    summary = f"Caller inquired regarding {intent.lower()}. Official university details provided."

    return {
        "intent": intent,
        "lead_status": lead_status,
        "sentiment": sentiment,
        "summary": summary,
    }


async def analyze_and_record_call(call_id: str, history: list[dict], db) -> dict:
    """
    Analyzes call transcript asynchronously and records college CRM disposition.
    Safe, non-blocking, zero impact on caller real-time audio latency.
    """
    if not call_id or not history or len(history) <= 1:
        fallback = {
            "intent": "Brief / Abandoned Call",
            "lead_status": "Unqualified",
            "sentiment": "Neutral",
            "summary": "Caller disconnected shortly after welcome message.",
        }
        try:
            db.update_call_analytics(call_id, fallback["intent"], fallback["lead_status"], fallback["sentiment"], fallback["summary"])
        except Exception:
            pass
        return fallback

    # Format transcript
    lines = []
    for turn in history:
        role = "Caller" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {turn.get('content', '')}")
    transcript_text = "\n".join(lines)

    result = None

    # Call LLM for extraction
    if config.LLM_PROVIDER == "openrouter" and config.OPENROUTER_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": config.LLM_MODEL,
                "messages": [
                    {"role": "system", "content": POST_CALL_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Concluded Call Transcript:\n{transcript_text}"},
                ],
                "temperature": 0.1,
                "max_tokens": 150,
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(config.OPENROUTER_URL, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    # Clean markdown codeblocks if present
                    clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
                    parsed = json.loads(clean_json)
                    if "intent" in parsed and "lead_status" in parsed:
                        result = parsed
        except Exception as e:
            print(f"[intelligence] LLM extraction notice: {e}")

    # Fallback heuristic if LLM unavailable or JSON parse failed
    if not result:
        result = _heuristic_fallback(transcript_text)

    # Persist in database
    try:
        db.update_call_analytics(
            call_id,
            result.get("intent", "General College Inquiry"),
            result.get("lead_status", "Prospective Applicant"),
            result.get("sentiment", "Neutral"),
            result.get("summary", "Call completed."),
        )
        print(f"📊 [intelligence] Call {call_id} analyzed: Intent='{result.get('intent')}', Lead='{result.get('lead_status')}'")
    except Exception as err:
        print(f"[intelligence] Failed to persist analytics for {call_id}: {err}")

    return result
