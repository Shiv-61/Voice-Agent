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
from config.prompt_loader import get_post_call_prompt

def get_post_call_system_prompt() -> str:
    """Returns the dynamically loaded post-call CRM audit prompt from config/prompts.yaml."""
    return get_post_call_prompt()


def _heuristic_fallback(transcript_text: str) -> dict:
    """Fast deterministic fallback if LLM extraction fails."""
    lower = transcript_text.lower()

    # Intent
    if any(k in lower for k in ["fee", "फीस", "फी", "cost", "scholarship", "छात्रवृत्ति"]):
        intent = "Fee Structure & Scholarships"
    elif any(k in lower for k in ["admission", "eligibility", "apply", "एडमिशन", "प्रवेश", "btech", "बीटेक"]):
        intent = "Admission & Eligibility"
    elif any(k in lower for k in ["curriculum", "syllabus", "subject", "semester", "credit", "course", "पाठ्यक्रम", "વિષય"]):
        intent = "College Curriculum & Academics"
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
        disposition = "Enrolled / Hot Lead"
        follow_up = "Send B.Tech online application link via SMS"
    elif any(k in lower for k in ["stu", "student", "roll", "attendance"]):
        lead_status = "Current Student"
        disposition = "Info Provided / Resolved"
        follow_up = "None"
    elif intent == "College Curriculum & Academics":
        lead_status = "Prospective Applicant"
        disposition = "Info Provided / Resolved"
        follow_up = "Share academic syllabus link if requested"
    else:
        lead_status = "Prospective Applicant"
        disposition = "Info Provided / Resolved"
        follow_up = "None"

    # Sentiment
    if any(k in lower for k in ["thank", "धन्यवाद", "great", "good", "helpful", "आभार"]):
        sentiment = "Positive"
    elif any(k in lower for k in ["bad", "wrong", "angry", "bekaar", "gussa"]):
        sentiment = "Frustrated"
        disposition = "Escalation Required"
        follow_up = "Connect with department coordinator at 0268-2520502"
    else:
        sentiment = "Neutral"

    summary = f"Caller inquired regarding {intent.lower()}. Official university details provided."

    return {
        "intent": intent,
        "disposition": disposition,
        "lead_status": lead_status,
        "sentiment": sentiment,
        "follow_up_action": follow_up,
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
    messages = [
        {"role": "system", "content": get_post_call_system_prompt()},
        {"role": "user", "content": f"Concluded Call Transcript:\n{transcript_text}"},
    ]

    req_url = None
    req_headers = {}
    req_payload = {}

    if config.LLM_PROVIDER == "sarvam" and config.SARVAM_API_KEY:
        req_url = config.SARVAM_CHAT_URL
        req_headers = {
            "api-subscription-key": config.SARVAM_API_KEY.strip(),
            "Content-Type": "application/json",
        }
        sarvam_model = config.LLM_MODEL if config.LLM_MODEL and "sarvam" in config.LLM_MODEL else "sarvam-105b-conversations"
        req_payload = {
            "model": sarvam_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 150,
        }
    elif config.LLM_PROVIDER == "groq" and config.GROQ_API_KEY:
        req_url = config.GROQ_URL
        req_headers = {
            "Authorization": f"Bearer {config.GROQ_API_KEY.strip()}",
            "Content-Type": "application/json",
        }
        groq_model = config.LLM_MODEL if config.LLM_MODEL and "llama" in config.LLM_MODEL.lower() else "llama-3.3-70b-versatile"
        req_payload = {
            "model": groq_model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 150,
        }
    elif config.LLM_PROVIDER == "openrouter" and config.OPENROUTER_API_KEY:
        req_url = config.OPENROUTER_URL
        req_headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY.strip()}",
            "Content-Type": "application/json",
        }
        req_payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 150,
        }
    elif config.LLM_PROVIDER == "ollama":
        req_url = config.OLLAMA_URL
        req_headers = {"Content-Type": "application/json"}
        req_payload = {
            "model": config.LLM_MODEL or "qwen2.5:3b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 150},
        }

    if req_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(req_url, headers=req_headers, json=req_payload)
                if resp.status_code == 200:
                    data = resp.json()
                    content = ""
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0].get("message", {}).get("content", "").strip()
                    elif "message" in data:
                        content = data["message"].get("content", "").strip()

                    if content:
                        clean_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
                        parsed = json.loads(clean_json)
                        if "intent" in parsed and "lead_status" in parsed:
                            result = parsed
        except Exception as e:
            print(f"[intelligence] LLM extraction notice ({config.LLM_PROVIDER}): {e}")

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
            disposition=result.get("disposition", ""),            # Fix #16
            follow_up_action=result.get("follow_up_action", ""),  # Fix #16
        )
        print(f"📊 [intelligence] Call {call_id} analyzed: Intent='{result.get('intent')}', Lead='{result.get('lead_status')}', Disposition='{result.get('disposition')}'")
    except Exception as err:
        print(f"[intelligence] Failed to persist analytics for {call_id}: {err}")

    return result
