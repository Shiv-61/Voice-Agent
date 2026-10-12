"""
Modular Prompt Loader with Dynamic Domain Triggering & Hot-Reloading.
Loads prompt templates, persona rules, and domain directives from config/prompts.yaml.
Automatically refreshes when the YAML file is modified without restarting the server.
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import yaml

# Path to the prompts YAML configuration
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.yaml"

_cached_prompts: dict[str, Any] = {}
_last_mtime: float = 0.0


def _load_prompts_yaml() -> dict[str, Any]:
    """Loads and caches the YAML file, refreshing if modified on disk."""
    global _cached_prompts, _last_mtime
    try:
        if not PROMPTS_FILE.exists():
            return _cached_prompts

        current_mtime = PROMPTS_FILE.stat().st_mtime
        if current_mtime != _last_mtime or not _cached_prompts:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            _cached_prompts = data
            _last_mtime = current_mtime
            # print(f"[prompt_loader] Loaded prompts configuration from {PROMPTS_FILE.name}")
    except Exception as e:
        print(f"[prompt_loader] Notice reading {PROMPTS_FILE.name}: {e}")

    return _cached_prompts


def load_prompts() -> dict[str, Any]:
    """Public helper to get the latest loaded prompt configuration dict."""
    return _load_prompts_yaml()


def get_welcome_message() -> str:
    """Returns the initial call greeting message."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "welcome_message",
        "નમસ્તે, હું ડીડીયુ આઈટી ડિપાર્ટમેન્ટમાંથી પ્રિયા વાત કરું છું. હું તમારી શું મદદ કરી શકું? તમે ગુજરાતી, હિન્દી અથવા અંગ્રેજીમાં વાત કરી શકો છો.",
    )


def get_call_hangup_prompt() -> str:
    """Returns the supervisory prompt for classifying call hangup / completion."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "call_hangup_prompt",
        """\
You are an intelligent call supervisor analyzing a voice call between a caller and a university voice assistant.
Determine whether the caller or the conversation has concluded and the telephone call should be hung up.
Return {"call_hangup": true} or {"call_hangup": false}.
""",
    )


def get_post_call_prompt() -> str:
    """Returns the CRM intelligence analysis prompt for finished calls."""
    cfg = _load_prompts_yaml()
    return cfg.get(
        "post_call_prompt",
        """\
You are an expert College Admissions & Student Affairs CRM Auditor analyzing a concluded voice call transcript.
Return ONLY a raw, valid JSON object with keys: intent, lead_status, sentiment, summary.
""",
    )


def get_fallback(key: str, default: str = "") -> str:
    """Retrieves a fallback phrase from the configuration."""
    cfg = _load_prompts_yaml()
    fallbacks = cfg.get("fallbacks", {})
    return fallbacks.get(key, default)


def detect_active_domains(user_text: str, retrieved_docs: Optional[list[dict[str, Any]]] = None) -> list[str]:
    """
    Intelligently detects which domain(s) should be activated based on:
    1. Caller's spoken question / keywords (English, Hindi, Gujarati).
    2. Uploaded / retrieved RAG documents (curriculum vs student records vs admissions).
    """
    cfg = _load_prompts_yaml()
    domains_cfg = cfg.get("domains", {})
    lower_text = (user_text or "").lower()

    active: set[str] = set()

    # 1. Inspect user query keywords using config domains or default fallback keywords
    if domains_cfg:
        for domain_key, domain_info in domains_cfg.items():
            keywords = domain_info.get("keywords", [])
            for kw in keywords:
                if kw.lower() in lower_text:
                    active.add(domain_key)
                    break
    else:
        default_keywords = {
            "student_records": ["student", "mark", "grade", "attendance", "result", "cpi", "spi", "stu", "હાજરી", "માર્ક્સ", "अटेंडेंस", "अंक"],
            "college_curriculum": ["syllabus", "curriculum", "subject", "semester", "credit", "course", "વિષય", "સિલેબસ", "સેમેસ્ટર", "पाठ्यक्रम", "सिलेबस"],
            "admissions_and_campus": ["admission", "fee", "hostel", "placement", "curfew", "scholarship", "gate", "ફી", "પ્રવેશ", "હોસ્ટેલ", "फीસ", "फी", "प्रवेश"],
        }
        for dom_key, kws in default_keywords.items():
            if any(kw in lower_text for kw in kws):
                active.add(dom_key)

    # 2. Inspect retrieved RAG documents
    if retrieved_docs:
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            category = (meta.get("category") or "").lower()
            filename = (meta.get("filename") or "").lower()
            text_snippet = (doc.get("text") or "").lower()

            if category == "student_records" or any(k in filename or k in text_snippet[:200] for k in ["student", "mark", "grade", "attendance", "transcript", "cgpa", "sgpa"]):
                active.add("student_records")
            if category == "college_curriculum" or any(k in filename or k in text_snippet[:200] for k in ["curriculum", "syllabus", "subject", "semester", "credit", "course"]):
                active.add("college_curriculum")
            if category in ("admissions", "campus", "admissions_and_campus", "placements_and_achievements") or any(k in filename or k in text_snippet[:200] for k in ["admission", "fee", "hostel", "placement", "hackathon", "gate", "achievement"]):
                active.add("admissions_and_campus")

    return list(active)


def get_system_prompt(active_domains: Optional[list[str]] = None) -> str:
    """
    Dynamically constructs the system prompt:
    - Base Persona & Description
    - Call Flow (Steps 1 to 6)
    - Multilingual Policy
    - Strict Voice Rules
    - Hangup and Silence Handling (Global Overrides)
    - Guardrails & Emotional Intelligence
    - Prompt Knowledge (Route A) & Answering Style
    - RAG Instructions (Route B)
    - Student Records (Route C)
    - Tool Instructions & Protocol
    """
    cfg = _load_prompts_yaml()
    persona = cfg.get("base_persona", {})
    p_name = persona.get("name", "Priya")
    p_role = persona.get("role", "AI Voice Assistant for DDU IT Department")
    p_inst = persona.get("institution", "Dharamsinh Desai University (DDU) IT Department in Nadiad, Gujarat")
    p_desc = persona.get("description", "You are on an active live telephone call with a student, applicant, or parent.")

    # Persona header
    parts = [
        f"You are {p_name}, the {p_role} for {p_inst}.",
        p_desc,
    ]

    # Call Flow
    call_flow = cfg.get("call_flow", "").strip()
    if call_flow:
        parts.append("")
        parts.append("CALL FLOW (FOLLOW IN ORDER ON EVERY CALL):")
        parts.append(call_flow)

    # Multilingual Priority & Dynamic In-Call Language Switching
    lang_policy = cfg.get("language_policy", {})
    if lang_policy:
        parts.append("")
        parts.append("MULTILINGUAL POLICY (PRIMARY: GUJARATI, SECONDARY: HINDI & ENGLISH):")
        parts.append(f"- Primary Language: {lang_policy.get('primary', 'Gujarati (ગુજરાતી)')}")
        secondaries = lang_policy.get('secondary', ['Hindi (हिंदी)', 'English'])
        parts.append(f"- Secondary Languages: {', '.join(secondaries)}")
        for r in lang_policy.get("rules", []):
            parts.append(f"- {r}")

    # Voice rules
    rules = cfg.get("voice_rules", [])
    if rules:
        parts.append("")
        parts.append("STRICT VOICE RULES:")
        for rule in rules:
            parts.append(rule)

    # Hangup and Silence Handling (Global Overrides)
    hangup_silence = cfg.get("hangup_and_silence_handling", "").strip()
    if hangup_silence:
        parts.append("")
        parts.append("HANGUP COMMAND AND SILENCE HANDLING (GLOBAL OVERRIDES):")
        parts.append(hangup_silence)

    # Guardrails
    guardrails = cfg.get("guardrails", {})
    if guardrails:
        parts.append("")
        parts.append("SAFETY & INTEGRITY GUARDRAILS:")
        for g_cat, g_rules in guardrails.items():
            if isinstance(g_rules, list):
                for gr in g_rules:
                    parts.append(f"- {gr}")

    # Emotional Intelligence & Empathy
    eq_cfg = cfg.get("emotional_intelligence", {})
    if eq_cfg:
        parts.append("")
        parts.append("EMPATHY & ADAPTIVE TONE:")
        for eq_key, eq_val in eq_cfg.items():
            if isinstance(eq_val, dict):
                g_line = eq_val.get("guideline", "").strip()
                if g_line:
                    cleaned_guideline = " ".join(line.strip().lstrip("- ") for line in g_line.splitlines() if line.strip())
                    parts.append(f"- [{eq_key.replace('_', ' ').title()}]: {cleaned_guideline}")
            elif isinstance(eq_val, str) and eq_val.strip():
                parts.append(f"- [{eq_key.replace('_', ' ').title()}]: {eq_val.strip()}")

    # Prompt Knowledge (Route A - Known Facts) & University Facts
    facts = cfg.get("prompt_knowledge", []) or cfg.get("university_facts", [])
    if facts:
        parts.append("")
        parts.append("ROUTE A - PROMPT KNOWLEDGE (KNOWN DIRECTLY):")
        for fact in facts:
            parts.append(f"- {fact}")

    answering_style = cfg.get("answering_style", "").strip()
    if answering_style:
        parts.append(f"Answering Style: {answering_style}")

    # Route B - RAG Instructions
    rag_inst = cfg.get("rag_instructions", "").strip()
    if rag_inst:
        parts.append("")
        parts.append("ROUTE B - RAG INSTRUCTIONS:")
        parts.append(rag_inst)

    # Route C - Student Records
    student_records = cfg.get("student_records", "").strip()
    if student_records:
        parts.append("")
        parts.append("ROUTE C - STUDENT DATABASE:")
        parts.append(student_records)

    # Active Domain Directives (Legacy compatibility)
    domains_cfg = cfg.get("domains", {})
    selected_domains = active_domains if active_domains is not None else list(domains_cfg.keys())
    if selected_domains and domains_cfg:
        for dom_key in selected_domains:
            if dom_key in domains_cfg:
                dom_inst = domains_cfg[dom_key].get("instruction", "").strip()
                if dom_inst:
                    parts.append("")
                    parts.append(dom_inst)

    # Tool Instructions / Protocol
    tools_str = cfg.get("tool_instructions", "").strip()
    if tools_str:
        parts.append("")
        parts.append("TOOL PROTOCOL:")
        parts.append(tools_str)

    return "\n".join(parts)

