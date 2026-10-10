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

    # 1. Inspect user query keywords
    for domain_key, domain_info in domains_cfg.items():
        keywords = domain_info.get("keywords", [])
        for kw in keywords:
            if kw.lower() in lower_text:
                active.add(domain_key)
                break

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
    - Base Persona & Strict Voice Rules
    - Core University Facts
    - Triggered Domain Directives (Student Data vs College Curriculum vs Admissions)
    - Tool Calling Signatures
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

    parts.append("")
    parts.append("STRICT VOICE RULES:")

    # Voice rules
    rules = cfg.get("voice_rules", [])
    for rule in rules:
        parts.append(rule)

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
                    # Clean bullets into readable compact format
                    cleaned_guideline = " ".join(line.strip().lstrip("- ") for line in g_line.splitlines() if line.strip())
                    parts.append(f"- [{eq_key.replace('_', ' ').title()}]: {cleaned_guideline}")

    parts.append("")
    parts.append("UNIVERSITY FACTS:")
    facts = cfg.get("university_facts", [])
    for fact in facts:
        parts.append(f"- {fact}")

    # Active Domain Directives
    domains_cfg = cfg.get("domains", {})
    selected_domains = active_domains if active_domains is not None else list(domains_cfg.keys())

    if selected_domains:
        for dom_key in selected_domains:
            if dom_key in domains_cfg:
                dom_inst = domains_cfg[dom_key].get("instruction", "").strip()
                if dom_inst:
                    parts.append("")
                    parts.append(dom_inst)

    # Tool Instructions
    tools_str = cfg.get("tool_instructions", "").strip()
    if tools_str:
        parts.append("")
        parts.append(tools_str)

    return "\n".join(parts)
