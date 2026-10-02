import sys
from pathlib import Path
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.prompt_loader import detect_active_domains, get_system_prompt, get_welcome_message
from llm.llm import LLM

def test_modular_prompts():
    print("=== 1. Test Welcome Message ===")
    welcome = get_welcome_message()
    print("Welcome:", welcome)
    assert len(welcome) > 10

    print("\n=== 2. Test Domain Triggers ===")
    test_cases = [
        ("What is the syllabus and curriculum for semester 3?", ["college_curriculum"]),
        ("What are the marks and attendance of STU101?", ["student_records"]),
        ("What is the admission fee and hostel curfew?", ["admissions_and_campus"]),
        ("Rahul Sharma ke marks kitne hain?", ["student_records"]),
        ("Diploma ke baad direct 2nd year syllabus kya hai?", ["college_curriculum"]),
    ]

    for q, expected_domains in test_cases:
        triggered = detect_active_domains(q)
        print(f"Query: '{q}' -> Triggered: {triggered}")
        for expected in expected_domains:
            assert expected in triggered, f"Expected {expected} in {triggered} for '{q}'"

    print("\n=== 3. Test Dynamic System Prompt Construction ===")
    curriculum_prompt = get_system_prompt(["college_curriculum"])
    assert "[ACTIVE DIRECTIVE: COLLEGE CURRICULUM & ACADEMICS]" in curriculum_prompt
    assert "You are Priya" in curriculum_prompt
    assert "SAFETY & INTEGRITY GUARDRAILS:" in curriculum_prompt
    assert "EMPATHY & ADAPTIVE TONE:" in curriculum_prompt
    assert "[Anxious Or Worried]:" in curriculum_prompt
    assert "DATABASE & KNOWLEDGE LOOKUP PROTOCOL:" in curriculum_prompt
    print("✓ Curriculum prompt assembled correctly with guardrails & emotional tone")

    student_prompt = get_system_prompt(["student_records"])
    assert "[ACTIVE DIRECTIVE: STUDENT RECORDS & PERFORMANCE]" in student_prompt
    assert "FERPA & DPDP Act compliance" in student_prompt
    print("✓ Student records prompt assembled correctly with privacy guardrails")

    admissions_prompt = get_system_prompt(["admissions_and_campus"])
    assert "[ACTIVE DIRECTIVE: ADMISSIONS & CAMPUS LIFE]" in admissions_prompt
    print("✓ Admissions prompt assembled correctly")

    print("\n=== 4. Test RAG Document Categorization ===")
    from rag import RAGStore
    rag = RAGStore()
    docs = rag.list_documents()
    print(f"Indexed documents in RAG ({len(docs)}):")
    for d in docs:
        print(f" - {d.get('filename')}: category='{d.get('category')}'")

    print("\n=== 5. Test Live LLM Message Preparation ===")
    llm = LLM()
    # Test curriculum query message construction
    msgs = llm._prepare_messages("What is the B.Tech curriculum?")
    sys_content = msgs[0]["content"]
    assert "[ACTIVE DIRECTIVE: COLLEGE CURRICULUM & ACADEMICS]" in sys_content
    print("✓ LLM _prepare_messages triggered curriculum directive dynamically")

    # Test student query message construction
    msgs = llm._prepare_messages("Check marks of STU101")
    sys_content = msgs[0]["content"]
    assert "[ACTIVE DIRECTIVE: STUDENT RECORDS & PERFORMANCE]" in sys_content
    print("✓ LLM _prepare_messages triggered student records directive dynamically")

    print("\n✅ All modular prompt tests passed 100%!")

if __name__ == "__main__":
    test_modular_prompts()
