# 🎓 University Admission & Student Info Voice Agent

A multi-lingual voice call agent for university admission inquiries and student academic assistance. Parents and prospective students call over **WebSockets** to ask about:
- **Admission Process, Eligibility, Fees, & Deadlines**
- **University Achievements & Department Placements**
- **Student Academic Marks & Subject-wise Attendance** (via database tool calling)

The agent speaks naturally in **English, Hindi, and Gujarati**.

---

## 🏗️ System Architecture

```mermaid
flowchart TB
    classDef telephony fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef gateway fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#0d47a1
    classDef stt fill:#fff3e0,stroke:#e65100,stroke-width:1.5px,color:#bf360c
    classDef llm fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1.5px,color:#4a148c
    classDef storage fill:#f5f5f5,stroke:#616161,stroke-dasharray:5 5,color:#424242
    classDef tts fill:#e0f7fa,stroke:#00695c,stroke-width:1.5px,color:#004d40

    subgraph Callers ["Parents & Students"]
        Caller1([Hindi Speaking Parent])
        Caller2([Gujarati Speaking Parent])
        Caller3([English Speaking Student])
    end
    class Caller1,Caller2,Caller3 telephony

    subgraph Gateway ["Call Connection Layer"]
        WSGateway["WebSocket Gateway\n(ws://0.0.0.0:8765)"]
    end
    class WSGateway gateway

    subgraph Core ["Voice Agent Core"]
        STT["STT Layer\n(sarvam)"]
        LLM["LLM Engine & Tool Agent\n(qwen)"]
        TTS["TTS Layer\n(sarvam)"]
    end
    class STT stt
    class LLM llm
    class TTS tts

    subgraph Data ["Database Layer"]
        Postgres[("PostgreSQL Database\n(Students, Marks, Attendance,\nPlacements, Admissions)")]
    end
    class Postgres storage

    Caller1 & Caller2 & Caller3 -- "WebSocket Connection" --> WSGateway
    WSGateway -- "Audio Input Stream" --> STT
    STT -- "Transcribed Text + Lang Tag" --> LLM
    LLM -- "Tool Queries" --> Postgres
    Postgres -- "Query Results" --> LLM
    LLM -- "Response Text" --> TTS
    TTS -- "Audio Output Stream" --> WSGateway
    WSGateway -- "Audio Response" --> Caller1 & Caller2 & Caller3
```

---

## 🛠️ Technology Stack

| Layer | Provider / Tool | Description |
|-------|----------------|-------------|
| **Call Connection** | `websockets` | Real-time audio streaming connection gateway |
| **STT (Speech-to-Text)** | `sarvam` | Multi-lingual speech recognition (EN, HI, GU) |
| **LLM Engine** | `qwen` (via Ollama) | Local Qwen 2.5 LLM with tool-calling support |
| **TTS (Text-to-Speech)** | `sarvam` | High quality multi-lingual voice synthesis |
| **Database** | `postgresql` | Stores student marks, attendance, placements & admission details |

---

## 🔍 Detailed Component Breakdown

### 1. Call Connectivity (`websockets`)
- Real-time bi-directional audio connection over WebSockets.
- Listens on `ws://0.0.0.0:8765`.
- Accepts binary audio streams from calling clients and yields synthesized audio chunks back for instant playback.

### 2. Speech-to-Text (`sarvam`)
- Converts caller audio into transcribed text in real-time.
- Automatically identifies spoken language (`hi-IN`, `gu-IN`, `en-IN`).
- Code reference:
```python
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="YOUR_SARVAM_API_KEY")

response = client.speech_to_text.transcribe(
    file=open("audio.wav", "rb"),
    mode="transcribe"
)
print(response)
```

### 3. LLM & Tool-Calling Agent (`qwen`)
- Powered by Qwen 2.5 (3B) running locally via Ollama.
- Personified as a university admission desk assistant.
- Dynamically parses questions, executes database query tools, and synthesizes natural responses.

#### 🧰 Available Agent Tools:
1. **`lookup_student(identifier)`**: Finds a student's ID, department, and semester.
2. **`get_student_marks(student_id)`**: Fetches subject-wise marks and grades.
3. **`get_student_attendance(student_id)`**: Fetches attendance records and percentages.
4. **`get_placement_stats(department)`**: Retrieves placement packages (LPA), placement rate %, and top recruiting companies.
5. **`get_admission_info(program)`**: Fetches eligibility criteria, tuition fees, and application deadlines.

### 4. Text-to-Speech (`sarvam`)
- Synthesizes text responses into natural sounding audio in English, Hindi, or Gujarati.
- Code reference:
```python
from sarvamai import SarvamAI

client = SarvamAI(api_subscription_key="YOUR_SARVAM_API_KEY")

response = client.text_to_speech.convert(
    text="नमस्ते, आज मैं आपकी क्या मदद कर सकता हूँ?",
    target_language_code="hi-IN",
    speaker="shubh",
)
print(response)
```

### 5. PostgreSQL Database
Stores structured university information. Schema includes:
- **`students`**: `student_id`, `name`, `department_id`, `semester`, `parent_phone`
- **`marks`**: `student_id`, `subject`, `marks_obtained`, `max_marks`, `grade`
- **`attendance`**: `student_id`, `subject`, `total_classes`, `classes_attended`, `attendance_percentage`
- **`placement_stats`**: `year`, `department_id`, `highest_package_lpa`, `average_package_lpa`, `top_recruiters`
- **`admission_info`**: `program`, `eligibility`, `fee_per_year`, `last_date_to_apply`

*(Note: Automatically falls back to local SQLite database if PostgreSQL is not active)*

---

## 📁 Project Structure

```text
├── README.md               # Documentation
├── pyproject.toml          # Project dependencies
├── project.mermaid         # Architecture diagrams
├── main.py                 # Application launcher (WebSocket Server & CLI Mode)
├── config/
│   └── config.py           # Sarvam API keys, DB URL, model configs
├── stt/
│   └── stt.py              # STT integration (sarvam)
├── tts/
│   └── tts.py              # TTS integration (sarvam)
├── llm/
│   └── llm.py              # Qwen LLM integration with DB Tool Calling
├── db/
│   ├── database.py         # PostgreSQL connection & query tool methods
│   └── schema.sql          # DB schema definition & sample student seed data
└── ws/
    └── server.py           # WebSocket Media Gateway server
```

---

## 🚀 Setup & Execution

### 1. Requirements
- Python 3.10+
- [Ollama](https://ollama.com/) running locally with Qwen:
  ```bash
  ollama pull qwen2.5:3b
  ```

### 2. Environment Configuration
Export your Sarvam API Key:
```bash
export SARVAM_API_KEY="sk_xc0xbbkb_xTxOQkUEFOY8iwucsJOgPWIA"
```

Optionally set custom PostgreSQL connection string:
```bash
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/university_agent"
```

### 3. Run the Voice Agent

#### Mode A: WebSocket Media Server (Default)
Starts WebSocket server for call connections on `ws://0.0.0.0:8765`:
```bash
.venv/bin/python main.py
```

#### Mode B: Terminal Interactive Mode (CLI)
Test the agent using your microphone & speakers directly in terminal:
```bash
.venv/bin/python main.py --cli
```

---

## 💬 Example Calls & Interactions

- **Parent (Hindi):** *"मेरे बेटे आरव पटेल के मार्क्स क्या आए हैं?"*
  - **Agent:** Executes `lookup_student("आरव पटेल")` -> `get_student_marks("STU101")`
  - **Agent Response (Hindi spoken audio):** *"आरव पटेल को डेटा स्ट्रक्चर्स में 88 मार्क्स और डेटाबेस मैनेजमेंट में 92 मार्क्स मिले हैं।"*

- **Parent (English):** *"What is the placement percentage for CSE department?"*
  - **Agent:** Executes `get_placement_stats("CSE")`
  - **Agent Response (English spoken audio):** *"The Computer Science department achieved a 96.5% placement rate with an average package of 12.5 LPA. Top recruiters include Google, Microsoft, and Amazon."*
