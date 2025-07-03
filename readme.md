# 🧠 LLMPatient: Simulated Patient Chat System for Therapist Training

## 📌 Overview

This project aims to develop a **chat-based system powered by a local LLM** that simulates a **realistic patient with a mental health disorder**. It is designed as a **training tool for psychologists**, enabling them to practice therapeutic conversations in multiple sessions with continuity, coherence, and eventual patient evolution.

---

## 🎯 Project Goals

- Create a **conversational LLM agent** that impersonates a specific mental health patient.
- Ensure **multi-session coherence**, allowing the simulated patient to "remember" previous interactions.
- Embed a **rich, structured patient persona** including symptoms, history, mood, and behavior.
- Support future **adaptive behavior changes** based on therapist performance or session progression.
- Provide a base system that is **modular and extensible** for future research and training enhancements.

---

## 🔧 Tech Stack

| Component          | Choice/Tool                                  |
|--------------------|----------------------------------------------|
| LLM Backend        | Local model (e.g., Mistral, LLaMA2, Phi-3)   |
| Inference Engine   |                   `vLLM`                     |
| Embedding Model    | `sentence-transformers` or `bge-small-en`    |
| Vector Store       | `FAISS`, `ChromaDB`                          |
| Memory Management  | Custom + LangChain Memory                    |
| Orchestration      | LangChain Agent with fallback behavior       |
| UI (MVP)           | `Streamlit`.                                 |
| Config & Secrets   | `python-dotenv`                              |
| (Optional) DB      | SQLite or MongoDB for logs/state             |

---

## 📁 Project Structure

```text
llm_patient_simulator/
├── app/
│   ├── main.py                  # Entry point (Streamlit or FastAPI interface)
│   ├── config.py                # Configuration and environment loading
│   ├── core/                    # Core logic of the system
│   │   ├── langgraph_builder.py # Builds the graph 
│   │   ├── prompt_builder.py    # Builds LLM prompt with memory and persona
│   │   ├── memory.py            # Short- and long-term memory management
│   │   ├── patient_profile.py   # Loads and tracks patient state/persona
│   │   ├── llm_runner.py        # Interfaces with local LLM backend
│   │   └── embeddings.py        # Embedding model wrapper for memory retrieval
│   ├── data/
│   │   ├── patients/            # JSON files with patient profiles
│   │   └── memory_store/        # Vector database (e.g., FAISS index)
│   └── utils/                   # Utility functions (logging, formatting, etc.)
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── config/                      # contains all of the configuration information
    ├── .env.                    # enviroment info
    └── prompt_template.py       # File containing the prompt templates
```

## 🗂️ Modules & Responsibilities

### `main.py`
Entry point for running the web UI or backend service.

### `core/orchestrator.py`
Central controller. Handles message flow, memory retrieval, persona loading, and agent interaction.

### `core/prompt_builder.py`
Builds the full prompt, combining:
- System instructions
- Persona traits
- Retrieved memory
- Current user input
- Recent chat history

### `core/memory.py`
Manages both:
- **Short-term memory** (in-session)
- **Long-term memory** (multi-session, vector-based)

### `core/patient_profile.py`
Stores and manages patient state, including:
- Demographics and diagnosis
- Symptom profiles
- Personality traits and behavior tendencies
- Therapy goals and progression markers

### `core/llm_runner.py`
Interfaces with the local LLM (via `transformers` or `llama-cpp`) to generate responses.

### `core/embeddings.py`
Encodes textual memory entries and queries for similarity-based retrieval.

### `data/patients/`
Patient JSON profiles with structured metadata.

### `data/memory_store/`
FAISS or ChromaDB indices for long-term memory retrieval.

### `utils/`
Utility functions (time, logging, formatting, etc.)

---

## 🚀 Future Extensions

> 🧠 You can start with a static agent behavior and later integrate dynamic components.

| Feature | Description | Status |
|--------|-------------|--------|
| Adaptive agent behavior | Switch patient tone based on therapist quality or session analysis | 🔜 Future |
| Patient state progression | Patient improves or worsens over time (based on symbolic or statistical logic) | 🔜 Future |
| Therapist feedback analysis | Log, rate, or analyze therapist input using rule-based or ML classifiers | 🔜 Future |
| Patient emotion tracking | Simulated mood evolution using symbolic or neural state models | 🔜 Future |
| GUI with memory timeline | Inspect patient memory and state visually | 🔜 Future |

---

## 🔄 System Workflow

```text
1. Therapist sends message to the simulated patient
2. System:
   - Retrieves relevant memory from vector store
   - Loads the patient's profile and session history
   - Assembles prompt (instructions + persona + memory + input)
   - Sends prompt to the local LLM via the orchestrator
3. LLM generates a coherent, persona-aligned response
4. Memory and patient state are updated
5. Chat is returned to the therapist, and logged