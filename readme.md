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

                ┌────────────────────────────────────────────────────┐
                │                 THERAPIST INPUT                    │
                │ (user enters message; may guide or probe patient)  │
                └────────────────────────────────────────────────────┘
                                        │
                                        ▼
                  ┌────────────────────────────────┐
                  │ 1️⃣ detect_intent_topic         │
                  │ • Encodes input via MiniLM     │
                  │ • Computes cosine similarity    │
                  │ • Selects or reuses topic       │
                  └────────────────────────────────┘
                                        │
                                        ▼
                  ┌────────────────────────────────┐
                  │ 2️⃣ build_prompt                │
                  │ • Assembles full context:       │
                  │   - Patient profile (JSON)      │
                  │   - Summary of past sessions    │
                  │   - Last 5 turns                │
                  │   - Therapist’s latest input    │
                  │ • Injects topic + instruction   │
                  │   → "Respond naturally, in English" │
                  └────────────────────────────────┘
                                        │
                                        ▼
                  ┌────────────────────────────────┐
                  │ 3️⃣ generate_response (LLM)     │
                  │ • Produces patient’s reply      │
                  │   reflecting tone + context     │
                  │ • Output = natural utterance    │
                  └────────────────────────────────┘
                                        │
                                        ▼
                  ┌────────────────────────────────┐
                  │ 4️⃣ update_memory               │
                  │ • Appends new turn (T ↔ P)     │
                  │ • Summarizes if >5 turns       │
                  │   (compresses into summary)    │
                  │ • Extracts emotional tone via LLM │
                  │   (e.g., "anxious but receptive") │
                  │ • Updates profile field         │
                  │   `current_emotional_state`     │
                  └────────────────────────────────┘
                                        │
                                        ▼
                  ┌────────────────────────────────┐
                  │ 5️⃣ display_response            │
                  │ • Prints patient’s message      │
                  │ • Logs emotional tone, memory   │
                  │   length, summary, topic        │
                  └────────────────────────────────┘
                                        │
                                        ▼
                ┌────────────────────────────────────────────────────┐
                │          LOOP BACK: NEXT THERAPIST TURN            │
                │ (reuses full updated state — continuity preserved) │
                └────────────────────────────────────────────────────┘