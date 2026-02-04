# LLMPatients-Agent API Usage

Minimal examples for creating a patient record, sending chat turns, and finalizing sessions.

## 1) Start the Service
- Launch FastAPI: `uvicorn agent.api.app:app --reload --port 8000`
- No authentication is required at this stage.

## 2) Create / Initialize a Patient
```bash
curl -X POST http://localhost:8000/patients \
  -H "Content-Type: application/json" \
  -d '{
        "id": "alex_martinez_001",
        "name": "Alex Martinez",
        "age": 32,
        "gender": "male",
        "diagnosis": "Generalized anxiety disorder",
        "difficulty_level": 2,
        "psychological_profile": "Alex reports chronic worry about work performance and finances, with muscle tension and poor sleep.",
        "background": "Software engineer under sustained job pressure",
        "current_medications": ["sertraline 50mg"],
        "therapy_goals": ["Reduce worry", "Improve sleep"],
        "previous_sessions": 0,
        "session_id": "intake-session"
      }'
```

## 3) Send a Chat Turn
```bash
curl -X POST http://localhost:8000/chat-response \
  -H "Content-Type: application/json" \
  -d '{
        "external_patient_id": "alex_martinez_001",
        "user_message": "How have you been sleeping this week?",
        "session_id": "intake-session",
        "step_id": 1,
        "therapist_id": "therapist0"
      }'
```

Example response:
```json
{
  "message": "Honestly, sleep has been rough. I keep waking up worrying about deadlines.",
  "reasoning_time": 0.842,
  "emotion": "seeking",
  "topic": "general",
  "timestamp": "2024-07-17T12:35:02.789012"
}
```

## 4) End a Session (finalize reflection + long-term summary)
```bash
curl -X POST http://localhost:8000/session-end \
  -H "Content-Type: application/json" \
  -d '{
        "external_patient_id": "alex_martinez_001",
        "session_id": "intake-session",
        "therapist_id": "therapist0"
      }'
```

## Notes
- Patient files are stored under `data/patients/<patient_id>.json`.
- Use a consistent `session_id` to keep context and summaries tied to the same conversation.
- `difficulty_level` maps to the internal volatility preset (low/medium/high) for emotion dynamics.
