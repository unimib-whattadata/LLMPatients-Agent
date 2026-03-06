# LLMPatients-Agent

Agente modulare e stateful per simulare un paziente in contesti di training psicoterapeutico.

Il progetto combina:
- orchestrazione conversazionale con LangGraph,
- profili clinici strutturati,
- dinamica emotiva simbolica (Panksepp),
- memoria multi-sessione (episodica + reflection + summary lungo termine),
- guardrail di sicurezza contro prompt injection e role-swap.

## Indice

1. [Stato Progetto](#stato-progetto)
2. [Setup Rapido](#setup-rapido)
3. [Esecuzione](#esecuzione)
4. [API Quick Reference](#api-quick-reference)
5. [Runtime e Memoria](#runtime-e-memoria)
6. [Inventario Completo File](#inventario-completo-file)
7. [Note Tecniche Importanti](#note-tecniche-importanti)
8. [Docker](#docker)
9. [Troubleshooting](#troubleshooting)
10. [Limitazioni Note](#limitazioni-note)
11. [Contributing](#contributing)

## Stato progetto

Stato corrente: prototipo di ricerca in sviluppo attivo.

Perimetro operativo attuale:
- CLI interattiva e scripted (`main.py`, `scripts/chat_cli.py`)
- API FastAPI (`agent/api/app.py`)
- persistenza memoria in JSONL (`data/memory/*.jsonl`)
- resume stato per terapeuta/paziente (`tests/runs/<therapist>.json`, generato a runtime)
- documentazione centralizzata in questo unico `readme.md` (nessuna cartella `docs/`)

## Setup rapido

### Requisiti
- Python 3.10+
- ambiente virtuale consigliato
- `model_provider=local`: GPU consigliata (`vllm`)
- `model_provider=vertex_ai`: progetto GCP + credenziali valide

### Installazione

```bash
git clone <your-repo-url>
cd LLMPatients-Agent

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r agent/requirements.txt
```

### Configurazione (`config/.env`)

```dotenv
model_provider=local
model_id=meta-llama/Llama-2-7b-chat-hf
temperature=0.7
max_tokens=512
max_model_len=8192
cache_path=/absolute/path/to/hf-cache

# Solo per Vertex AI
GCP_PROJECT=your-gcp-project
GCP_LOCATION=us-central1
GOOGLE_APPLICATION_CREDENTIALS=config/vertex-ai-api-key.json
```

Note:
- i parametri modello sono letti con nomi lowercase (`model_provider`, `model_id`, ...);
- `max_tokens` limita i token di output per singola generazione, mentre `max_model_len` controlla la finestra totale prompt+output (utile per questionari lunghi come SNAP-2);
- al primo avvio viene scaricato `all-MiniLM-L6-v2` per embedding topic/memory retrieval.

## Esecuzione

### CLI interattiva minima

```bash
PYTHONPATH=. python3 scripts/chat_cli.py --patient juanita_delgado_001 --therapist therapist0
```

### CLI completa (interactive + scripted)

```bash
PYTHONPATH=. python3 main.py --patient juanita_delgado_001 --therapist therapist0
```

Opzioni utili (`main.py`):
- `--session <id>`
- `--messages "..." "..."`
- `--messages-file <path>`

### API server

```bash
PYTHONPATH=. uvicorn agent.api.app:app --reload --port 8000
```

Endpoint esposti:
- `POST /patients`
- `POST /chat-response`
- `POST /session-end`

## API quick reference

### Endpoints

| Endpoint | Metodo | Scopo |
|---|---|---|
| `/patients` | `POST` | Crea/inizializza un profilo paziente JSON in `data/patients/`. |
| `/chat-response` | `POST` | Invia un turno terapeuta e riceve la risposta paziente. |
| `/session-end` | `POST` | Finalizza memoria sessione (reflection + summary lungo termine). |

### Esempi `curl`

#### 1) Creazione paziente

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
    "psychological_profile": "Alex reports chronic worry about work performance and finances.",
    "background": "Software engineer under sustained job pressure",
    "current_medications": ["sertraline 50mg"],
    "therapy_goals": ["Reduce worry", "Improve sleep"],
    "previous_sessions": 0,
    "session_id": "intake-session"
  }'
```

#### 2) Chat turn

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

#### 3) Fine sessione

```bash
curl -X POST http://localhost:8000/session-end \
  -H "Content-Type: application/json" \
  -d '{
    "external_patient_id": "alex_martinez_001",
    "session_id": "intake-session",
    "therapist_id": "therapist0"
  }'
```

Regola operativa critica:
- mantenere stabili `therapist_id` e `session_id` durante la sessione;
- chiamare sempre `/session-end` a fine sessione.

## Runtime e memoria

### Pipeline LangGraph per turno

1. `load_profile`
2. `sanitize_input`
3. `classify_topic_and_emotion`
4. `hydrate_memory`
5. `update_emotions`
6. `build_prompt`
7. `generate`
8. `append_messages`
9. `trim_messages`
10. `update_memory`
11. `display`

### Tipi di memoria persistiti

- `episode_summary`
- `session_reflection`
- `long_term_summary`

Path memoria:
- `data/memory/<therapist_id>__<patient_id>.jsonl`

## Inventario completo file

Inventario aggiornato ai file presenti nel workspace.

### Root

| File | Ruolo |
|---|---|
| `.gitignore` | Regole ignore locali (`config/`, `.env`, `tests/`, e alcuni script). |
| `Dockerfile` | Build immagine API (`uvicorn agent.api.app:app`). |
| `main.py` | CLI principale: interactive + scripted + finalize session memory. |
| `readme.md` | Documentazione unica del progetto. |

### CI / automazione

| File | Ruolo |
|---|---|
| `.github/workflows/eval-suite.yml` | Workflow settimanale/manuale per evaluation scripts e report. |

### Config IDE / metadati locali

| File | Ruolo |
|---|---|
| `.vscode/settings.json` | Config VS Code (`git.ignoreLimitWarning`). |
| `agent/.DS_Store` | Metadato macOS, non funzionale. |
| `data/.DS_Store` | Metadato macOS, non funzionale. |

### Package `agent/`

| File | Ruolo |
|---|---|
| `agent/__init__.py` | Marker package (vuoto). |
| `agent/requirements.txt` | Dipendenze Python pinned. |
| `agent/api/__init__.py` | Marker subpackage (vuoto). |
| `agent/api/app.py` | FastAPI app e contratti API. |
| `agent/core/__init__.py` | Marker subpackage (vuoto). |
| `agent/core/langgraph_builder.py` | Stato LangGraph, nodi pipeline, retrieval, memoria async, finalize. |
| `agent/core/prompt_builder.py` | Composizione prompt (identità, memoria, topic sections, safety, affect). |
| `agent/core/patient_profile.py` | Modello Pydantic paziente + normalizzazione input JSON. |
| `agent/core/emotion_model.py` | Dinamica emotiva: baseline + rumore + modificatori + smoothing. |
| `agent/core/safety.py` | Guardrail + regex injection/role-swap + keyword contestuali. |
| `agent/core/llm_runner.py` | Provider abstraction (`local` vLLM, `vertex_ai` Gemini). |
| `agent/core/memory_store.py` | Store append-only JSONL con locking POSIX opzionale. |
| `agent/utils/run_logger.py` | Logging sessioni e snapshot stato per resume. |
| `agent/utils/session_opening.py` | Greeting contestuale su stato restaurato. |

### `data/`

| File | Ruolo |
|---|---|
| `data/topics_tree.json` | Tassonomia topic e mapping `profile_fields` per prompt dinamico. |
| `data/eval/scenarios.json` | Scenari goal/red-team per evaluation suite. |
| `data/icd11_templates/6D11_5.json` | Template clinico ICD-11 (Borderline pattern). |
| `data/patients/juanita_delgado_001.json` | Profilo paziente runtime-ready (caricato dal codice). |
| `data/patients/juanita_delgado_001.yaml` | Variante YAML del caso Juanita (supporto). |
| `data/patients/crystal_smith_001.yaml` | Caso aggiuntivo YAML. |
| `data/patients/normal_user_001.yaml` | Profilo non-clinico YAML. |
| `data/memory/therapist0__juanita_delgado_001.jsonl` | Memoria persistita campione (`therapist0`). |
| `data/memory/therapist23__juanita_delgado_001.jsonl` | Memoria persistita campione (`therapist23`). |
| `data/memory/therapist77__juanita_delgado_001.jsonl` | Memoria persistita campione (`therapist77`). |
| `data/memory/therapist234__juanita_delgado_001.jsonl` | Memoria persistita campione (`therapist234`). |
| `data/memory/therapist2345__juanita_delgado_001.jsonl` | Memoria persistita campione (`therapist2345`). |

### `scripts/`

| File | Ruolo |
|---|---|
| `scripts/chat_cli.py` | REPL conversazionale minimale. |
| `scripts/run_eval_suite.py` | Runner scenari evaluation (richiede modulo `agent.eval`). |
| `scripts/eval_report.py` | Report/benchmark confronto summary evaluation. |

### `tests/`

| File | Ruolo |
|---|---|
| `tests/api-google-test.py` | Smoke test manuale per `create_llm_runner()` e generazione. |

### `notebooks/`

| File | Ruolo |
|---|---|
| `notebooks/data_management.ipynb` | Parsing HTML DSM + produzione JSON elaborato. |
| `notebooks/inference.ipynb` | Notebook sperimentale di inferenza/generazione. |
| `notebooks/data/source/depression_dsm_cases.html` | Sorgente HTML DSM (depressione). |
| `notebooks/data/source/anxiety_dsm_cases.html` | Sorgente HTML DSM (ansia). |
| `notebooks/data/source/test.json` | Dataset test persona CBT sintetiche. |
| `notebooks/data/processed/extracted_conditions.json` | Output elaborato data-management. |

## Note tecniche importanti

- Runtime profili paziente:
  il caricamento in `agent/utils/session_opening.py` usa `data/patients/<id>.json`; i file YAML non vengono caricati direttamente dalla pipeline.

- Coerenza label emozionali:
  il runtime usa `SEEKING, FEAR, RAGE, LUST, CARE, PANIC_GRIEF, PLAY`; nei YAML di esempio compaiono anche label legacy (`ANGER`, `SADNESS`) che non entrano direttamente nel set finale.

- Evaluation suite:
  `scripts/run_eval_suite.py` importa `agent.eval` (modulo non presente nella tree corrente).

- `.gitignore` e file tracciati:
  anche se `.gitignore` include `tests/` e alcuni script, i file già versionati continuano a funzionare.

- Smoke test API:
  `tests/api-google-test.py` modifica `sys.path` per importare `core.llm_runner` da `agent/`; è un test manuale rapido, non una suite automatica completa.

- Logging CLI:
  le CLI stampano output con icone/telemetria; è solo osservabilità, non logica di business.

## Docker

```bash
docker build -t llmpatients-agent .
docker run --rm -p 8000:8000 -e PYTHONPATH=/app llmpatients-agent
```

Se usi Vertex AI nel container, monta/inietta credenziali e variabili ambiente.

## Troubleshooting

- `Missing model_id for LocalLLMRunner`:
  manca `model_id` in env/.env.

- Errori Vertex:
  verificare `GCP_PROJECT`, `GCP_LOCATION`, `GOOGLE_APPLICATION_CREDENTIALS`.

- Prima risposta lenta:
  download iniziale embeddings/modello.

- Continuità sessione assente:
  controllare coerenza `therapist_id`, `session_id` e chiamata `/session-end`.

## Limitazioni note

- Non c'è file `LICENSE` nel repository.
- I file in `data/memory/` sono esempi reali di memoria persistita: trattarli come dati sensibili di simulazione.
- Le pipeline notebook sono sperimentali e non fanno parte del runtime API/CLI.

## Contributing

1. Aprire branch dedicato.
2. Mantenere questo README allineato ai cambiamenti file-level.
3. Eseguire smoke test CLI/API prima della PR.
4. Documentare nuove variabili ambiente e nuovi path dati.
