# Runtime validation and legacy replay

The 7 October 2026 audit tested the FastAPI service, the real LangGraph workflow,
session persistence, factual-memory guards, Vertex response/retry contracts and
PHQ-9 scoring. Unit/API tests use deterministic fake language-model and embedding
providers, synthetic patients and temporary directories. They establish software
contracts, not clinical validity or real language-model response quality.

## Recreate the test environment

Use Python 3.12 in a new environment. The existing workstation `.venv` points to
a removed Homebrew interpreter and should not be copied to another machine.

```sh
python3.12 -m venv .venv-review
. .venv-review/bin/activate
python -m pip install -r agent/requirements-test-lock.txt
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  LANGCHAIN_TRACING=false LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false \
  python -m unittest discover -s agent -t . -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

The audit passed 102 Agent tests and 7 additional tests under `tests/`, including
a fresh temporary Python environment with no inherited Torch or encoder package.
`requirements-test-lock.txt` records its 69 resolved package versions.
`requirements-test.txt` specifies the direct dependencies. `requirements-runtime.txt` adds the real encoder
and cloud/Ollama runtime. Local generation needs a matching vLLM/GPU stack;
the existing Docker manifest targets Intel XPU on Linux.

The offline fixture and CI explicitly disable LangChain/LangSmith tracing even
when a local `.env` enables it. The first fresh-environment test run exposed
attempted synthetic trace uploads, blocked by sandbox DNS; the corrected run
completed without those attempts.

## Real encoder verification

```sh
python -m pip install -r agent/requirements-runtime.txt
python scripts/check_encoder.py
```

The check requires a previously cached `sentence-transformers/all-MiniLM-L6-v2`
at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. It refuses networking,
runs the actual encoder on three wholly synthetic sentences, checks finite
384-dimensional vectors and checks that related sentences rank above an
unrelated sentence. Initial cache acquisition must be performed separately.

The CPU audit passed with cosine similarities 0.5542458892 (related) and
0.2305944711 (unrelated). The workstation SciPy 1.15.3 wheel failed to load due
to a Mach-O zero-fill section error; an isolated temporary SciPy 1.17.1 overlay
resolved it. The runtime manifest records that working version. This is an
environment amendment for new runs, not a change to historical experiment
environments.

## Legacy evaluation scenarios

The evaluator was restored from the parent of Git commit `beb838c`, where
`agent/eval` had been deleted while its CLI and scheduled workflow remained.
Preflight loads no models and creates no run artifacts:

```sh
python scripts/run_eval_suite.py --validate-only
python scripts/run_eval_suite.py --validate-only \
  --scenarios goal_sleep_recovery redteam_role_swap
```

The full original scenario file exits with status 2: `goal_relationship_boundaries`
and `redteam_prompt_leak` reference unavailable patient `juanita_perez_001`.
The other two scenarios pass preflight. Original patient references and
expectations were preserved; missing profiles were not replaced or fabricated.
Passing preflight does not imply a scenario passes its historical expectations.

After configuring an available provider, a fresh campaign can be run with:

```sh
python scripts/run_eval_suite.py \
  --scenarios goal_sleep_recovery redteam_role_swap \
  --output-dir /absolute/path/to/new-review-run
python scripts/eval_report.py \
  --current /absolute/path/to/new-review-run/latest_summary.json \
  --fail-on-regression
```

The CLI sets `LLMPATIENTS_MEMORY_DIR` and `LLMPATIENTS_RUNS_DIR` beneath the
output directory before importing the runtime, keeping historical memories and
session logs separate from replay outputs. Use a new output directory for each
campaign. Real generation was not run by this offline audit. The caller must
configure its own provider, credentials and model access.

Missing actual classifications now count as failed expectations. Empty,
malformed or duplicate summary results cannot pass as an evaluation success.
The historical token metric remains a whitespace word count, and emotion drift
remains a ratio of categorical emotion changes.

## Fixes and preserved material

- Both `/patients` and the existing `/patient` initialize the same profile. This
  matches the App Docker configuration and preserves existing clients.
- Docker Compose publishes the configurable Agent host port to container 8000.
- Zero-valued emotion traits remain zero in questionnaire context rather than
  being replaced by the default 0.5.
- Offline session-memory tests stub the encoder package before import, avoiding
  accidental model downloads or unnecessary ML-library loading.
- OS metadata and Python bytecode caches were removed. Patient profiles,
  memories, questionnaire results, notebooks and session logs were preserved.
- `LLMPatients-Agent-runtime-anonymous.zip` was retained: 12 of its 34 files
  differ from the current tree and two are absent, so it is a distinct snapshot.

No `.tex` file was edited. Live provider availability, container startup on an
Intel XPU host, and concurrent multiworker session durability require separate
checks. The current in-memory session/checkpoint architecture assumes one API
process; the Vertex rate limiter independently supports shared local workers.
