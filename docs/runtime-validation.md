# Runtime validation

The 7 October 2026 audit passed 102 Agent tests and seven additional tests in a
fresh Python 3.12 environment. API tests exercise the real FastAPI/LangGraph path
with explicit synthetic model/encoder fixtures and temporary files. They verify
software contracts, not live generation quality or clinical validity.

## Offline checks

From the repository root:

```sh
python3.12 -m venv .venv-test
. .venv-test/bin/activate
python -m pip install -r agent/requirements-test-lock.txt
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LANGCHAIN_TRACING=false LANGCHAIN_TRACING_V2=false LANGSMITH_TRACING=false
python -m unittest discover -s agent -t . -p 'test_*.py' -v
python -m unittest discover -s tests -p 'test_*.py' -v
```

The lock records 69 resolved packages; the test fixtures need no model weights
or credentials and explicitly disable tracing. Tests cover API initialization,
chat, session closure/reopening, memory provenance, quarantine, questionnaire
scoring and Vertex retry/rate-limit behavior.

## Real encoder and generation

Use a separate [runtime environment](../readme.md) to run:

```sh
python scripts/check_encoder.py
```

This loads cached MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, forbids networking and checks finite
384-dimensional vectors and related-text ranking. The CPU audit passed using
SciPy 1.17.1, recorded in the runtime manifest. A working encoder does not establish
availability of a generation provider.

Legacy input preflight is `python scripts/run_eval_suite.py --validate-only`.
Two original scenarios reference an unavailable patient; this remains an explicit
gap. Valid replay runs isolate logs and memory under their output directory.
Historical experimental snapshots retain their original configurations.

Recreate the removed environment from current manifests; obsolete notebooks and
the runtime ZIP remain in Git commits `05c3a84` and `abb6cfe`, respectively.
