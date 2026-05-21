ARG BASE_IMAGE=intel/vllm
FROM ${BASE_IMAGE}

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

COPY ./agent/requirements-xpu.txt ./requirements.txt
COPY ./filter_reqs.py ./filter_reqs.py

# vLLM and XPU-compatible torch may already be installed in the base image.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 filter_reqs.py requirements.txt requirements.filtered.txt && \
    python3 -m pip install --no-cache-dir -r requirements.filtered.txt

COPY ./agent ./agent
COPY ./data ./data
COPY ./scripts ./scripts
COPY ./main.py ./main.py
COPY ./readme.md ./readme.md

RUN mkdir -p /app/config /app/tests/runs /root/.cache/huggingface

EXPOSE 8000

CMD ["sh", "-c", "uvicorn agent.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
