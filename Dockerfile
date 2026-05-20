FROM intel/vllm

WORKDIR /app

COPY ./agent/requirements.txt ./requirements.txt
COPY ./filter_reqs.py ./filter_reqs.py

# vLLM and XPU-compatible torch may already be installed in the base image.
RUN python3 filter_reqs.py requirements.txt requirements.filtered.txt && \
    pip install --no-cache-dir -r requirements.filtered.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

ENTRYPOINT ["sh", "-c", "uvicorn agent.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
