FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn agent.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
