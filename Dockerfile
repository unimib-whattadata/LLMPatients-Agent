FROM yanwk/comfyui-boot:xpu

WORKDIR /app

COPY ./agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn agent.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
