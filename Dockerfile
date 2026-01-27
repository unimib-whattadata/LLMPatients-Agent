FROM intel/vllm

WORKDIR /app

COPY ./agent/requirements.txt ./requirements.txt
# Note: vllm is excluded from requirements.txt to avoid conflict with XPU torch.
# Please install XPU-compatible vllm separately if needed, e.g.:
# RUN pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["sh", "-c", "uvicorn agent.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
