import logging
import os
from typing import Optional

from agent.core.llm_provider_base import LLMRunnerBase, STOP_SEQUENCES

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # Placeholder for type hints
    SamplingParams = None

try:
    import torch
except ImportError:
    torch = None

# === Configure Logging ===
logger = logging.getLogger(__name__)

DEFAULT_HF_CACHE_DIR = "~/.cache/huggingface"


def _preferred_local_device() -> tuple[int, str]:
    """Return tensor parallel count and vLLM device for the available accelerator."""
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        n_devices = torch.xpu.device_count()
        logger.info("Intel XPU detected. Using %s XPU(s).", n_devices)
        return n_devices, "xpu"

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    n_devices = 1 if not cuda_devices else cuda_devices.count(",") + 1
    logger.info("Using %s GPU(s) (CUDA_VISIBLE_DEVICES=%s)", n_devices, cuda_devices)
    return n_devices, "auto"


class LocalLLMRunner(LLMRunnerBase):
    """Adapter that executes prompts against a local vLLM engine."""
    def __init__(
        self,
        model_id: str,
        cache_path: Optional[str],
        temperature: float,
        max_tokens: int,
        max_model_len: Optional[int] = None,
    ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "Local provider requires the `vllm` package plus compatible `torch`. "
                "Install a matching local stack, use the Docker image, or set model_provider=vertex_ai."
            )
        
        super().__init__(temperature, max_tokens)

        if not model_id:
            raise ValueError("Missing `model_id` for LocalLLMRunner")

        self.model_id = model_id
        self.cache_path = cache_path
        # Keep generation length separate from the model context window.
        inferred_context_len = max(self.max_tokens * 4, 4096)
        self.max_model_len = max_model_len if max_model_len is not None else inferred_context_len
        if self.max_model_len < self.max_tokens:
            logger.warning(
                "max_model_len (%s) is lower than max_tokens (%s); raising it to max_tokens.",
                self.max_model_len,
                self.max_tokens,
            )
            self.max_model_len = self.max_tokens
        self.llm = self._build_llm()

    def _build_llm(self) -> LLM:
        """Instantiate the vLLM object with sane defaults and logging."""
        try:
            download_dir = (
                self.cache_path if self.cache_path and os.path.isdir(self.cache_path)
                else os.getenv("HF_HOME", os.path.expanduser(DEFAULT_HF_CACHE_DIR))
            )
            os.makedirs(download_dir, exist_ok=True)

            n_gpus, device = _preferred_local_device()
            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Loading model: {self.model_id}")

            # Allow configuring model precision/dtype (e.g. 'half' or 'float16' for Intel Arc A770)
            model_dtype = os.getenv("model_dtype", "auto").strip().lower()

            try:
                return LLM(
                    model=self.model_id,
                    tokenizer_mode="auto",
                    trust_remote_code=True,
                    enable_prefix_caching=True,
                    max_model_len=self.max_model_len,
                    download_dir=download_dir,
                    tensor_parallel_size=n_gpus,
                    device=device,
                    dtype=model_dtype,
                )
            except TypeError as te:
                if "device" in str(te):
                    logger.warning("vLLM does not accept 'device' argument. Retrying without it.")
                    return LLM(
                        model=self.model_id,
                        tokenizer_mode="auto",
                        trust_remote_code=True,
                        enable_prefix_caching=True,
                        max_model_len=self.max_model_len,
                        download_dir=download_dir,
                        tensor_parallel_size=n_gpus,
                        dtype=model_dtype,
                    )
                else:
                    raise
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate text locally, trimming to stop tokens and handling transient failures."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                stop=STOP_SEQUENCES,
            )
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else "[NO RESPONSE]"
        except Exception as e:
            logger.error(f"Local vLLM generation error: {e}")
            return "[ERROR] Local vLLM failed to generate response."
