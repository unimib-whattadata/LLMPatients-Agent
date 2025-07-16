import os
import logging
from typing import Optional

from dotenv import load_dotenv
from vllm import LLM, SamplingParams

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

# === LLM Runner ===
class LLMRunner:
    def __init__(self):
        self.model_id = os.getenv("model_id")
        self.temperature = float(os.getenv("temperature", 0.7))
        self.max_tokens = int(os.getenv("max_tokens", 2048))
        self.cache_path = os.getenv("cache_path")  # optional

        if not self.model_id:
            raise ValueError("Missing `model_id` in .env file.")

        self.llm = self._build_llm(self.model_id, self.cache_path)
        if self.llm is None:
            raise RuntimeError("Failed to initialize LLM.")

    def _build_llm(self, model_id: str, cache_path: Optional[str]) -> Optional[LLM]:
        try:
            if cache_path and os.path.isdir(cache_path):
                download_dir = cache_path
            else:
                download_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
                os.makedirs(download_dir, exist_ok=True)

            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            n_gpus = 1 if not cuda_devices else cuda_devices.count(",") + 1
            logger.info(f"Using {n_gpus} GPU(s) (CUDA_VISIBLE_DEVICES={cuda_devices})")
            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Loading model: {model_id}")

            return LLM(
                model=model_id,
                tokenizer_mode="auto",
                trust_remote_code=True,
                enable_prefix_caching=True,
                max_model_len=2048,
                download_dir=download_dir,
                tensor_parallel_size=n_gpus
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def generate(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        try:
            temp = temperature if temperature is not None else self.temperature
            max_tok = max_tokens if max_tokens is not None else self.max_tokens

            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=max_tok,
                stop=["\nTherapist:", "Therapist:",]  # Adjust as needed
            )

            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            logger.info("vLLM generation completed successfully.")
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else "[NO RESPONSE]"
        except Exception as e:
            logger.error(f"Failed during vLLM generation: {e}")
            return "[ERROR] vLLM failed to generate response."