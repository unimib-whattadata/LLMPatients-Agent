import os
import logging
import vertexai

from typing import Optional
from dotenv import load_dotenv
from abc import ABC, abstractmethod
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None  # Placeholder for type hints
    SamplingParams = None

from vertexai.generative_models import GenerativeModel, SafetySetting
from vertexai.generative_models import HarmCategory, HarmBlockThreshold

# === Configure Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Load Environment ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
rel_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "config/vertex-ai-api-key.json")
abs_path = os.path.join(PROJECT_ROOT, rel_path)

# Set the final environment variable for GCP auth
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_path

# === Base LLM Runner ===
class LLMRunnerBase(ABC):
    """Minimal interface all backing LLM providers must implement."""
    def __init__(self, temperature: float, max_tokens: int):
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Return a text completion for the provided prompt."""


# === Local vLLM Runner ===
class LocalLLMRunner(LLMRunnerBase):
    """Adapter that executes prompts against a local vLLM engine."""
    def __init__(self, model_id: str, cache_path: Optional[str], temperature: float, max_tokens: int):
        if not VLLM_AVAILABLE:
            raise ImportError("Checking for execution: 'vllm' module is not installed. This installation requires Python <= 3.12 (approx) and compatible 'torch' version. Please use 'vertex_ai' provider or install 'vllm' manually in a compatible environment.")
        
        super().__init__(temperature, max_tokens)

        if not model_id:
            raise ValueError("Missing `model_id` for LocalLLMRunner")

        self.model_id = model_id
        self.cache_path = cache_path
        self.llm = self._build_llm()


    def _build_llm(self) -> LLM:
        """Instantiate the vLLM object with sane defaults and logging."""
        try:
            download_dir = (
                self.cache_path if self.cache_path and os.path.isdir(self.cache_path)
                else os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            )
            os.makedirs(download_dir, exist_ok=True)

            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            n_gpus = 1 if not cuda_devices else cuda_devices.count(",") + 1
            logger.info(f"Using {n_gpus} GPU(s) (CUDA_VISIBLE_DEVICES={cuda_devices})")
            logger.info(f"Download dir: {download_dir}")
            logger.info(f"Loading model: {self.model_id}")

            return LLM(
                model=self.model_id,
                tokenizer_mode="auto",
                trust_remote_code=True,
                enable_prefix_caching=True,
                max_model_len=self.max_tokens,
                download_dir=download_dir,
                tensor_parallel_size=n_gpus
            )
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
                stop=["\nTherapist:", "Therapist:"]
            )
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
            return outputs[0].outputs[0].text.strip() if outputs and outputs[0].outputs else "[NO RESPONSE]"
        except Exception as e:
            logger.error(f"Local vLLM generation error: {e}")
            return "[ERROR] Local vLLM failed to generate response."
        
# === Vertex AI Runner ===
class VertexLLMRunner(LLMRunnerBase):
    """Adapter for Google Vertex AI's text-generation APIs with safety tuning."""
    def __init__(self, model_id: str, temperature: float, max_tokens: int):
        super().__init__(temperature, max_tokens)

        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_LOCATION", "us-central1")

        if not model_id or not project:
            raise ValueError("Missing required Vertex AI configuration.")

        vertexai.init(project=project, location=location)
        logger.info(f"Initialized Vertex AI (project={project}, location={location})")

        self.model = GenerativeModel(model_id)
        
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: HarmBlockThreshold.BLOCK_NONE,
        }

    @staticmethod
    def _extract_text_from_candidates(response) -> str:
        text_chunks = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                value = getattr(part, "text", None)
                if value:
                    text_chunks.append(value)
        return "\n".join(text_chunks).strip()

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Proxy prompt execution to Vertex AI with consistent config and error handling."""
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": temp,
                    "max_output_tokens": max_tok,
                    "stop_sequences": ["\nTherapist:", "Therapist:"],
                    "top_p": 0.95,
                    "top_k": 40,
                },
                safety_settings=self.safety_settings
            )
            try:
                return response.text.strip()
            except Exception as text_error:
                extracted = self._extract_text_from_candidates(response)
                if extracted:
                    logger.warning(
                        "Falling back to candidate-part extraction after response.text error: %s",
                        text_error,
                    )
                    return extracted
                raise text_error
        except Exception as e:
            logger.error(f"Vertex AI (Gemini) generation error: {e}")
            return "[ERROR] Vertex AI Gemini failed to generate response."
        

# === Factory Function to Create LLM Runner ===
def create_llm_runner() -> LLMRunnerBase:
    """Factory that instantiates the correct runner based on environment configuration."""
    provider = os.getenv("model_provider", "local").lower()
    model_id = os.getenv("model_id")
    temperature = float(os.getenv("temperature", 0.7))
    max_tokens = int(os.getenv("max_tokens", 512))
    cache_path = os.getenv("cache_path")

    if provider == "local":
        return LocalLLMRunner(model_id, cache_path, temperature, max_tokens)
    elif provider == "vertex_ai":
        return VertexLLMRunner(model_id, temperature, max_tokens)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
