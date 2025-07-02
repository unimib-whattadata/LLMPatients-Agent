import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

class LLMRunner:
    def __init__(self):
        self.model_id = os.getenv("model_id")
        self.cache_dir = os.getenv("cache_dir", None)
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        if not self.model_id:
            raise ValueError("Missing `model_id` in .env file.")

        try:
            logger.info(f"Initializing LLMRunner with model: {self.model_id}")
            self._ensure_model_downloaded()

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

            logger.info("Transformers model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Transformers model: {e}")
            raise

    def _ensure_model_downloaded(self):
        # Forces early download for cache verification
        try:
            AutoTokenizer.from_pretrained(self.model_id, cache_dir=self.cache_dir, token=self.hf_token)
            AutoModelForCausalLM.from_pretrained(self.model_id, cache_dir=self.cache_dir, token=self.hf_token)
            logger.info(f"Model {self.model_id} verified/downloaded successfully.")
        except Exception as e:
            logger.warning(f"Model download failed or partially cached: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 300) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            logger.error(f"Failed during generation: {e}")
            return "[ERROR] Model failed to generate response."