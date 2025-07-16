import os
import logging

from google import genai
from dotenv import load_dotenv
from google.genai.types import GenerationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from config/.env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../config/.env"))

class LLMRunner:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_id = os.getenv("model_id")  # e.g. gemini-2.5-flash

        if not self.api_key:
            raise ValueError("Missing `GOOGLE_API_KEY` in .env file.")
        if not self.model_id:
            raise ValueError("Missing `model_id` in .env file.")

        try:
            logger.info(f"Initializing LLMRunner with model: {self.model_id}")
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Gemini client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    def generate(self, prompt: str, temperature: float = 0.8, max_tokens: int = 300) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed during generation: {e}")
            return "[ERROR] Gemini API failed to generate response."
            