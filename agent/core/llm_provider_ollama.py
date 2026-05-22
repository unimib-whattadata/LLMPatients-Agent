import logging
from typing import Optional

from agent.core.llm_provider_base import LLMRunnerBase, STOP_SEQUENCES

# === Configure Logging ===
logger = logging.getLogger(__name__)


class OllamaLLMRunner(LLMRunnerBase):
    """Adapter that executes prompts against an Ollama server."""
    def __init__(self, model_id: str, base_url: str, temperature: float, max_tokens: int):
        super().__init__(temperature, max_tokens)
        if not model_id:
            raise ValueError("Missing `model_id` for OllamaLLMRunner")
        self.model_id = model_id
        
        # Clean the base URL
        self.base_url = base_url.rstrip("/")
        
        # Auto-detect if we need to append port 11434 when no port is specified and port 80 fails/404s
        if "://" in self.base_url:
            parts = self.base_url.split("://", 1)
            protocol = parts[0]
            host_port = parts[1]
        else:
            protocol = "http"
            host_port = self.base_url
            
        if ":" not in host_port:
            test_url_80 = self.base_url
            test_url_11434 = f"{protocol}://{host_port}:11434"
            import requests
            url_to_use = self.base_url
            try:
                r80 = requests.get(f"{test_url_80}/api/tags", timeout=3)
                if r80.status_code == 200:
                    url_to_use = test_url_80
                else:
                    try:
                        r11434 = requests.get(f"{test_url_11434}/api/tags", timeout=5)
                        if r11434.status_code == 200:
                            url_to_use = test_url_11434
                    except Exception:
                        url_to_use = test_url_11434
            except Exception:
                try:
                    r11434 = requests.get(f"{test_url_11434}/api/tags", timeout=5)
                    if r11434.status_code == 200:
                        url_to_use = test_url_11434
                except Exception:
                    url_to_use = test_url_11434
            if url_to_use == test_url_11434:
                logger.info(f"Ollama auto-detected port 11434. Setting base_url to {test_url_11434}")
            self.base_url = url_to_use
                
        logger.info(f"Initialized OllamaLLMRunner (model={self.model_id}, base_url={self.base_url})")

    def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate text using Ollama's HTTP API."""
        import requests
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "stop": STOP_SEQUENCES
            }
        }
        
        url = f"{self.base_url}/api/generate"
        try:
            logger.info(f"Sending request to Ollama: {url} with model {self.model_id}")
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            return f"[ERROR] Ollama failed to generate response: {e}"
