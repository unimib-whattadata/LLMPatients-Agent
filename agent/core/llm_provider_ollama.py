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
        """Generate text using Ollama's HTTP API (chat mode)."""
        import requests
        import re
        
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        # Format the system + user message framing for a natural conversation history to prevent CoT/planning output
        system_lines = []
        therapist_input = None
        for line in prompt.splitlines():
            if line.strip().startswith("• Therapist's latest message:"):
                # Extract whatever is inside the quotes or after the colon
                content_part = line.strip().split(":", 1)[1].strip()
                # Strip leading/trailing quotes if present
                if content_part.startswith('"') and content_part.endswith('"') and len(content_part) >= 2:
                    therapist_input = content_part[1:-1]
                else:
                    therapist_input = content_part
            else:
                system_lines.append(line)
        
        if therapist_input is not None:
            system_prompt = "\n".join(system_lines).strip()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": therapist_input
                }
            ]
            logger.info(f"Ollama runner: Structured patient roleplay conversation with therapist input: '{therapist_input}'")
        else:
            messages = [
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": "Please generate the output according to the instructions above."
                }
            ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
                "stop": STOP_SEQUENCES
            }
        }
        
        url = f"{self.base_url}/api/chat"
        try:
            logger.info(f"Sending request to Ollama chat: {url} with model {self.model_id}")
            response = requests.post(url, json=payload, timeout=90)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "").strip()
            
            # Robust parsing: remove any reasoning/thinking tags (e.g. <think>...</think>)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"<thought>.*?</thought>", "", content, flags=re.DOTALL).strip()
            
            # Clean up common conversational prefixes that some models generate
            content = re.sub(r"^(Patient|Alex|Alex Carter|🧍 Patient|🧍 Alex):\s*", "", content, flags=re.IGNORECASE).strip()
            
            return content
        except Exception as e:
            logger.error(f"Ollama chat generation error: {e}")
            return f"[ERROR] Ollama failed to generate response: {e}"
