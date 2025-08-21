import json
import os
import time
from typing import Optional

import requests


class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: float = 10.0):
        # Default to localhost for non-Docker runs; Docker Compose provides OLLAMA_HOST= http://ollama:11434
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:0.5b"
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 1024,
            },
        }
        resp = requests.post(
            url,
            data=json.dumps(payload),
            timeout=self.timeout,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

    def pull_model(self, name: Optional[str] = None, retries: int = 12, delay: float = 5.0, timeout: float = 600.0) -> None:
        """Ensure model is available by calling Ollama pull. Retries while Ollama starts.

        - retries x delay ~= max wait before Ollama is ready
        - timeout: per request timeout for long pulls
        """
        model = name or self.model
        url = f"{self.host}/api/pull"
        payload = {"name": model}

        last_err: Optional[Exception] = None
        for _ in range(max(1, retries)):
            try:
                with requests.post(url, json=payload, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    # Consume stream to completion; ignore content
                    for _line in r.iter_lines():
                        pass
                return
            except Exception as e:
                last_err = e
                time.sleep(delay)
        # Best effort: if still failing, raise last error
        if last_err:
            raise last_err
