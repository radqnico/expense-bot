import json
import os
from typing import Optional

import requests


class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: float = 10.0):
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://ollama:11434").rstrip("/")
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
        resp = requests.post(url, data=json.dumps(payload), timeout=self.timeout, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()

