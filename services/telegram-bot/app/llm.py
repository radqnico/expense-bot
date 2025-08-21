import json
import os
import time
from typing import Optional

import requests


class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None, timeout: Optional[float] = None):
        # Default to localhost for non-Docker runs; Docker Compose provides OLLAMA_HOST= http://ollama:11434
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "qwen2.5:0.5b"
        # Use env override for timeout; default to 60s to avoid server-side context cancellations
        env_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS") or os.getenv("OLLAMA_REQUEST_TIMEOUT")
        self.timeout = float(timeout if timeout is not None else (env_timeout or 300))

    def generate(self, prompt: str) -> str:
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
            },
            "keep_alive": "24h"
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

    def has_model(self, name: Optional[str] = None, timeout: float = 5.0) -> bool:
        """Return True if the model is present locally, using /api/show.

        This is a quick check; if Ollama doesn't support /api/show, we fallback to False.
        """
        model = name or self.model
        url = f"{self.host}/api/show"
        try:
            r = requests.post(url, json={"name": model}, timeout=timeout)
            if not r.ok:
                return False
            data = r.json() if r.headers.get("Content-Type", "").startswith("application/json") else None
            # If we get any JSON back, assume model exists
            return bool(data)  # conservative True on non-empty JSON
        except Exception:
            return False
