"""vLLM provider for fast GPU inference."""

import asyncio
import aiohttp
import time
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VLLMProvider:
    """Provider for vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = None

    async def _get_session(self):
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def generate_async(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async generation via vLLM."""
        session = await self._get_session()

        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        start = time.time()
        async with session.post(url, json=payload, timeout=self.timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"vLLM error {resp.status}: {text}")
            data = await resp.json()

        latency_ms = int((time.time() - start) * 1000)

        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return {
            "text": message.get("content", ""),
            "model": model,
            "tokens": usage.get("completion_tokens", 0),
            "latency_ms": latency_ms,
        }

    def generate(self, **kwargs) -> Dict[str, Any]:
        """Sync wrapper for async generation."""
        return asyncio.get_event_loop().run_until_complete(
            self.generate_async(**kwargs)
        )

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None


class MultiGPUVLLMProvider:
    """Manages multiple vLLM instances across GPUs."""

    def __init__(self, gpu_configs: List[Dict[str, Any]]):
        """
        gpu_configs: List of {gpu_id, port, model} dicts
        Example: [
            {"gpu_id": 0, "port": 8000, "model": "microsoft/phi-3-mini-4k-instruct"},
            {"gpu_id": 1, "port": 8001, "model": "mistralai/Mistral-7B-Instruct-v0.2"},
        ]
        """
        self.providers = {}
        for cfg in gpu_configs:
            model = cfg["model"]
            port = cfg["port"]
            self.providers[model] = VLLMProvider(f"http://localhost:{port}")

    async def generate_async(self, model: str, **kwargs) -> Dict[str, Any]:
        provider = self.providers.get(model)
        if not provider:
            raise ValueError(f"No provider for model {model}")
        return await provider.generate_async(model=model, **kwargs)

    async def close_all(self):
        for provider in self.providers.values():
            await provider.close()
