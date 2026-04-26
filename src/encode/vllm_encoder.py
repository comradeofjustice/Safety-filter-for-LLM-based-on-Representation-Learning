"""vLLM API-based encoder for embedding models."""

import logging
import numpy as np
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VLLMEncoder:
    """Calls a vLLM /v1/embeddings endpoint to encode texts."""

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model_name: str = "qwen3-embedding-8B",
        batch_size: int = 8,
        max_retries: int = 3,
        timeout: int = 120,
        max_chars: int = 8000,
    ):
        self.url = f"{base_url}/v1/embeddings"
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_chars = max_chars

    def _encode_single(self, text: str) -> np.ndarray:
        # Progressively halve text until it fits within the model's token limit
        for max_chars in [self.max_chars, self.max_chars // 2, self.max_chars // 4, 512]:
            truncated = text[:max_chars]
            payload = {"model": self.model_name, "input": [truncated], "encoding_format": "float"}
            try:
                resp = requests.post(self.url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)
            except Exception:
                continue
        raise RuntimeError(f"Failed to encode text even at 512 chars: {text[:100]!r}")

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        texts = [t[:self.max_chars] for t in texts]
        payload = {"model": self.model_name, "input": texts, "encoding_format": "float"}
        try:
            resp = requests.post(self.url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()["data"]
            return np.array([item["embedding"] for item in data], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Batch of {len(texts)} failed ({e}), falling back to per-sample encoding...")
            return np.stack([self._encode_single(t) for t in texts])

    def encode(self, texts: list[str]) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding via vLLM"):
            batch = texts[i : i + self.batch_size]
            emb = self._encode_batch(batch)
            all_embeddings.append(emb)
        return np.vstack(all_embeddings)
