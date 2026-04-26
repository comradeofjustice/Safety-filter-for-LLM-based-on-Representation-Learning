"""Qwen3 Embedding encoder with mean pooling and L2 normalization."""

import logging
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class QwenEncoder:
    """Encoder for Qwen3-Embedding models."""

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 512,
        batch_size: int = 16,
        torch_dtype: str = "bfloat16",
        device: str = "auto",
    ):
        self.max_length = max_length
        self.batch_size = batch_size

        # Auto-detect batch size for 8B model
        if "8B" in model_name_or_path and batch_size == 16:
            self.batch_size = 8
            logger.info("Detected 8B model, reducing batch_size to 8")

        logger.info(f"Loading model from {model_name_or_path} on device={device}...")
        dtype = getattr(torch, torch_dtype) if device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        logger.info(f"Model loaded on device: {self.model.device}")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings with mean pooling and L2 normalization."""
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Encoding",
            total=len(texts) // self.batch_size + 1,
        ):
            batch_texts = texts[i : i + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.model.device)

            outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

            # Mean pooling with attention mask
            mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            all_embeddings.append(pooled.cpu().float().numpy())

        return np.vstack(all_embeddings)
