#!/usr/bin/env python3
"""Pre-encode all benchmark texts with Qwen3-Embedding-8B and save to disk.
This avoids re-encoding the same texts for each seed, speeding up R5 inference.
"""
import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path("/root/autodl-tmp/llm-safety-classifier")
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

EMBED_MODEL = "qwen3-embedding-8B"
EMBED_DIR = PROJECT_DIR / "embeddings" / EMBED_MODEL / "benchmarks"
EMBED_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = [
    "AegisAI-Content-Safety-1.0",
    "AegisAI-Content-Safety-2.0",
    "BeaverTails",
    "CRiskEval",
    "DoNotAnswer",
    "MM-SafetyBench",
    "SafetyBench",
    "ToxicChat",
    "XSTest",
]


def load_benchmark_data(benchmark_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load text + binary label for a held-out benchmark."""
    bm_dir = PROJECT_DIR / "benchmark" / benchmark_name

    def get_text_label(df, bm_name):
        if bm_name == "AegisAI-Content-Safety-1.0":
            if "text" in df.columns:
                df["text"] = df["text"].fillna("").astype(str)
            else:
                df["text"] = df.apply(lambda r: f"{r.get('prompt','')}\n{r.get('response','')}", axis=1)
            def aegis_v1_label(row):
                for k in ["labels_0", "labels_1", "labels_2", "labels_3", "labels_4"]:
                    if k in row and str(row[k]).lower() == "safe":
                        return 0
                return 1
            df["label"] = df.apply(aegis_v1_label, axis=1)
        elif bm_name == "AegisAI-Content-Safety-2.0":
            df["text"] = df.apply(lambda r: f"{r.get('prompt','')}\n{r.get('response','')}", axis=1)
            df["label"] = df.get("prompt_label", df.get("label", 0)).apply(
                lambda x: 0 if str(x).lower() in ("safe", "0") else 1
            )
        elif bm_name == "BeaverTails":
            df["text"] = df.apply(lambda r: f"{r.get('prompt','')}\n{r.get('response','')}", axis=1)
            df["label"] = df["is_safe"].apply(lambda x: 0 if x else 1)
        elif bm_name == "CRiskEval":
            df["text"] = df["question"].fillna("")
            df["label"] = 1
        elif bm_name == "DoNotAnswer":
            df["text"] = df["question"].fillna("")
            df["label"] = 1
        elif bm_name == "MM-SafetyBench":
            df["text"] = df["question"].fillna("")
            df["label"] = 1
        elif bm_name == "SafetyBench":
            df["text"] = df["question"].fillna("")
            df["label"] = 1
        elif bm_name == "ToxicChat":
            df["text"] = df["user_input"].fillna("")
            df["label"] = df["toxicity"].apply(lambda x: 0 if x == 0 else 1)
        elif bm_name == "XSTest":
            df["text"] = df["prompt"].fillna("")
            df["label"] = df["label"].apply(lambda x: 0 if str(x).lower() == "safe" else 1)
        return df[["text", "label"]]

    for fname in ["data.parquet", "test.parquet", "train.parquet"]:
        fpath = bm_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df = get_text_label(df, benchmark_name)
            texts = df["text"].tolist()
            labels = np.array(df["label"].tolist())
            return np.array(texts), labels, texts

    # SafetyBench special case
    import json
    sb_repo = bm_dir / "_repo" / "data"
    if sb_repo.exists():
        records = []
        for f in sorted(sb_repo.glob("*.json")):
            try:
                with open(f) as fh:
                    items = json.load(fh)
                    if isinstance(items, list):
                        for item in items:
                            text = str(item.get("question", item)) if isinstance(item, dict) else str(item)
                            records.append({"text": text, "label": 1})
            except Exception:
                pass
        if records:
            df = pd.DataFrame(records)
            texts = df["text"].tolist()
            labels = np.array(df["label"].tolist())
            return np.array(texts), labels, texts

    logger.warning(f"No data found for {benchmark_name}")
    return np.array([]), np.array([]), []


def main():
    import torch
    from src.encode.qwen_encoder import QwenEncoder

    encoder_path = str(PROJECT_DIR / "pretrained/qwen/Qwen3-Embedding-8B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = QwenEncoder(encoder_path, device=device, batch_size=32)

    t0 = time.time()
    for bm_name in BENCHMARKS:
        texts_arr, labels, texts = load_benchmark_data(bm_name)
        if len(texts) == 0:
            logger.warning(f"Skipping {bm_name} (no data)")
            continue

        # Check if already encoded
        emb_path = EMBED_DIR / f"{bm_name}.npy"
        label_path = EMBED_DIR / f"{bm_name}_labels.npy"
        if emb_path.exists() and label_path.exists():
            logger.info(f"{bm_name}: already encoded ({len(texts)} samples), skipping")
            continue

        logger.info(f"Encoding {bm_name}: {len(texts)} samples...")
        embeddings = encoder.encode(texts)
        np.save(emb_path, embeddings)
        np.save(label_path, labels)
        logger.info(f"  Saved {emb_path} ({embeddings.shape})")

    elapsed = time.time() - t0
    logger.info(f"Pre-encoding complete in {elapsed/60:.0f}min")


if __name__ == "__main__":
    main()
