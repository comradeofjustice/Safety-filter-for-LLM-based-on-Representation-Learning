#!/usr/bin/env python3
"""Prepare augmented training data: merge original embeddings + 70% benchmark data.

Uses existing benchmark loaders from run_benchmark.py for consistency.
Splits evaluation sets 70/30 for fair comparison.
Also loads larger benchmark training sets.
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from valuation.run_benchmark import (
    load_wildguardmix, load_100poisonmpts, load_xstest,
    load_aegis2, load_beavertails, load_toxicchat,
    load_donotanswer, load_xguard,
)
from src.encode.qwen_encoder import QwenEncoder
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent


def compute_embeddings_8b(texts, batch_size=32):
    """Compute 8B embeddings for a list of texts."""
    embed_path = PROJ / "pretrained" / "qwen" / "Qwen3-Embedding-8B"
    encoder = QwenEncoder(str(embed_path), device="cuda", batch_size=batch_size)
    return encoder.encode(texts)


def prepare_augmented_data():
    """Load original data, add 70% of eval benchmarks + external training data."""
    bench_dir = str(PROJ / "benchmark")

    # 1. Load original 8B embeddings
    logger.info("Loading original 8B embeddings...")
    train_emb = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "train.npy")
    train_labels = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "train_labels.npy")
    test_emb = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "test.npy")
    test_labels = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "test_labels.npy")

    # Load intent labels
    intent_path = PROJ / "data" / "processed" / "train_intent.parquet"
    if intent_path.exists():
        intent_df = pd.read_parquet(intent_path)
        train_intent = intent_df["intent"].values
    else:
        train_intent = train_labels.copy()

    logger.info(f"Original: train={train_emb.shape}, test={test_emb.shape}")

    # 2. Split evaluation benchmarks 70/30 and save 30% for testing
    np.random.seed(42)
    eval_heldout = {}

    for bench_name, loader in [
        ("WildGuardMix", lambda: load_wildguardmix(bench_dir)),
        ("XSTest", lambda: load_xstest(bench_dir)),
        ("100PoisonMpts", lambda: load_100poisonmpts(bench_dir)),
    ]:
        data = loader()
        if data is None:
            continue

        texts = data["texts"]
        labels = data["labels"]
        n = len(texts)
        n_train = int(n * 0.7)

        indices = np.random.permutation(n)
        tr_idx = indices[:n_train]
        te_idx = indices[n_train:]

        eval_heldout[bench_name] = {
            "texts": [texts[i] for i in te_idx],
            "labels": [labels[i] for i in te_idx],
        }
        logger.info(f"{bench_name}: {n_train}/{n} for training, {len(te_idx)} held out")

    # Save held-out for later evaluation
    heldout_dir = PROJ / "valuation" / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    for name, data in eval_heldout.items():
        pd.DataFrame(data).to_parquet(heldout_dir / f"{name}_30pct.parquet")
        logger.info(f"  Saved {name} held-out: {len(data['texts'])} samples")

    # 3. Collect ALL new texts to embed
    new_texts = []
    new_labels = []
    new_intents = []

    def add_data(texts, labels, source_name, default_intent_map=None):
        """Add texts + labels to new data lists."""
        nonlocal new_texts, new_labels, new_intents
        for i, (t, l) in enumerate(zip(texts, labels)):
            t = str(t).strip()
            if not t or len(t) < 3:
                continue
            new_texts.append(t)
            new_labels.append(int(l))
            if default_intent_map:
                new_intents.append(default_intent_map.get(int(l), int(l)))
            else:
                new_intents.append(0 if int(l) == 0 else 2)
        logger.info(f"  {source_name}: added {len(texts)} texts")

    # 3a. Add 70% of evaluation splits
    for bench_name, loader in [
        ("WildGuardMix", lambda: load_wildguardmix(bench_dir)),
        ("XSTest", lambda: load_xstest(bench_dir)),
        ("100PoisonMpts", lambda: load_100poisonmpts(bench_dir)),
    ]:
        data = loader()
        if data is None:
            continue
        n = len(data["texts"])
        n_train = int(n * 0.7)
        indices = np.random.permutation(n)[:n_train]
        add_data(
            [data["texts"][i] for i in indices],
            [data["labels"][i] for i in indices],
            f"{bench_name}_eval_70pct",
        )

    # 3b. Add large training sets from benchmarks
    # WildGuardMix training set (86k samples)
    wgm_path = PROJ / "benchmark" / "WildGuardMix" / "train" / "data.parquet"
    if wgm_path.exists():
        df = pd.read_parquet(wgm_path)
        wgm_texts = df["prompt"].tolist()
        wgm_labels = (df["prompt_harm_label"] == "harmful").astype(int).tolist()
        # Sample up to 50k to manage embedding time
        n_sample = min(50000, len(wgm_texts))
        idx = np.random.permutation(len(wgm_texts))[:n_sample]
        add_data(
            [wgm_texts[i] for i in idx],
            [wgm_labels[i] for i in idx],
            f"WildGuardMix_train_{n_sample}",
        )

    # AegisAI v2 (33k samples)
    aegis2_path = PROJ / "benchmark" / "AegisAI-Content-Safety-2.0" / "data.parquet"
    if aegis2_path.exists():
        df = pd.read_parquet(aegis2_path)
        a2_texts = df["prompt"].fillna("").tolist()
        a2_labels = (df["prompt_label"] == "unsafe").astype(int).tolist()
        n_sample = min(20000, len(a2_texts))
        idx = np.random.permutation(len(a2_texts))[:n_sample]
        valid_idx = [i for i in idx if len(str(a2_texts[i]).strip()) >= 3]
        add_data(
            [a2_texts[i] for i in valid_idx],
            [a2_labels[i] for i in valid_idx],
            f"AegisAI_v2_{len(valid_idx)}",
        )

    # 4. Compute embeddings
    logger.info(f"\nTotal new texts to embed: {len(new_texts)}")
    logger.info(f"  Labels: safe={sum(1 for l in new_labels if l==0)}, unsafe={sum(1 for l in new_labels if l==1)}")

    new_embeddings = compute_embeddings_8b(new_texts, batch_size=32)
    new_embeddings = np.array(new_embeddings, dtype=np.float32)
    new_labels_arr = np.array(new_labels, dtype=np.int64)
    new_intents_arr = np.array(new_intents, dtype=np.int64)

    # 5. Merge
    merged_emb = np.vstack([train_emb, new_embeddings])
    merged_labels = np.concatenate([train_labels, new_labels_arr])
    merged_intent = np.concatenate([train_intent, new_intents_arr])

    logger.info(f"\nFinal training set: {merged_emb.shape[0]} samples, dim={merged_emb.shape[1]}")
    logger.info(f"  Safe: {(merged_labels==0).sum()}, Unsafe: {(merged_labels==1).sum()}")
    logger.info(f"  Intent: safe={(merged_intent==0).sum()}, benign={(merged_intent==1).sum()}, malicious={(merged_intent==2).sum()}")

    # 6. Save
    aug_dir = PROJ / "embeddings" / "deepsafe_v3_augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)
    np.save(aug_dir / "train.npy", merged_emb)
    np.save(aug_dir / "train_labels.npy", merged_labels)
    np.save(aug_dir / "train_intent.npy", merged_intent)
    np.save(aug_dir / "test.npy", test_emb)
    np.save(aug_dir / "test_labels.npy", test_labels)
    np.save(aug_dir / "new_bench_labels.npy", new_labels_arr)

    logger.info(f"Saved augmented data to {aug_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    set_seed(42)
    prepare_augmented_data()
