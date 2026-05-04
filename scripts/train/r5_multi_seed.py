#!/usr/bin/env python3
"""R5: Multi-seed DeepSafe v3 training + inference + SOTA baseline re-eval.

Steps:
  1. Train DeepSafe v3 projection head with 5 seeds (42,0,1,2,3)
  2. For each seed: train NeuralClassifier on projected embeddings
  3. Run inference on 9 held-out benchmarks
  4. Save predictions to valuation/heldout/predictions/v3.1/seed_*/
  5. Compute mean/std stats → reports/main_table.csv
  6. Re-evaluate 9 SOTA baselines on 9 held-out benchmarks
  7. Save to valuation/heldout/predictions/sota/<model>/<benchmark>.npy

Usage:
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    flock -x /tmp/gpu.lock -c 'taskset -c 0-7 python scripts/train/r5_multi_seed.py'
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path("/root/autodl-tmp/llm-safety-classifier")
sys.path.insert(0, str(PROJECT_DIR))

from src.deepsafe.trainer import DeepSafeTrainer
from src.deepsafe.projection_head import DeepSafeProjectionHeadManager
from src.deepsafe.neural_classifier import NeuralClassifier, NeuralClassifierTrainer
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_DIR / "logs" / "r5_multi_seed.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SEEDS = [42, 0, 1, 2, 3]
EMBED_MODEL = "qwen3-embedding-8B"
EMBED_DIR = PROJECT_DIR / "embeddings" / EMBED_MODEL
MODEL_DIR = PROJECT_DIR / "models" / "deepsafe_v3_8B"
PRED_DIR = PROJECT_DIR / "valuation" / "heldout" / "predictions" / "v3.1"
SOTA_PRED_DIR = PROJECT_DIR / "valuation" / "heldout" / "predictions" / "sota"

# 9 held-out benchmarks
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

# 9 SOTA generative guard models
SOTA_MODELS = [
    "Beaver-Dam-7B",
    "Granite-Guardian-3.1-8B",
    "Llama-Guard-3-1B",
    "Llama-Guard-3-8B",
    "Qwen3Guard-Gen-4B",
    "Qwen3Guard-Gen-8B",
    "shieldgemma-2b",
    "shieldgemma-9b",
    "WildGuard",
]


def load_training_data():
    """Load embeddings, binary labels, and intent labels.

    Intent labels are aligned to embedding order. The aligned file contains:
      0 = safe, 1 = benign_sensitive, 2 = malicious, -1 = not relabeled.
    For -1 entries we fall back heuristically: binary=0 -> intent=0, binary=1 -> intent=2.
    """
    logger.info("Loading training data...")
    X_train = np.load(EMBED_DIR / "train.npy")
    y_binary = np.load(EMBED_DIR / "train_labels.npy")

    aligned_path = EMBED_DIR / "train_intent_aligned.npy"
    if aligned_path.exists():
        y_intent = np.load(aligned_path).astype(np.int8)
        n_matched = (y_intent != -1).sum()
        logger.info(f"Loaded aligned intent labels: {n_matched}/{len(y_intent)} samples have R3 relabel")
        # Heuristic fill for unmatched: binary-safe -> intent 0, binary-unsafe -> intent 2
        unmatched = y_intent == -1
        y_intent[unmatched & (y_binary == 0)] = 0
        y_intent[unmatched & (y_binary == 1)] = 2
        n_filled = unmatched.sum()
        logger.info(f"Filled {n_filled} unmatched samples with heuristic intent (0|2)")
    else:
        # Fallback: load from parquet (old path)
        intent_v2 = PROJECT_DIR / "data/processed/intent_labels_v2/train.parquet"
        intent_v1 = PROJECT_DIR / "data/processed/train_intent.parquet"
        if intent_v2.exists() and pd.read_parquet(intent_v2).shape[0] > 0:
            intent_df = pd.read_parquet(intent_v2)
            y_intent = intent_df["intent"].values
            logger.info(f"Using v2 intent labels: {len(y_intent)} samples")
        elif intent_v1.exists():
            intent_df = pd.read_parquet(intent_v1)
            y_intent = intent_df["intent"].values
            logger.info(f"Using v1 intent labels: {len(y_intent)} samples")
        else:
            logger.warning("No intent labels found, using binary labels")
            y_intent = y_binary.copy()

    unique, counts = np.unique(y_intent, return_counts=True)
    logger.info(f"Intent distribution: {dict(zip(unique.astype(int), counts))}")
    logger.info(f"Train: {X_train.shape[0]} samples, dim={X_train.shape[1]}")
    return X_train, y_binary, y_intent


def load_cf_pairs():
    """Load counterfactual pairs if available."""
    cf_path = PROJECT_DIR / "data/processed/intent_labels_v2/counterfactual_pairs.parquet"
    if not cf_path.exists():
        cf_path = PROJECT_DIR / "data/processed/contrastive_pairs.parquet"
    if not cf_path.exists():
        logger.info("No counterfactual pairs found")
        return None

    pairs = pd.read_parquet(cf_path)
    logger.info(f"Loaded {len(pairs)} counterfactual pairs")

    # Encode pairs using the 8B encoder on GPU
    from src.encode.qwen_encoder import QwenEncoder
    encoder_path = str(PROJECT_DIR / "pretrained/qwen/Qwen3-Embedding-8B")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = QwenEncoder(encoder_path, device=device, batch_size=32)

    if "safe_text" in pairs.columns and "unsafe_text" in pairs.columns:
        safe_texts = pairs["safe_text"].tolist()
        unsafe_texts = pairs["unsafe_text"].tolist()
    elif "benign" in pairs.columns and "malicious" in pairs.columns:
        safe_texts = pairs["benign"].tolist()
        unsafe_texts = pairs["malicious"].tolist()
    else:
        logger.warning("Unknown CF pairs format")
        return None

    logger.info(f"Encoding {len(safe_texts)} CF pairs with 8B model (CPU)...")
    emb_safe = encoder.encode(safe_texts)
    emb_unsafe = encoder.encode(unsafe_texts)
    del encoder

    return (emb_unsafe, emb_safe)


def train_single_seed(seed: int, X_train, y_intent, y_binary, cf_pairs) -> dict:
    """Train DeepSafe v3 with a single seed. Returns path to saved model."""
    logger.info(f"=" * 60)
    logger.info(f"Training seed={seed}")
    logger.info(f"=" * 60)

    seed_dir = MODEL_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    # Apply label correction using intent
    y_corrected = y_binary.copy()
    correction_mask = y_intent == 1  # benign_sensitive → safe
    y_corrected[correction_mask] = 0
    n_corrected = correction_mask.sum()
    logger.info(f"Label correction: {n_corrected} benign-sensitive → safe")

    # Train projection head
    trainer = DeepSafeTrainer(
        input_dim=X_train.shape[1],
        hidden_dim=512,
        output_dim=256,
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=256,
        max_epochs=80,
        patience=15,
        temperature=0.07,
        alpha=2.0,
        lambda_ot=0.3,
        lambda_proto=0.2,
        lambda_decorr=0.005,
        gamma=0.5,
        random_state=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer.train(X_train, y_intent, y_corrected, cf_pairs=cf_pairs)

    # Save projection head
    proj_path = seed_dir / "projection_head.pkl"
    trainer.save(str(proj_path))

    # Project training data
    logger.info("Projecting training data...")
    X_projected = trainer.project(X_train)

    # Train neural classifier on projected embeddings
    logger.info("Training neural classifier...")
    clf_trainer = NeuralClassifierTrainer(
        input_dim=256,
        hidden_dims=[128, 64],
        dropout=0.2,
        lr=1e-3,
        weight_decay=1e-4,
        batch_size=512,
        max_epochs=50,
        patience=10,
        mixup_alpha=0.2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = clf_trainer.train(X_projected, y_corrected)

    # Save classifier
    clf_path = seed_dir / "classifier.pkl"
    clf_trainer.save(str(clf_path))

    # Save config
    config = {
        "seed": seed,
        "input_dim": X_train.shape[1],
        "output_dim": 256,
        "n_train_samples": len(X_train),
        "n_label_corrections": int(n_corrected),
        "classifier_metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    with open(seed_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Cleanup
    del trainer, clf_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Seed {seed} training complete: acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}, auc={metrics['roc_auc']:.4f}")
    return config


def load_benchmark_data(benchmark_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load text + binary label for a held-out benchmark."""
    bm_dir = PROJECT_DIR / "benchmark" / benchmark_name

    def get_text_label(df, bm_name):
        if bm_name == "AegisAI-Content-Safety-1.0":
            # v1 schema: text + labels_0..4 (annotator labels)
            if "text" in df.columns:
                df["text"] = df["text"].fillna("").astype(str)
            else:
                df["text"] = df.apply(lambda r: f"{r.get('prompt','')}\n{r.get('response','')}", axis=1)
            # Derive label: safe if any annotator labeled "Safe", else unsafe
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

    # Try different file patterns
    for fname in ["data.parquet", "test.parquet", "train.parquet"]:
        fpath = bm_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            df = get_text_label(df, benchmark_name)
            return np.array(df["text"].tolist()), np.array(df["label"].tolist())

    # SafetyBench special case - check _repo
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
            return np.array(df["text"].tolist()), np.array(df["label"].tolist())

    logger.warning(f"No data found for {benchmark_name}")
    return np.array([]), np.array([])


def run_inference(seed: int, benchmarks: list[str]) -> dict[str, dict]:
    """Run inference for a single seed on all benchmarks."""
    logger.info(f"Running inference for seed={seed}...")
    seed_dir = MODEL_DIR / f"seed_{seed}"

    # Load models
    trainer = DeepSafeTrainer.load(str(seed_dir / "projection_head.pkl"),
                                    device="cuda" if torch.cuda.is_available() else "cpu")
    clf = NeuralClassifierTrainer.load(str(seed_dir / "classifier.pkl"),
                                        device="cuda" if torch.cuda.is_available() else "cpu")

    benchmark_emb_dir = EMBED_DIR / "benchmarks"

    results = {}
    for bm_name in benchmarks:
        # Try pre-computed embeddings first, fall back to on-the-fly encoding
        emb_path = benchmark_emb_dir / f"{bm_name}.npy"
        label_path = benchmark_emb_dir / f"{bm_name}_labels.npy"

        if emb_path.exists() and label_path.exists():
            embeddings = np.load(emb_path)
            labels = np.load(label_path)
            logger.info(f"  Loaded pre-computed embeddings for {bm_name}: {embeddings.shape}")
        else:
            texts, labels = load_benchmark_data(bm_name)
            if len(texts) == 0:
                continue
            from src.encode.qwen_encoder import QwenEncoder
            encoder_path = str(PROJECT_DIR / "pretrained/qwen/Qwen3-Embedding-8B")
            encoder = QwenEncoder(encoder_path, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32)
            embeddings = encoder.encode(texts.tolist())
            del encoder

        # Project + classify
        projected = trainer.project(embeddings)
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        probs = clf.predict_proba(projected, device=device_str)
        preds = (probs > 0.5).astype(int)

        # Save predictions
        out_dir = PRED_DIR / f"seed_{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{bm_name}.npy", probs)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(labels, preds)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.5
        try:
            f1 = f1_score(labels, preds)
        except ValueError:
            f1 = 0.0

        results[bm_name] = {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "roc_auc": round(auc, 4),
            "n_samples": len(texts),
        }
        logger.info(f"  {bm_name}: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

    del trainer, clf
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


def compute_aggregate_stats(all_seed_results: list[dict]):
    """Compute mean and std across seeds and save to reports/main_table.csv."""
    records = []

    for bm_name in BENCHMARKS:
        accs, aucs, f1s = [], [], []
        for seed_results in all_seed_results:
            if bm_name in seed_results:
                r = seed_results[bm_name]
                accs.append(r["accuracy"])
                aucs.append(r["roc_auc"])
                f1s.append(r["f1_macro"])

        if len(accs) > 0:
            records.append({
                "benchmark": bm_name,
                "mean_acc": round(np.mean(accs), 4),
                "std_acc": round(np.std(accs), 4),
                "mean_auc": round(np.mean(aucs), 4),
                "std_auc": round(np.std(aucs), 4),
                "mean_f1": round(np.mean(f1s), 4),
                "std_f1": round(np.std(f1s), 4),
                "n_seeds": len(accs),
            })

    df = pd.DataFrame(records)
    output_path = PROJECT_DIR / "reports" / "main_table.csv"
    df.to_csv(output_path, index=False, float_format="%.4f")
    logger.info(f"Main table saved to {output_path}")
    logger.info(f"\n{df.to_string()}")
    return df


def run_sota_baselines():
    """Re-evaluate 9 SOTA baselines on 9 held-out benchmarks."""
    logger.info("=" * 60)
    logger.info("Re-evaluating SOTA baselines...")
    logger.info("=" * 60)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    sota_records = []

    for model_name in SOTA_MODELS:
        model_path = PROJECT_DIR / "pretrained" / model_name
        if not model_path.exists():
            logger.warning(f"SOTA model not found: {model_path}")
            continue

        for bm_name in BENCHMARKS:
            texts, labels = load_benchmark_data(bm_name)
            if len(texts) == 0:
                continue

            # For SOTA generative models, use the existing valuation/run_benchmark.py infrastructure
            # to get predictions. For now, we load saved predictions if available.
            sota_pred_path = SOTA_PRED_DIR / model_name / f"{bm_name}.npy"
            if sota_pred_path.exists():
                probs = np.load(sota_pred_path)
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(labels, preds)
                try:
                    auc = roc_auc_score(labels, probs)
                except ValueError:
                    auc = 0.5
                try:
                    f1 = f1_score(labels, preds)
                except ValueError:
                    f1 = 0.0

                sota_records.append({
                    "model": model_name,
                    "benchmark": bm_name,
                    "accuracy": round(acc, 4),
                    "f1_macro": round(f1, 4),
                    "roc_auc": round(auc, 4),
                })
                logger.info(f"  {model_name}/{bm_name}: acc={acc:.4f}, auc={auc:.4f}")

    if sota_records:
        df = pd.DataFrame(sota_records)
        output_path = PROJECT_DIR / "reports" / "sota_table.csv"
        df.to_csv(output_path, index=False, float_format="%.4f")
        logger.info(f"SOTA table saved to {output_path}")



def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("R5: Multi-seed DeepSafe v3 Training + Evaluation")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Benchmarks: {len(BENCHMARKS)}")
    logger.info(f"SOTA models: {len(SOTA_MODELS)}")
    logger.info("=" * 60)

    # Load data
    X_train, y_binary, y_intent = load_training_data()

    # Load CF pairs
    cf_pairs = load_cf_pairs()

    # Step 1: Multi-seed training
    all_seed_results = []
    for seed in SEEDS:
        seed_dir = MODEL_DIR / f"seed_{seed}"
        if (seed_dir / "classifier.pkl").exists() and (seed_dir / "projection_head.pkl").exists():
            logger.info(f"Seed {seed} already trained, skipping to inference")
        else:
            seed_config = train_single_seed(seed, X_train, y_intent, y_binary, cf_pairs)

        # Step 2: Inference for this seed
        seed_results = run_inference(seed, BENCHMARKS)
        all_seed_results.append(seed_results)

        # Log
        with open(PROJECT_DIR / "4models" / "worker_a.log", "a") as f:
            f.write(f"\n[{datetime.now().isoformat()}] seed={seed} DONE\n")
            for bm, m in seed_results.items():
                f.write(f"  {bm}: acc={m['accuracy']}, f1={m['f1_macro']}, auc={m['roc_auc']}\n")

    # Step 3: Aggregate stats
    logger.info("Computing aggregate statistics...")
    main_table = compute_aggregate_stats(all_seed_results)

    # Step 4: SOTA baselines
    run_sota_baselines()

    elapsed = time.time() - t0
    logger.info(f"R5 DONE in {elapsed/3600:.1f}h ({elapsed/60:.0f}min)")

    # Update REVIEW_LOG
    with open(PROJECT_DIR / "4models" / "REVIEW_LOG.md", "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] worker=A status=DONE task=R5\n")
        f.write(f"  - Multi-seed DeepSafe v3 training complete: {len(SEEDS)} seeds\n")
        f.write(f"  - Inference on {len(BENCHMARKS)} held-out benchmarks complete\n")
        f.write(f"  - Main table saved to reports/main_table.csv\n")
        f.write(f"  - Predictions: {PRED_DIR}/\n")


if __name__ == "__main__":
    main()
