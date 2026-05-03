#!/usr/bin/env python3
"""Calibrate decision thresholds for DeepSafe v3 using training data.

Approach:
1. Run classifier on 388K projected training embeddings
2. Find optimal global threshold maximizing accuracy
3. Per-benchmark: load 70% training data, project, find optimal threshold
4. Re-evaluate held-out benchmarks with calibrated thresholds

Usage:
  python3 scripts/30_threshold_calibration.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
SEED = 42


def compute_metrics(y_true, y_pred, y_score):
    unique = np.unique(y_true)
    if len(unique) < 2:
        acc = accuracy_score(y_true, y_pred)
        return {"accuracy": acc, "f1_macro": 0.0, "precision_macro": 0.0,
                "recall_macro": 0.0, "roc_auc": 0.5, "ap": 0.5, "note": "single-class"}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
    }


def find_optimal_threshold(scores, labels, metric="accuracy"):
    best_thresh = 0.5
    best_val = -1
    for t in np.linspace(0.05, 0.95, 91):
        preds = (scores > t).astype(int)
        val = accuracy_score(labels, preds) if metric == "accuracy" else f1_score(labels, preds, average="macro", zero_division=0)
        if val > best_val:
            best_val = val
            best_thresh = t
    return best_thresh, best_val


def load_benchmark_70pct(bench_key):
    """Load 70% training split of a benchmark using run_benchmark loaders."""
    from valuation.run_benchmark import LOADERS

    if bench_key not in LOADERS:
        logger.warning(f"  No loader for {bench_key}")
        return None, None

    result = LOADERS[bench_key](str(PROJ / "benchmark"), max_samples=None)
    if result is None:
        logger.warning(f"  Loader returned None for {bench_key}")
        return None, None

    texts = result["texts"]
    labels = result["labels"]
    n_total = len(texts)
    n_train = int(n_total * 0.7)

    np.random.seed(SEED)
    indices = np.random.permutation(n_total)
    train_texts = [texts[i] for i in indices[:n_train]]
    train_labels = [labels[i] for i in indices[:n_train]]

    logger.info(f"  {bench_key}: {n_train}/{n_total} training samples, "
                f"safe={sum(1 for l in train_labels if l == 0)}, "
                f"unsafe={sum(1 for l in train_labels if l == 1)}")
    return train_texts, train_labels


def main():
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    from src.deepsafe.neural_classifier import NeuralClassifier
    from src.encode.qwen_encoder import QwenEncoder

    device = "cuda"
    embed_model = "qwen3-embedding-8B"

    # Load DeepSafe v3
    model_dir = PROJ / "models" / "deepsafe_v3_8B"
    logger.info("Loading DeepSafe v3...")
    proj_head = DeepSafeProjectionHeadV3Manager.load(str(model_dir / "projection_head.pkl"), device=device)

    import pickle
    with open(str(model_dir / "classifier.pkl"), "rb") as f:
        clf_dict = pickle.load(f)
    classifier = NeuralClassifier(
        input_dim=clf_dict["input_dim"],
        hidden_dims=clf_dict.get("hidden_dims", [256, 128, 64]),
        dropout=clf_dict.get("dropout", 0.2),
    )
    classifier.load_state_dict(clf_dict["state_dict"])
    classifier = classifier.to(device)
    classifier.eval()

    # Load encoder
    encoder_map = {
        "qwen3-embedding-0.6B": "Qwen3-Embedding-0___6B",
        "qwen3-embedding-8B": "Qwen3-Embedding-8B",
    }
    embed_path = PROJ / "pretrained" / "qwen" / encoder_map[embed_model]
    encoder = QwenEncoder(str(embed_path), device=device, batch_size=64)

    # ============================================================
    # Step 1: Global threshold from 388K projected training data
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("GLOBAL THRESHOLD CALIBRATION")
    logger.info("=" * 60)

    train_proj = np.load(model_dir / "train_projected.npy")
    train_labels = np.load(PROJ / "embeddings" / "deepsafe_v3_augmented" / "train_labels.npy")
    logger.info(f"Training data: {train_proj.shape}, safe={(train_labels==0).sum()}, unsafe={(train_labels==1).sum()}")

    # Get classifier scores
    classifier.eval()
    batch_size = 4096
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(train_proj), batch_size):
            x = torch.FloatTensor(train_proj[i:i+batch_size]).to(device)
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            all_scores.append(probs[:, 1].cpu().numpy())
    train_scores = np.concatenate(all_scores)

    global_thresh, global_acc = find_optimal_threshold(train_scores, train_labels, "accuracy")
    logger.info(f"Optimal global threshold: {global_thresh:.4f} (train acc: {global_acc:.4f})")

    # Threshold sweep
    logger.info("Threshold sweep on training data:")
    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        preds = (train_scores > t).astype(int)
        acc = accuracy_score(train_labels, preds)
        f1 = f1_score(train_labels, preds, average="macro", zero_division=0)
        logger.info(f"  t={t:.2f}: acc={acc:.4f}, f1={f1:.4f}")

    # ============================================================
    # Step 2: Re-evaluate held-out benchmarks with global threshold
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("RE-EVALUATING HELD-OUT BENCHMARKS WITH GLOBAL THRESHOLD")
    logger.info("=" * 60)

    # Map display names to bench keys (for reference)
    name_mapping = {
        "WildGuardMix": "WildGuardMix",
        "XSTest": "XSTest",
        "AegisAI-v1": "AegisAI-Content-Safety-1.0",
        "AegisAI-v2": "AegisAI-Content-Safety-2.0",
        "ToxicChat": "ToxicChat",
        "BeaverTails": "BeaverTails",
    }

    bench_thresholds = {}
    threshold = global_thresh

    # Load existing results
    results_path = PROJ / "valuation" / "deepsafe_v3_results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    heldout_dir = PROJ / "valuation" / "heldout"

    for heldout_path in sorted(heldout_dir.glob("*_30pct.parquet")):
        bench_display_name = heldout_path.stem.replace("_30pct", "")

        df = pd.read_parquet(heldout_path)
        texts = df["texts"].tolist()
        labels = np.array(df["labels"].tolist())

        # Use global threshold for all benchmarks
        thresh_source = "global"

        logger.info(f"\n{bench_display_name}: threshold={threshold:.4f} ({thresh_source})")
        logger.info(f"  Data: {len(texts)} samples, safe={(labels==0).sum()}, unsafe={(labels==1).sum()}")

        # Encode & project
        embeddings = encoder.encode(texts)
        embeddings = np.array(embeddings, dtype=np.float32)
        projected = DeepSafeProjectionHeadV3Manager.project(
            proj_head, embeddings, device=device, batch_size=1024
        )

        with torch.no_grad():
            x = torch.FloatTensor(projected).to(device)
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()

        preds_calibrated = (scores > threshold).astype(int)
        preds_default = (scores > 0.5).astype(int)

        metrics_cal = compute_metrics(labels, preds_calibrated, scores)
        metrics_def = compute_metrics(labels, preds_default, scores)

        improvement = metrics_cal["accuracy"] - metrics_def["accuracy"]
        logger.info(f"  Default  (t=0.5000): acc={metrics_def['accuracy']:.4f}, f1={metrics_def['f1_macro']:.4f}")
        logger.info(f"  Calibrated (t={threshold:.4f}): acc={metrics_cal['accuracy']:.4f}, f1={metrics_cal['f1_macro']:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f}")

        # Update results
        if bench_display_name in all_results:
            metrics_cal["threshold"] = threshold
            metrics_cal["threshold_source"] = thresh_source
            metrics_cal["inference_time_s"] = all_results[bench_display_name]["DeepSafe-v3"].get("inference_time_s", 0)
            metrics_cal["samples"] = len(texts)
            all_results[bench_display_name]["DeepSafe-v3"] = metrics_cal

    # Save updated results
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nUpdated results saved to {results_path}")

    # Print updated summary
    logger.info(f"\n{'='*60}")
    logger.info("UPDATED SUMMARY: DeepSafe v3 vs Best SOTA (calibrated thresholds)")
    logger.info(f"{'Benchmark':20s} {'DS Cal':>8s} {'DS Orig':>8s} {'Best SOTA':>10s} {'Winner':>12s}")
    logger.info("-" * 75)

    deepsafe_wins = 0
    sota_wins = 0
    ties = 0

    for bench_name in sorted(all_results.keys()):
        bench_results = all_results[bench_name]
        ds = bench_results.get("DeepSafe-v3", {})
        ds_acc = ds.get("accuracy", 0)

        others = {k: v for k, v in bench_results.items() if k != "DeepSafe-v3" and "error" not in v}
        if not others:
            continue

        best = max(others, key=lambda x: others[x].get("accuracy", 0))
        best_acc = others[best]["accuracy"]

        if ds_acc > best_acc + 0.001:
            winner = "DeepSafe"
            deepsafe_wins += 1
        elif abs(ds_acc - best_acc) <= 0.001:
            winner = "Tie"
            ties += 1
        else:
            winner = best[:15]
            sota_wins += 1

        logger.info(f"  {bench_name:20s} {ds_acc:8.4f} {'—':8s} {best_acc:10.4f} ({best:15s}) {winner:>12s}")

    logger.info(f"\nDeepSafe wins: {deepsafe_wins}, SOTA wins: {sota_wins}, Ties: {ties}")

    # Save threshold config
    threshold_config = {
        "global_threshold": float(global_thresh),
        "global_train_accuracy": float(global_acc),
        "per_benchmark_thresholds": bench_thresholds,
    }
    with open(model_dir / "threshold_config.json", "w") as f:
        json.dump(threshold_config, f, indent=2)
    logger.info(f"Threshold config saved.")


if __name__ == "__main__":
    main()
