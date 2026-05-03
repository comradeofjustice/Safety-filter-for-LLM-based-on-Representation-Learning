#!/usr/bin/env python3
"""Calibrate per-benchmark thresholds from 70% training data and re-evaluate.

Uses the run_benchmark loaders to correctly load benchmark data,
samples up to 5000 per benchmark for fast threshold estimation,
then applies learned thresholds to held-out evaluation.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, average_precision_score)

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
                "recall_macro": 0.0, "roc_auc": 0.5, "ap": 0.5}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
    }


def find_best_threshold(scores, labels):
    best_t, best_acc = 0.5, 0
    for t in np.linspace(0.05, 0.95, 91):
        acc = accuracy_score(labels, (scores > t).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_t = t
    return best_t


def main():
    from valuation.run_benchmark import LOADERS
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    from src.deepsafe.neural_classifier import NeuralClassifier
    from src.encode.qwen_encoder import QwenEncoder

    device = "cuda"
    embed_model = "qwen3-embedding-8B"

    # Load model
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
    classifier = classifier.to(device).eval()

    encoder_map = {
        "qwen3-embedding-0.6B": "Qwen3-Embedding-0___6B",
        "qwen3-embedding-8B": "Qwen3-Embedding-8B",
    }
    embed_path = PROJ / "pretrained" / "qwen" / encoder_map[embed_model]
    encoder = QwenEncoder(str(embed_path), device=device, batch_size=64)

    # ============================================================
    # Step 1: Learn per-benchmark thresholds from 70% training data
    # ============================================================
    logger.info("=" * 60)
    logger.info("LEARNING PER-BENCHMARK THRESHOLDS (70% training data)")
    logger.info("=" * 60)

    # Map display names -> loader keys
    bench_map = {
        "WildGuardMix": "WildGuardMix",
        "XSTest": "XSTest",
        "AegisAI-v1": "AegisAI-Content-Safety-1.0",
        "AegisAI-v2": "AegisAI-Content-Safety-2.0",
        "ToxicChat": "ToxicChat",
        "BeaverTails": "BeaverTails",
        "100PoisonMpts": "100PoisonMpts",
        "DoNotAnswer": "DoNotAnswer",
        "XGuard": "XGuard-Train-Open-200K",
    }

    thresholds = {}
    MAX_CAL_SAMPLES = 5000  # Cap for speed

    for display_name, loader_key in bench_map.items():
        if loader_key not in LOADERS:
            logger.warning(f"No loader for {display_name} ({loader_key})")
            continue

        logger.info(f"\n{display_name} ({loader_key}):")
        result = LOADERS[loader_key](str(PROJ / "benchmark"), max_samples=None)
        if result is None:
            logger.warning(f"  No data loaded")
            continue

        texts = result["texts"]
        labels = result["labels"]
        n_total = len(texts)

        # 70% training split
        n_train = int(n_total * 0.7)
        np.random.seed(SEED)
        indices = np.random.permutation(n_total)
        train_texts = [texts[i] for i in indices[:n_train]]
        train_labels = np.array([labels[i] for i in indices[:n_train]])

        # Cap for speed
        if n_train > MAX_CAL_SAMPLES:
            np.random.seed(SEED)
            idx = np.random.choice(n_train, MAX_CAL_SAMPLES, replace=False)
            train_texts = [train_texts[i] for i in idx]
            train_labels = train_labels[idx]
            logger.info(f"  Sampled {MAX_CAL_SAMPLES}/{n_train} training samples")
        else:
            logger.info(f"  Using all {n_train} training samples")

        logger.info(f"  Safe: {(train_labels==0).sum()}, Unsafe: {(train_labels==1).sum()}")

        # Encode
        embeddings = encoder.encode(train_texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Project
        projected = DeepSafeProjectionHeadV3Manager.project(
            proj_head, embeddings, device=device, batch_size=1024
        )

        # Classifier scores
        with torch.no_grad():
            x = torch.FloatTensor(projected).to(device)
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()

        best_t = find_best_threshold(scores, train_labels)
        default_acc = accuracy_score(train_labels, (scores > 0.5).astype(int))
        best_acc = accuracy_score(train_labels, (scores > best_t).astype(int))
        logger.info(f"  Default (t=0.5): acc={default_acc:.4f}")
        logger.info(f"  Best (t={best_t:.4f}): acc={best_acc:.4f} (gain={best_acc-default_acc:+.4f})")

        thresholds[display_name] = float(best_t)

    # For benchmarks without training data, use global threshold
    global_thresh = 0.53  # From earlier analysis
    for display_name in ["100PoisonMpts", "DoNotAnswer", "XGuard"]:
        if display_name not in thresholds:
            if display_name in bench_map and bench_map[display_name] in LOADERS:
                continue  # Already tried
            thresholds[display_name] = global_thresh  # Fallback

    logger.info(f"\nLearned thresholds: {json.dumps(thresholds, indent=2)}")

    # ============================================================
    # Step 2: Re-evaluate held-out benchmarks
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("RE-EVALUATING WITH CALIBRATED THRESHOLDS")
    logger.info("=" * 60)

    results_path = PROJ / "valuation" / "deepsafe_v3_results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    heldout_dir = PROJ / "valuation" / "heldout"
    orig_accs = {}  # Store original accuracies for summary

    for hp in sorted(heldout_dir.glob("*_30pct.parquet")):
        bench = hp.stem.replace("_30pct", "")
        df = pd.read_parquet(hp)
        texts = df["texts"].tolist()
        labels = np.array(df["labels"].tolist())

        # Get threshold
        thresh = thresholds.get(bench, global_thresh)
        logger.info(f"\n{bench}: t={thresh:.4f}, {len(texts)} samples")

        # Store original accuracy
        if bench in all_results and "DeepSafe-v3" in all_results[bench]:
            orig_accs[bench] = all_results[bench]["DeepSafe-v3"].get("accuracy", 0)

        # Encode, project, classify
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

        # Evaluate with calibrated threshold
        preds = (scores > thresh).astype(int)
        metrics = compute_metrics(labels, preds, scores)
        default_preds = (scores > 0.5).astype(int)
        default_metrics = compute_metrics(labels, default_preds, scores)

        improvement = metrics["accuracy"] - default_metrics["accuracy"]
        logger.info(f"  Original (t=0.5):  acc={default_metrics['accuracy']:.4f}, f1={default_metrics['f1_macro']:.4f}")
        logger.info(f"  Calibrated (t={thresh:.4f}): acc={metrics['accuracy']:.4f}, f1={metrics['f1_macro']:.4f}")
        logger.info(f"  Improvement: {improvement:+.4f}")

        # Update results
        if bench in all_results:
            metrics["threshold"] = thresh
            metrics["inference_time_s"] = all_results[bench]["DeepSafe-v3"].get("inference_time_s", 0)
            metrics["samples"] = len(texts)
            all_results[bench]["DeepSafe-v3"] = metrics

    # Save
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY: DeepSafe v3 (calibrated) vs Best SOTA")
    logger.info(f"{'Benchmark':20s} {'DS Cal':>8s} {'DS Orig':>8s} {'Best SOTA':>10s} {'Winner':>12s}")
    logger.info("-" * 75)

    wins, sota_wins, ties = 0, 0, 0
    for bench in sorted(all_results.keys()):
        ds = all_results[bench].get("DeepSafe-v3", {})
        ds_acc = ds.get("accuracy", 0)
        orig = orig_accs.get(bench, ds_acc)
        others = {k: v for k, v in all_results[bench].items() if k != "DeepSafe-v3" and "error" not in v}
        if not others:
            continue
        best = max(others, key=lambda x: others[x].get("accuracy", 0))
        best_acc = others[best]["accuracy"]

        if ds_acc > best_acc + 0.001:
            winner = "DeepSafe"
            wins += 1
        elif abs(ds_acc - best_acc) <= 0.001:
            winner = "Tie"
            ties += 1
        else:
            winner = best[:15]
            sota_wins += 1

        logger.info(f"  {bench:20s} {ds_acc:8.4f} {orig:8.4f} {best_acc:10.4f} ({best:15s}) {winner:>12s}")

    logger.info(f"\nDeepSafe wins: {wins}, SOTA wins: {sota_wins}, Ties: {ties}")
    logger.info(f"(vs original: DeepSafe wins: 1, Ties: 1)")


if __name__ == "__main__":
    main()
