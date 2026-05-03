#!/usr/bin/env python3
"""Fast calibrated evaluation using pre-computed per-benchmark thresholds.

Loads DeepSafe v3 once, encodes all heldout benchmarks with large batch size,
applies learned thresholds, and updates the results JSON.
"""
import json
import logging
import sys
import time
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

# Pre-computed optimal thresholds from 70% training data
THRESHOLDS = {
    "WildGuardMix": 0.38,
    "XSTest": 0.43,
    "AegisAI-v1": 0.24,
    "AegisAI-v2": 0.47,
    "ToxicChat": 0.87,
    "BeaverTails": 0.85,
    "100PoisonMpts": 0.89,
    "DoNotAnswer": 0.95,
    "XGuard": 0.08,
}


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


def main():
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

    # Load existing results
    results_path = PROJ / "valuation" / "deepsafe_v3_results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    heldout_dir = PROJ / "valuation" / "heldout"
    orig_accs = {}

    logger.info("=" * 70)
    logger.info("FAST CALIBRATED EVALUATION")
    logger.info("=" * 70)

    for hp in sorted(heldout_dir.glob("*_30pct.parquet")):
        bench = hp.stem.replace("_30pct", "")
        df = pd.read_parquet(hp)
        texts = df["texts"].tolist()
        labels = np.array(df["labels"].tolist())

        thresh = THRESHOLDS.get(bench, 0.5)
        n = len(texts)

        # Store original accuracy
        if bench in all_results and "DeepSafe-v3" in all_results[bench]:
            orig_accs[bench] = all_results[bench]["DeepSafe-v3"].get("accuracy", 0)

        logger.info(f"\n{bench}: {n} samples, t={thresh:.2f} (safe={(labels==0).sum()}, unsafe={(labels==1).sum()})")

        t0 = time.time()

        # Encode
        embeddings = encoder.encode(texts)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Project
        projected = DeepSafeProjectionHeadV3Manager.project(
            proj_head, embeddings, device=device, batch_size=4096
        )

        # Classify
        with torch.no_grad():
            x = torch.FloatTensor(projected).to(device)
            logits = classifier(x)
            probs = F.softmax(logits, dim=-1)
            scores = probs[:, 1].cpu().numpy()

        # Evaluate
        cal_preds = (scores > thresh).astype(int)
        cal_metrics = compute_metrics(labels, cal_preds, scores)
        def_preds = (scores > 0.5).astype(int)
        def_metrics = compute_metrics(labels, def_preds, scores)

        elapsed = time.time() - t0
        improvement = cal_metrics["accuracy"] - def_metrics["accuracy"]

        # Determine winner
        others = {k: v for k, v in all_results[bench].items() if k != "DeepSafe-v3" and "error" not in v}
        if others:
            best_name = max(others, key=lambda x: others[x].get("accuracy", 0))
            best_acc = others[best_name]["accuracy"]
        else:
            best_name, best_acc = "N/A", 0

        if cal_metrics["accuracy"] > best_acc + 0.001:
            winner = "DeepSafe-v3"
        elif abs(cal_metrics["accuracy"] - best_acc) <= 0.001:
            winner = "TIE"
        else:
            winner = best_name[:20]

        logger.info(f"  Default:  acc={def_metrics['accuracy']:.4f} f1={def_metrics['f1_macro']:.4f} auc={def_metrics['roc_auc']:.4f}")
        logger.info(f"  Calibrated: acc={cal_metrics['accuracy']:.4f} f1={cal_metrics['f1_macro']:.4f} auc={cal_metrics['roc_auc']:.4f}  (+{improvement:+.4f})")
        logger.info(f"  Best SOTA: {best_name} acc={best_acc:.4f}  => {winner}  [{elapsed:.0f}s]")

        # Update results
        if bench in all_results:
            cal_metrics["threshold"] = thresh
            cal_metrics["inference_time_s"] = all_results[bench]["DeepSafe-v3"].get("inference_time_s", 0)
            cal_metrics["samples"] = n
            all_results[bench]["DeepSafe-v3"] = cal_metrics

    # Save
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL SUMMARY: DeepSafe v3 (calibrated) vs Best SOTA")
    logger.info(f"{'Benchmark':22s} {'DS Cal':>8s} {'DS Orig':>8s} {'Best SOTA':>10s} {'Winner':>15s}")
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
            winner, wins = "DeepSafe-v3", wins + 1
        elif abs(ds_acc - best_acc) <= 0.001:
            winner, ties = "TIE", ties + 1
        else:
            winner, sota_wins = best[:15], sota_wins + 1

        logger.info(f"  {bench:22s} {ds_acc:8.4f} {orig:8.4f} {best_acc:10.4f} ({best:15s}) {winner:>15s}")

    logger.info(f"\nDeepSafe wins: {wins}, SOTA wins: {sota_wins}, Ties: {ties}")


if __name__ == "__main__":
    main()
