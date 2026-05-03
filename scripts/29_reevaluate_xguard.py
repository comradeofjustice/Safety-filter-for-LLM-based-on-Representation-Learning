#!/usr/bin/env python3
"""Re-evaluate XGuard with corrected labels (sec=safe) for DeepSafe v3 and SOTA.

Bug fix: Previously labels were inverted — "sec" (secure/safe) was mapped to 1 (unsafe).
Correct mapping: "sec" → 0 (safe), all other 30+ label types → 1 (unsafe).

Usage:
  python3 scripts/29_reevaluate_xguard.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent


def compute_metrics(y_true, y_pred, y_score):
    unique = np.unique(y_true)
    if len(unique) < 2:
        acc = accuracy_score(y_true, y_pred)
        return {
            "accuracy": acc, "f1_macro": 0.0, "precision_macro": 0.0,
            "recall_macro": 0.0, "roc_auc": 0.5, "ap": 0.5, "note": "single-class"
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "ap": float(average_precision_score(y_true, y_score)),
    }


def evaluate_deepsafe(proj_head, classifier, texts, labels, embed_model, device="cuda"):
    from src.encode.qwen_encoder import QwenEncoder
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    import torch.nn.functional as F

    encoder_map = {
        "qwen3-embedding-0.6B": "Qwen3-Embedding-0___6B",
        "qwen3-embedding-8B": "Qwen3-Embedding-8B",
    }
    embed_path = PROJ / "pretrained" / "qwen" / encoder_map[embed_model]
    encoder = QwenEncoder(str(embed_path), device=device, batch_size=64)
    embeddings = encoder.encode(texts)
    embeddings = np.array(embeddings, dtype=np.float32)

    projected = DeepSafeProjectionHeadV3Manager.project(
        proj_head, embeddings, device=device, batch_size=1024
    )

    classifier.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(projected).to(device)
        logits = classifier(x_t)
        probs = F.softmax(logits, dim=-1)
        scores = probs[:, 1].cpu().numpy()
        preds = (scores > 0.5).astype(int)

    labels = np.array(labels)
    return compute_metrics(labels, preds, scores)


def evaluate_sota_xguard(texts, labels, max_samples=2000):
    """Re-evaluate all SOTA models on XGuard data with corrected labels."""
    from valuation.run_benchmark import (
        Qwen3GuardEvaluator, LlamaGuard3Evaluator, ShieldGemmaEvaluator,
        WildGuardEvaluator, GraniteGuardianEvaluator, BeaverDamEvaluator,
    )

    pretrained = PROJ / "pretrained"
    evaluators = []

    if (pretrained / "Qwen3Guard-Gen-4B").exists():
        evaluators.append(Qwen3GuardEvaluator("Qwen3Guard-Gen-4B", str(pretrained / "Qwen3Guard-Gen-4B")))
    if (pretrained / "Qwen3Guard-Gen-8B").exists():
        evaluators.append(Qwen3GuardEvaluator("Qwen3Guard-Gen-8B", str(pretrained / "Qwen3Guard-Gen-8B")))
    if (pretrained / "Llama-Guard-3-8B").exists():
        evaluators.append(LlamaGuard3Evaluator("Llama-Guard-3-8B", str(pretrained / "Llama-Guard-3-8B")))
    if (pretrained / "Llama-Guard-3-1B").exists():
        evaluators.append(LlamaGuard3Evaluator("Llama-Guard-3-1B", str(pretrained / "Llama-Guard-3-1B")))
    if (pretrained / "WildGuard").exists():
        evaluators.append(WildGuardEvaluator("WildGuard", str(pretrained / "WildGuard")))
    if (pretrained / "shieldgemma-9b").exists():
        evaluators.append(ShieldGemmaEvaluator("ShieldGemma-9B", str(pretrained / "shieldgemma-9b")))
    if (pretrained / "shieldgemma-2b").exists():
        evaluators.append(ShieldGemmaEvaluator("ShieldGemma-2B", str(pretrained / "shieldgemma-2b")))
    if (pretrained / "Granite-Guardian-3.1-8B").exists():
        evaluators.append(GraniteGuardianEvaluator("Granite-Guardian-3.1-8B", str(pretrained / "Granite-Guardian-3.1-8B")))
    if (pretrained / "Beaver-Dam-7B").exists():
        evaluators.append(BeaverDamEvaluator("Beaver-Dam-7B", str(pretrained / "Beaver-Dam-7B")))

    # Sample for speed
    n_total = len(texts)
    if n_total > max_samples:
        idx = np.random.RandomState(42).choice(n_total, max_samples, replace=False)
        sample_texts = [texts[i] for i in idx]
        sample_labels = [labels[i] for i in idx]
    else:
        sample_texts = texts
        sample_labels = labels

    results = {}
    for evaluator in evaluators:
        try:
            evaluator.load()
            t0 = time.time()
            preds, scores = evaluator.predict(sample_texts, batch_size=16)
            elapsed = time.time() - t0
            evaluator.unload()

            results[evaluator.name] = compute_metrics(np.array(sample_labels), preds, scores)
            results[evaluator.name]["inference_time_s"] = round(elapsed, 1)
            results[evaluator.name]["samples"] = len(sample_texts)
            logger.info(f"  {evaluator.name}: acc={results[evaluator.name]['accuracy']:.4f}, "
                       f"auc={results[evaluator.name]['roc_auc']:.4f}, time={elapsed:.0f}s")
        except Exception as e:
            logger.error(f"  {evaluator.name} ERROR: {str(e)[:200]}")
            results[evaluator.name] = {"error": str(e)[:200]}
    return results


def main():
    # Load DeepSafe v3
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    from src.deepsafe.neural_classifier import NeuralClassifier

    model_dir = PROJ / "models" / "deepsafe_v3_8B"
    proj_head = DeepSafeProjectionHeadV3Manager.load(str(model_dir / "projection_head.pkl"), device="cuda")

    import pickle
    with open(str(model_dir / "classifier.pkl"), "rb") as f:
        clf_dict = pickle.load(f)
    classifier = NeuralClassifier(
        input_dim=clf_dict["input_dim"],
        hidden_dims=clf_dict.get("hidden_dims", [256, 128, 64]),
        dropout=clf_dict.get("dropout", 0.2),
    )
    classifier.load_state_dict(clf_dict["state_dict"])
    classifier = classifier.to("cuda")
    classifier.eval()

    # Load XGuard held-out data (already has corrected labels from the updated run_benchmark.py)
    heldout_path = PROJ / "valuation" / "heldout" / "XGuard_30pct.parquet"
    df = pd.read_parquet(heldout_path)
    texts = df["texts"].tolist()
    labels = df["labels"].tolist()

    n_total = len(texts)
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"XGuard held-out: {n_total} samples")
    for u, c in zip(unique, counts):
        logger.info(f"  {'Safe' if u == 0 else 'Unsafe'}: {c}")

    # Evaluate DeepSafe v3
    logger.info("Evaluating DeepSafe v3 on XGuard...")
    t0 = time.time()
    our_results = evaluate_deepsafe(
        proj_head, classifier, texts, labels, "qwen3-embedding-8B"
    )
    our_results["inference_time_s"] = round(time.time() - t0, 1)
    our_results["samples"] = n_total
    logger.info(f"  DeepSafe-v3: acc={our_results['accuracy']:.4f}, "
               f"f1={our_results['f1_macro']:.4f}, auc={our_results['roc_auc']:.4f}")

    # Re-evaluate SOTA
    logger.info("Re-evaluating SOTA on XGuard...")
    sota_results = evaluate_sota_xguard(texts, labels, max_samples=2000)

    # Merge into existing results
    results_path = PROJ / "valuation" / "deepsafe_v3_results.json"
    with open(results_path) as f:
        all_results = json.load(f)

    all_results["XGuard"] = {"DeepSafe-v3": our_results}
    all_results["XGuard"].update(sota_results)

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Updated results saved to {results_path}")

    # Print corrected XGuard summary
    logger.info(f"\n{'='*60}")
    logger.info("CORRECTED XGuard RESULTS")
    logger.info(f"{'Model':30s} {'Accuracy':>10s} {'F1':>10s} {'AUC':>10s}")
    logger.info("-" * 60)
    sorted_models = sorted(
        all_results["XGuard"].items(),
        key=lambda x: x[1].get("accuracy", 0),
        reverse=True,
    )
    for name, metrics in sorted_models:
        if "error" in metrics:
            logger.info(f"  {name:30s} ERROR")
        else:
            logger.info(f"  {name:30s} {metrics['accuracy']:10.4f} {metrics['f1_macro']:10.4f} {metrics['roc_auc']:10.4f}")


if __name__ == "__main__":
    main()
