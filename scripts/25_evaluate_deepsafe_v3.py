#!/usr/bin/env python3
"""Evaluate DeepSafe v3 on 30% held-out benchmark splits + re-evaluate SOTA for fair comparison.

This script:
  1. Loads DeepSafe v3 projection head + neural classifier
  2. Evaluates on held-out 30% of WildGuardMix, 100PoisonMpts, XSTest
  3. Re-evaluates SOTA models on the same held-out data
  4. Computes comprehensive metrics
  5. Saves results for paper
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent


def load_deepsafe_v3(model_dir: str):
    """Load DeepSafe v3 projection head + neural classifier."""
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    from src.deepsafe.neural_classifier import NeuralClassifier

    model_dir = Path(model_dir)

    # Load projection head
    proj_path = model_dir / "projection_head.pkl"
    proj_head = DeepSafeProjectionHeadV3Manager.load(str(proj_path), device="cuda")
    logger.info(f"Loaded projection head: {proj_head.input_dim}→{proj_head.output_dim}")

    # Load neural classifier
    clf_path = model_dir / "classifier.pkl"
    import pickle
    with open(str(clf_path), "rb") as f:
        clf_dict = pickle.load(f)
    classifier = NeuralClassifier(
        input_dim=clf_dict["input_dim"],
        hidden_dims=clf_dict.get("hidden_dims", [256, 128, 64]),
        dropout=clf_dict.get("dropout", 0.2),
    )
    classifier.load_state_dict(clf_dict["state_dict"])
    classifier = classifier.to("cuda")
    classifier.eval()
    logger.info(f"Loaded classifier: hidden_dims={clf_dict.get('hidden_dims')}")

    return proj_head, classifier


def evaluate_model(proj_head, classifier, texts, labels, embed_model, device="cuda"):
    """Evaluate DeepSafe v3 on given texts."""
    from src.encode.qwen_encoder import QwenEncoder

    # Encode
    encoder_map = {
        "qwen3-embedding-0.6B": "Qwen3-Embedding-0___6B",
        "qwen3-embedding-8B": "Qwen3-Embedding-8B",
    }
    embed_path = PROJ / "pretrained" / "qwen" / encoder_map[embed_model]
    encoder = QwenEncoder(str(embed_path), device=device, batch_size=64)

    embeddings = encoder.encode(texts)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Project
    from src.deepsafe.projection_head_v3 import DeepSafeProjectionHeadV3Manager
    projected = DeepSafeProjectionHeadV3Manager.project(
        proj_head, embeddings, device=device, batch_size=1024
    )

    # Classify
    import torch.nn.functional as F
    classifier.eval()
    with torch.no_grad():
        x_t = torch.FloatTensor(projected).to(device)
        logits = classifier(x_t)
        probs = F.softmax(logits, dim=-1)
        scores = probs[:, 1].cpu().numpy()
        preds = (scores > 0.5).astype(int)

    labels = np.array(labels)
    results = compute_metrics(labels, preds, scores)
    return results


def compute_metrics(y_true, y_pred, y_score):
    """Compute all classification metrics."""
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


def evaluate_sota_on_heldout(bench_name: str, texts: list, labels: list):
    """Re-evaluate SOTA models on held-out data using existing evaluators."""
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

    results = {}
    for evaluator in evaluators:
        try:
            evaluator.load()
            t0 = time.time()
            preds, scores = evaluator.predict(texts, batch_size=16)
            elapsed = time.time() - t0
            evaluator.unload()

            results[evaluator.name] = compute_metrics(np.array(labels), preds, scores)
            results[evaluator.name]["inference_time_s"] = round(elapsed, 1)
            results[evaluator.name]["samples"] = len(texts)
            logger.info(f"  {evaluator.name}: acc={results[evaluator.name]['accuracy']:.4f}, "
                       f"auc={results[evaluator.name]['roc_auc']:.4f}, time={elapsed:.0f}s")
        except Exception as e:
            logger.error(f"  {evaluator.name} ERROR: {str(e)[:200]}")
            results[evaluator.name] = {"error": str(e)[:200]}
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/deepsafe_v3_8B")
    parser.add_argument("--embed-model", default="qwen3-embedding-8B")
    parser.add_argument("--skip-sota", action="store_true")
    parser.add_argument("--output", default="valuation/deepsafe_v3_results.json")
    parser.add_argument("--max-sota-samples", type=int, default=2000,
                        help="Max samples for SOTA evaluation (speed)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("DeepSafe v3 EVALUATION")
    logger.info(f"  Model: {args.model_dir}")
    logger.info(f"  Embed: {args.embed_model}")
    logger.info("=" * 60)

    # Load model
    proj_head, classifier = load_deepsafe_v3(args.model_dir)

    all_results = {}
    heldout_dir = PROJ / "valuation" / "heldout"

    # Scan ALL held-out files
    heldout_files = sorted(heldout_dir.glob("*_30pct.parquet"))
    if not heldout_files:
        logger.error("No held-out files found!")
        return

    logger.info(f"Found {len(heldout_files)} held-out benchmark files")

    for heldout_path in heldout_files:
        bench_name = heldout_path.stem.replace("_30pct", "")

        df = pd.read_parquet(heldout_path)
        texts = df["texts"].tolist()
        labels = df["labels"].tolist()

        # Sample for SOTA evaluation if too large
        n_total = len(texts)
        sota_texts = texts
        sota_labels = labels
        if n_total > args.max_sota_samples and not args.skip_sota:
            idx = np.random.RandomState(42).choice(n_total, args.max_sota_samples, replace=False)
            sota_texts = [texts[i] for i in idx]
            sota_labels = [labels[i] for i in idx]

        logger.info(f"\n{'='*60}")
        logger.info(f"{bench_name}: {n_total} held-out samples")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            logger.info(f"  {'Safe' if u==0 else 'Unsafe'}: {c}")

        bench_results = {}

        # Our model (evaluate on FULL held-out)
        logger.info(f"  Evaluating DeepSafe v3 on {n_total} samples...")
        t0 = time.time()
        our_results = evaluate_model(proj_head, classifier, texts, labels, args.embed_model)
        our_results["inference_time_s"] = round(time.time() - t0, 1)
        our_results["samples"] = n_total
        bench_results["DeepSafe-v3"] = our_results
        logger.info(f"  DeepSafe-v3: acc={our_results['accuracy']:.4f}, "
                   f"f1={our_results.get('f1_macro', 0):.4f}, "
                   f"auc={our_results.get('roc_auc', 0):.4f}")

        # SOTA re-evaluation (on sampled subset for speed)
        if not args.skip_sota:
            logger.info(f"  Re-evaluating SOTA on {len(sota_texts)} samples...")
            sota_results = evaluate_sota_on_heldout(bench_name, sota_texts, sota_labels)
            for sota_name, sota_metrics in sota_results.items():
                sota_metrics["samples"] = len(sota_texts)
            bench_results.update(sota_results)

        all_results[bench_name] = bench_results

        all_results[bench_name] = bench_results

    # Save
    output_path = PROJ / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    for bench_name, bench_results in all_results.items():
        logger.info(f"\n--- {bench_name} ---")
        sorted_models = sorted(
            bench_results.items(),
            key=lambda x: x[1].get("accuracy", 0),
            reverse=True,
        )
        for model_name, metrics in sorted_models:
            if "error" in metrics:
                logger.info(f"  {model_name:30s} ERROR: {metrics['error'][:80]}")
            else:
                logger.info(f"  {model_name:30s} acc={metrics['accuracy']:.4f}  "
                          f"f1={metrics.get('f1_macro', 0):.4f}  auc={metrics.get('roc_auc', 0):.4f}")

    # Print SOTA comparison summary (DeepSafe-v3 vs best SOTA per benchmark)
    logger.info(f"\n{'='*60}")
    logger.info("DEEPSAFE v3 vs BEST SOTA (per benchmark)")
    logger.info(f"{'Benchmark':25s} {'DeepSafe-v3':>10s} {'Best SOTA':>10s} {'Winner':>12s}")
    logger.info("-" * 60)
    for bench_name, bench_results in all_results.items():
        our_acc = bench_results.get("DeepSafe-v3", {}).get("accuracy", 0)
        sota_models = {k: v for k, v in bench_results.items() if k != "DeepSafe-v3" and "error" not in v}
        best_sota_name = max(sota_models, key=lambda x: sota_models[x].get("accuracy", 0)) if sota_models else "N/A"
        best_sota_acc = sota_models[best_sota_name]["accuracy"] if sota_models else 0
        winner = "DeepSafe" if our_acc > best_sota_acc else best_sota_name[:12]
        logger.info(f"  {bench_name:25s} {our_acc:10.4f} {best_sota_acc:10.4f} {winner:>12s}")


if __name__ == "__main__":
    main()
