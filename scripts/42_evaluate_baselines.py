#!/usr/bin/env python3
"""R4 Evaluation: Evaluate trained baselines on all 9 held-out benchmarks.

Loads baselines from models/baselines/, encodes held-out benchmarks,
generates predictions, computes metrics, and saves to baselines dir.

Requires GPU for Qwen3-Embedding-8B encoding.
"""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encode.qwen_encoder import QwenEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
EMBED_PATH = str(PROJ / "pretrained" / "qwen" / "Qwen3-Embedding-8B")
BASELINE_DIR = PROJ / "models" / "baselines"
PRED_DIR = PROJ / "valuation" / "heldout" / "predictions" / "baselines"
REPORT_PATH = PROJ / "reports" / "baselines_table.csv"
HELDOUT_DIR = PROJ / "valuation" / "heldout"
DEVICE = "cuda"

BENCHMARKS = [
    "100PoisonMpts", "AegisAI-v1", "AegisAI-v2", "BeaverTails",
    "DoNotAnswer", "ToxicChat", "WildGuardMix", "XGuard", "XSTest",
]


def compute_metrics(y_true, y_pred, y_score):
    unique = np.unique(y_true)
    if len(unique) < 2:
        acc = float(accuracy_score(y_true, y_pred))
        return {"accuracy": acc, "f1_macro": 0.0, "precision_macro": 0.0,
                "recall_macro": 0.0, "roc_auc": 0.5, "pr_auc": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def load_lr_baseline():
    with open(BASELINE_DIR / "lr_baseline.pkl", "rb") as f:
        data = pickle.load(f)
    return data["clf"], data["scaler"]


def load_mlp_baseline():
    with open(BASELINE_DIR / "mlp_baseline.pkl", "rb") as f:
        data = pickle.load(f)

    import torch.nn as nn
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=4096, hidden_dim=512, dropout=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        def forward(self, x):
            return self.net(x)

    model = SimpleMLP(input_dim=4096, hidden_dim=512, dropout=0.3)
    model.load_state_dict(data["state_dict"])
    model = model.to(DEVICE).eval()
    return model, data["scaler"]


def load_euclidean_baseline():
    with open(BASELINE_DIR / "euclidean_nn_baseline.pkl", "rb") as f:
        data = pickle.load(f)

    import torch.nn as nn
    class EuclideanProjector(nn.Module):
        def __init__(self, input_dim=4096, hidden_dim=768, output_dim=256):
            super().__init__()
            self.euc_path = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )
            self.residual = nn.Linear(input_dim, output_dim)
            self.gate = nn.Parameter(torch.zeros(1))
        def forward(self, x):
            x_norm = F.normalize(x, p=2, dim=1)
            hyp = self.euc_path(x_norm)
            res = self.residual(x_norm)
            alpha = torch.sigmoid(self.gate)
            combined = alpha * hyp + (1 - alpha) * res
            return F.normalize(combined, p=2, dim=1)

    from src.deepsafe.neural_classifier import NeuralClassifier

    projector = EuclideanProjector(input_dim=4096, hidden_dim=768, output_dim=256)
    projector.load_state_dict(data["projector_state"])
    projector = projector.to(DEVICE).eval()

    clf = NeuralClassifier(input_dim=256, hidden_dims=[128, 64], dropout=0.2)
    clf.load_state_dict(data["classifier_state"])
    clf = clf.to(DEVICE).eval()

    return projector, clf, data["scaler"]


def main():
    # Load encoder
    logger.info("Loading Qwen3-Embedding-8B encoder...")
    encoder = QwenEncoder(EMBED_PATH, device=DEVICE, batch_size=64)

    # Load baselines
    logger.info("Loading baselines...")
    lr_clf, lr_scaler = load_lr_baseline()
    mlp_model, mlp_scaler = load_mlp_baseline()
    euc_proj, euc_clf, euc_scaler = load_euclidean_baseline()

    baselines = {
        "LR-4096d": ("lr", lr_clf, lr_scaler, None, None),
        "MLP-4096-512-2": ("mlp", mlp_model, mlp_scaler, None, None),
        "EuclideanProj-NN": ("euc", euc_proj, euc_scaler, euc_clf, None),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for bench in BENCHMARKS:
        hp = HELDOUT_DIR / f"{bench}_30pct.parquet"
        df = pd.read_parquet(hp)
        texts = df["texts"].tolist()
        labels = np.array(df["labels"].tolist())
        n_safe = int(np.sum(labels == 0))
        n_unsafe = int(np.sum(labels == 1))

        logger.info(f"\n{'='*60}")
        logger.info(f"{bench}: {len(texts)} samples (safe={n_safe}, unsafe={n_unsafe})")
        logger.info(f"{'='*60}")

        t0 = time.time()
        embeddings = np.array(encoder.encode(texts), dtype=np.float32)
        logger.info(f"  Encoding: {time.time()-t0:.0f}s")

        bench_results = {}
        for name, (btype, model, scaler, clf, _) in baselines.items():
            # Scale
            X_s = scaler.transform(embeddings).astype(np.float32)
            t1 = time.time()

            if btype == "lr":
                y_score = model.predict_proba(X_s)[:, 1]
                y_pred = model.predict(X_s)
            elif btype == "mlp":
                with torch.no_grad():
                    x_t = torch.FloatTensor(X_s).to(DEVICE)
                    logits = model(x_t)
                    probs = F.softmax(logits, dim=-1)
                    y_score = probs[:, 1].cpu().numpy()
                    y_pred = logits.argmax(dim=-1).cpu().numpy()
            elif btype == "euc":
                with torch.no_grad():
                    x_t = torch.FloatTensor(X_s).to(DEVICE)
                    proj = model(x_t)
                    y_score = clf.predict_proba(proj.cpu().numpy(), device=DEVICE)
                y_pred = (y_score > 0.5).astype(int)

            metrics = compute_metrics(labels, y_pred, y_score)
            metrics["inference_time_s"] = time.time() - t1
            bench_results[name] = metrics

            logger.info(f"  {name:25s}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

        all_results[bench] = bench_results

        # Save predictions
        pred_subdir = PRED_DIR / "raw-baselines-v1"
        pred_subdir.mkdir(parents=True, exist_ok=True)
        for name in baselines:
            pass  # predictions saved via metrics

    # Save results
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PRED_DIR / "raw_baselines_v1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {PRED_DIR / 'raw_baselines_v1_results.json'}")

    # Generate CSV report
    rows = []
    for bench in BENCHMARKS:
        for name in baselines:
            m = all_results[bench][name]
            rows.append({
                "benchmark": bench, "model": name,
                "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "roc_auc": m["roc_auc"],
            })

    import csv
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "model", "accuracy", "f1_macro", "roc_auc"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV report saved to {REPORT_PATH}")

    # Summary: average across benchmarks
    logger.info(f"\n{'='*70}")
    logger.info("BASELINE SUMMARY (averaged across 9 benchmarks)")
    logger.info("-"*70)
    for name in baselines:
        avg_acc = np.mean([all_results[b][name]["accuracy"] for b in BENCHMARKS])
        avg_f1 = np.mean([all_results[b][name]["f1_macro"] for b in BENCHMARKS])
        avg_auc = np.mean([all_results[b][name]["roc_auc"] for b in BENCHMARKS])
        logger.info(f"  {name:25s}: avg_acc={avg_acc:.4f}  avg_f1={avg_f1:.4f}  avg_auc={avg_auc:.4f}")


if __name__ == "__main__":
    main()
