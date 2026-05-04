#!/usr/bin/env python3
"""Finish baseline evaluation: encode remaining 3 benchmarks + save all results."""
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.encode.qwen_encoder import QwenEncoder
from src.deepsafe.neural_classifier import NeuralClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
EMBED_PATH = str(PROJ / "pretrained" / "qwen" / "Qwen3-Embedding-8B")
BASELINE_DIR = PROJ / "models" / "baselines"
HELDOUT_DIR = PROJ / "valuation" / "heldout"
OUT_DIR = PROJ / "valuation" / "heldout" / "predictions" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"

# Reconstruct SimpleMLP
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )
    def forward(self, x): return self.net(x)

# Reconstruct EuclideanProjector
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


def compute_metrics(y_true, y_pred, y_score):
    unique = np.unique(y_true)
    if len(unique) < 2:
        acc = float(accuracy_score(y_true, y_pred))
        return {"accuracy": acc, "f1_macro": 0.0, "precision_macro": 0.0,
                "recall_macro": 0.0, "roc_auc": 0.5}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def load_baselines():
    with open(BASELINE_DIR / "lr_baseline.pkl", "rb") as f:
        lr_data = pickle.load(f)
    with open(BASELINE_DIR / "mlp_baseline.pkl", "rb") as f:
        mlp_data = pickle.load(f)
    with open(BASELINE_DIR / "euclidean_nn_baseline.pkl", "rb") as f:
        euc_data = pickle.load(f)

    mlp = SimpleMLP().to(DEVICE).eval()
    mlp.load_state_dict(mlp_data["state_dict"])

    euc_proj = EuclideanProjector().to(DEVICE).eval()
    euc_proj.load_state_dict(euc_data["projector_state"])
    euc_clf = NeuralClassifier(input_dim=256, hidden_dims=[128, 64], dropout=0.2).to(DEVICE).eval()
    euc_clf.load_state_dict(euc_data["classifier_state"])

    return {
        "LR-4096d": ("lr", lr_data["clf"], lr_data["scaler"]),
        "MLP-4096-512-2": ("mlp", mlp, mlp_data["scaler"]),
        "EuclideanProj-NN": ("euc", (euc_proj, euc_clf), euc_data["scaler"]),
    }


# Previously computed results (from first 6 benchmarks in earlier run)
KNOWN = {
    "100PoisonMpts": {
        "LR-4096d": {"accuracy": 0.2684, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "roc_auc": 0.5},
        "MLP-4096-512-2": {"accuracy": 0.2574, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "roc_auc": 0.5},
        "EuclideanProj-NN": {"accuracy": 0.4375, "f1_macro": 0.0, "precision_macro": 0.0, "recall_macro": 0.0, "roc_auc": 0.5},
    },
    "AegisAI-v1": {
        "LR-4096d": {"accuracy": 0.7498, "f1_macro": 0.6881, "precision_macro": 0.7319, "recall_macro": 0.6881, "roc_auc": 0.7988},
        "MLP-4096-512-2": {"accuracy": 0.7488, "f1_macro": 0.7409, "precision_macro": 0.7335, "recall_macro": 0.7484, "roc_auc": 0.8683},
        "EuclideanProj-NN": {"accuracy": 0.7161, "f1_macro": 0.7095, "precision_macro": 0.7118, "recall_macro": 0.7038, "roc_auc": 0.7989},
    },
    "AegisAI-v2": {
        "LR-4096d": {"accuracy": 0.7212, "f1_macro": 0.6850, "precision_macro": 0.7331, "recall_macro": 0.6850, "roc_auc": 0.8052},
        "MLP-4096-512-2": {"accuracy": 0.7782, "f1_macro": 0.7738, "precision_macro": 0.7733, "recall_macro": 0.7765, "roc_auc": 0.8633},
        "EuclideanProj-NN": {"accuracy": 0.7221, "f1_macro": 0.7206, "precision_macro": 0.7148, "recall_macro": 0.7270, "roc_auc": 0.7680},
    },
    "BeaverTails": {
        "LR-4096d": {"accuracy": 0.6140, "f1_macro": 0.5248, "precision_macro": 0.6288, "recall_macro": 0.5248, "roc_auc": 0.7134},
        "MLP-4096-512-2": {"accuracy": 0.6607, "f1_macro": 0.6327, "precision_macro": 0.6358, "recall_macro": 0.6327, "roc_auc": 0.7125},
        "EuclideanProj-NN": {"accuracy": 0.6147, "f1_macro": 0.6090, "precision_macro": 0.6091, "recall_macro": 0.6102, "roc_auc": 0.6332},
    },
    "DoNotAnswer": {
        "LR-4096d": {"accuracy": 0.1851, "f1_macro": 0.1646, "precision_macro": 0.4589, "recall_macro": 0.1646, "roc_auc": 0.6616},
        "MLP-4096-512-2": {"accuracy": 0.2420, "f1_macro": 0.2058, "precision_macro": 0.4781, "recall_macro": 0.2058, "roc_auc": 0.5235},
        "EuclideanProj-NN": {"accuracy": 0.3523, "f1_macro": 0.2726, "precision_macro": 0.4760, "recall_macro": 0.2726, "roc_auc": 0.5514},
    },
    "ToxicChat": {
        "LR-4096d": {"accuracy": 0.6563, "f1_macro": 0.5263, "precision_macro": 0.7263, "recall_macro": 0.5263, "roc_auc": 0.8974},
        "MLP-4096-512-2": {"accuracy": 0.9262, "f1_macro": 0.7873, "precision_macro": 0.7363, "recall_macro": 0.7873, "roc_auc": 0.9552},
        "EuclideanProj-NN": {"accuracy": 0.9141, "f1_macro": 0.7586, "precision_macro": 0.8228, "recall_macro": 0.7586, "roc_auc": 0.9192},
    },
}


def main():
    # Load encoder
    logger.info("Loading Qwen3-Embedding-8B encoder...")
    encoder = QwenEncoder(EMBED_PATH, device=DEVICE, batch_size=64)

    # Load baselines
    logger.info("Loading baselines...")
    baselines = load_baselines()

    # Start with known results
    all_results = dict(KNOWN)

    # Encode remaining 3 benchmarks
    remaining = ["WildGuardMix", "XGuard", "XSTest"]
    for bench in remaining:
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
        for name, (btype, model, scaler) in baselines.items():
            X_s = scaler.transform(embeddings).astype(np.float32)

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
                euc_proj, euc_clf = model
                with torch.no_grad():
                    x_t = torch.FloatTensor(X_s).to(DEVICE)
                    proj = euc_proj(x_t)
                    y_score = euc_clf.predict_proba(proj.cpu().numpy(), device=DEVICE)
                y_pred = (y_score > 0.5).astype(int)

            metrics = compute_metrics(labels, y_pred, y_score)
            bench_results[name] = metrics
            logger.info(f"  {name:25s}: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

        all_results[bench] = bench_results

    # Save
    with open(OUT_DIR / "raw_baselines_v1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {OUT_DIR / 'raw_baselines_v1_results.json'}")

    # CSV report
    REPORT_PATH = PROJ / "reports" / "baselines_table.csv"
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "model", "accuracy", "f1_macro", "roc_auc"])
        writer.writeheader()
        for bench in sorted(all_results.keys()):
            for name in ["LR-4096d", "MLP-4096-512-2", "EuclideanProj-NN"]:
                m = all_results[bench][name]
                writer.writerow({"benchmark": bench, "model": name,
                    "accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "roc_auc": m["roc_auc"]})
    logger.info(f"CSV report saved to {REPORT_PATH}")

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("BASELINE SUMMARY (averaged across all 9 benchmarks)")
    logger.info("-"*70)
    for name in ["LR-4096d", "MLP-4096-512-2", "EuclideanProj-NN"]:
        accs = [all_results[b][name]["accuracy"] for b in sorted(all_results.keys())]
        f1s = [all_results[b][name]["f1_macro"] for b in sorted(all_results.keys())]
        aucs = [all_results[b][name]["roc_auc"] for b in sorted(all_results.keys())]
        logger.info(f"  {name:25s}: avg_acc={np.mean(accs):.4f}  avg_f1={np.mean(f1s):.4f}  avg_auc={np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
