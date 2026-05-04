#!/usr/bin/env python3
"""R8: Hyperbolic vs Euclidean necessity ablation at [64, 128, 256, 512] hidden dims.

Compares:
  - Euclidean projector: Linear(4096→hdim) → GELU → Linear(hdim→256) → L2Norm
  - Hyperbolic projector: Linear(4096→128) → exp_map → HyperbolicLinear(128→128) → log_map → Linear(128→256) → L2Norm

Both with residual skip connections and same training setup.
Uses pre-computed 8B embeddings.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deepsafe.hyperbolic import HyperbolicLinear, HyperbolicActivation, HyperbolicToEuclidean, exp_map

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
EMBED_DIR = PROJ / "embeddings" / "qwen3-embedding-8B"
OUT_DIR = PROJ / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = PROJ / "models" / "ablation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"


class EuclideanProjector(nn.Module):
    """Euclidean projector: dim→hdim→256 with residual + gate."""

    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=256):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.residual = nn.Linear(input_dim, output_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        main_out = self.main(x_norm)
        res_out = self.residual(x_norm)
        alpha = torch.sigmoid(self.gate)
        combined = alpha * main_out + (1 - alpha) * res_out
        return F.normalize(combined, p=2, dim=1)


class HyperbolicProjector(nn.Module):
    """Hyperbolic projector with Poincare ball operations."""

    def __init__(self, input_dim=4096, hyperbolic_dim=128, output_dim=256, curvature=1.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hyperbolic_dim)
        self.input_ln = nn.LayerNorm(hyperbolic_dim)
        self.hyp_linear = HyperbolicLinear(hyperbolic_dim, hyperbolic_dim, c=curvature)
        self.hyp_act = HyperbolicActivation(activation=nn.GELU(), c=curvature)
        self.hyp_to_euc = HyperbolicToEuclidean(hyperbolic_dim, output_dim, c=curvature)
        self.residual = nn.Linear(input_dim, output_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.curvature = curvature

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=1)
        # Hyperbolic pathway
        h = self.input_ln(self.input_proj(x_norm))
        h = exp_map(h, c=self.curvature)
        h = self.hyp_act(self.hyp_linear(h))
        hyp_out = self.hyp_to_euc(h)
        # Residual pathway
        res_out = self.residual(x_norm)
        alpha = torch.sigmoid(self.gate)
        combined = alpha * hyp_out + (1 - alpha) * res_out
        return F.normalize(combined, p=2, dim=1)


def train_projector(projector, X_train, y_train, X_val, y_val, epochs=50, lr=5e-4):
    """Train a projector using binary cross-entropy classification."""
    projector = projector.to(DEVICE)

    # Classification head
    head = nn.Linear(256, 2).to(DEVICE)

    # Combined optimizer
    optimizer = torch.optim.AdamW(
        list(projector.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.LongTensor(y_train)
    train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.LongTensor(y_val).to(DEVICE)

    best_acc, patience = 0.0, 0
    best_state = None

    for epoch in range(epochs):
        projector.train()
        head.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            z = projector(bx)
            logits = head(z)
            loss = F.cross_entropy(logits, by)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(projector.parameters()) + list(head.parameters()), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        projector.eval()
        head.eval()
        with torch.no_grad():
            val_logits = head(projector(X_val_t))
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            best_state = {
                "projector": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"    Early stop at epoch {epoch+1}, best_val_acc={best_acc:.4f}")
                break

    if best_state:
        projector.load_state_dict(best_state["projector"])
        head.load_state_dict(best_state["head"])
        projector = projector.to(DEVICE)
        head = head.to(DEVICE)

    return projector, head, best_acc


def evaluate_model(projector, head, X_test, y_test):
    """Evaluate on test set."""
    projector.eval()
    head.eval()
    X_t = torch.FloatTensor(X_test).to(DEVICE)
    with torch.no_grad():
        logits = head(projector(X_t))
        probs = F.softmax(logits, dim=-1)
        y_score = probs[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=-1).cpu().numpy()

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
    }


def main():
    logger.info("Loading pre-computed 8B embeddings...")
    X_train = np.load(EMBED_DIR / "train.npy", mmap_mode="r")
    y_train = np.load(EMBED_DIR / "train_labels.npy")
    X_test = np.load(EMBED_DIR / "test.npy", mmap_mode="r")
    y_test = np.load(EMBED_DIR / "test_labels.npy")

    # Use 50K train / 20K test for faster ablation
    n_train, n_val, n_test = 50000, 15000, 20000
    rng = np.random.RandomState(42)
    idx_all = rng.permutation(len(X_train))[:n_train + n_val]
    train_idx = idx_all[:n_train]
    val_idx = idx_all[n_train:]
    test_idx = rng.choice(len(X_test), n_test, replace=False)

    X_tr = np.array(X_train[train_idx], dtype=np.float32)
    y_tr = y_train[train_idx].astype(np.int64)
    X_val = np.array(X_train[val_idx], dtype=np.float32)
    y_val = y_train[val_idx].astype(np.int64)
    X_te = np.array(X_test[test_idx], dtype=np.float32)
    y_te = y_test[test_idx].astype(np.int64)

    # Standardize
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    logger.info(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
    logger.info(f"Train class dist: safe={np.sum(y_tr==0)}, unsafe={np.sum(y_tr==1)}")

    HIDDEN_DIMS = [64, 128, 256, 512]
    results = {}

    for hdim in HIDDEN_DIMS:
        logger.info(f"\n{'='*60}")
        logger.info(f"HIDDEN DIM = {hdim}")
        logger.info(f"{'='*60}")

        # Euclidean projector
        logger.info(f"  Training Euclidean projector (4096→{hdim}→256)...")
        t0 = time.time()
        euc_proj = EuclideanProjector(input_dim=4096, hidden_dim=hdim, output_dim=256)
        euc_proj, euc_head, euc_val_acc = train_projector(euc_proj, X_tr_s, y_tr, X_val_s, y_val, epochs=50)
        euc_time = time.time() - t0
        euc_metrics = evaluate_model(euc_proj, euc_head, X_te_s, y_te)
        euc_params = sum(p.numel() for p in euc_proj.parameters())
        logger.info(f"    Euclidean: val_acc={euc_val_acc:.4f} test_acc={euc_metrics['accuracy']:.4f} "
                    f"f1={euc_metrics['f1_macro']:.4f} auc={euc_metrics['roc_auc']:.4f} "
                    f"params={euc_params:,} time={euc_time:.0f}s")

        # Hyperbolic projector
        logger.info(f"  Training Hyperbolic projector (4096→128 hyperbolic→256)...")
        t0 = time.time()
        hyp_proj = HyperbolicProjector(input_dim=4096, hyperbolic_dim=128, output_dim=256)
        hyp_proj, hyp_head, hyp_val_acc = train_projector(hyp_proj, X_tr_s, y_tr, X_val_s, y_val, epochs=50, lr=3e-4)
        hyp_time = time.time() - t0
        hyp_metrics = evaluate_model(hyp_proj, hyp_head, X_te_s, y_te)
        hyp_params = sum(p.numel() for p in hyp_proj.parameters())
        logger.info(f"    Hyperbolic: val_acc={hyp_val_acc:.4f} test_acc={hyp_metrics['accuracy']:.4f} "
                    f"f1={hyp_metrics['f1_macro']:.4f} auc={hyp_metrics['roc_auc']:.4f} "
                    f"params={hyp_params:,} time={hyp_time:.0f}s")

        results[str(hdim)] = {
            "euclidean": euc_metrics,
            "hyperbolic": hyp_metrics,
            "euclidean_val_acc": euc_val_acc,
            "hyperbolic_val_acc": hyp_val_acc,
            "euclidean_params": euc_params,
            "hyperbolic_params": hyp_params,
            "euclidean_time_s": euc_time,
            "hyperbolic_time_s": hyp_time,
        }

        # Save checkpoint
        torch.save({"euc_proj": euc_proj.state_dict(), "hyp_proj": hyp_proj.state_dict()},
                   MODEL_DIR / f"ablation_hdim_{hdim}.pt")

    # Print summary table
    logger.info(f"\n{'='*80}")
    logger.info("ABLATION SUMMARY: Euclidean vs Hyperbolic Projection")
    logger.info(f"{'='*80}")
    logger.info(f"{'HDim':>6s} {'Euc Acc':>10s} {'Hyp Acc':>10s} {'Δ (Hyp-Euc)':>12s} "
                f"{'Euc F1':>10s} {'Hyp F1':>10s} {'Euc AUC':>10s} {'Hyp AUC':>10s}")
    logger.info(f"{'-'*80}")
    for hdim in HIDDEN_DIMS:
        r = results[str(hdim)]
        delta = r["hyperbolic"]["accuracy"] - r["euclidean"]["accuracy"]
        logger.info(f"{hdim:6d} {r['euclidean']['accuracy']:10.4f} {r['hyperbolic']['accuracy']:10.4f} "
                    f"{delta:+12.4f} {r['euclidean']['f1_macro']:10.4f} {r['hyperbolic']['f1_macro']:10.4f} "
                    f"{r['euclidean']['roc_auc']:10.4f} {r['hyperbolic']['roc_auc']:10.4f}")

    # Save results
    with open(OUT_DIR / "ablation_table.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {OUT_DIR / 'ablation_table.json'}")

    # Generate CSV
    import csv
    with open(OUT_DIR / "ablation_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hidden_dim", "type", "accuracy", "f1_macro", "roc_auc", "val_acc", "params", "time_s"])
        for hdim in HIDDEN_DIMS:
            r = results[str(hdim)]
            for typ in ["euclidean", "hyperbolic"]:
                m = r[typ]
                writer.writerow([hdim, typ, m["accuracy"], m["f1_macro"], m["roc_auc"],
                                r[f"{typ}_val_acc"], r[f"{typ}_params"], r[f"{typ}_time_s"]])
    logger.info(f"CSV saved to {OUT_DIR / 'ablation_table.csv'}")


if __name__ == "__main__":
    main()
