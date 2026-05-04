#!/usr/bin/env python3
"""Train Euclidean projection baseline (4096→256) + NeuralClassifier."""
import json
import logging
import pickle
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

from src.deepsafe.losses import DeepSafeLoss
from src.deepsafe.neural_classifier import NeuralClassifier, NeuralClassifierTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
EMBED_DIR = PROJ / "embeddings" / "qwen3-embedding-8B"
OUT_DIR = PROJ / "models" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda"


class EuclideanProjector(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=768, output_dim=256):
        super().__init__()
        self.euc_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
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
        return {"accuracy": float(accuracy_score(y_true, y_pred)), "f1_macro": 0.0, "roc_auc": 0.5}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def main():
    X_train_full = np.load(EMBED_DIR / "train.npy", mmap_mode="r")
    y_train_full = np.load(EMBED_DIR / "train_labels.npy")
    X_test_full = np.load(EMBED_DIR / "test.npy", mmap_mode="r")
    y_test_full = np.load(EMBED_DIR / "test_labels.npy")

    n_train, n_test = 100000, 30000
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_train_full), n_train, replace=False)
    test_idx = rng.choice(len(X_test_full), n_test, replace=False)
    X_tr = np.array(X_train_full[train_idx], dtype=np.float32)
    y_tr = y_train_full[train_idx].astype(np.int64)
    X_te = np.array(X_test_full[test_idx], dtype=np.float32)
    y_te = y_test_full[test_idx].astype(np.int64)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_test_s = scaler.transform(X_te).astype(np.float32)

    projector = EuclideanProjector(input_dim=4096, hidden_dim=768, output_dim=256).to(DEVICE)
    criterion = DeepSafeLoss(
        feature_dim=256, temperature=0.07, alpha=2.0,
        lambda_ot=0.3, lambda_proto=0.2, lambda_decorr=0.005, gamma=0.0,
    ).to(DEVICE)

    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.LongTensor(y_tr)
    train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    best_loss, patience_counter = float("inf"), 0
    best_state = None

    t0 = time.time()
    for epoch in range(80):
        projector.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            z = projector(bx)
            intent_labels = torch.where(by == 1,
                torch.tensor(2, device=DEVICE), torch.tensor(0, device=DEVICE))
            loss = criterion(z, by, intent_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
                logger.info(f"  Early stop at epoch {epoch+1}, best_loss={best_loss:.4f}")
                break
        if (epoch + 1) % 20 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    if best_state:
        projector.load_state_dict(best_state)
        projector = projector.to(DEVICE)
    logger.info(f"  Projector training: {time.time()-t0:.0f}s")

    # Phase 2: Project and train classifier
    projector.eval()
    with torch.no_grad():
        X_tr_proj = projector(torch.FloatTensor(X_train_s).to(DEVICE)).cpu().numpy()
        X_te_proj = projector(torch.FloatTensor(X_test_s).to(DEVICE)).cpu().numpy()

    trainer = NeuralClassifierTrainer(
        input_dim=256, hidden_dims=[128, 64], dropout=0.2,
        lr=1e-3, batch_size=512, max_epochs=50, patience=10,
        mixup_alpha=0.2, device=DEVICE,
    )
    clf_metrics = trainer.train(X_tr_proj, y_tr, X_te_proj, y_te)

    clf = trainer.model
    y_score = clf.predict_proba(X_te_proj, device=DEVICE)
    y_pred = (y_score > 0.5).astype(int)
    metrics = compute_metrics(y_te, y_pred, y_score)
    logger.info(f"  Final: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

    # Save
    with open(OUT_DIR / "euclidean_nn_baseline.pkl", "wb") as f:
        pickle.dump({
            "projector_state": {k: v.cpu() for k, v in projector.state_dict().items()},
            "classifier_state": {k: v.cpu() for k, v in clf.state_dict().items()},
            "scaler": scaler, "input_dim": 4096, "hidden_dim": 768, "output_dim": 256,
        }, f)
    logger.info(f"  Saved to {OUT_DIR / 'euclidean_nn_baseline.pkl'}")

    # Update summary
    with open(OUT_DIR / "baseline_results.json") as f:
        results = json.load(f)
    results["euclidean_projection_nn"] = metrics
    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Updated baseline_results.json")


if __name__ == "__main__":
    main()
