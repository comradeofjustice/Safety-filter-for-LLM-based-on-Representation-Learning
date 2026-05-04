#!/usr/bin/env python3
"""R4: Train missing baselines on raw Qwen3-Embedding-8B (4096-d) embeddings.

Produces:
  (a) Logistic Regression on raw 4096-d
  (b) 2-layer MLP (4096→512→2) on raw 4096-d
  (c) Euclidean projection head (4096→256) + NeuralClassifier

All training is CPU-only using pre-computed embeddings.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deepsafe.neural_classifier import NeuralClassifier, NeuralClassifierTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
EMBED_DIR = PROJ / "embeddings" / "qwen3-embedding-8B"
OUT_DIR = PROJ / "models" / "baselines"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def compute_metrics(y_true, y_pred, y_score):
    unique = np.unique(y_true)
    if len(unique) < 2:
        acc = float(accuracy_score(y_true, y_pred))
        return {"accuracy": acc, "f1_macro": 0.0, "roc_auc": 0.5}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def train_lr_baseline(X_train, y_train, X_test, y_test):
    """Baseline (a): Logistic Regression on raw 4096-d embeddings."""
    logger.info("=" * 60)
    logger.info("BASELINE (a): Logistic Regression on raw 4096-d")
    logger.info("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        C=1.0, max_iter=2000, solver="saga", penalty="l2",
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    t0 = time.time()
    clf.fit(X_train_s, y_train)
    elapsed = time.time() - t0
    logger.info(f"  Training: {elapsed:.0f}s")

    y_score = clf.predict_proba(X_test_s)[:, 1]
    y_pred = clf.predict(X_test_s)
    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(f"  Test: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

    # Save
    import pickle
    with open(OUT_DIR / "lr_baseline.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    logger.info(f"  Saved to {OUT_DIR / 'lr_baseline.pkl'}")

    return metrics


class SimpleMLP(nn.Module):
    """2-layer MLP: 4096 → 512 → 2 with ReLU + Dropout."""

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


def train_mlp_baseline(X_train, y_train, X_test, y_test):
    """Baseline (b): 2-layer MLP on raw 4096-d embeddings."""
    logger.info("=" * 60)
    logger.info("BASELINE (b): 2-layer MLP 4096→512→2")
    logger.info("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    model = SimpleMLP(input_dim=4096, hidden_dim=512, dropout=0.3).to(DEVICE)

    # Class weights
    unique, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    class_weights = torch.FloatTensor([n_samples / c for c in counts]).to(DEVICE)

    # Data
    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.LongTensor(y_train)
    X_te_t = torch.FloatTensor(X_test_s).to(DEVICE)
    y_te_t = torch.LongTensor(y_test).to(DEVICE)

    train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    best_acc, patience = 0.0, 0
    best_state = None

    t0 = time.time()
    for epoch in range(50):
        model.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by, weight=class_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_te_t)
            val_preds = val_logits.argmax(dim=-1)
            val_acc = (val_preds == y_te_t).float().mean().item()

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"  Early stop at epoch {epoch+1}, best val_acc={best_acc:.4f}")
                break

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: loss={epoch_loss/len(train_loader):.4f} val_acc={val_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    elapsed = time.time() - t0
    logger.info(f"  Training: {elapsed:.0f}s")

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        probs = F.softmax(logits, dim=-1)
        y_score = probs[:, 1].cpu().numpy()
        y_pred = logits.argmax(dim=-1).cpu().numpy()

    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(f"  Test: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

    # Save
    import pickle
    with open(OUT_DIR / "mlp_baseline.pkl", "wb") as f:
        pickle.dump({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "scaler": scaler,
            "input_dim": 4096,
            "hidden_dim": 512,
        }, f)
    logger.info(f"  Saved to {OUT_DIR / 'mlp_baseline.pkl'}")

    return metrics


class EuclideanProjector(nn.Module):
    """Euclidean projection head: 4096 → 256 with residual path and gate.

    Same architecture as DeepSafeProjectionHeadV3 but without hyperbolic geometry.
    Uses only Euclidean layers with the same depth and capacity.
    """

    def __init__(self, input_dim=4096, hidden_dim=768, output_dim=256):
        super().__init__()
        # Deep Euclidean pathway (replaces hyperbolic layers)
        self.euc_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Residual shortcut
        self.residual = nn.Linear(input_dim, output_dim)
        # Gate
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # L2 normalize input
        x_norm = F.normalize(x, p=2, dim=1)
        hyp = self.euc_path(x_norm)
        res = self.residual(x_norm)
        alpha = torch.sigmoid(self.gate)
        combined = alpha * hyp + (1 - alpha) * res
        # Output head
        out = F.normalize(combined, p=2, dim=1)
        return out


def train_euclidean_nn(X_train, y_train, X_test, y_test):
    """Baseline (c): Euclidean projection head + NeuralClassifier on raw 4096-d."""
    logger.info("=" * 60)
    logger.info("BASELINE (c): Euclidean projection 4096→256 + NeuralClassifier")
    logger.info("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    # Phase 1: Train Euclidean projector with contrastive + OT + proto + decorr loss
    from src.deepsafe.losses import DeepSafeLoss

    projector = EuclideanProjector(input_dim=4096, hidden_dim=768, output_dim=256).to(DEVICE)
    criterion = DeepSafeLoss(
        feature_dim=256, temperature=0.07, alpha=2.0,
        lambda_ot=0.3, lambda_proto=0.2, lambda_decorr=0.005, gamma=0.0,  # no CF pairs for baselines
    ).to(DEVICE)

    X_tr_t = torch.FloatTensor(X_train_s)
    y_tr_t = torch.LongTensor(y_train)
    train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-6)

    best_loss, patience = float("inf"), 0
    best_state = None

    t0 = time.time()
    for epoch in range(80):
        projector.train()
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            z = projector(bx)
            intent_labels = torch.where(by == 1, torch.tensor(2, device=DEVICE), torch.tensor(0, device=DEVICE))
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
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in projector.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
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
    clf_metrics = trainer.train(X_tr_proj, y_train, X_te_proj, y_test)

    # Evaluate
    clf = trainer.model
    y_score = clf.predict_proba(X_te_proj, device=DEVICE)
    y_pred = (y_score > 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_score)
    logger.info(f"  Final: acc={metrics['accuracy']:.4f} f1={metrics['f1_macro']:.4f} auc={metrics['roc_auc']:.4f}")

    # Save
    import pickle
    with open(OUT_DIR / "euclidean_nn_baseline.pkl", "wb") as f:
        pickle.dump({
            "projector_state": {k: v.cpu() for k, v in projector.state_dict().items()},
            "classifier_state": {k: v.cpu() for k, v in clf.state_dict().items()},
            "scaler": scaler,
            "input_dim": 4096,
            "hidden_dim": 768,
            "output_dim": 256,
        }, f)
    logger.info(f"  Saved to {OUT_DIR / 'euclidean_nn_baseline.pkl'}")

    return metrics


def main():
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Loading embeddings from {EMBED_DIR}")

    X_train = np.load(EMBED_DIR / "train.npy", mmap_mode="r")
    y_train = np.load(EMBED_DIR / "train_labels.npy")
    X_test = np.load(EMBED_DIR / "test.npy", mmap_mode="r")
    y_test = np.load(EMBED_DIR / "test_labels.npy")

    # Use subset for faster training (full set is ~257K samples)
    n_train = min(len(X_train), 100000)
    n_test = min(len(X_test), 30000)
    rng = np.random.RandomState(42)
    train_idx = rng.choice(len(X_train), n_train, replace=False)
    test_idx = rng.choice(len(X_test), n_test, replace=False)
    X_tr = np.array(X_train[train_idx], dtype=np.float32)
    y_tr = y_train[train_idx].astype(np.int64)
    X_te = np.array(X_test[test_idx], dtype=np.float32)
    y_te = y_test[test_idx].astype(np.int64)

    logger.info(f"Using {n_train} train, {n_test} test samples")
    logger.info(f"Train class dist: safe={np.sum(y_tr==0)}, unsafe={np.sum(y_tr==1)}")
    logger.info(f"Test  class dist: safe={np.sum(y_te==0)}, unsafe={np.sum(y_te==1)}")

    results = {}

    # (a) LR baseline
    results["lr_4096d"] = train_lr_baseline(X_tr, y_tr, X_te, y_te)

    # (b) MLP baseline
    results["mlp_4096_512_2"] = train_mlp_baseline(X_tr, y_tr, X_te, y_te)

    # (c) Euclidean projection + NN
    results["euclidean_projection_nn"] = train_euclidean_nn(X_tr, y_tr, X_te, y_te)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("BASELINE SUMMARY (validation set only)")
    logger.info("-" * 70)
    for name, m in results.items():
        logger.info(f"  {name:30s}: acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}  auc={m['roc_auc']:.4f}")

    # Save summary
    with open(OUT_DIR / "baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSummary saved to {OUT_DIR / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
