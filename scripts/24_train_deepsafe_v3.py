#!/usr/bin/env python3
"""Train DeepSafe v3: Enhanced hyperbolic projection head + neural classifier.

Two-phase training:
  1. Train DeepSafeProjectionHeadV3 with combined loss on augmented data
  2. Train NeuralClassifier on frozen projected embeddings

Supports data augmentation from benchmark datasets (up to 70% merged into training).

Usage:
  python scripts/24_train_deepsafe_v3.py
  python scripts/24_train_deepsafe_v3.py --embed-model qwen3-embedding-8B
  python scripts/24_train_deepsafe_v3.py --embed-model qwen3-embedding-0.6B --no-augment
"""

import argparse
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

from src.deepsafe.projection_head_v3 import (
    DeepSafeProjectionHeadV3,
    DeepSafeProjectionHeadV3Manager,
)
from src.deepsafe.losses import DeepSafeLoss
from src.deepsafe.neural_classifier import NeuralClassifierTrainer
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent


def load_augmented_data(embed_model: str, augment_benchmarks: bool = True):
    """Load embeddings + labels, optionally augmented with benchmark data.

    Checks for pre-computed augmented data first (from prepare_augmented_data.py).
    Falls back to on-the-fly computation if unavailable.
    """
    # Priority 1: Pre-computed augmented data
    aug_dir = PROJ / "embeddings" / "deepsafe_v3_augmented"
    if aug_dir.exists() and (aug_dir / "train.npy").exists():
        logger.info("=" * 40)
        logger.info("Loading pre-computed augmented data...")
        train_emb = np.load(aug_dir / "train.npy")
        train_labels = np.load(aug_dir / "train_labels.npy")
        train_intent = np.load(aug_dir / "train_intent.npy")
        test_emb = np.load(aug_dir / "test.npy")
        test_labels = np.load(aug_dir / "test_labels.npy")
        logger.info(f"Augmented training: {train_emb.shape[0]} samples, dim={train_emb.shape[1]}")
        logger.info(f"  Safe: {(train_labels==0).sum()}, Unsafe: {(train_labels==1).sum()}")
        logger.info(f"  Intent: safe={(train_intent==0).sum()}, "
                    f"benign={(train_intent==1).sum()}, malicious={(train_intent==2).sum()}")
        logger.info(f"Test: {test_emb.shape[0]} samples")

        # Load counterfactual pairs from original embed dir
        cf_pairs = None
        cf_path = PROJ / "embeddings" / embed_model / "pairs.npy"
        if cf_path.exists():
            cf_emb = np.load(cf_path)
            half = len(cf_emb) // 2
            cf_pairs = (cf_emb[:half], cf_emb[half:])
            logger.info(f"Loaded {half} counterfactual pairs")
        return train_emb, train_labels, train_intent, cf_pairs, test_emb, test_labels

    # Priority 2: Original data with on-the-fly augmentation
    embed_dir = PROJ / "embeddings" / embed_model
    train_emb = np.load(embed_dir / "train.npy")
    train_labels = np.load(embed_dir / "train_labels.npy")
    test_emb = np.load(embed_dir / "test.npy")
    test_labels = np.load(embed_dir / "test_labels.npy")

    logger.info(f"Base training data: {train_emb.shape[0]} samples, dim={train_emb.shape[1]}")
    logger.info(f"Base test data: {test_emb.shape[0]} samples")

    # Load intent labels
    intent_path = PROJ / "data/processed/train_intent.parquet"
    if intent_path.exists():
        intent_df = pd.read_parquet(intent_path)
        train_intent = intent_df["intent"].values
        logger.info(f"Intent labels: safe={(train_intent==0).sum()}, "
                    f"benign={(train_intent==1).sum()}, malicious={(train_intent==2).sum()}")
    else:
        train_intent = train_labels.copy()

    # Load counterfactual pairs
    cf_path = embed_dir / "pairs.npy"
    cf_pairs = None
    if cf_path.exists():
        cf_emb = np.load(cf_path)
        half = len(cf_emb) // 2
        cf_pairs = (cf_emb[:half], cf_emb[half:])
        logger.info(f"Loaded {half} counterfactual pairs")

    # Augment with benchmark data
    if augment_benchmarks:
        bench_emb, bench_labels, bench_intent = augment_from_benchmarks(embed_model)
        if bench_emb is not None:
            train_emb = np.vstack([train_emb, bench_emb])
            train_labels = np.concatenate([train_labels, bench_labels])
            train_intent = np.concatenate([train_intent, bench_intent])
            logger.info(f"After augmentation: {train_emb.shape[0]} samples")

    return train_emb, train_labels, train_intent, cf_pairs, test_emb, test_labels


def augment_from_benchmarks(embed_model: str):
    """Load benchmark data, compute embeddings, return 70% for training.

    Returns (embeddings, binary_labels, intent_labels) or (None, None, None).
    """
    from datasets import load_from_disk, load_dataset
    import pyarrow.parquet as pq

    bench_dir = PROJ / "benchmark"
    encoder = None

    all_texts = []
    all_labels = []
    all_intents = []

    bench_configs = [
        ("WildGuardMix", "train"),  # HF dataset
        ("XSTest", "data.parquet"),  # parquet
        ("AegisAI-Content-Safety-1.0", None),
        ("AegisAI-Content-Safety-2.0", None),
        ("ToxicChat", None),
        ("BeaverTails", "train"),
    ]

    np.random.seed(42)

    for bench_name, subset in bench_configs:
        try:
            texts, labels = load_single_benchmark(bench_name, str(bench_dir), subset)
            if texts is None or len(texts) == 0:
                continue

            # Take 70% for training
            n_total = len(texts)
            n_train = int(n_total * 0.7)
            indices = np.random.permutation(n_total)

            train_texts = [texts[i] for i in indices[:n_train]]
            train_labels_list = [labels[i] for i in indices[:n_train]]

            all_texts.extend(train_texts)
            all_labels.extend(train_labels_list)
            # Default intent: 0 for safe, 2 for unsafe
            all_intents.extend([0 if l == 0 else 2 for l in train_labels_list])

            logger.info(f"  {bench_name}: added {n_train}/{n_total} samples to training")
        except Exception as e:
            logger.warning(f"  Skipping {bench_name}: {e}")

    if len(all_texts) == 0:
        return None, None, None

    # Compute embeddings
    logger.info(f"Computing embeddings for {len(all_texts)} benchmark samples...")
    all_embeddings = compute_embeddings(all_texts, embed_model)

    return (
        np.array(all_embeddings, dtype=np.float32),
        np.array(all_labels, dtype=np.int64),
        np.array(all_intents, dtype=np.int64),
    )


def load_single_benchmark(name: str, bench_dir: str, subset: str = None):
    """Load a single benchmark dataset. Returns (texts, labels) or (None, None)."""
    import pyarrow.parquet as pq
    from datasets import load_from_disk

    texts = []
    labels = []

    bench_path = os.path.join(bench_dir, name)

    # Try HF datasets format
    try:
        ds = load_from_disk(bench_path)
        if hasattr(ds, 'keys') and not isinstance(ds, (list, tuple)):
            # DatasetDict - pick first split
            key = list(ds.keys())[0]
            ds = ds[key]
        texts = ds["text"] if "text" in ds.column_names else ds["prompt"]
        if "label" in ds.column_names:
            labels = ds["label"]
        elif "labels" in ds.column_names:
            labels = ds["labels"]
        else:
            labels = [0] * len(texts)  # default safe
        return list(texts), list(int(l) for l in labels)
    except Exception:
        pass

    # Try parquet
    parquet_path = os.path.join(bench_path, "data.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(bench_path, "train", "data-00000-of-00001.parquet")
    if os.path.exists(parquet_path):
        df = pq.read_table(parquet_path).to_pandas()
        texts = df["text"].tolist() if "text" in df.columns else df["prompt"].tolist()
        labels = df["label"].tolist() if "label" in df.columns else [0] * len(texts)
        return texts, [int(l) for l in labels]

    return None, None


def compute_embeddings(texts: list, embed_model: str) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    from src.encode.qwen_encoder import QwenEncoder

    encoder_map = {
        "qwen3-embedding-0.6B": "Qwen3-Embedding-0___6B",
        "qwen3-embedding-4B": "Qwen3-Embedding-4B",
        "qwen3-embedding-8B": "Qwen3-Embedding-8B",
    }
    embed_dir = encoder_map.get(embed_model, "Qwen3-Embedding-0___6B")
    embed_path = PROJ / "pretrained" / "qwen" / embed_dir

    encoder = QwenEncoder(str(embed_path), device="cuda", batch_size=32)
    return encoder.encode(texts)


def train_projection_head(
    train_emb, train_intent, train_labels, cf_pairs, test_emb, test_labels, args, embed_model
):
    """Phase 1: Train the projection head."""
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split

    logger.info("=" * 60)
    logger.info("PHASE 1: Training DeepSafe v3 Projection Head")
    logger.info(f"  Input dim: {train_emb.shape[1]}, Samples: {train_emb.shape[0]}")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DeepSafeProjectionHeadV3(
        input_dim=train_emb.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        hyperbolic_dim=args.hyp_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = DeepSafeLoss(
        feature_dim=args.output_dim,
        temperature=args.temperature,
        alpha=args.alpha,
        lambda_ot=args.lambda_ot,
        lambda_proto=args.lambda_proto,
        lambda_decorr=args.lambda_decorr,
        gamma=args.gamma,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # Split (5% validation to keep N² contrastive matrix manageable)
    indices = np.arange(len(train_emb))
    tr_idx, val_idx = train_test_split(
        indices, test_size=0.05, stratify=train_intent, random_state=42
    )

    X_tr = train_emb[tr_idx]
    X_val = train_emb[val_idx]
    y_tr_intent = train_intent[tr_idx]
    y_val_intent = train_intent[val_idx]
    y_tr_binary = train_labels[tr_idx]
    y_val_binary = train_labels[val_idx]

    tr_dataset = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.LongTensor(y_tr_intent),
        torch.LongTensor(y_tr_binary),
    )
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation data stays on CPU; we sample a subset each epoch to avoid OOM
    # (contrastive loss is O(N²) — 3000² ≈ 9M elements ≈ 36 MB, safe)
    VAL_SAMPLE_SIZE = 3000
    X_val_cpu = torch.FloatTensor(X_val)
    y_val_intent_cpu = torch.LongTensor(y_val_intent)
    y_val_binary_cpu = torch.LongTensor(y_val_binary)

    # CF pairs
    cf_tensors = None
    if cf_pairs is not None:
        cf_tensors = (
            torch.FloatTensor(cf_pairs[0]).to(device),
            torch.FloatTensor(cf_pairs[1]).to(device),
        )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    t_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tr_loader:
            batch_x = batch[0].to(device)
            batch_intent = batch[1].to(device)
            batch_binary = batch[2].to(device)

            optimizer.zero_grad()
            features = model(batch_x)
            loss, components = criterion(
                features, batch_binary, batch_intent, return_components=True
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += components["total"]
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # CF interleaving
        if cf_tensors is not None:
            model.train()
            cf_i, cf_j = cf_tensors
            cf_bs = min(args.batch_size, len(cf_i))
            for cf_s in range(0, len(cf_i), cf_bs):
                cf_e = min(cf_s + cf_bs, len(cf_i))
                z_i = model(cf_i[cf_s:cf_e])
                z_j = model(cf_j[cf_s:cf_e])
                cf_loss = criterion.counterfactual(z_i, z_j) * args.gamma
                optimizer.zero_grad()
                cf_loss.backward()
                optimizer.step()

        scheduler.step()

        # Validation on random subset to avoid OOM (contrastive loss is O(N²))
        model.eval()
        with torch.no_grad():
            n_val = len(X_val_cpu)
            n_sample = min(VAL_SAMPLE_SIZE, n_val)
            val_idx = torch.randperm(n_val)[:n_sample]
            val_x = X_val_cpu[val_idx].to(device)
            val_y_b = y_val_binary_cpu[val_idx].to(device)
            val_y_i = y_val_intent_cpu[val_idx].to(device)
            val_feat = model(val_x)
            val_loss, val_comp = criterion(
                val_feat, val_y_b, val_y_i, return_components=True
            )

        lr = scheduler.get_last_lr()[0]
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1}/{args.epochs}: "
                f"train={epoch_loss:.4f}, val={val_comp['total']:.4f}, "
                f"hier={val_comp.get('hierarchical', 0):.4f}, lr={lr:.2e}"
            )

        if val_comp["total"] < best_loss:
            best_loss = val_comp["total"]
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch+1}, best={best_epoch+1}")
                break

        if device == "cuda":
            torch.cuda.empty_cache()

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    elapsed = time.time() - t_start
    logger.info(f"  Phase 1 complete: best_val={best_loss:.4f} at epoch {best_epoch+1}, {elapsed:.0f}s")

    # Save
    output_dir = PROJ / "models" / f"deepsafe_v3_{embed_model.split('-')[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    proj_path = output_dir / "projection_head.pkl"
    DeepSafeProjectionHeadV3Manager.save(model, str(proj_path))

    # Save config
    config = {
        "timestamp": datetime.now().isoformat(),
        "embed_model": embed_model,
        "input_dim": train_emb.shape[1],
        "output_dim": args.output_dim,
        "hidden_dim": args.hidden_dim,
        "hyp_dim": args.hyp_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": best_epoch + 1,
        "best_val_loss": best_loss,
        "train_time_s": elapsed,
        "n_train_samples": len(train_emb),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Saved to {output_dir}")

    # Project all embeddings
    logger.info("  Projecting train/test embeddings...")
    model.eval()
    train_proj = DeepSafeProjectionHeadV3Manager.project(
        model, train_emb, device=device, batch_size=1024
    )
    test_proj = DeepSafeProjectionHeadV3Manager.project(
        model, test_emb, device=device, batch_size=1024
    )

    np.save(output_dir / "train_projected.npy", train_proj)
    np.save(output_dir / "test_projected.npy", test_proj)
    logger.info(f"  Projected: train={train_proj.shape}, test={test_proj.shape}")

    return model, train_proj, train_labels, test_proj, test_labels


def train_classifier(train_proj, train_labels, test_proj, test_labels, args, embed_model):
    """Phase 2: Train neural classifier on projected embeddings."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Training Neural Classifier")
    logger.info("=" * 60)

    trainer = NeuralClassifierTrainer(
        input_dim=train_proj.shape[1],
        hidden_dims=args.cls_hidden_dims,
        dropout=args.cls_dropout,
        lr=args.cls_lr,
        weight_decay=args.cls_weight_decay,
        batch_size=args.cls_batch_size,
        max_epochs=args.cls_epochs,
        patience=args.cls_patience,
        mixup_alpha=args.mixup_alpha,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    results = trainer.train(train_proj, train_labels, test_proj, test_labels)

    # Save
    output_dir = PROJ / "models" / f"deepsafe_v3_{embed_model.split('-')[-1]}"
    trainer.save(str(output_dir / "classifier.pkl"))

    logger.info(f"  Phase 2 complete: acc={results['accuracy']:.4f}, "
                f"f1={results['f1_macro']:.4f}, auc={results['roc_auc']:.4f}")

    return trainer.model, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-model", default="qwen3-embedding-8B")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--hyp-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--lambda-ot", type=float, default=0.3)
    parser.add_argument("--lambda-proto", type=float, default=0.2)
    parser.add_argument("--lambda-decorr", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--cls-hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--cls-dropout", type=float, default=0.2)
    parser.add_argument("--cls-lr", type=float, default=1e-3)
    parser.add_argument("--cls-weight-decay", type=float, default=1e-5)
    parser.add_argument("--cls-batch-size", type=int, default=512)
    parser.add_argument("--cls-epochs", type=int, default=50)
    parser.add_argument("--cls-patience", type=int, default=10)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--skip-phase1", action="store_true")
    parser.add_argument("--skip-phase2", action="store_true")
    args = parser.parse_args()

    set_seed(42)

    logger.info("=" * 60)
    logger.info(f"DeepSafe v3 Training: {args.embed_model}")
    logger.info(f"  Hidden: {args.hidden_dim}, Output: {args.output_dim}, Hyp: {args.hyp_dim}")
    logger.info(f"  Augment: {not args.no_augment}")
    logger.info("=" * 60)

    # Load data
    train_emb, train_labels, train_intent, cf_pairs, test_emb, test_labels = \
        load_augmented_data(args.embed_model, augment_benchmarks=not args.no_augment)

    if not args.skip_phase1:
        _, train_proj, train_labels_used, test_proj, test_labels_used = train_projection_head(
            train_emb, train_intent, train_labels, cf_pairs, test_emb, test_labels, args, args.embed_model
        )
    else:
        output_dir = PROJ / "models" / f"deepsafe_v3_{args.embed_model.split('-')[-1]}"
        train_proj = np.load(output_dir / "train_projected.npy")
        test_proj = np.load(output_dir / "test_projected.npy")
        train_labels_used = train_labels
        test_labels_used = test_labels

    if not args.skip_phase2:
        classifier, cls_results = train_classifier(
            train_proj, train_labels_used, test_proj, test_labels_used, args, args.embed_model
        )

    logger.info("=" * 60)
    logger.info("DeepSafe v3 Training Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
