#!/usr/bin/env python3
"""
Figure 5: Side-by-side t-SNE — raw frozen embeddings (left) vs DeepSafe-projected (right).
Silhouette scores annotated on each panel. No "40×" text.
"""
import os, sys, pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "paper" / "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_projection_head_cpu():
    """Load the DeepSafe projection head and run it on CPU."""
    sys.path.insert(0, str(PROJ))
    from src.deepsafe.projection_head import DeepSafeProjectionHead

    proj_path = PROJ / "models" / "deepsafe_v3_8B" / "seed_42" / "projection_head.pkl"
    with open(proj_path, "rb") as f:
        saved = pickle.load(f)

    sd = saved["state_dict"]
    model = DeepSafeProjectionHead(
        input_dim=saved["input_dim"],
        output_dim=saved["output_dim"],
        hidden_dim=saved["hidden_dim"],
        curvature=saved.get("curvature", 1.0),
        hyperbolic_dim=saved.get("hyperbolic_dim", 128),
    )
    model.load_state_dict(sd)
    model.eval()
    return model


def compute_silhouette_2d(emb_2d, labels):
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    return silhouette_score(emb_2d, labels)


def main():
    print("Generating side-by-side t-SNE comparison...")

    # Load embeddings
    raw = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "test.npy")
    raw_labels = np.load(PROJ / "embeddings" / "qwen3-embedding-8B" / "test_labels.npy")

    # Subsample
    n_max = 3000
    rng = np.random.RandomState(42)
    idx = rng.choice(len(raw), min(n_max, len(raw)), replace=False)
    raw_sample = raw[idx]
    raw_labels_sample = raw_labels[idx]

    # Project through DeepSafe head on CPU
    print("  Projecting embeddings through DeepSafe projection head (CPU)...")
    model = load_projection_head_cpu()
    with torch.no_grad():
        X_t = torch.tensor(raw_sample, dtype=torch.float32)
        projected = model(X_t).cpu().numpy()
    print(f"  Projected: {projected.shape}")

    # t-SNE on both
    print("  Computing t-SNE for raw embeddings...")
    tsne_raw = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=6)
    raw_2d = tsne_raw.fit_transform(raw_sample)

    print("  Computing t-SNE for projected embeddings...")
    tsne_proj = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=6)
    proj_2d = tsne_proj.fit_transform(projected)

    # Silhouette scores
    sil_raw = compute_silhouette_2d(raw_2d, raw_labels_sample)
    sil_proj = compute_silhouette_2d(proj_2d, raw_labels_sample)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    colors = {0: "#2E86AB", 1: "#A23B72"}
    labels = {0: "Safe", 1: "Unsafe"}

    for ax, emb_2d, title, sil in [
        (ax1, raw_2d, "Original Frozen Embeddings\n(Qwen3-Embedding-8B)", sil_raw),
        (ax2, proj_2d, "DeepSafe-v3 Projected Embeddings\n(Hyperbolic + Neural Classifier)", sil_proj),
    ]:
        for label_val in sorted(np.unique(raw_labels_sample)):
            mask = raw_labels_sample == label_val
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=colors.get(label_val, "#999"),
                       label=labels.get(label_val, f"Class {label_val}"),
                       alpha=0.45, s=6, edgecolors="none", rasterized=True)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE 1", fontsize=9)
        ax.set_ylabel("t-SNE 2", fontsize=9)

        # Silhouette box
        ax.text(0.03, 0.97, f"Silhouette = {sil:.3f}",
                transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="gray", alpha=0.9))

    ax1.legend(loc="lower right", fontsize=8, markerscale=2.5)
    ax2.legend(loc="lower right", fontsize=8, markerscale=2.5)

    fig.savefig(OUT_DIR / "tsne_comparison.pdf", facecolor="white")
    fig.savefig(OUT_DIR / "tsne_comparison.png", facecolor="white", dpi=300)
    plt.close(fig)

    print(f"  Saved tsne_comparison.pdf")
    print(f"  Raw silhouette (2D t-SNE):     {sil_raw:.4f}")
    print(f"  Projected silhouette (2D t-SNE): {sil_proj:.4f}")


if __name__ == "__main__":
    main()
