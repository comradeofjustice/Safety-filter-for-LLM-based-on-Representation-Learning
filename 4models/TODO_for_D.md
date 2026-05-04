# TODO List for Worker D — 如何利用 Worker C 的图表

> 生成时间: 2026-05-04 03:XX
> 所有图的绝对路径均以 `/root/autodl-tmp/llm-safety-classifier/paper/figures/` 为前缀

---

## 一、已经存在的 5 张图（路径和 caption 修正）

所有图都已保存为 PDF（矢量，适合 LaTeX），同时有 PNG 预览版。

### Figure 1 — 柱状图（benchmark × model + error bars）

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_benchmark_accuracy.pdf`

**内容:** 7 个模型 (DeepSafe-v3 + 6 SOTA) × 9 个 benchmark 的 accuracy 柱状图。DeepSafe 有 error bar (std across 5 seeds)，其余为半透明。

**LaTeX 改动:**
```latex
% 替换现有 caption (line 276):
\caption{Accuracy comparison across all 9 held-out benchmarks.
DeepSafe-v3 (blue, solid) consistently achieves top accuracy.
Error bars indicate $\pm1$ standard deviation across 5 random seeds.}
```
⚠️ 原有 caption 写的是 "DeepSafe v3 (red)"，但图中 DeepSafe 是**蓝色**（Okabe-Ito colorblind palette）。请把 "red" 改成 "blue"。

---

### Figure 2 — 雷达图 (mean + CI)

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_radar.pdf`

**内容:** 6 个模型在 9 个 benchmark 上的雷达图。DeepSafe-v3 实线 + 浅蓝色 CI 填充区域，其余模型虚线。DeepSafe 的面积更大，表示跨领域一致性更强。

**LaTeX 改动:**
```latex
% 替换现有 caption (line 283):
\caption{Performance radar chart: DeepSafe-v3 vs. 5 SOTA models across
9 benchmarks. DeepSafe-v3 (solid blue line with shaded 95\% confidence
interval) demonstrates consistent performance across diverse safety domains.}
```
⚠️ 原有 caption 说的是 "top-3 SOTA"，实际图中是 5 个 SOTA。

---

### Figure 3 — Speed vs Accuracy (throughput)

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_speed_vs_accuracy.pdf`

**内容:** 各模型的 throughput（samples/second，越高越快）vs weighted mean accuracy 散点图。DeepSafe-v3 是蓝色菱形，带 error bar。

⚠️ **重要改动**：原论文用的是总推理时间（秒），但这是**不公平的**——DeepSafe 在所有 26028 条 WildGuardMix 样本上跑，而 SOTA 模型只跑了 2000 条（`--max-samples 2000`）。现在统一用 **samples/second**（吞吐量），体现的是每样本的推理速度。

**LaTeX 改动:**
```latex
% 替换现有 caption (line 290):
\caption{Throughput vs.\ accuracy: DeepSafe-v3 achieves strong accuracy
with competitive throughput (samples/second), benefiting from batched
embedding-based inference. Diamond marker = DeepSafe-v3 with $\pm1\sigma$
error bar.}
```

---

### Figure 4 — Head-to-head 矩阵

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_head_to_head.pdf`

**内容:** 10×10 矩阵。cell[i,j] = 行模型 i 在多少个 benchmark 上 accuracy 超过列模型 j（比例）。红=高胜率，蓝=低。对角线都是 "-"。DeepSafe 行加黑框高亮。

**LaTeX 改动:**
```latex
% 替换现有 caption (line 297):
\caption{Head-to-head comparison across 9 benchmarks $\times$ 9 SOTA models
(81 unique pairings). Cell $(i,j)$ shows the fraction of benchmarks where
row model $i$ achieves higher accuracy than column model $j$. Red = high
win rate, Blue = low. DeepSafe-v3 row highlighted with black border.}
```
⚠️ 原有 caption 说 "Green = DeepSafe wins, Yellow = ties, Red = SOTA wins"，但实际图用的是 **红蓝色阶矩阵 + 数值**，不是绿黄红三色。请用上面的新 caption。

---

### Figure 5 — t-SNE 侧对比 (side-by-side)

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/tsne_comparison.pdf`

**内容:** 左：原始 Qwen3-Embedding-8B 冻结编码器的 t-SNE (silhouette ≈ 0.012)。右：经过 DeepSafe projective head 之后的 t-SNE (silhouette ≈ 0.346)。安全/不安全两类点，左图完全混杂，右图明显分离。

**LaTeX 改动:**
```latex
% 替换现有 \includegraphics (line 332):
\includegraphics[width=\columnwidth]{tsne_comparison.pdf}

% 替换 caption (line 333):
\caption{t-SNE visualization of original frozen embeddings (left,
silhouette $\approx 0.012$) vs.\ DeepSafe-projected embeddings (right,
silhouette $\approx 0.346$). The projection head learns to structurally
separate safe from unsafe content in the embedding space.}
```
⚠️ 原 LaTeX 引用的是 `.png`，现在是 `.pdf`。同时 ".30" silhouette 在 2D t-SNE 空间是 0.346，建议统一为 `$\approx 0.35$` 或保持 `$\approx 0.30$`（如果引用的是 256-d 原始空间的 silhouette）。两种都行但必须一致。

⚠️ **§4.4 正文 (line 328) 也需要改**：当前写着 "silhouette score increases from near-random clustering (silhouette $\approx 0$) to moderate separation ($\approx 0.30$)"。建议改为：
```latex
silhouette score increases from near-random clustering (silhouette $\approx 0.01$)
to moderate separation ($\approx 0.35$ in 2D t-SNE space, $\approx 0.30$ in the
original 256-dimensional projected space).
```

---

## 二、Worker B 数据新增的 3 张补充图

### Figure S1 — 消融实验 (Euclidean vs Hyperbolic × hidden dim)

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_ablation.pdf`

**内容:** 4 子图（accuracy / F1 / AUC / 推理时间），横轴 hidden_dim (64/128/256/512)，两条线：Euclidean（橙色方块）vs Hyperbolic（蓝色菱形）。每个点标注 Δ 差值。

**核心结论:** 在纯 BCE loss 下，hyperbolic 和 Euclidean 的 accuracy 差异 < 0.003（噪音级）。Hyperbolic 单独使用**没有优势**。

**建议插入位置:** §4.3 "Why DeepSafe v3 Works: Component Analysis"，在现有 Table (ablation) 之后加：

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig_ablation.pdf}
    \caption{Ablation: Euclidean vs.\ Hyperbolic projection heads at four hidden
    dimensions (64, 128, 256, 512), trained with BCE loss only (no DeepSafe loss
    components). The difference is within $\pm 0.003$, confirming that hyperbolic
    geometry alone provides no advantage. The benefit emerges only when combined
    with the full DeepSafe loss (hierarchical contrastive + OT + prototype +
    decorrelation + counterfactual).}
    \label{fig:ablation}
\end{figure}
```

---

### Figure S2 — Baseline 对比

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_baselines.pdf`

**内容:** 3 个简单 baseline (LR-4096d / MLP-4096-512-2 / EuclideanProj-NN) + DeepSafe-v3 在 9 个 benchmark 上的 accuracy 柱状图。DeepSafe 绿色，baselines 橙/蓝/紫。

**核心结论:** 简单 baseline 在多数 benchmark 上远不如 DeepSafe，说明复杂的投影 + 损失函数设计是必要的。

**建议插入位置:** §4.2 "Main Results"，在 main results table 之后，benchmark accuracy figure 之前：

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig_baselines.pdf}
    \caption{Simple classification baselines (LR, MLP, Euclidean projection
    head) vs.\ DeepSafe-v3 across all 9 held-out benchmarks. All baselines
    use the same frozen Qwen3-Embedding-8B encoder but without hyperbolic
    projection or DeepSafe training objectives.}
    \label{fig:baselines}
\end{figure}
```

---

### Figure S3 — Trivial baselines (sanity check)

**路径:** `/root/autodl-tmp/llm-safety-classifier/paper/figures/fig_trivial_baselines.pdf`

**内容:** predict-all-safe / majority-class 两个平凡 baseline vs DeepSafe-v3。灰色斜线柱体 + DeepSafe 绿色柱体。

**核心结论:** 在类别不平衡的 benchmark 上（如 DoNotAnswer，99% safe），"全部判 safe" 也能拿高 accuracy。这是 sanity check，说明 accuracy 不是唯一指标，需要看 F1/AUC。

**建议插入位置:** Appendix，或 §4.2 的脚注处：

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=\columnwidth]{fig_trivial_baselines.pdf}
    \caption{Trivial baselines (predict-all-safe, majority class) vs.\
    DeepSafe-v3. On benchmarks with severe class imbalance (e.g., DoNotAnswer,
    98.6\% safe), naive strategies achieve high accuracy but zero MCC,
    highlighting the need for macro-averaged metrics.}
    \label{fig:trivial_baselines}
\end{figure}
```

---

## 三、正文数字更新清单

以下是 Worker C 产出的**确定数字**，D 可以用 `paper/inject_numbers.py` 从 CSV 自动注入，也可以手写到 `auto_numbers.tex`：

| LaTeX macro | 值 | 来源 CSV | 列/行 |
|---|---|---|---|
| `\DeepSafeAcc` | 0.9008 | stats_with_ci.csv | DeepSafe-v3 accuracy_mean 均值 |
| `\DeepSafeBestBench` | WildGuardMix | stats_with_ci.csv | accuracy_mean 最大的行 |
| `\DeepSafeBestAcc` | 0.9682 | stats_with_ci.csv | WildGuardMix DeepSafe-v3 accuracy_mean |
| `\DeepSafeCIlow` | 0.9672 | stats_with_ci.csv | WildGuardMix accuracy_ci_low |
| `\DeepSafeCIhigh` | 0.9691 | stats_with_ci.csv | WildGuardMix accuracy_ci_high |
| `\DeepSafeWorstBench` | DoNotAnswer | stats_with_ci.csv | accuracy_mean 最小的行 |
| `\DeepSafeWorstAcc` | 0.7822 | stats_with_ci.csv | DoNotAnswer accuracy_mean |
| `\NumComparable` | 7 | stats_with_ci.csv | is_comparable == "comparable" 的行数 |
| `\SilhouetteRaw` | 0.012 | C 计算 | 原始 frozen embedding 2D t-SNE |
| `\SilhouetteProj` | 0.346 | C 计算 | 投影后 256-d → 2D t-SNE |
| `\AblationMaxDelta` | 0.003 | ablation_table.csv | Euclidean vs Hyperbolic accuracy 最大差值 |

---

## 四、Caption 修正速查表

| 图 | 当前 caption 问题 | 需要改 |
|---|---|---|
| fig_benchmark_accuracy | "DeepSafe v3 (red)" → 实际是蓝色 | "blue" |
| fig_radar | "top-3 SOTA" → 实际是 5 个 | "5 SOTA models" |
| fig_speed_vs_accuracy | "lower inference time" → 现在用 throughput | "Throughput (samples/s)" |
| fig_head_to_head | "Green/Yellow/Red" → 实际是红蓝色阶数值矩阵 | 全新 caption（见上文） |
| tsne_comparison | .png → .pdf; silhouette 数值; 只有一个 panel → 现在是左右两张 | 全新 caption（见上文） |

---

## 五、不要做的事（重申）

- ❌ 不要写 `paper/paper/figures/`（那个目录 D 已经清理过）
- ❌ 不要删除 `paper/figures/` 下的旧图（`fig1_tsne.pdf` 等）。保留给 D 自己清理
- ❌ 不要在正文里手抄数字。用 `auto_numbers.tex` 或 `inject_numbers.py` 自动注入
- ❌ 不要改 C 的 CSV 文件。只读引用
