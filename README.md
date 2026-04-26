# LLM Safety Classifier

基于 Qwen3-Embedding 的大模型 prompt 安全/不安全二分类项目。

## 运行流程

### 1. 下载模型

```bash
source /etc/network_turbo  # 如需加速
bash scripts/00_download_models.sh
```

### 2. 收集数据

```bash
source /etc/network_turbo  # 如需加速 HuggingFace 数据集
python scripts/01_collect_data.py

# 只跑 XGuard 单数据集：
python scripts/01_collect_data.py --only xguard
```

### 3. 预处理数据

```bash
python scripts/02_preprocess.py
```

### 4. 编码 + 训练 + 评估

```bash
source /etc/network_turbo  # 如需
bash scripts/run_all.sh
```

## 目录结构

```
llm-safety-classifier/
  configs/config.yaml
  data/{raw,interim,processed}/
  embeddings/{qwen3-embedding-0.6B,4B,8B}/
  models/{qwen3-embedding-0.6B,4B,8B}/
  reports/{figures}/
  scripts/{00_download_models.sh, 01_collect_data.py, 02_preprocess.py,
           03_encode.py, 04_train.py, 05_evaluate.py, run_all.sh}
  src/{data/, encode/, classifiers/, utils/}
```

## 数据

- **主数据**: XGuard-Train-Open-200K (200K)
- **辅助数据**: BeaverTails, ToxicChat, Do-Not-Answer, Aegis (网络可达时自动融合)
- **Schema**: `text, label (0=safe/1=unsafe), source, meta`
- **切分**: train:test = 8:2, stratified, random_state=42

## 模型

| 模型 | 维度 | 下载位置 |
|------|------|----------|
| qwen3-embedding-0.6B | 1024 | pretrained/qwen3-embedding-0.6B |
| qwen3-embedding-4B | 2560 | pretrained/qwen3-embedding-4B |
| qwen3-embedding-8B | 4096 | pretrained/qwen3-embedding-8B |

## 分类器

- Logistic Regression
- MLP (2-layer, 512→128→2)
- Linear SVM
- RBF SVM (高维时自动 PCA→256)

## 输出

- `reports/metrics.csv` — 所有模型×分类器的指标汇总
- `reports/data_stats.csv` — 各数据集 safe/unsafe 计数
- `reports/figures/` — 混淆矩阵图
- `models/{model_size}/{classifier}.pkl` — 训练好的分类器
