#!/usr/bin/env bash
# scripts/run_all.sh - Run encoding (parallel), training, and evaluation
set -e

# Use conda environment Python
PYTHON="/root/miniconda3/envs/llm-safety/bin/python"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

EMBED_MODELS=(
  "qwen3-embedding-0.6B"
)

CLASSIFIERS=("logistic" "mlp" "linear_svm" "rbf_svm")

echo "========================================"
echo "LLM Safety Classifier Pipeline"
echo "========================================"

# Step 1: Encode with 0.6B on GPU
echo ""
echo "========================================"
echo "Step 03: Encoding (qwen3-embedding-0.6B on GPU)"
echo "========================================"
$PYTHON scripts/03_encode.py --embed-model "qwen3-embedding-0.6B" --device "cuda:0"
echo "  Encoding complete!"
echo ""

# Step 2: Train and evaluate
echo ""
echo "========================================"
echo "Processing: qwen3-embedding-0.6B"
echo "========================================"

# Train each classifier
echo ""
echo "--- Step 04: Training ---"
for clf in "${CLASSIFIERS[@]}"; do
  echo ""
  echo "  Training: $clf"
  $PYTHON scripts/04_train.py --embed-model "qwen3-embedding-0.6B" --classifier "$clf"
done

# Evaluate
echo ""
echo "--- Step 05: Evaluation ---"
$PYTHON scripts/05_evaluate.py --embed-model "qwen3-embedding-0.6B" --classifiers "${CLASSIFIERS[@]}"

echo ""
echo "========================================"
echo "Pipeline complete!"
echo "Results in reports/metrics.csv"
echo "========================================"
