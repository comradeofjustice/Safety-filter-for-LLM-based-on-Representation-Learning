#!/bin/bash
# Full DeepSafe v3 pipeline: train, evaluate ALL benchmarks, generate figures, update paper
set -euo pipefail

cd /root/autodl-tmp/llm-safety-classifier
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/pipeline_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "DEEPSAFE v3 FULL PIPELINE - $(date)"
echo "Logs: ${LOG_DIR}"
echo "========================================================================"

# Step 1: Train DeepSafe v3
echo ""
echo "STEP 1: Training DeepSafe v3..."
python3 scripts/24_train_deepsafe_v3.py \
    --embed-model qwen3-embedding-8B \
    --epochs 100 \
    --patience 20 \
    --batch-size 256 \
    --lr 5e-4 \
    --hidden-dim 768 \
    --output-dim 256 \
    --hyp-dim 128 \
    --temperature 0.07 \
    --alpha 2.0 \
    --lambda-ot 0.3 \
    --lambda-proto 0.2 \
    --lambda-decorr 0.005 \
    --gamma 0.5 \
    --cls-hidden-dims 256 128 64 \
    --cls-epochs 50 \
    --cls-patience 10 \
    2>&1 | tee "${LOG_DIR}/01_train.log"
echo "Training done: $(date)"

# Step 2: Evaluate on ALL 9 held-out benchmarks + re-evaluate SOTA
echo ""
echo "STEP 2: Evaluating on ALL held-out benchmarks..."
python3 scripts/25_evaluate_deepsafe_v3.py \
    --model-dir models/deepsafe_v3_8B \
    --embed-model qwen3-embedding-8B \
    --output valuation/deepsafe_v3_results.json \
    --max-sota-samples 2000 \
    2>&1 | tee "${LOG_DIR}/02_evaluate.log"
echo "Evaluation done: $(date)"

# Step 3: Generate figures
echo ""
echo "STEP 3: Generating paper figures..."
python3 scripts/28_generate_figures.py \
    --results valuation/deepsafe_v3_results.json \
    2>&1 | tee "${LOG_DIR}/03_figures.log"
echo "Figures done: $(date)"

# Step 4: Generate LaTeX tables
echo ""
echo "STEP 4: Generating paper tables..."
python3 scripts/27_update_paper_tables.py \
    --results valuation/deepsafe_v3_results.json \
    --output paper/tables_v3.tex \
    2>&1 | tee "${LOG_DIR}/04_tables.log"
echo "Tables done: $(date)"

echo ""
echo "========================================================================"
echo "PIPELINE COMPLETE: $(date)"
echo "Results: valuation/deepsafe_v3_results.json"
echo "Figures: paper/figures/fig_*.{pdf,png}"
echo "Tables: paper/tables_v3.tex"
echo "========================================================================"
