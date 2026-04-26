#!/bin/bash
# Wait for 8B embeddings, then train all classifiers
set -e
cd /root/autodl-tmp/llm-safety-classifier
PYTHON=/root/miniconda3/envs/llm-safety/bin/python
EMBED_MODEL=qwen3-embedding-8B
LOG_DIR=logs

echo "[$(date)] Waiting for embeddings to be ready..."
while [ ! -f "embeddings/$EMBED_MODEL/test.npy" ]; do sleep 30; done
echo "[$(date)] Embeddings found. Starting training..."

for clf in logistic linear_svm mlp rbf_svm; do
    echo "[$(date)] === Training $clf ==="
    $PYTHON scripts/04_train.py --embed-model $EMBED_MODEL --classifier $clf \
        2>&1 | tee $LOG_DIR/train_8b_${clf}.log
    echo "[$(date)] === $clf done ==="
done

echo "[$(date)] All classifiers trained. Running evaluation..."
$PYTHON scripts/05_evaluate.py \
    --embed-model $EMBED_MODEL \
    --classifiers logistic mlp linear_svm rbf_svm \
    2>&1 | tee $LOG_DIR/evaluate_8b.log

echo "[$(date)] Done!"
