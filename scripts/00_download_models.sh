#!/usr/bin/env bash
# scripts/00_download_models.sh - Download Qwen3 Embedding models in parallel using tmux
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p pretrained

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "========================================"
echo "Downloading Qwen3 Embedding Models"
echo "========================================"

# Kill existing sessions if they exist
for sess in dl_06b dl_4b dl_8b; do
    tmux kill-session -t "$sess" 2>/dev/null || true
done

# Download 0.6B
echo "Starting download: qwen3-embedding-0.6B"
tmux new-session -d -s dl_06b \
    "huggingface-cli download Qwen/Qwen3-Embedding-0.6B \
     --local-dir ./pretrained/qwen3-embedding-0.6B \
     --local-dir-use-symlinks False && \
     echo 'Download complete: 0.6B' && \
     tmux kill-session -t dl_06b"

# Download 4B
echo "Starting download: qwen3-embedding-4B"
tmux new-session -d -s dl_4b \
    "huggingface-cli download Qwen/Qwen3-Embedding-4B \
     --local-dir ./pretrained/qwen3-embedding-4B \
     --local-dir-use-symlinks False && \
     echo 'Download complete: 4B' && \
     tmux kill-session -t dl_4b"

# Download 8B
echo "Starting download: qwen3-embedding-8B"
tmux new-session -d -s dl_8b \
    "huggingface-cli download Qwen/Qwen3-Embedding-8B \
     --local-dir ./pretrained/qwen3-embedding-8B \
     --local-dir-use-symlinks False && \
     echo 'Download complete: 8B' && \
     tmux kill-session -t dl_8b"

echo ""
echo "========================================"
echo "Downloads started in background tmux sessions"
echo "========================================"
echo "View progress with:"
echo "  tmux attach -t dl_06b   # 0.6B model"
echo "  tmux attach -t dl_4b    # 4B model"
echo "  tmux attach -t dl_8b    # 8B model"
echo ""
echo "Active sessions:"
tmux ls
