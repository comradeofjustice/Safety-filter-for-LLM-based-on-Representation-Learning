import os
import logging
from modelscope import snapshot_download

# Configure logging
log_dir = '/root/CSY/Safety-filter-for-LLM-based-on-Representation-Learning/0.6B'
log_file = os.path.join(log_dir, 'download.log')

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Target directory
target_dir = os.path.join(log_dir, 'embedding_model')

logger.info(f"Starting download to: {target_dir}")

# Download with progress bar
model_dir = snapshot_download(
    'Qwen/Qwen3-Embedding-0.6B',
    cache_dir=target_dir
)

logger.info(f"Download completed. Model saved to: {model_dir}")
