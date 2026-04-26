#!/usr/bin/env python
import os
import sys

os.chdir('/root/autodl-tmp/llm-safety-classifier')

from modelscope import snapshot_download

cache_dir = 'pretrained'
print('Downloading Qwen3-Embedding-8B...')
model_dir = snapshot_download('qwen/Qwen3-Embedding-8B', cache_dir=cache_dir)
print(f'Downloaded to: {model_dir}')

target = os.path.join(cache_dir, 'qwen3-embedding-8B')
if not os.path.exists(target):
    os.symlink(model_dir, target)
    print(f'Created symlink: {target} -> {model_dir}')

import subprocess
result = subprocess.run(['du', '-sh', model_dir], capture_output=True, text=True)
print(f'Model size: {result.stdout.strip()}')
print('Done!')
