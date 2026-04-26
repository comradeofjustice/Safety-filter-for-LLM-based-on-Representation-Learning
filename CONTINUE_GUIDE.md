# 项目继续指引 - LLM Safety Classifier

下次对话时，直接将以下内容发给新助手：

---

## 项目背景
- **项目名**: LLM Safety Classifier（基于 Qwen3-Embedding 的文本安全分类）
- **代码位置**: `/root/autodl-tmp/llm-safety-classifier/`
- **conda 环境**: `llm-safety` (Python 3.10)
- **数据盘**: `/root/autodl-tmp/` (200G, 可读写)
- **系统盘**: `/` (30G overlay, 无法扩容)

## 已完成
1. ✅ 项目结构搭建完毕 (src/, scripts/, configs/, README.md)
2. ✅ conda 环境已创建，依赖已安装
3. ✅ XGuard 数据集已下载并预处理:
   - `data/processed/train.parquet` (155,536 条)
   - `data/processed/test.parquet` (38,885 条)
   - `reports/data_stats.csv` 已生成
4. ✅ 0.6B 和 4B 模型已下载:
   - `pretrained/qwen3-embedding-0.6B/`
   - `pretrained/qwen3-embedding-4B/`
5. ✅ 硬编码路径已替换 (`/root/llm-safety-classifier` → `/root/autodl-tmp/llm-safety-classifier`)

## 未完成（需要继续）

### 第一步：下载 8B 模型
```bash
cd /root/autodl-tmp/llm-safety-classifier
/root/miniconda3/envs/llm-safety/bin/python download_8b.py
```
其中 `download_8b.py` 内容：
```python
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
```

### 第二步：一键跑完全流程
```bash
cd /root/autodl-tmp/llm-safety-classifier
bash scripts/run_all.sh
```

## ⚠️ 重要注意事项
1. **Shell CWD 问题**: 如果新助手启动时报 `chdir(2) failed`，说明 shell 工作目录指向了不存在的旧路径 `/root/llm-safety-classifier`。解决方法：
   ```bash
   ln -s /root/autodl-tmp/llm-safety-classifier /root/llm-safety-classifier
   ```
   或者让助手在每次命令中显式指定目录。

2. **不要动系统盘**: 系统盘 `/` 只有 30G，大文件都放在 `/root/autodl-tmp/`。

3. **不要移动项目**: 项目已经稳定在 `/root/autodl-tmp/llm-safety-classifier/`，不要再移动。

---

## 快速粘贴版本（复制以下内容给新助手）：

```
项目路径: /root/autodl-tmp/llm-safety-classifier/
conda 环境: llm-safety (Python 3.10)

已完成: 项目搭建、数据处理(train/test parquet)、0.6B和4B模型下载、路径替换

需要继续:
1. 下载 8B 模型: cd /root/autodl-tmp/llm-safety-classifier && /root/miniconda3/envs/llm-safety/bin/python download_8b.py
2. 跑全流程: bash scripts/run_all.sh

注意: 如果报 chdir 错误，先执行: ln -s /root/autodl-tmp/llm-safety-classifier /root/llm-safety-classifier
```
