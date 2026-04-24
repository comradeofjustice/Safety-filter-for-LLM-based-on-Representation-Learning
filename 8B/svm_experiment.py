#!/usr/bin/env python3
"""
SVM 安全分类器完整实验流程 (sklearn CPU 版 - 无 K 折 CV)
数据编码 → 标准化 → 划分数据集 → 逐组超参数训练 → 评估 → 保存权重和结果

使用 LinearSVC 进行 CPU 训练，支持多核并行
对于高维 embedding (4096d)，线性 SVM 是 NeurIPS 标准做法

Usage:
    python svm_experiment.py --training_ratio 1.0 --optimizer grid
    python svm_experiment.py --training_ratio 1.0 --optimizer bayesian
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ============== 配置路径 ==============
# 以脚本所在目录为基准
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录（存放 embeddings, scaler 等预处理数据）
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# 日志目录
LOG_DIR = os.path.join(SCRIPT_DIR, 'log')
os.makedirs(LOG_DIR, exist_ok=True)

# 权重目录
WEIGHT_DIR = os.path.join(SCRIPT_DIR, 'weight')
os.makedirs(WEIGHT_DIR, exist_ok=True)

# 结果目录
RESULT_DIR = os.path.join(SCRIPT_DIR, 'result')
os.makedirs(RESULT_DIR, exist_ok=True)

# 模型路径（Qwen8B Embedding，与脚本同级）
MODEL_PATH = os.path.join(SCRIPT_DIR, 'embedding_model')

# 当前输出目录（每个超参数组合的子目录）
OUTPUT_DIR = RESULT_DIR

# 数据路径
DATA_PATH = '/root/CSY/SafestAI/dataset/merged_safety_dataset.csv'

# 默认参数
RANDOM_STATE = 42
DEFAULT_TRAINING_RATIO = 1.0
DEFAULT_OPTIMIZER = 'grid'

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'experiment.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SVM 安全分类器实验')
    parser.add_argument('--training_ratio', type=float, default=DEFAULT_TRAINING_RATIO,
                        help=f'训练数据比例 (0.0-1.0], 默认 {DEFAULT_TRAINING_RATIO}')
    parser.add_argument('--optimizer', type=str, default=DEFAULT_OPTIMIZER,
                        choices=['grid', 'bayesian'],
                        help=f'优化器类型：grid 或 bayesian, 默认 {DEFAULT_OPTIMIZER}')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='实验名称前缀，用于输出文件夹命名')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE,
                        help=f'随机种子，默认 {RANDOM_STATE}')
    parser.add_argument('--bayesian_iterations', type=int, default=50,
                        help='贝叶斯优化迭代次数，默认 50')
    return parser.parse_args()


def log_section(title):
    """记录章节分隔线"""
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def load_and_prepare_data(training_ratio=1.0, random_state=RANDOM_STATE):
    """加载并准备数据（优先使用缓存）"""
    log_section("1. 数据加载与准备")

    sampled_data_path = os.path.join(DATA_DIR, 'sampled_data.csv')

    # 检查缓存
    if os.path.exists(sampled_data_path):
        logger.info(f"发现缓存数据：{sampled_data_path}，直接加载")
        df_sample = pd.read_csv(sampled_data_path)
        logger.info(f"缓存数据大小：{len(df_sample)}")
    else:
        logger.info(f"加载数据集：{DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"原始数据集大小：{len(df)}")

        if training_ratio < 1.0:
            n_samples = int(len(df) * training_ratio)
            df_sample = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
            logger.info(f"训练比例：{training_ratio:.2%}, 采样数量：{n_samples}")
        else:
            df_sample = df.reset_index(drop=True)
            logger.info(f"全量训练：使用全部 {len(df_sample)} 个样本")

        # 保存缓存
        df_sample.to_csv(sampled_data_path, index=False)
        logger.info(f"数据已缓存：{sampled_data_path}")

    logger.info(f"实际使用数据集大小：{len(df_sample)}")

    label_dist = df_sample['label'].value_counts().to_dict()
    logger.info(f"标签分布：{label_dist}")

    y = (df_sample['label'] == 'unsafe').astype(int).values
    logger.info(f"标签编码：safe=0, unsafe=1")
    logger.info(f"unsafe 比例：{y.mean():.4f}")

    return df_sample, y


def load_embeddings(df, y):
    """加载 embeddings（优先使用缓存，否则编码）"""
    log_section("2. 文本编码")

    embeddings_path = os.path.join(DATA_DIR, 'embeddings.npy')

    # 检查缓存
    if os.path.exists(embeddings_path):
        logger.info(f"发现缓存 embeddings：{embeddings_path}，直接加载")
        embeddings = np.load(embeddings_path)
        nan_count = np.isnan(embeddings).sum()
        if nan_count > 0:
            logger.warning(f"缓存 embeddings 包含 {nan_count} 个 NaN，用 0 填充")
            embeddings = np.nan_to_num(embeddings, nan=0.0)
        logger.info(f"Embeddings 形状：{embeddings.shape}")
        return embeddings

    # 缓存不存在，执行编码
    logger.info("缓存不存在，开始编码...")
    return encode_texts_local(df, y)


def encode_texts_local(df, y):
    """使用 Transformers 加载本地模型编码文本"""
    import torch
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm

    logger.info("加载 Qwen Embedding 模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    logger.info("模型加载完成，开始编码")

    texts = df['content'].tolist()
    all_embeddings = []
    batch_size = 64

    logger.info(f"开始编码 {len(texts)} 个样本 (batch_size={batch_size})...")
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            batch_texts = [t[:512] if len(t) > 512 else t for t in batch_texts]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)

            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)

            if (i // batch_size + 1) % 50 == 0:
                progress = (i + len(batch_texts)) / len(texts) * 100
                elapsed = time.time() - start_time
                logger.info(f"  进度：{progress:.1f}%, 耗时：{elapsed:.1f}s")

    embeddings = np.array(all_embeddings)
    elapsed = time.time() - start_time
    logger.info(f"编码完成！形状：{embeddings.shape}, 总耗时：{elapsed:.1f}s")

    # 保存缓存
    np.save(embeddings_path, embeddings)
    logger.info(f"Embeddings 已缓存：{embeddings_path}")

    return embeddings


def load_or_prepare_split(X, y, random_state=RANDOM_STATE):
    """加载或准备标准化数据（优先使用缓存）"""
    log_section("3. 数据标准化与划分")

    x_train_path = os.path.join(DATA_DIR, 'X_train_scaled.npy')

    # 检查缓存
    if os.path.exists(x_train_path):
        logger.info(f"发现缓存的划分数据，直接加载")
        X_train_scaled = np.load(os.path.join(DATA_DIR, 'X_train_scaled.npy'))
        X_test_scaled = np.load(os.path.join(DATA_DIR, 'X_test_scaled.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"训练集大小：{len(X_train_scaled)}, 测试集大小：{len(X_test_scaled)}")
        logger.info(f"训练集 unsafe 比例：{y_train.mean():.4f}")
        logger.info(f"测试集 unsafe 比例：{y_test.mean():.4f}")

        # 检查并修复 NaN
        train_nan = np.isnan(X_train_scaled).sum()
        test_nan = np.isnan(X_test_scaled).sum()
        if train_nan > 0 or test_nan > 0:
            logger.warning(f"缓存数据包含 NaN (train: {train_nan}, test: {test_nan})，重新生成...")
            os.remove(os.path.join(DATA_DIR, 'X_train_scaled.npy'))
            os.remove(os.path.join(DATA_DIR, 'X_test_scaled.npy'))
            os.remove(os.path.join(DATA_DIR, 'y_train.npy'))
            os.remove(os.path.join(DATA_DIR, 'y_test.npy'))
            os.remove(os.path.join(DATA_DIR, 'scaler.pkl'))
            return load_or_prepare_split(X, y, random_state)
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    # 缓存不存在，重新处理
    logger.info("缓存不存在，重新标准化和划分...")

    # 处理 NaN：用 0 填充（embedding 维度）
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        logger.warning(f"原始数据包含 {nan_count} 个 NaN，用 0 填充")
        X = np.nan_to_num(X, nan=0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    logger.info(f"训练集大小：{len(X_train)}, 测试集大小：{len(X_test)}")
    logger.info(f"训练集 unsafe 比例：{y_train.mean():.4f}")
    logger.info(f"测试集 unsafe 比例：{y_test.mean():.4f}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 标准化后再次检查 NaN
    train_nan = np.isnan(X_train_scaled).sum()
    test_nan = np.isnan(X_test_scaled).sum()
    if train_nan > 0 or test_nan > 0:
        logger.warning(f"标准化后仍有 NaN (train: {train_nan}, test: {test_nan})，用 0 填充")
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

    logger.info("标准化完成 (StandardScaler)")

    # 保存缓存
    np.save(os.path.join(DATA_DIR, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(DATA_DIR, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'y_test.npy'), y_test)
    with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"数据已缓存到 {DATA_DIR}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def make_param_name(params):
    """生成超参数组合的文件夹名称"""
    c = params['C']
    c_str = f"{c:.6f}".rstrip('0').rstrip('.') if isinstance(c, float) else str(c)
    return f"C{c_str}"


def train_in_tmux(X_train_path, X_test_path, y_train_path, y_test_path, params, result_dir, weight_dir, log_dir, data_dir, random_state=42):
    """在 tmux 会话中训练单个超参数组合"""
    param_name = make_param_name(params)
    tmux_session = f"{param_name}_train"
    combo_log = os.path.join(log_dir, f"{param_name}.log")

    # 如果 tmux 会话已存在，先清理
    subprocess.run(['tmux', 'kill-session', '-t', tmux_session], capture_output=True)

    logger.info(f"创建 tmux 会话: {tmux_session}")
    logger.info(f"日志输出: {combo_log}")

    # 构建训练命令
    train_cmd = (
        f"python {os.path.join(SCRIPT_DIR, 'train_single.py')} "
        f"--c {params['C']} "
        f"--data_dir {data_dir} "
        f"--result_dir {result_dir} "
        f"--weight_dir {weight_dir} "
        f"--log_dir {log_dir} "
        f"--random_state {random_state}"
    )

    # 创建 tmux  detached session 并运行训练命令
    # 使用 tee 将输出同时写入终端和日志文件
    tmux_cmd = [
        'tmux', 'new-session', '-d', '-s', tmux_session,
        f'{train_cmd} 2>&1 | tee {combo_log}'
    ]

    subprocess.run(tmux_cmd, check=True)
    logger.info(f"tmux 会话 '{tmux_session}' 已启动，正在训练...")

    # 等待训练完成（轮询 tmux 会话状态）
    while True:
        time.sleep(5)
        check = subprocess.run(['tmux', 'has-session', '-t', tmux_session], capture_output=True)
        if check.returncode != 0:
            break

    logger.info(f"tmux 会话 '{tmux_session}' 已结束")

    # 读取结果
    results_path = os.path.join(result_dir, 'results.json')
    if not os.path.exists(results_path):
        logger.error(f"训练失败，未找到结果文件: {results_path}")
        return None

    with open(results_path, 'r') as f:
        eval_results = json.load(f)

    logger.info(f"训练完成 | F1: {eval_results['f1_score']:.4f}, Accuracy: {eval_results['accuracy']:.4f}")
    return eval_results


def plot_evaluation(y_test, y_pred, y_proba, cm, fpr, tpr, roc_auc, precision, recall, pr_auc, save_dir):
    """绘制评估图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 混淆矩阵
    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['safe', 'unsafe'], yticklabels=['safe', 'unsafe'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # 2. ROC 曲线
    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 3. PR 曲线
    ax = axes[0, 2]
    ax.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.4f})', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # 4. 预测概率分布
    ax = axes[1, 0]
    safe_proba = y_proba[y_test == 0]
    unsafe_proba = y_proba[y_test == 1]
    ax.hist(safe_proba, bins=50, alpha=0.7, label='safe', color='blue')
    ax.hist(unsafe_proba, bins=50, alpha=0.7, label='unsafe', color='red')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted Probability (unsafe)')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 空位
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'See support vector plot',
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.set_title('Support Vector Analysis')
    ax.axis('off')

    # 6. 指标对比
    ax = axes[1, 2]
    metrics = ['Accuracy', 'F1 Score', 'ROC AUC', 'PR AUC']
    values = [accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), roc_auc, pr_auc]
    colors = ['steelblue', 'forestgreen', 'darkorange', 'purple']
    bars = ax.bar(metrics, values, color=colors)
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_support_vectors_simple(model, X_train_scaled, y_train, save_dir):
    """绘制支持向量可视化"""
    from sklearn.decomposition import PCA

    support_indices = model.support_

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_train_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    safe_mask = y_train == 0
    unsafe_mask = y_train == 1

    # 左图：PCA 空间
    ax = axes[0]
    ax.scatter(X_pca[safe_mask, 0], X_pca[safe_mask, 1],
               alpha=0.3, s=10, c='blue', label='safe (train)')
    ax.scatter(X_pca[unsafe_mask, 0], X_pca[unsafe_mask, 1],
               alpha=0.3, s=10, c='red', label='unsafe (train)')

    sv_safe = support_indices[y_train[support_indices] == 0]
    sv_unsafe = support_indices[y_train[support_indices] == 1]

    ax.scatter(X_pca[sv_safe, 0], X_pca[sv_safe, 1],
               s=100, c='darkblue', marker='X', edgecolors='white', linewidths=2,
               label='Support Vectors (safe)')
    ax.scatter(X_pca[sv_unsafe, 0], X_pca[sv_unsafe, 1],
               s=100, c='darkred', marker='X', edgecolors='white', linewidths=2,
               label='Support Vectors (unsafe)')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Support Vectors in PCA Space')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右图：原始 embedding 空间（前两个维度）
    ax = axes[1]
    # 注意：这里需要原始 embeddings，我们使用标准化后的前两个维度代替
    ax.scatter(X_train_scaled[safe_mask, 0], X_train_scaled[safe_mask, 1],
               alpha=0.3, s=10, c='blue', label='safe')
    ax.scatter(X_train_scaled[unsafe_mask, 0], X_train_scaled[unsafe_mask, 1],
               alpha=0.3, s=10, c='red', label='unsafe')

    ax.scatter(X_train_scaled[sv_safe, 0], X_train_scaled[sv_safe, 1],
               s=100, c='darkblue', marker='X', edgecolors='white', linewidths=2,
               label='SV (safe)')
    ax.scatter(X_train_scaled[sv_unsafe, 0], X_train_scaled[sv_unsafe, 1],
               s=100, c='darkred', marker='X', edgecolors='white', linewidths=2,
               label='SV (unsafe)')

    ax.set_xlabel('Feature Dim 1')
    ax.set_ylabel('Feature Dim 2')
    ax.set_title('Support Vectors in Feature Space')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'support_vectors.png'), dpi=150, bbox_inches='tight')
    plt.close()


def grid_search_svm(X_train, X_test, y_train, y_test):
    """Grid Search - 逐组训练，无 K 折 CV (LinearSVC, tmux)"""
    log_section("4. Grid Search 超参数搜索 (无 K 折 CV, LinearSVC, tmux)")

    # 参数网格 - LinearSVC 只有 C
    param_grid = {
        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    }

    total_combos = len(param_grid['C'])
    logger.info(f"参数网格：C={param_grid['C']}")
    logger.info(f"总参数组合数：{total_combos}")

    best_f1 = -1
    best_params = None
    best_result_dir = None
    all_results = []

    start_time = time.time()

    for i, C in enumerate(param_grid['C']):
        idx = i + 1
        params = {'C': C}
        param_name = make_param_name(params)

        logger.info(f"\n[{idx}/{total_combos}] 训练 C={C}")

        result_dir = os.path.join(RESULT_DIR, param_name)
        weight_name = param_name

        eval_results = train_in_tmux(
            None, None, None, None, params, result_dir, WEIGHT_DIR, LOG_DIR, DATA_DIR, RANDOM_STATE
        )

        if eval_results is None:
            logger.error(f"训练失败: C={C}")
            continue

        all_results.append({
            'param_name': param_name,
            'params': params,
            'f1': eval_results['f1_score'],
            'accuracy': eval_results['accuracy'],
            'roc_auc': eval_results['roc_auc'],
            'pr_auc': eval_results['pr_auc']
        })

        if eval_results['f1_score'] > best_f1:
            best_f1 = eval_results['f1_score']
            best_params = params
            best_result_dir = result_dir

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"Grid Search 完成！总耗时：{elapsed:.1f}s")
    logger.info(f"最佳参数：{best_params}")
    logger.info(f"最佳 F1：{best_f1:.4f}")

    # 保存所有结果汇总
    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('f1', ascending=False)
    summary_df.to_csv(os.path.join(RESULT_DIR, 'grid_search_summary.csv'), index=False)
    logger.info(f"结果汇总已保存：{os.path.join(RESULT_DIR, 'grid_search_summary.csv')}")

    # 保存最佳模型信息
    best_info = {
        'best_params': best_params,
        'best_f1': float(best_f1),
        'best_result_dir': best_result_dir,
        'total_combinations': total_combos,
        'elapsed_time': elapsed
    }
    with open(os.path.join(RESULT_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2)

    return best_params, best_f1


def bayesian_optimization_svm(X_train, X_test, y_train, y_test, n_iterations=50):
    """贝叶斯优化 - 逐次训练，无 K 折 CV (LinearSVC, tmux)"""
    log_section("4. 贝叶斯优化超参数搜索 (无 K 折 CV, LinearSVC, tmux)")

    from skopt import gp_minimize
    from skopt.space import Real

    param_space = [
        Real(1e-4, 1e3, prior='log-uniform', name='C'),
    ]

    logger.info(f"参数空间：C=[1e-4, 1e3] (log-uniform)")
    logger.info(f"优化迭代次数：{n_iterations}")

    all_results = []
    best_f1 = -1
    best_params = None

    start_time = time.time()

    def objective(params):
        nonlocal best_f1, best_params
        C = params[0]
        params_dict = {'C': float(C)}
        param_name = make_param_name(params_dict)

        logger.info(f"\n[Iteration {len(all_results)+1}/{n_iterations}] C={C:.6f}")

        result_dir = os.path.join(RESULT_DIR, param_name)
        weight_name = param_name

        eval_results = train_in_tmux(
            None, None, None, None, params_dict, result_dir, WEIGHT_DIR, LOG_DIR, DATA_DIR, RANDOM_STATE
        )

        if eval_results is None:
            return 0  # 训练失败

        all_results.append({
            'param_name': param_name,
            'params': params_dict,
            'f1': eval_results['f1_score'],
            'accuracy': eval_results['accuracy'],
            'roc_auc': eval_results['roc_auc'],
            'pr_auc': eval_results['pr_auc']
        })

        if eval_results['f1_score'] > best_f1:
            best_f1 = eval_results['f1_score']
            best_params = params_dict

        return -eval_results['f1_score']

    result = gp_minimize(
        objective,
        dimensions=param_space,
        n_calls=n_iterations,
        random_state=RANDOM_STATE,
        verbose=True
    )

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*70}")
    logger.info(f"贝叶斯优化完成！总耗时：{elapsed:.1f}s")
    logger.info(f"最佳参数：{best_params}")
    logger.info(f"最佳 F1：{best_f1:.4f}")

    summary_df = pd.DataFrame(all_results)
    summary_df = summary_df.sort_values('f1', ascending=False)
    summary_df.to_csv(os.path.join(RESULT_DIR, 'bayesian_optimization_summary.csv'), index=False)
    logger.info(f"结果汇总已保存：{os.path.join(RESULT_DIR, 'bayesian_optimization_summary.csv')}")

    best_info = {
        'best_params': best_params,
        'best_f1': float(best_f1),
        'total_iterations': n_iterations,
        'elapsed_time': elapsed
    }
    with open(os.path.join(RESULT_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(best_info, f, indent=2)

    return best_params, best_f1


def generate_final_report(args, best_params, best_f1, total_time):
    """生成最终实验报告"""
    log_section("5. 生成实验报告")

    summary_path = os.path.join(RESULT_DIR, 'grid_search_summary.csv')
    if not os.path.exists(summary_path):
        summary_path = os.path.join(RESULT_DIR, 'bayesian_optimization_summary.csv')

    report = f"""# SVM 安全分类器实验报告

## 实验时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验配置
- 优化器：{args.optimizer}
- 训练比例：{args.training_ratio:.2%}
- 随机种子：{args.random_state}
- 交叉验证：无（直接训练评估）
- 模型：LinearSVC + CalibratedClassifierCV

## 数据集
- 数据来源：{DATA_PATH}
- 特征维度：4096 (Qwen Embedding)

## 最佳模型参数
- C：{best_params['C']}

## 最佳性能
- F1 分数：{best_f1:.4f}

## 输出目录
- 结果目录：{RESULT_DIR}
- 权重目录：{WEIGHT_DIR}
- 日志目录：{LOG_DIR}
- 数据缓存：{DATA_DIR}

## 结果汇总
见 {summary_path}
"""

    report_path = os.path.join(RESULT_DIR, 'experiment_report.md')
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"实验报告已保存：{report_path}")


def main():
    """主函数"""
    args = parse_args()

    # 重新配置日志处理器（确保日志写入正确目录）
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, 'experiment.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 18 + "SVM 安全分类器实验流程 (无 K 折 CV)" + " " * 16 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"实验配置：优化器={args.optimizer}, 训练比例={args.training_ratio:.2%}")
    logger.info(f"数据目录：{DATA_DIR}")
    logger.info(f"权重目录：{WEIGHT_DIR}")
    logger.info(f"结果目录：{RESULT_DIR}")
    logger.info(f"日志目录：{LOG_DIR}")

    try:
        # 1. 数据加载（优先使用缓存）
        df, y = load_and_prepare_data(args.training_ratio, args.random_state)

        # 2. 加载 embeddings（优先使用缓存）
        embeddings = load_embeddings(df, y)

        # 3. 标准化和划分（优先使用缓存）
        X_train, X_test, y_train, y_test, scaler = load_or_prepare_split(embeddings, y, args.random_state)

        # 4. 超参数搜索（无 K 折 CV）
        if args.optimizer == 'bayesian':
            best_params, best_f1 = bayesian_optimization_svm(
                X_train, X_test, y_train, y_test, n_iterations=args.bayesian_iterations
            )
        else:  # grid
            best_params, best_f1 = grid_search_svm(X_train, X_test, y_train, y_test)

        # 5. 生成报告
        total_time = time.time() - start_time
        generate_final_report(args, best_params, best_f1, total_time)

        logger.info("╔" + "═" * 68 + "╗")
        logger.info(f"║  实验完成！总耗时：{total_time/60:.1f} 分钟{' ' * 35}║")
        logger.info("╚" + "═" * 68 + "╝")
        logger.info(f"结果目录：{RESULT_DIR}")
        logger.info(f"权重目录：{WEIGHT_DIR}")

    except Exception as e:
        logger.exception(f"实验出错：{e}")
        raise


if __name__ == "__main__":
    main()
