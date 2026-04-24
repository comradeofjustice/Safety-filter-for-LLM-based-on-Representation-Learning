#!/usr/bin/env python3
"""
SVM 安全分类器完整实验流程 (sklearn CPU 版)
数据编码 → 标准化 → 划分数据集 → GridSearchCV/Bayesian Optimization 调参（5 折 CV）
→ 评估（F1 + AUC + 混淆矩阵）→ 支持向量分析 + UMAP 可视化

使用 sklearn SVC 进行 CPU 训练，支持多核并行 (n_jobs=-1)

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
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# ============== 配置路径 ==============
# 基础目录（位于 /root/CSY/Safety-filter-for-LLM-based-on-Representation-Learning）
BASE_DIR = '/root/CSY/Safety-filter-for-LLM-based-on-Representation-Learning'

# 日志目录
LOG_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(LOG_DIR, exist_ok=True)

# 模型路径（Qwen8B Embedding）
MODEL_PATH = '/root/CSY/SafestAI/SAPL/Qwen8BEmbedding'

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据路径（保持不变）
DATA_PATH = '/root/CSY/SafestAI/dataset/merged_safety_dataset.csv'

# 默认参数
RANDOM_STATE = 42
DEFAULT_TRAINING_RATIO = 0.8  # 默认全量训练
DEFAULT_OPTIMIZER = 'grid'  # grid 或 bayesian

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
                        choices=['grid', 'bayesian', 'compare'],
                        help=f'优化器类型：grid, bayesian 或 compare, 默认 {DEFAULT_OPTIMIZER}')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='实验名称前缀，用于输出文件夹命名')
    parser.add_argument('--random_state', type=int, default=RANDOM_STATE,
                        help=f'随机种子，默认 {RANDOM_STATE}')
    return parser.parse_args()


def create_output_folder(experiment_name, optimizer, training_ratio):
    """根据实验条件创建输出文件夹"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 构建实验条件标识
    ratio_str = f"{int(training_ratio * 100)}pct" if training_ratio < 1.0 else "full"
    opt_str = optimizer
    
    # 实验条件前缀
    if experiment_name:
        condition = f"{experiment_name}_{opt_str}_{ratio_str}"
    else:
        condition = f"{opt_str}_{ratio_str}"
    
    # 输出文件夹命名：[实验条件]+output
    folder_name = f"{condition}+output"
    output_path = os.path.join(BASE_DIR, folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    return output_path


def log_section(title):
    """记录章节分隔线"""
    logger.info("=" * 70)
    logger.info(f"  {title}")
    logger.info("=" * 70)


def load_and_prepare_data(training_ratio=1.0, random_state=RANDOM_STATE):
    """加载并准备数据"""
    log_section("1. 数据加载与准备")

    logger.info(f"加载数据集：{DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"原始数据集大小：{len(df)}")

    # 根据比例采样（全量训练时 training_ratio=1.0）
    if training_ratio < 1.0:
        n_samples = int(len(df) * training_ratio)
        df_sample = df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        logger.info(f"训练比例：{training_ratio:.2%}, 采样数量：{n_samples}")
    else:
        df_sample = df.reset_index(drop=True)
        logger.info(f"全量训练：使用全部 {len(df_sample)} 个样本")

    logger.info(f"实际使用数据集大小：{len(df_sample)}")

    # 标签分布
    label_dist = df_sample['label'].value_counts().to_dict()
    logger.info(f"标签分布：{label_dist}")

    # 转换标签
    y = (df_sample['label'] == 'unsafe').astype(int).values
    logger.info(f"标签编码：safe=0, unsafe=1")
    logger.info(f"unsafe 比例：{y.mean():.4f}")

    # 保存样本索引用于后续分析
    df_sample['label_encoded'] = y
    df_sample.to_csv(os.path.join(OUTPUT_DIR, 'sampled_data.csv'), index=False)
    logger.info(f"采样数据已保存：{os.path.join(OUTPUT_DIR, 'sampled_data.csv')}")

    return df_sample, y


def encode_texts_with_api(df, y, port=8811):
    """使用 VLLM API 服务编码文本"""
    log_section("2. 文本编码 (Qwen Embedding via VLLM API)")

    import requests
    from tqdm import tqdm

    api_url = f"http://localhost:{port}/v1/embeddings"
    logger.info(f"VLLM API 地址：{api_url}")

    # 批量编码
    texts = df['content'].tolist()
    all_embeddings = []
    batch_size = 16

    logger.info(f"开始编码 {len(texts)} 个样本 (batch_size={batch_size})...")
    start_time = time.time()

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i+batch_size]
        # 截断过长的文本
        batch_texts = [t[:8000] if len(t) > 8000 else t for t in batch_texts]
        
        payload = {
            "model": "Qwen8BEmbedding",
            "input": batch_texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 提取 embeddings
            batch_embeddings = [np.array(item['embedding']) for item in result['data']]
            all_embeddings.extend(batch_embeddings)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API 请求失败：{e}")
            raise

    embeddings = np.array(all_embeddings)
    elapsed = time.time() - start_time
    logger.info(f"编码完成！形状：{embeddings.shape}, 总耗时：{elapsed:.1f}s")

    # 保存 embeddings
    np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
    logger.info(f"Embeddings 已保存：{os.path.join(OUTPUT_DIR, 'embeddings.npy')}")

    return embeddings


def encode_texts_local(df, y):
    """使用 Transformers 加载本地模型编码文本"""
    log_section("2. 文本编码 (Qwen Embedding - Local)")

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

    # 批量编码
    texts = df['content'].tolist()
    all_embeddings = []
    batch_size = 64  # 增大批量加速

    logger.info(f"开始编码 {len(texts)} 个样本 (batch_size={batch_size})...")
    start_time = time.time()

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i:i+batch_size]
            # 截断过长的文本
            batch_texts = [t[:512] if len(t) > 512 else t for t in batch_texts]
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            ).to(model.device)
            
            outputs = model(**inputs)
            # 使用 [CLS] token 的隐藏状态作为句子嵌入
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend(embeddings)

            if (i // batch_size + 1) % 50 == 0:
                progress = (i + len(batch_texts)) / len(texts) * 100
                elapsed = time.time() - start_time
                logger.info(f"  进度：{progress:.1f}%, 耗时：{elapsed:.1f}s")

    embeddings = np.array(all_embeddings)
    elapsed = time.time() - start_time
    logger.info(f"编码完成！形状：{embeddings.shape}, 总耗时：{elapsed:.1f}s")

    # 保存 embeddings
    np.save(os.path.join(OUTPUT_DIR, 'embeddings.npy'), embeddings)
    logger.info(f"Embeddings 已保存：{os.path.join(OUTPUT_DIR, 'embeddings.npy')}")

    return embeddings


def standardize_and_split(X, y, random_state=RANDOM_STATE):
    """标准化并划分数据集"""
    log_section("3. 数据标准化与划分")

    # 划分训练集和测试集 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    logger.info(f"训练集大小：{len(X_train)}, 测试集大小：{len(X_test)}")
    logger.info(f"训练集 unsafe 比例：{y_train.mean():.4f}")
    logger.info(f"测试集 unsafe 比例：{y_test.mean():.4f}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info("标准化完成 (StandardScaler)")

    # 保存
    np.save(os.path.join(OUTPUT_DIR, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)

    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    logger.info("数据已保存")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def grid_search_svm(X_train, y_train, random_state=RANDOM_STATE):
    """GridSearchCV 调参（5 折 CV）- sklearn CPU 版"""
    log_section("4. GridSearchCV 超参数调优 (5 折 CV) - sklearn CPU")

    from sklearn.svm import SVC

    # 参数网格
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf']
    }

    logger.info(f"参数网格：{param_grid}")
    logger.info(f"总参数组合数：{len(param_grid['C']) * len(param_grid['gamma']) * len(param_grid['kernel'])}")

    # 5 折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    logger.info("使用 5 折 StratifiedKFold 交叉验证")

    base_svm = SVC(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)

    logger.info(f"GridSearchCV 完成！")
    logger.info(f"最佳参数：{grid_search.best_params_}")
    logger.info(f"最佳 F1 分数 (CV): {grid_search.best_score_:.4f}")

    # 保存结果
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'grid_search_results.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'best_svm.pkl'), 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    return grid_search


def bayesian_optimization_svm(X_train, y_train, n_iterations=50, random_state=RANDOM_STATE):
    """贝叶斯优化 SVM 超参数 - sklearn CPU 版"""
    log_section("4. 贝叶斯优化超参数调优 (5 折 CV) - sklearn CPU")

    from sklearn.svm import SVC

    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Categorical, Integer
        logger.info("skopt 库可用，开始贝叶斯优化...")
    except ImportError:
        logger.warning("skopt 库不可用，尝试安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-optimize", "-q"])
            from skopt import BayesSearchCV
            from skopt.space import Real, Categorical, Integer
            logger.info("scikit-optimize 安装成功")
        except Exception as e:
            logger.error(f"无法安装 skopt，请手动安装：pip install scikit-optimize")
            raise

    # 参数空间
    param_space = {
        'C': Real(1e-3, 1e4, prior='log-uniform'),
        'gamma': Real(1e-4, 1e3, prior='log-uniform'),
        'kernel': Categorical(['rbf'])
    }

    logger.info(f"参数空间：{param_space}")
    logger.info(f"优化迭代次数：{n_iterations}")

    # 5 折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    logger.info("使用 5 折 StratifiedKFold 交叉验证")

    base_svm = SVC(random_state=random_state)

    bayes_search = BayesSearchCV(
        estimator=base_svm,
        search_spaces=param_space,
        n_iter=n_iterations,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
        random_state=random_state
    )
    bayes_search.fit(X_train, y_train)

    logger.info(f"贝叶斯优化完成！")
    logger.info(f"最佳参数：{bayes_search.best_params_}")
    logger.info(f"最佳 F1 分数 (CV): {bayes_search.best_score_:.4f}")

    # 保存结果
    optimization_history = pd.DataFrame(bayes_search.cv_results_)
    optimization_history.to_csv(os.path.join(OUTPUT_DIR, 'bayesian_optimization_results.csv'), index=False)

    with open(os.path.join(OUTPUT_DIR, 'best_svm.pkl'), 'wb') as f:
        pickle.dump(bayes_search.best_estimator_, f)

    return bayes_search


def compare_optimizers(X_train, y_train, random_state=RANDOM_STATE):
    """对比 Grid Search 和 Bayesian Optimization"""
    log_section("4. 优化器对比实验")
    
    logger.info("同时运行 Grid Search 和 Bayesian Optimization 进行对比...")
    
    # Grid Search
    logger.info("\n" + "="*50)
    logger.info("运行 Grid Search...")
    logger.info("="*50)
    grid_result = grid_search_svm(X_train, y_train, random_state)
    
    # Bayesian Optimization
    logger.info("\n" + "="*50)
    logger.info("运行 Bayesian Optimization...")
    logger.info("="*50)
    bayes_result = bayesian_optimization_svm(X_train, y_train, n_iterations=50, random_state=random_state)
    
    # 对比结果
    logger.info("\n" + "="*50)
    logger.info("优化器对比结果")
    logger.info("="*50)
    logger.info(f"Grid Search - 最佳参数：{grid_result.best_params_}")
    logger.info(f"Grid Search - 最佳 F1 (CV): {grid_result.best_score_:.4f}")
    logger.info(f"Bayesian - 最佳参数：{bayes_result.best_params_}")
    logger.info(f"Bayesian - 最佳 F1 (CV): {bayes_result.best_score_:.4f}")
    
    # 保存对比结果
    comparison = {
        'grid_search': {
            'best_params': grid_result.best_params_,
            'best_f1': float(grid_result.best_score_)
        },
        'bayesian': {
            'best_params': bayes_result.best_params_,
            'best_f1': float(bayes_result.best_score_)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, 'optimizer_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"对比结果已保存：{os.path.join(OUTPUT_DIR, 'optimizer_comparison.json')}")
    
    # 选择更好的模型
    if bayes_result.best_score_ > grid_result.best_score_:
        logger.info(f"\n贝叶斯优化表现更好！选择贝叶斯优化的模型。")
        return bayes_result, 'bayesian'
    else:
        logger.info(f"\nGrid Search 表现更好或相同！选择 Grid Search 的模型。")
        return grid_result, 'grid'


def evaluate_model(model, X_test, y_test, X_train, y_train):
    """模型评估（F1 + AUC + 混淆矩阵）"""
    log_section("5. 模型评估")

    # 预测
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # 基本指标
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    logger.info("=" * 50)
    logger.info("基本指标")
    logger.info("=" * 50)
    logger.info(f"测试集准确率：{accuracy:.4f}")
    logger.info(f"测试集 F1 分数：{f1:.4f}")
    logger.info(f"训练集准确率：{accuracy_score(y_train, y_train_pred):.4f}")

    # 分类报告
    logger.info("\n" + "=" * 50)
    logger.info("分类报告")
    logger.info("=" * 50)
    report = classification_report(y_test, y_pred, target_names=['safe', 'unsafe'])
    logger.info(report)

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    logger.info("\n" + "=" * 50)
    logger.info("混淆矩阵")
    logger.info("=" * 50)
    logger.info(f"\n{cm}")

    # 概率预测
    y_proba = None
    roc_auc = None
    pr_auc = None

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        logger.info(f"\nROC AUC: {roc_auc:.4f}")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        logger.info(f"PR AUC: {pr_auc:.4f}")
    except Exception as e:
        logger.warning(f"无法计算概率预测：{e}")
        # 使用决策函数值代替
        try:
            y_scores = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            roc_auc = auc(fpr, tpr)
            logger.info(f"\nROC AUC (使用 decision_function): {roc_auc:.4f}")
        except Exception as e2:
            logger.warning(f"无法计算 decision_function: {e2}")

    # 保存评估结果
    eval_results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': float(roc_auc) if roc_auc else None,
        'pr_auc': float(pr_auc) if pr_auc else None,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'best_params': model.get_params()
    }

    with open(os.path.join(OUTPUT_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    logger.info(f"评估结果已保存：{os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")

    # 绘制评估图表
    plot_evaluation(y_test, y_pred, y_proba, cm, fpr if roc_auc else None, tpr if roc_auc else None, 
                    roc_auc, precision if pr_auc else None, recall if pr_auc else None, pr_auc)

    return eval_results


def plot_evaluation(y_test, y_pred, y_proba, cm, fpr, tpr, roc_auc, precision, recall, pr_auc):
    """绘制评估图表"""
    log_section("6. 评估可视化")

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

    # 5. 支持向量分析（用 PCA 可视化）
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Support Vector Analysis\n(see separate figure)', 
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"评估图已保存：{os.path.join(OUTPUT_DIR, 'evaluation_plots.png')}")


def analyze_support_vectors(model, X_train, y_train, embeddings):
    """支持向量分析"""
    log_section("7. 支持向量分析")

    # 获取支持向量
    support_indices = model.support_
    support_vectors = model.support_vectors_

    logger.info(f"支持向量数量：{len(support_indices)}")
    logger.info(f"支持向量占训练集比例：{len(support_indices) / len(X_train) * 100:.2f}%")

    # 支持向量的标签分布
    support_labels = y_train[support_indices]
    safe_sv = np.sum(support_labels == 0)
    unsafe_sv = np.sum(support_labels == 1)
    logger.info(f"支持向量中 safe 数量：{safe_sv}, unsafe 数量：{unsafe_sv}")

    # 保存支持向量索引
    np.save(os.path.join(OUTPUT_DIR, 'support_vector_indices.npy'), support_indices)
    logger.info(f"支持向量索引已保存")

    # 绘制支持向量可视化
    plot_support_vectors(model, X_train, y_train, support_indices, embeddings)

    return support_indices


def plot_support_vectors(model, X_train_scaled, y_train, support_indices, embeddings_original):
    """绘制支持向量可视化"""
    logger.info("生成支持向量可视化...")

    # 使用 PCA 降维到 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_train_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. PCA 空间中的支持向量
    ax = axes[0]

    # 所有训练样本
    safe_mask = y_train == 0
    unsafe_mask = y_train == 1

    ax.scatter(X_pca[safe_mask, 0], X_pca[safe_mask, 1],
               alpha=0.3, s=10, c='blue', label='safe (train)')
    ax.scatter(X_pca[unsafe_mask, 0], X_pca[unsafe_mask, 1],
               alpha=0.3, s=10, c='red', label='unsafe (train)')

    # 支持向量
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

    # 2. 支持向量在原始 embedding 空间的分布（用前两个维度）
    ax = axes[1]

    ax.scatter(embeddings_original[safe_mask, 0], embeddings_original[safe_mask, 1],
               alpha=0.3, s=10, c='blue', label='safe')
    ax.scatter(embeddings_original[unsafe_mask, 0], embeddings_original[unsafe_mask, 1],
               alpha=0.3, s=10, c='red', label='unsafe')

    ax.scatter(embeddings_original[sv_safe, 0], embeddings_original[sv_safe, 1],
               s=100, c='darkblue', marker='X', edgecolors='white', linewidths=2,
               label='SV (safe)')
    ax.scatter(embeddings_original[sv_unsafe, 0], embeddings_original[sv_unsafe, 1],
               s=100, c='darkred', marker='X', edgecolors='white', linewidths=2,
               label='SV (unsafe)')

    ax.set_xlabel('Embedding Dim 1')
    ax.set_ylabel('Embedding Dim 2')
    ax.set_title('Support Vectors in Original Embedding Space')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'support_vectors_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"支持向量分析图已保存：{os.path.join(OUTPUT_DIR, 'support_vectors_analysis.png')}")


def umap_visualization(X_train, y_train, model, support_indices):
    """UMAP 可视化"""
    log_section("8. UMAP 可视化")

    try:
        import umap
        logger.info("UMAP 库可用，开始降维可视化...")
    except ImportError:
        logger.warning("UMAP 库不可用，跳过 UMAP 可视化。安装：pip install umap-learn")
        return

    # UMAP 降维
    logger.info("执行 UMAP 降维 (n_components=2)...")
    start_time = time.time()

    reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_jobs=-1)
    X_umap = reducer.fit_transform(X_train)

    elapsed = time.time() - start_time
    logger.info(f"UMAP 降维完成！耗时：{elapsed:.1f}s")

    # 保存 UMAP 结果
    np.save(os.path.join(OUTPUT_DIR, 'umap_embedding.npy'), X_umap)

    # 绘制 UMAP 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 所有样本的 UMAP
    ax = axes[0]
    safe_mask = y_train == 0
    unsafe_mask = y_train == 1

    scatter = ax.scatter(X_umap[safe_mask, 0], X_umap[safe_mask, 1],
                         alpha=0.5, s=10, c='blue', label='safe')
    ax.scatter(X_umap[unsafe_mask, 0], X_umap[unsafe_mask, 1],
               alpha=0.5, s=10, c='red', label='unsafe')

    ax.set_title(f'UMAP Visualization (all samples)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 支持向量高亮
    ax = axes[1]

    # 背景 - 所有样本
    ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.2, s=5, c='gray', label='other samples')

    # 支持向量
    sv_safe = support_indices[y_train[support_indices] == 0]
    sv_unsafe = support_indices[y_train[support_indices] == 1]

    ax.scatter(X_umap[sv_safe, 0], X_umap[sv_safe, 1],
               s=100, c='darkblue', marker='X', edgecolors='white', linewidths=2,
               label='Support Vectors (safe)')
    ax.scatter(X_umap[sv_unsafe, 0], X_umap[sv_unsafe, 1],
               s=100, c='darkred', marker='X', edgecolors='white', linewidths=2,
               label='Support Vectors (unsafe)')

    ax.set_title(f'UMAP Visualization (Support Vectors Highlighted)')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'umap_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"UMAP 可视化已保存：{os.path.join(OUTPUT_DIR, 'umap_visualization.png')}")


def generate_final_report(eval_results, support_indices, X_train, args):
    """生成最终实验报告"""
    log_section("9. 生成实验报告")

    report = f"""
# SVM 安全分类器实验报告

## 实验时间
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 实验配置
- 优化器：{args.optimizer}
- 训练比例：{args.training_ratio:.2%}
- 随机种子：{args.random_state}

## 数据集
- 数据来源：{DATA_PATH}
- 样本数量：{eval_results.get('n_samples', 'N/A')}
- 特征维度：4096 (Qwen Embedding)

## 模型参数
- 最佳参数：{eval_results['best_params']}

## 性能指标
| 指标 | 分数 |
|------|------|
| 准确率 | {eval_results['accuracy']:.4f} |
| F1 分数 | {eval_results['f1_score']:.4f} |
| ROC AUC | {eval_results['roc_auc']:.4f} |
| PR AUC | {eval_results['pr_auc']:.4f} |

## 混淆矩阵
```
{np.array(eval_results['confusion_matrix'])}
```

## 支持向量分析
- 支持向量数量：{len(support_indices)}
- 支持向量占比：{len(support_indices) / len(X_train) * 100:.2f}%

## 输出文件
- 最佳模型：{OUTPUT_DIR}/best_svm.pkl
- 评估结果：{OUTPUT_DIR}/evaluation_results.json
- 支持向量索引：{OUTPUT_DIR}/support_vector_indices.npy
- 可视化图表：{OUTPUT_DIR}/evaluation_plots.png, {OUTPUT_DIR}/support_vectors_analysis.png
"""

    with open(os.path.join(OUTPUT_DIR, 'experiment_report.md'), 'w') as f:
        f.write(report)

    logger.info(f"实验报告已保存：{os.path.join(OUTPUT_DIR, 'experiment_report.md')}")


def main():
    """主函数"""
    args = parse_args()
    
    # 根据实验条件创建输出文件夹
    global OUTPUT_DIR
    OUTPUT_DIR = create_output_folder(args.experiment_name, args.optimizer, args.training_ratio)
    
    # 重新配置日志处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(OUTPUT_DIR, 'experiment.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    
    start_time = time.time()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 20 + "SVM 安全分类器实验流程" + " " * 21 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info(f"实验配置：优化器={args.optimizer}, 训练比例={args.training_ratio:.2%}")
    logger.info(f"输出目录：{OUTPUT_DIR}")

    try:
        # 1. 数据加载
        df, y = load_and_prepare_data(args.training_ratio, args.random_state)

        # 2. 文本编码（使用本地 VLLM 方式）
        logger.info("使用本地 VLLM 方式进行文本编码...")
        embeddings = encode_texts_local(df, y)

        # 3. 标准化和划分
        X_train, X_test, y_train, y_test, scaler = standardize_and_split(embeddings, y, args.random_state)

        # 4. 超参数调优
        if args.optimizer == 'compare':
            # 对比两种优化器
            search_result, best_optimizer = compare_optimizers(X_train, y_train, args.random_state)
        elif args.optimizer == 'bayesian':
            search_result = bayesian_optimization_svm(X_train, y_train, random_state=args.random_state)
        else:  # grid
            search_result = grid_search_svm(X_train, y_train, args.random_state)
        
        best_model = search_result.best_estimator_

        # 5. 模型评估
        eval_results = evaluate_model(best_model, X_test, y_test, X_train, y_train)
        eval_results['n_samples'] = len(df)

        # 6. 支持向量分析
        support_indices = analyze_support_vectors(best_model, X_train, y_train, embeddings)

        # 7. UMAP 可视化
        umap_visualization(X_train, y_train, best_model, support_indices)

        # 8. 生成报告
        generate_final_report(eval_results, support_indices, X_train, args)

        total_time = time.time() - start_time
        logger.info("╔" + "═" * 68 + "╗")
        logger.info(f"║  实验完成！总耗时：{total_time/60:.1f} 分钟{' ' * 35}║")
        logger.info("╚" + "═" * 68 + "╝")
        logger.info(f"输出目录：{OUTPUT_DIR}")

    except Exception as e:
        logger.exception(f"实验出错：{e}")
        raise


if __name__ == "__main__":
    main()
