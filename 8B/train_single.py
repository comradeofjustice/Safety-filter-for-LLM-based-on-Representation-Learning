#!/usr/bin/env python3
"""
训练单个超参数组合的 SVM 模型 (由 tmux 调用)
用法:
    python train_single.py --c 0.01 --result_dir result/C0.01 --weight_dir weight --log_dir log
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_recall_curve, average_precision_score
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=float, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--weight_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    param_name = f"C{args.c:.6f}".rstrip('0').rstrip('.')
    print(f"{'='*70}")
    print(f"开始训练 | C={args.c}")
    print(f"{'='*70}")

    # 加载数据
    print(f"加载训练数据...")
    X_train = np.load(os.path.join(args.data_dir, 'X_train_scaled.npy'))
    X_test = np.load(os.path.join(args.data_dir, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
    y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"NaN check - train: {np.isnan(X_train).sum()}, test: {np.isnan(X_test).sum()}")

    # 训练 LinearSVC
    print(f"训练 LinearSVC (C={args.c})...")
    base_model = LinearSVC(C=args.c, loss='squared_hinge', penalty='l2', max_iter=10000, random_state=args.random_state)
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train, y_train)
    print(f"训练完成")

    # 保存权重
    weight_path = os.path.join(args.weight_dir, f"{param_name}.pkl")
    with open(weight_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"权重已保存: {weight_path}")

    # 预测
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['safe', 'unsafe'])
    cm = confusion_matrix(y_test, y_pred)

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"\n{'='*50}")
    print(f"分类报告:")
    print(report)
    print(f"混淆矩阵:")
    print(cm)
    print(f"{'='*50}")
    print(f"测试集 Accuracy:  {accuracy:.4f}")
    print(f"测试集 F1 Score:  {f1:.4f}")
    print(f"测试集 ROC AUC:   {roc_auc:.4f}")
    print(f"测试集 PR AUC:    {pr_auc:.4f}")
    print(f"训练集 Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"{'='*70}")

    # 保存结果
    os.makedirs(args.result_dir, exist_ok=True)
    eval_results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'params': {'C': args.c},
    }
    with open(os.path.join(args.result_dir, 'results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 绘制评估图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    ax = axes[0, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['safe', 'unsafe'], yticklabels=['safe', 'unsafe'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    ax = axes[0, 1]
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(recall, precision, label=f'PR Curve (AP = {pr_auc:.4f})', linewidth=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

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

    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'C={args.c}\nF1={f1:.4f}',
            ha='center', va='center', fontsize=14, transform=ax.transAxes)
    ax.set_title('Model Summary')
    ax.axis('off')

    ax = axes[1, 2]
    metrics = ['Accuracy', 'F1 Score', 'ROC AUC', 'PR AUC']
    values = [accuracy, f1, roc_auc, pr_auc]
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
    plt.savefig(os.path.join(args.result_dir, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(args.result_dir, 'evaluation_plots.png')}")

    print(f"结果已保存: {args.result_dir}")
    print(f"{'='*70}")
    print(f"训练完成 | C={args.c} | F1={f1:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
