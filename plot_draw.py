import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, confusion_matrix, f1_score, average_precision_score
import numpy as np
import os
from config import OUTPUT_DIR

# 统一输出图片分辨率为 300 dpi
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
# 中文字体，保证方法名称/市场名称正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_confusion_matrix(y_true, y_pred, name):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(OUTPUT_DIR, f'cm_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def plot_pr_curve_multi(y_true, score_dict, name):
    """
    绘制四种方法的合并 PR 曲线：
    - 每条曲线标注对应方法的 AP 值
    - 使用图例区分四种方法
    - 图中统一使用英文标注，避免中文乱码
    """
    # 方法名的英文显示映射
    EN_LABELS = {
        '3σ 原则': '3-Sigma Rule',
        'Isolation Forest': 'Isolation Forest',
        '动态阈值规则': 'Dynamic Threshold Rule',
        'XGBoost 融合模型': 'XGBoost Fusion Model'
    }
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    markers = ['o', 's', '^', 'D']
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ap_lines = []

    for idx, (method, scores) in enumerate(score_dict.items()):
        en_name = EN_LABELS.get(method, method)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        color = colors[idx % len(colors)]
        plt.plot(recall, precision, color=color, linewidth=2,
                 marker=markers[idx % len(markers)], markevery=50, markersize=4,
                 label=f'{en_name} (AP = {ap:.4f})')
        ap_lines.append(f'{en_name}: AP = {ap:.4f}')

    # 在图内补充标注各方法的 AP 值
    plt.text(0.02, 0.02, '\n'.join(ap_lines), transform=ax.transAxes, fontsize=9,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

    plt.title('Precision-Recall Curve - Four Anomaly Detection Methods')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=9, frameon=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f'pr_{name}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_transfer_f1(transfer_results):
    """绘制留一市场迁移实验 F1 对比柱状图（市场名称统一使用英文，避免乱码）"""
    # 市场名的英文显示映射
    EN_MARKETS = {
        'A股': 'CN (A-Share)',
        '港股': 'HK (Hang Seng)',
        '蒙古': 'MN (MSE TOP-20)'
    }
    x_labels = [EN_MARKETS.get(m, m) for m in transfer_results['target_market']]

    plt.figure(figsize=(10, 6))
    x = range(len(transfer_results))
    width = 0.35
    
    plt.bar(x, transfer_results['local_f1'], width, label='Local Model')
    plt.bar([i + width for i in x], transfer_results['transfer_f1'], width, label='Transfer Model')
    
    plt.xlabel('Market')
    plt.ylabel('F1 Score')
    plt.title('Leave-one-out Market Transfer Performance')
    plt.xticks([i + width/2 for i in x], x_labels)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'leave_one_out_transfer_f1.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_sensitivity(val_df, model, feature_cols):
    """绘制阈值敏感性曲线图 (基于验证集)"""
    y_probs = model.predict_proba(val_df[feature_cols])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_scores = []
    
    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(val_df['label'], y_pred, zero_division=0))
        
    best_idx = np.argmax(f1_scores)
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, marker='o', linestyle='-', markersize=4, label='F1 Score')
    
    # 标注最佳阈值
    plt.axvline(x=best_t, color='r', linestyle='--', label=f'Best Threshold ≈ {best_t:.2f}')
    plt.scatter(best_t, best_f1, color='red', s=100, zorder=5)
    
    # 高亮最佳阈值附近区间
    plt.axvspan(max(0.05, best_t - 0.05), min(0.95, best_t + 0.05), color='yellow', alpha=0.2,
                label=f'Sensitivity Range ({max(0.05, best_t - 0.05):.2f}-{min(0.95, best_t + 0.05):.2f})')
    
    plt.title('Anomaly Detection Threshold Sensitivity Analysis')
    plt.xlabel('Classification Threshold')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_sensitivity_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
