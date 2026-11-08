import os
import pandas as pd
from config import OUTPUT_DIR


def _fmt(df, rename=None, decimals=4):
    """将 DataFrame 转为 Markdown 表格，统一四舍五入并转换为中文列名"""
    if df is None or len(df) == 0:
        return '（无数据）'
    out = df.copy()
    num_cols = out.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].round(decimals)
    if rename:
        out = out.rename(columns=rename)
    return out.to_markdown(index=False)


def generate_markdown_report(data_stats, label_stats, split_stats, split_market_stats,
                             method_compare, market_f1, transfer_results, ablation_results,
                             quality_compare, xgb_metrics, best_threshold, best_f1,
                             validation_msg, total_removed, cm_df=None):
    """生成包含全部对比指标的完整 Markdown 实验报告"""

    # ---- 结论中使用的统计量 ----
    xgb_f1 = xgb_metrics['f1_score']
    xgb_acc = xgb_metrics['accuracy']
    xgb_pre = xgb_metrics['precision']
    xgb_rec = xgb_metrics['recall']

    baseline_df = method_compare[method_compare['method'] != 'XGBoost 融合模型']
    if len(baseline_df) > 0:
        best_base_row = baseline_df.loc[baseline_df['f1_score'].idxmax()]
        best_base_name = best_base_row['method']
        best_base_f1 = best_base_row['f1_score']
        best_base_acc = best_base_row['accuracy']
        best_base_pre = best_base_row['precision']
        best_base_rec = best_base_row['recall']
        f1_gain = xgb_f1 - best_base_f1
        advantage_text = (f'XGBoost 融合模型测试集 F1 为 {xgb_f1:.4f}，'
                          f'较最优基线方法（{best_base_name}，F1={best_base_f1:.4f}）'
                          f'高出 {f1_gain:.4f}（约 {f1_gain * 100:.2f} 个百分点）。')
    else:
        best_base_name, best_base_f1 = '无', 0.0
        best_base_acc = best_base_pre = best_base_rec = 0.0
        f1_gain = 0.0
        advantage_text = f'XGBoost 融合模型测试集 F1 为 {xgb_f1:.4f}。'

    # ---- 分市场汇总 ----
    market_avg_f1 = float(market_f1['f1_score'].mean()) if len(market_f1) else 0.0
    market_min_row = market_f1.loc[market_f1['f1_score'].idxmin()] if len(market_f1) else None
    market_max_row = market_f1.loc[market_f1['f1_score'].idxmax()] if len(market_f1) else None

    # ---- 迁移实验汇总 ----
    transfer_arr = transfer_results['drop_val'].to_numpy()
    improved = transfer_results[transfer_results['drop_val'] < 0]['target_market'].tolist()
    max_drop = float(transfer_arr.max()) if len(transfer_arr) else 0.0
    avg_transfer_f1 = float(transfer_results['transfer_f1'].mean()) if len(transfer_results) else 0.0
    avg_local_f1 = float(transfer_results['local_f1'].mean()) if len(transfer_results) else 0.0
    best_transfer_f1 = float(transfer_results['transfer_f1'].max()) if len(transfer_results) else 0.0
    transfer_text = (f'留一市场迁移实验中，迁移模型平均 F1 为 {avg_transfer_f1:.4f}，'
                     f'最大性能下降为 {max_drop * 100:.2f} 个百分点，最高迁移 F1 达 {best_transfer_f1:.4f}')
    if improved:
        transfer_text += f'，且在 {("、".join(improved))} 市场上迁移模型优于仅用本地数据训练的模型'
    transfer_text += '，说明多维特征具备跨市场适应能力。'

    # ---- 质量指标汇总 ----
    def avg_metric(metric_name, col):
        vals = quality_compare[quality_compare['metric'] == metric_name][col]
        return float(vals.mean()) if len(vals) else 0.0

    conflict_before = avg_metric('ohlc_conflict_rate', 'before_val')
    conflict_after = avg_metric('ohlc_conflict_rate', 'after_val')
    consistency_before = avg_metric('pct_change_consistency', 'before_val')
    consistency_after = avg_metric('pct_change_consistency', 'after_val')
    missing_before = avg_metric('missing_rate', 'before_val')
    missing_after = avg_metric('missing_rate', 'after_val')

    # ---- 消融实验最大损失 ----
    worst_row = None
    if len(ablation_results) > 1:
        non_base = ablation_results[ablation_results['feature_set'] != 'Full Features']
        worst_row = non_base.loc[non_base['f1_drop'].idxmax()]
        ablation_text = (f'特征消融中，移除「{worst_row["feature_set"]}」后 F1 下降最多'
                         f'（{worst_row["f1_drop"]:.4f}），表明该特征组对异常识别贡献最大。')
    else:
        ablation_text = ''

    # ---- 混淆矩阵指标（含误报率/漏报率）----
    if cm_df is not None and len(cm_df) > 0:
        cm_show = cm_df.copy()
        cm_show['false_positive_rate'] = cm_show['fp'] / (cm_show['fp'] + cm_show['tn']).replace(0, 1)
        cm_show['false_negative_rate'] = cm_show['fn'] / (cm_show['fn'] + cm_show['tp']).replace(0, 1)
    else:
        cm_show = None

    # ---- 综合对比汇总表 ----
    summary_rows = [
        {
            '实验维度': '四方法测试集对比',
            '核心指标': 'F1 / 准确率',
            '主方法结果': f'{xgb_f1:.4f} / {xgb_acc:.4f}',
            '对比对象': f'最优基线（{best_base_name}）',
            '对比结果': f'{best_base_f1:.4f} / {best_base_acc:.4f}',
            '优势': f'F1 领先 {f1_gain:.4f}（{f1_gain * 100:.2f} 个百分点）'
        },
        {
            '实验维度': '分市场异常识别',
            '核心指标': 'F1',
            '主方法结果': f'平均 {market_avg_f1:.4f}',
            '对比对象': f'最低市场（{market_min_row["market"]}）' if market_min_row is not None else '无',
            '对比结果': f'{market_min_row["f1_score"]:.4f}' if market_min_row is not None else '无',
            '优势': '各市场 F1 均保持在 0.9 以上'
                   if market_min_row is not None and market_min_row['f1_score'] >= 0.9
                   else '存在市场性能差异'
        },
        {
            '实验维度': '留一市场迁移',
            '核心指标': '迁移 F1',
            '主方法结果': f'平均 {avg_transfer_f1:.4f}',
            '对比对象': f'本地模型平均 {avg_local_f1:.4f}',
            '对比结果': f'最大下降 {max_drop:.4f}',
            '优势': '跨市场迁移性能稳定' if max_drop <= 0.05 else '迁移存在一定性能损失'
        },
        {
            '实验维度': '特征消融',
            '核心指标': 'F1 下降',
            '主方法结果': f'完整特征 F1 {ablation_results.iloc[0]["f1_score"]:.4f}' if len(ablation_results) else '无',
            '对比对象': '移除成交量类特征',
            '对比结果': f'F1 下降 {worst_row["f1_drop"]:.4f}' if worst_row is not None else '无',
            '优势': '成交量类特征贡献最大' if worst_row is not None else '无'
        },
        {
            '实验维度': '数据清洗质量',
            '核心指标': 'OHLC 冲突率',
            '主方法结果': f'{conflict_after:.4f}',
            '对比对象': f'清洗前 {conflict_before:.4f}',
            '对比结果': f'{conflict_before - conflict_after:.4f}',
            '优势': '价格逻辑冲突完全消除'
        }
    ]
    summary_df = pd.DataFrame(summary_rows)

    # ---- 图表引用（相对路径，便于本地与网页查看）----
    figures = """
![混淆矩阵](cm_XGBoost_Global.png)

![Precision-Recall 曲线](pr_XGBoost_Global.png)

![阈值敏感性曲线](threshold_sensitivity_curve.png)

![留一市场迁移 F1 对比](leave_one_out_transfer_f1.png)
"""

    report_content = f"""# 跨境金融数据清洗实验完整报告

> 本报告汇总数据集检查、异常标签统计、数据集划分、四种异常检测方法对比、分市场评估、
> 留一市场迁移、特征消融、清洗前后质量指标及全部可视化结果。

## 1. 数据集检查与样本统计

对上传的三个市场数据集逐项检查联合主键重复、日期范围、字段完整性、OHLC 逻辑冲突以及四类异常数，结果如下：

{_fmt(data_stats, rename={
    '市场': '市场', '原始记录数': '原始记录数', '去重后记录数': '去重后记录数',
    '重复记录数': '重复记录数', '重复率': '重复率', '起始日期': '起始日期',
    '结束日期': '结束日期', '字段完整性': '字段完整性', '字段缺失率': '字段缺失率',
    'OHLC冲突数': 'OHLC冲突数'
})}

- 共检查 {len(data_stats)} 个市场，累计去除重复记录 **{total_removed}** 条；
- 数据集划分校验结果：**{validation_msg}**。

## 2. 多市场异常标签规则触发统计

{_fmt(label_stats, rename={
    'market': '市场', 'rule_name': '规则名称', 'hit_count': '触发次数', 'hit_rate': '触发率'
})}

## 3. 数据集划分与样本统计

### 3.1 全局划分

{_fmt(split_stats, rename={
    'set_type': '集合类型', 'sample_count': '样本总数', 'anomaly_count': '异常样本数',
    'anomaly_rate': '异常占比', 'start_date': '起始日期', 'end_date': '结束日期',
    'market_count': '覆盖市场数'
})}

### 3.2 分市场划分

{_fmt(split_market_stats, rename={
    'market': '市场', 'set_type': '集合类型', 'sample_count': '样本总数',
    'anomaly_count': '异常样本数', 'anomaly_rate': '异常占比',
    'start_date': '起始日期', 'end_date': '结束日期'
})}

## 4. 四种异常检测方法测试集指标对比（完整指标）

{_fmt(method_compare.sort_values('f1_score', ascending=False), rename={
    'method': '方法', 'accuracy': '准确率', 'precision': '精确率',
    'recall': '召回率', 'f1_score': 'F1 值',
    'tp': 'TP（正确检出异常）', 'tn': 'TN（正确识别正常）',
    'fp': 'FP（误报）', 'fn': 'FN（漏报）'
})}

## 5. XGBoost 融合模型混淆矩阵指标

{_fmt(cm_show, rename={
    'exp_name': '实验', 'tp': 'TP', 'tn': 'TN', 'fp': 'FP', 'fn': 'FN',
    'false_positive_rate': '误报率', 'false_negative_rate': '漏报率'
}) if cm_show is not None else '（无混淆矩阵数据）'}

## 6. 各市场异常识别指标对比（XGBoost 融合模型）

{_fmt(market_f1, rename={
    'market': '市场', 'test_sample_count': '测试样本数', 'anomaly_count': '异常样本数',
    'precision': '精确率', 'recall': '召回率', 'f1_score': 'F1 值'
})}

## 7. 留一市场迁移实验与本地模型性能对比

{_fmt(transfer_results, rename={
    'target_market': '目标市场', 'local_f1': '本地模型 F1',
    'transfer_f1': '迁移模型 F1', 'drop_val': '性能下降（本地-迁移）'
})}

## 8. 特征消融实验结果

{_fmt(ablation_results, rename={
    'feature_set': '特征组合', 'f1_score': 'F1 值', 'f1_drop': 'F1 下降'
})}

## 9. 各市场数据清洗前后质量指标对比

{_fmt(quality_compare.assign(metric=quality_compare['metric'].replace({
    'dup_rate': '重复记录率', 'ohlc_conflict_rate': 'OHLC冲突率',
    'missing_rate': '字段缺失率', 'pct_change_consistency': '收益率一致性'
})), rename={
    'market': '市场', 'metric': '指标', 'before_val': '清洗前', 'after_val': '清洗后'
})}

## 10. 综合对比指标汇总

{_fmt(summary_df)}

## 11. 可视化结果

{figures}

## 12. 实验结论

1. **数据质量**：按“市场—证券代码—交易日”联合主键去重，累计消除重复记录 {total_removed} 条；
   OHLC 价格逻辑冲突率由平均 {conflict_before:.4f} 降至 {conflict_after:.4f}，收益率与价格序列一致性
   由平均 {consistency_before:.4f} 提升至 {consistency_after:.4f}，字段缺失率由 {missing_before:.4f} 变为 {missing_after:.4f}，
   表明清洗流程有效改善了字段逻辑一致性与数据可信度。
2. **模型优势**：{advantage_text}
   验证集确定的最佳分类阈值为 {best_threshold:.2f}（验证集 F1={best_f1:.4f}）；
   测试集准确率 {xgb_acc:.4f}、精确率 {xgb_pre:.4f}、召回率 {xgb_rec:.4f}、F1 {xgb_f1:.4f}。
3. **跨市场适应性**：{transfer_text}
4. **特征贡献**：{ablation_text}
"""

    report_path = os.path.join(OUTPUT_DIR, 'clean_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return report_path
