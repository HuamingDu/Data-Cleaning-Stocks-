import os
import sys
import pandas as pd
import numpy as np
import json

# 将当前目录添加到系统路径，修复导入问题
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, OUTPUT_DIR, DATASETS, ANOMALY_RULES_THRESHOLD, SPLIT_RATIO
from utils.db_helper import init_db, save_df_to_db, insert_log
from utils.preprocess import load_and_preprocess, compute_quality_metrics, apply_cleaning, REQUIRED_COLS
from utils.label_builder import build_silver_labels, get_label_stats, compute_anomaly_type_counts
from utils.feature_build import build_features
from utils.model_exp import (time_series_split, run_baselines, run_ablation, run_transfer_exp,
                             evaluate_metrics, METHOD_XGBOOST, predict_with_threshold)
from utils.plot_draw import plot_confusion_matrix, plot_pr_curve_multi, plot_transfer_f1, plot_threshold_sensitivity
from utils.report_generator import generate_markdown_report
from sklearn.metrics import f1_score

# 清洗前后对比使用的质量指标（顺序固定，便于报告展示）
QUALITY_METRIC_KEYS = ['dup_rate', 'ohlc_conflict_rate', 'missing_rate', 'pct_change_consistency']


def resolve_market_name(filename):
    """通过配置文件映射文件名到市场名称，无法匹配时按文件名规则回退"""
    for market, ds_name in DATASETS.items():
        if ds_name == filename:
            return market
    stem = filename.rsplit('.', 1)[0]
    parts = stem.split('_')
    name = parts[1] if len(parts) > 1 else stem
    return name.replace(' ', '')


def validate_split(train_df, val_df, test_df, full_df):
    """
    校验数据集划分与样本统计是否正确：
    1) 三个子集样本数之和等于全量样本数
    2) 每个市场在三个子集中均有样本
    3) 每个市场内部严格按时间顺序，无未来信息泄漏
    4) 全局划分比例与配置比例一致
    """
    errors = []

    total = len(train_df) + len(val_df) + len(test_df)
    if total != len(full_df):
        errors.append(f'样本数之和不等于全量({total}!={len(full_df)})')

    markets = sorted(full_df['market'].unique())
    for m in markets:
        for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            if (d['market'] == m).sum() == 0:
                errors.append(f'{m}-{name} 无样本')

        tr = train_df[train_df['market'] == m]
        va = val_df[val_df['market'] == m]
        te = test_df[test_df['market'] == m]
        if len(tr) and len(va):
            if set(tr['date']).intersection(set(va['date'])):
                errors.append(f'{m} Train/Val 存在相同交易日')
            if tr['date'].max() > va['date'].min():
                errors.append(f'{m} Train/Val 时间倒置')
        if len(va) and len(te):
            if set(va['date']).intersection(set(te['date'])):
                errors.append(f'{m} Val/Test 存在相同交易日')
            if va['date'].max() > te['date'].min():
                errors.append(f'{m} Val/Test 时间倒置')

    for name, d, target in [('Train', train_df, SPLIT_RATIO['train']),
                            ('Val', val_df, SPLIT_RATIO['val']),
                            ('Test', test_df, SPLIT_RATIO['test'])]:
        ratio = len(d) / len(full_df) if len(full_df) else 0
        if abs(ratio - target) > 0.01:
            errors.append(f'{name} 占比 {ratio:.4f} 偏离目标 {target:.2f}')

    if errors:
        msg = '未通过：' + '；'.join(errors)
    else:
        msg = f'通过：三市场均按 {SPLIT_RATIO["train"]:.0%}/{SPLIT_RATIO["val"]:.0%}/{SPLIT_RATIO["test"]:.0%} 时间顺序划分，无重叠与泄漏'
    print(f'>>> 数据集划分校验: {msg}')
    return msg


def run_full_pipeline(custom_threshold=None):
    print(">>> 启动跨境金融数据清洗流水线...")

    # 0. 初始化数据库
    try:
        init_db()
        print("数据库初始化完成。")
    except Exception as e:
        print(f"数据库初始化失败: {e}。")
        return

    # 1. 加载与预处理（含数据集检查与统计）
    all_market_dfs = []
    data_stats_list = []
    raw_by_market = {}
    before_metrics_by_market = {}

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    if len(csv_files) < 3:
        print(f"警告: 当前 data 目录下仅有 {len(csv_files)} 个 CSV 文件，建议上传至少三个市场的数据。")

    for filename in csv_files:
        path = os.path.join(DATA_DIR, filename)
        market = resolve_market_name(filename)

        raw_df = pd.read_csv(path, encoding='utf-8-sig')
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        raw_df['market'] = market

        missing_cols = [c for c in REQUIRED_COLS if c not in raw_df.columns]
        if missing_cols:
            print(f"跳过文件 {filename}: 缺失字段 {missing_cols}")
            continue

        # 清洗前质量检查（基于原始上传数据）
        before_metrics = compute_quality_metrics(raw_df)
        raw_by_market[market] = raw_df
        before_metrics_by_market[market] = before_metrics

        # 预处理：联合主键去重 + 字段标准化 + OHLC 校验
        df_clean = load_and_preprocess(path, market)

        data_stats_list.append({
            '市场': market,
            '原始记录数': len(raw_df),
            '去重后记录数': len(df_clean),
            '重复记录数': len(raw_df) - len(df_clean),
            '重复率': (len(raw_df) - len(df_clean)) / len(raw_df) if len(raw_df) else 0.0,
            '起始日期': str(raw_df['date'].min().date()),
            '结束日期': str(raw_df['date'].max().date()),
            '字段完整性': '通过',
            '字段缺失率': float(raw_df[REQUIRED_COLS].isnull().mean().mean()),
            'OHLC冲突数': int(before_metrics['ohlc_conflict_rate'] * len(raw_df))
        })

        all_market_dfs.append(df_clean)

    if not all_market_dfs:
        print("错误: 没有可用的数据集进行实验。请将 CSV 文件放入 data 目录。")
        return

    full_df = pd.concat(all_market_dfs, ignore_index=True)
    save_df_to_db(full_df, 'raw_data')

    # 四类异常数统计（需三市场合并后计算跨市场价差异动）
    anomaly_counts = compute_anomaly_type_counts(full_df).rename(columns={'market': '市场'})
    print(f">>> 数据集检查完成：共 {len(full_df)} 条有效记录，覆盖 {full_df['market'].nunique()} 个市场。")

    # 2. 银标签生成
    full_df = build_silver_labels(full_df, threshold=ANOMALY_RULES_THRESHOLD)
    label_stats = get_label_stats(full_df)
    save_df_to_db(label_stats, 'label_stat_table')

    # 3. 特征工程
    full_df, feature_cols = build_features(full_df)

    # 4. 数据集划分（按市场内时间顺序）
    train_df, val_df, test_df = time_series_split(full_df)

    split_stats = pd.DataFrame([
        {
            'set_type': name,
            'sample_count': len(d),
            'anomaly_count': int(d['label'].sum()),
            'anomaly_rate': d['label'].mean(),
            'start_date': str(d['date'].min().date()),
            'end_date': str(d['date'].max().date()),
            'market_count': int(d['market'].nunique())
        }
        for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]
    ])

    split_market_stats = pd.DataFrame([
        {
            'market': m,
            'set_type': name,
            'sample_count': len(d[(d['market'] == m)]),
            'anomaly_count': int(d[d['market'] == m]['label'].sum()),
            'anomaly_rate': d[d['market'] == m]['label'].mean() if (d['market'] == m).sum() else 0.0,
            'start_date': str(d[d['market'] == m]['date'].min().date()) if (d['market'] == m).sum() else '',
            'end_date': str(d[d['market'] == m]['date'].max().date()) if (d['market'] == m).sum() else ''
        }
        for m in sorted(full_df['market'].unique())
        for name, d in [('Train', train_df), ('Val', val_df), ('Test', test_df)]
    ])

    validation_msg = validate_split(train_df, val_df, test_df, full_df)
    save_df_to_db(split_stats, 'dataset_split_stat')
    save_df_to_db(split_market_stats, 'split_market_stat')

    # 5. 四种方法对比（3σ / Isolation Forest / 动态阈值 / XGBoost 融合模型）
    method_compare, xgb_model, compare_scores = run_baselines(train_df, test_df, feature_cols)

    # --- 在验证集上搜索最佳分类阈值 ---
    print(">>> 正在验证集上搜索最佳分类阈值...")
    val_probs = xgb_model.predict_proba(val_df[feature_cols])[:, 1]

    if custom_threshold is not None:
        best_threshold = custom_threshold
        best_f1 = f1_score(val_df['label'], (val_probs >= best_threshold).astype(int), zero_division=0)
        print(f">>> 使用手动指定的阈值: {best_threshold:.2f}, 验证集 F1: {best_f1:.4f}")
    else:
        search_thresholds = np.linspace(0.05, 0.95, 181)
        f1_scores = [f1_score(val_df['label'], (val_probs >= t).astype(int), zero_division=0)
                     for t in search_thresholds]
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(search_thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])
        print(f">>> 自动确定最佳阈值: {best_threshold:.2f}, 验证集 F1: {best_f1:.4f}")

    # 使用最佳阈值更新测试集指标
    y_test_probs = xgb_model.predict_proba(test_df[feature_cols])[:, 1]
    y_test_pred = (y_test_probs >= best_threshold).astype(int)
    xgb_metrics = evaluate_metrics(test_df['label'], y_test_pred)

    # 更新四种方法对比表中的 XGBoost 行（使用优化阈值后的结果）
    xgb_row = method_compare['method'] == METHOD_XGBOOST
    method_compare.loc[xgb_row, ['accuracy', 'precision', 'recall', 'f1_score', 'tp', 'tn', 'fp', 'fn']] = \
        [xgb_metrics['accuracy'], xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1_score'],
         xgb_metrics['tp'], xgb_metrics['tn'], xgb_metrics['fp'], xgb_metrics['fn']]
    save_df_to_db(method_compare, 'method_compare_result')

    # 6. 分市场 F1 评估（使用最佳阈值）
    market_f1_list = []
    for m in sorted(test_df['market'].unique()):
        m_test = test_df[test_df['market'] == m]
        m_pred = (xgb_model.predict_proba(m_test[feature_cols])[:, 1] >= best_threshold).astype(int)
        metrics = evaluate_metrics(m_test['label'], m_pred)
        market_f1_list.append({
            'market': m,
            'test_sample_count': len(m_test),
            'anomaly_count': int(m_test['label'].sum()),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    market_f1 = pd.DataFrame(market_f1_list)
    save_df_to_db(market_f1, 'market_f1_result')

    # 7. 留一市场迁移实验
    transfer_results = run_transfer_exp(full_df, sorted(full_df['market'].unique()), feature_cols,
                                        threshold=best_threshold)
    save_df_to_db(transfer_results, 'leave_one_out_result')
    plot_transfer_f1(transfer_results)

    # 8. 特征消融实验（与主模型同一阈值）
    ablation_results = run_ablation(train_df, test_df, feature_cols, threshold=best_threshold)
    save_df_to_db(ablation_results, 'ablation_result')

    # --- 核心清洗功能：执行数据修复与清洗 ---
    print(">>> 正在执行核心数据清洗与修复逻辑...")
    full_df = full_df.reset_index(drop=True)
    full_probs = xgb_model.predict_proba(full_df[feature_cols])[:, 1]
    full_preds = (full_probs >= best_threshold).astype(int)

    cleaned_full_df = apply_cleaning(full_df, anomaly_preds=full_preds)
    cleaned_full_df.to_csv(os.path.join(OUTPUT_DIR, 'clean_all_markets_full.csv'), index=False, encoding='utf-8-sig')
    save_df_to_db(cleaned_full_df, 'cleaned_full_dataset')

    print(f">>> 数据清洗完成，共处理 {len(cleaned_full_df)} 条记录，已保存至 clean_all_markets_full.csv")

    # 9. 清洗前后质量指标（真实计算，不含模拟值）
    quality_rows = []
    for market in sorted(full_df['market'].unique()):
        before = before_metrics_by_market[market]
        after = compute_quality_metrics(cleaned_full_df[cleaned_full_df['market'] == market])
        for key in QUALITY_METRIC_KEYS:
            quality_rows.append({
                'market': market,
                'metric': key,
                'before_val': before.get(key, 0.0),
                'after_val': after.get(key, 0.0)
            })
    quality_compare = pd.DataFrame(quality_rows)
    save_df_to_db(quality_compare, 'clean_quality_compare')

    # 10. 混淆矩阵与绘图
    cm = plot_confusion_matrix(test_df['label'], y_test_pred, 'XGBoost_Global')
    plot_threshold_sensitivity(val_df, xgb_model, feature_cols)

    cm_df = pd.DataFrame([{
        'exp_name': 'XGBoost_Global',
        'tp': int(cm[1, 1]), 'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]), 'fn': int(cm[1, 0])
    }])
    save_df_to_db(cm_df, 'confusion_matrix_table', if_exists='append')
    plot_pr_curve_multi(test_df['label'], compare_scores, 'XGBoost_Global')

    # 11. 保存指标 JSON
    with open(os.path.join(OUTPUT_DIR, 'test_true_label_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({**xgb_metrics, 'best_threshold': best_threshold}, f, ensure_ascii=False)

    # 12. 汇总表格输出
    data_stats = pd.DataFrame(data_stats_list).merge(anomaly_counts, on='市场', how='left')
    total_removed = int(data_stats['重复记录数'].sum())

    generate_markdown_report(
        data_stats, label_stats, split_stats, split_market_stats,
        method_compare, market_f1, transfer_results, ablation_results,
        quality_compare, xgb_metrics, best_threshold, best_f1, validation_msg,
        total_removed, cm_df
    )

    data_stats.to_csv(os.path.join(OUTPUT_DIR, 'table1_去重前后三市场数据表.csv'), index=False, encoding='utf-8-sig')
    label_stats.to_csv(os.path.join(OUTPUT_DIR, 'table2_多市场异常标签规则触发统计表.csv'), index=False, encoding='utf-8-sig')
    split_stats.to_csv(os.path.join(OUTPUT_DIR, 'table3_数据集划分与样本统计.csv'), index=False, encoding='utf-8-sig')
    split_market_stats.to_csv(os.path.join(OUTPUT_DIR, 'table3b_分市场数据集划分统计.csv'), index=False, encoding='utf-8-sig')
    method_compare.to_csv(os.path.join(OUTPUT_DIR, 'table4_不同异常检测方法测试集指标对比.csv'), index=False, encoding='utf-8-sig')
    market_f1.to_csv(os.path.join(OUTPUT_DIR, 'table5_各模型分跨境市场F1对比.csv'), index=False, encoding='utf-8-sig')
    transfer_results.to_csv(os.path.join(OUTPUT_DIR, 'table6_留一市场迁移实验本地模型对比.csv'), index=False, encoding='utf-8-sig')
    ablation_results.to_csv(os.path.join(OUTPUT_DIR, 'table7_特征消融实验结果.csv'), index=False, encoding='utf-8-sig')
    quality_compare.to_csv(os.path.join(OUTPUT_DIR, 'table8_各市场清洗前后质量指标.csv'), index=False, encoding='utf-8-sig')

    save_df_to_db(data_stats, 'dataset_check_table')
    save_df_to_db(method_compare, 'method_compare_result')

    print(f">>> 流水线运行完成。结果已输出到 {OUTPUT_DIR} 并写入数据库。")


if __name__ == "__main__":
    run_full_pipeline()
