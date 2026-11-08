import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from config import SPLIT_RATIO

# 四种对比方法（与论文表格保持一致）
METHOD_3SIGMA = '3σ 原则'
METHOD_ISOFOREST = 'Isolation Forest'
METHOD_DYNAMIC = '动态阈值规则'
METHOD_XGBOOST = 'XGBoost 融合模型'
COMPARE_METHODS = [METHOD_3SIGMA, METHOD_ISOFOREST, METHOD_DYNAMIC, METHOD_XGBOOST]


def time_series_split(df):
    """
    按“市场—时间”严格切分数据集：
    每个市场内部先提取交易日序列，再按比例按“整日”切分 Train / Val / Test，
    保证同一交易日的记录不会同时出现在两个子集中，且不存在未来信息泄漏。
    """
    df = df.sort_values(['market', 'stock_code', 'date'])
    parts = {'train': [], 'val': [], 'test': []}

    for market, g in df.groupby('market', sort=True):
        g = g.sort_values('date')
        dates = pd.Series(sorted(g['date'].unique()))
        n_dates = len(dates)
        train_end = int(n_dates * SPLIT_RATIO['train'])
        val_end = int(n_dates * (SPLIT_RATIO['train'] + SPLIT_RATIO['val']))

        train_dates = dates.iloc[:train_end]
        val_dates = dates.iloc[train_end:val_end]
        test_dates = dates.iloc[val_end:]

        parts['train'].append(g[g['date'].isin(train_dates)])
        parts['val'].append(g[g['date'].isin(val_dates)])
        parts['test'].append(g[g['date'].isin(test_dates)])

    train_df = pd.concat(parts['train'], ignore_index=True).sort_values(['market', 'date']).reset_index(drop=True)
    val_df = pd.concat(parts['val'], ignore_index=True).sort_values(['market', 'date']).reset_index(drop=True)
    test_df = pd.concat(parts['test'], ignore_index=True).sort_values(['market', 'date']).reset_index(drop=True)

    return train_df, val_df, test_df


def evaluate_metrics(y_true, y_pred):
    """计算评估指标（准确率、精确率、召回率、F1）及混淆矩阵计数"""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def _dynamic_threshold_predict(train_df, test_df, window=20, k=3):
    """
    动态阈值规则：对每只股票使用历史（不含当日）滚动均值与滚动标准差，
    当 |收益率 - 滚动均值| > k * 滚动标准差 时判定为异常。
    阈值随个股波动率自适应，避免静态 3σ 在高低波动期失效。
    """
    cols = ['market', 'stock_code', 'date', 'pct_change']
    hist = pd.concat([train_df[cols], test_df[cols]])
    hist = hist.sort_values(['market', 'stock_code', 'date'])
    hist = hist.reset_index(drop=True)

    grp = hist.groupby(['market', 'stock_code'])['pct_change']
    roll_mean = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
    roll_std = grp.transform(lambda x: x.shift(1).rolling(window, min_periods=5).std())
    z = (hist['pct_change'] - roll_mean) / (roll_std + 1e-6)
    hist['pred'] = (z.abs() > k).astype(int)

    # 通过 (market, stock_code, date) 映射回测试集
    test_key = test_df[['market', 'stock_code', 'date']].copy()
    test_key = test_key.merge(hist[['market', 'stock_code', 'date', 'pred']],
                              on=['market', 'stock_code', 'date'], how='left')
    pred = test_key['pred'].fillna(0).astype(int).to_numpy()

    # 同时返回异常分数（|z|），用于绘制 PR 曲线
    score_key = test_df[['market', 'stock_code', 'date']].copy()
    score_key = score_key.merge(hist[['market', 'stock_code', 'date']].assign(score=z.abs().to_numpy()),
                                on=['market', 'stock_code', 'date'], how='left')
    score = score_key['score'].fillna(0.0).to_numpy()

    return pred, score


def run_baselines(train_df, test_df, feature_cols):
    """
    运行四种异常检测方法：
    1) 3σ 原则           —— 静态统计阈值
    2) Isolation Forest  —— 无监督孤立森林
    3) 动态阈值规则       —— 滚动窗口自适应阈值
    4) XGBoost 融合模型   —— 多维特征监督学习（本文方法）
    """
    results = []
    scores = {}

    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    # 1. 3σ 原则 (基于训练集收益率分布，双侧检验)
    mean_ret = train_df['pct_change'].mean()
    std_ret = train_df['pct_change'].std()
    score_3sigma = ((test_df['pct_change'] - mean_ret).abs() / (std_ret + 1e-6)).to_numpy()
    y_pred_3sigma = (score_3sigma > 3).astype(int)
    scores[METHOD_3SIGMA] = score_3sigma
    results.append({'method': METHOD_3SIGMA, **evaluate_metrics(y_test, y_pred_3sigma)})

    # 2. Isolation Forest (污染率与训练集异常比例对齐，避免人为放大误报)
    contamination = float(y_train.mean()) if y_train.mean() > 0 else 0.01
    contamination = min(max(contamination, 0.005), 0.1)
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
    iso.fit(X_train)
    score_iso = -iso.decision_function(X_test)
    y_pred_iso = (iso.predict(X_test) == -1).astype(int)
    scores[METHOD_ISOFOREST] = score_iso
    results.append({'method': METHOD_ISOFOREST, **evaluate_metrics(y_test, y_pred_iso)})

    # 3. 动态阈值规则
    y_pred_dynamic, score_dynamic = _dynamic_threshold_predict(train_df, test_df)
    scores[METHOD_DYNAMIC] = score_dynamic
    results.append({'method': METHOD_DYNAMIC, **evaluate_metrics(y_test, y_pred_dynamic)})

    # 4. XGBoost 融合模型 (本文方法)
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric='logloss',
    )
    model.fit(X_train, y_train)
    y_pred_xgb = model.predict(X_test)
    scores[METHOD_XGBOOST] = model.predict_proba(X_test)[:, 1]
    results.append({'method': METHOD_XGBOOST, **evaluate_metrics(y_test, y_pred_xgb)})

    return pd.DataFrame(results), model, scores


def predict_with_threshold(model, df, feature_cols, threshold):
    """按指定阈值输出 0/1 预测"""
    probs = model.predict_proba(df[feature_cols])[:, 1]
    return (probs >= threshold).astype(int), probs


def run_ablation(train_df, test_df, all_features, threshold=0.5):
    """特征消融实验（统一使用验证集确定的最佳阈值，保证可比性）"""
    groups = {
        'Full Features': all_features,
        'Only Raw Price/Vol': ['open', 'close', 'high', 'low', 'volume', 'pct_change'],
        'No Rolling': [c for c in all_features if 'mean' not in c and 'std' not in c and 'lag' not in c
                       and 'ratio' not in c and 'zscore' not in c],
        'No Cross-Sectional': [c for c in all_features if c not in ('market_ret_rank', 'market_vol_rank')],
        'No Volume': [c for c in all_features if c not in ('volume', 'vol_ratio', 'market_vol_rank')],
    }
    # 过滤空特征组，避免无意义实验
    groups = {k: v for k, v in groups.items() if len(v) > 0}

    results = []
    base_f1 = None
    for name, cols in groups.items():
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='logloss'
        )
        model.fit(train_df[cols], train_df['label'])
        probs = model.predict_proba(test_df[cols])[:, 1]
        y_pred = (probs >= threshold).astype(int)
        f1 = f1_score(test_df['label'], y_pred, zero_division=0)

        if name == 'Full Features':
            base_f1 = f1
            drop = 0.0
        else:
            drop = (base_f1 - f1) if base_f1 is not None else 0.0

        results.append({'feature_set': name, 'f1_score': f1, 'f1_drop': drop})

    return pd.DataFrame(results)


def run_transfer_exp(full_df, markets, feature_cols, threshold=0.5):
    """
    留一市场迁移实验：
    - 本地模型：仅使用目标市场训练集训练
    - 迁移模型：使用其余市场训练集训练，在目标市场测试集上评估
    """
    results = []
    for target in markets:
        target_df = full_df[full_df['market'] == target]
        target_train, _, target_test = time_series_split(target_df)

        local_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='logloss'
        )
        local_model.fit(target_train[feature_cols], target_train['label'])
        local_pred, _ = predict_with_threshold(local_model, target_test, feature_cols, threshold)
        local_f1 = f1_score(target_test['label'], local_pred, zero_division=0)

        others_df = full_df[full_df['market'] != target]
        others_train, _, _ = time_series_split(others_df)
        transfer_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='logloss'
        )
        transfer_model.fit(others_train[feature_cols], others_train['label'])
        transfer_pred, _ = predict_with_threshold(transfer_model, target_test, feature_cols, threshold)
        transfer_f1 = f1_score(target_test['label'], transfer_pred, zero_division=0)

        results.append({
            'target_market': target,
            'local_f1': local_f1,
            'transfer_f1': transfer_f1,
            'drop_val': local_f1 - transfer_f1
        })

    return pd.DataFrame(results)
