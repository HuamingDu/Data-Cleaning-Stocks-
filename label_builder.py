import pandas as pd
import numpy as np

def build_silver_labels(df, threshold=1):
    """
    基于 4 条共识规则构建银标签
    df: 包含 stock_code, date, close, volume, pct_change, market 等字段
    threshold: 命中规则数阈值，根据最新要求改为 1
    """
    df = df.sort_values(['market', 'stock_code', 'date'])
    
    # 1. 收益跳变 (Price Jump)
    # 定义：当日收益率绝对值超过 10% (对于大多数市场是极端情况，或者根据 MAD 计算)
    # 这里使用简单的固定阈值 + 滚动 MAD
    df['rule_1_jump'] = (df['pct_change'].abs() > 0.1).astype(int)
    
    # 2. 相对市场偏离 (Market Deviation)
    # 计算市场每日平均收益率
    market_avg = df.groupby(['market', 'date'])['pct_change'].transform('mean')
    df['rule_2_dev'] = ((df['pct_change'] - market_avg).abs() > 0.08).astype(int)
    
    # 3. 成交量突变 (Volume Jump)
    # 当日成交量是过去 5 日平均成交量的 3 倍以上或 1/3 以下
    vol_mean_5 = df.groupby(['market', 'stock_code'])['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df['rule_3_vol'] = ((df['volume'] > 3 * vol_mean_5) | (df['volume'] < vol_mean_5 / 3)).astype(int)
    
    # 4. 量价共振 (Price-Volume Resonance)
    # 价格大幅变动但成交量极小，或者价格平稳但成交量巨大
    df['rule_4_pv'] = (((df['pct_change'].abs() > 0.05) & (df['volume'] < vol_mean_5 * 0.2)) | \
                       ((df['pct_change'].abs() < 0.01) & (df['volume'] > vol_mean_5 * 5))).astype(int)
    
    # 计算命中规则总数
    df['hit_rules_count'] = df[['rule_1_jump', 'rule_2_dev', 'rule_3_vol', 'rule_4_pv']].sum(axis=1)
    
    # 生成银标签 y
    df['label'] = (df['hit_rules_count'] >= threshold).astype(int)
    
    return df

def get_label_stats(df):
    """统计各规则触发情况"""
    stats = []
    markets = df['market'].unique()
    rules = ['rule_1_jump', 'rule_2_dev', 'rule_3_vol', 'rule_4_pv']
    
    for m in markets:
        m_df = df[df['market'] == m]
        total = len(m_df)
        for r in rules:
            hits = m_df[r].sum()
            stats.append({
                'market': m,
                'rule_name': r,
                'hit_count': int(hits),
                'hit_rate': hits / total if total > 0 else 0
            })
    return pd.DataFrame(stats)


def compute_anomaly_type_counts(df):
    """
    按市场统计四类异常记录数（用于《去重前后三市场数据表》）：
    - 跳价异常数：|当日收益率| > 10%
    - 异常波动率数：20 日滚动收益率标准差超过该市场 95% 分位
    - 跨市场价差异动数：市场日均收益率相对当日跨市场均值的偏离超过全样本 95% 分位
    - 时序离群点数：20 日滚动 Z 分数绝对值 > 3
    """
    df = df.sort_values(['market', 'stock_code', 'date']).copy()

    # 1. 跳价异常
    df['_jump'] = (df['pct_change'].abs() > 0.10).astype(int)

    # 2. 异常波动率（20 日滚动标准差的市场 95% 分位阈值）
    roll_std = df.groupby(['market', 'stock_code'])['pct_change'].transform(
        lambda x: x.rolling(20, min_periods=20).std()
    )
    vol_threshold = roll_std.quantile(0.95)
    df['_vol_anomaly'] = (roll_std > vol_threshold).astype(int)

    # 3. 跨市场价差异动（按交易日比较各市场平均收益率）
    daily = df.groupby(['market', 'date'], as_index=False)['pct_change'].mean()
    pivot = daily.pivot(index='date', columns='market', values='pct_change')
    cross_mean = pivot.mean(axis=1)
    deviation = pivot.sub(cross_mean, axis=0).abs()
    # 以全样本偏离度的 95% 分位为阈值（三市场情形下 2σ 无法被触发）
    dev_threshold = deviation.stack().quantile(0.95)
    flag_matrix = deviation.gt(dev_threshold)
    flags = flag_matrix.stack().rename('_cross_flag').reset_index()
    flags.columns = ['date', 'market', '_cross_flag']
    df = df.merge(flags, on=['date', 'market'], how='left')
    df['_cross_flag'] = df['_cross_flag'].fillna(False).astype(int)

    # 4. 时序离群点（基于历史 20 日滚动 Z 分数）
    roll_mean = df.groupby(['market', 'stock_code'])['pct_change'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).mean()
    )
    roll_std_z = df.groupby(['market', 'stock_code'])['pct_change'].transform(
        lambda x: x.shift(1).rolling(20, min_periods=20).std()
    )
    z_score = (df['pct_change'] - roll_mean) / (roll_std_z + 1e-6)
    df['_ts_outlier'] = (z_score.abs() > 3).astype(int)

    # 5. 按市场汇总
    agg = df.groupby('market').agg(
        跳价异常数=('_jump', 'sum'),
        异常波动率数=('_vol_anomaly', 'sum'),
        跨市场价差异动数=('_cross_flag', 'sum'),
        时序离群点数=('_ts_outlier', 'sum')
    ).reset_index()

    for col in ['跳价异常数', '异常波动率数', '跨市场价差异动数', '时序离群点数']:
        agg[col] = agg[col].astype(int)

    return agg
