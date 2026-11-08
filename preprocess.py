import pandas as pd
import numpy as np
from utils.db_helper import insert_log

REQUIRED_COLS = ['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'pct_change']


def load_and_preprocess(file_path, market_name):
    """加载并预处理单个市场的数据"""
    df = pd.read_csv(file_path, encoding='utf-8-sig')

    # 1. 时间转换
    df['date'] = pd.to_datetime(df['date'])
    df['market'] = market_name

    # 记录原始数量
    raw_count = len(df)

    # 2. 联合主键 (market, stock_code, date) 去重
    # 论文要求：按“市场—证券代码—交易日”联合主键去重
    df_dedup = df.drop_duplicates(subset=['market', 'stock_code', 'date'], keep='first')
    dedup_count = len(df_dedup)

    if raw_count > dedup_count:
        removed = raw_count - dedup_count
        insert_log('PREPROCESS', 'ALL', 'N/A', 'duplicates', raw_count, dedup_count, f'Removed {removed} duplicate records in {market_name}')

    # 3. OHLC 逻辑校验
    # high >= max(open, close), low <= min(open, close)
    df_dedup = df_dedup.copy()
    df_dedup['ohlc_conflict'] = 0

    mask = (df_dedup['high'] < df_dedup[['open', 'close']].max(axis=1)) | \
           (df_dedup['low'] > df_dedup[['open', 'close']].min(axis=1))

    df_dedup.loc[mask, 'ohlc_conflict'] = 1

    # 记录逻辑冲突日志
    conflict_count = int(mask.sum())
    if conflict_count > 0:
        insert_log('PREPROCESS', 'ALL', 'N/A', 'ohlc_logic', 0, conflict_count, f'Found {conflict_count} OHLC logic conflicts in {market_name}')

    # 4. 字段标准化
    cols = ['open', 'close', 'high', 'low', 'volume', 'turnover', 'pct_change']
    for col in cols:
        df_dedup[col] = pd.to_numeric(df_dedup[col], errors='coerce')

    df_dedup['market'] = market_name

    return df_dedup


def compute_quality_metrics(df):
    """
    计算数据质量指标（用于清洗前后对比）：
    - dup_rate              重复记录率（按市场—证券代码—交易日联合主键）
    - ohlc_conflict_rate    OHLC 价格逻辑冲突率
    - missing_rate          必要字段缺失率
    - pct_change_consistency 收益率与价格序列的一致性（|报告收益率-价格推算收益率|<=0.001）
    """
    if len(df) == 0:
        return {'dup_rate': 0.0, 'ohlc_conflict_rate': 0.0,
                'missing_rate': 0.0, 'pct_change_consistency': 0.0}

    df = df.sort_values(['market', 'stock_code', 'date'])

    dup_rate = float(df.duplicated(subset=['market', 'stock_code', 'date']).mean())

    if {'open', 'close', 'high', 'low'}.issubset(df.columns):
        conflict = ((df['high'] < df[['open', 'close']].max(axis=1)) |
                    (df['low'] > df[['open', 'close']].min(axis=1)))
        ohlc_conflict_rate = float(conflict.mean())
    else:
        ohlc_conflict_rate = 0.0

    present_cols = [c for c in REQUIRED_COLS if c in df.columns]
    missing_rate = float(df[present_cols].isnull().mean().mean()) if present_cols else 0.0

    derived = df.groupby(['market', 'stock_code'])['close'].pct_change()
    diff = (df['pct_change'] - derived).abs()
    valid = derived.notna()
    pct_change_consistency = float((diff[valid] <= 1e-3).mean()) if valid.any() else 1.0

    return {
        'dup_rate': dup_rate,
        'ohlc_conflict_rate': ohlc_conflict_rate,
        'missing_rate': missing_rate,
        'pct_change_consistency': pct_change_consistency
    }


def apply_cleaning(df, anomaly_preds=None):
    """
    执行核心数据清洗功能：
    1. OHLC 逻辑修复：纠正最高价/最低价逻辑冲突
    2. 异常数据修复：基于模型预测结果对异常样本的收盘价做滚动平滑
    3. 收益率一致性修复：由修复后的收盘价重算 pct_change，保证收益率与价格序列一致
    """
    df_clean = df.copy()
    df_clean = df_clean.sort_values(['market', 'stock_code', 'date']).reset_index(drop=True)

    # 记录异常标记（与传入预测结果一一对应）
    if anomaly_preds is not None:
        df_clean['_is_anomaly'] = np.asarray(anomaly_preds).astype(int)
    else:
        df_clean['_is_anomaly'] = 0

    # 1. OHLC 逻辑修复
    def repair_ohlc(frame):
        frame['high'] = frame[['open', 'close', 'high', 'low']].max(axis=1)
        frame['low'] = frame[['open', 'close', 'low']].min(axis=1)
        return frame

    df_clean = repair_ohlc(df_clean)
    insert_log('CLEANING', 'ALL', 'N/A', 'ohlc_repair', 'N/A', 'N/A', 'Repaired OHLC logic conflicts')

    # 2. 异常数据修复：使用该股票过去 5 日（不含当日）收盘价均值替代异常点
    anomaly_mask = df_clean['_is_anomaly'] == 1
    if anomaly_mask.any():
        prev_mean = df_clean.groupby(['market', 'stock_code'])['close'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        df_clean.loc[anomaly_mask, 'close'] = prev_mean[anomaly_mask].fillna(df_clean.loc[anomaly_mask, 'close'])
        # 收盘价变化后再次修复 OHLC 逻辑
        df_clean = repair_ohlc(df_clean)
        insert_log('CLEANING', 'ALL', 'N/A', 'anomaly_fix', 0, int(anomaly_mask.sum()),
                   f'Fixed {int(anomaly_mask.sum())} anomalies using rolling mean smoothing')

    # 3. 收益率一致性修复：由修复后的收盘价重算 pct_change（每只股票首日记为 0）
    df_clean['pct_change'] = df_clean.groupby(['market', 'stock_code'])['close'].pct_change().fillna(0.0)

    # 重置冲突标记
    df_clean['ohlc_conflict'] = 0
    df_clean = df_clean.drop(columns=['_is_anomaly'])

    return df_clean
