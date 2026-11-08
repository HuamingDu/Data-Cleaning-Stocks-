import pandas as pd
import numpy as np

def build_features(df):
    """
    多维特征工程
    价格类、成交量类、横截面统计特征、日历特征
    """
    df = df.sort_values(['market', 'stock_code', 'date'])
    
    # 1. 价格类特征
    # 原始价格标准化 (简单的 Min-Max 或 Z-score，这里使用相对变动)
    df['price_mean_5'] = df.groupby(['market', 'stock_code'])['close'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['price_std_5'] = df.groupby(['market', 'stock_code'])['close'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    df['price_zscore'] = (df['close'] - df['price_mean_5']) / (df['price_std_5'] + 1e-6)
    
    # 收益率特征
    df['ret_lag1'] = df.groupby(['market', 'stock_code'])['pct_change'].shift(1)
    df['ret_mean_5'] = df.groupby(['market', 'stock_code'])['pct_change'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    # 2. 成交量类特征
    df['vol_mean_5'] = df.groupby(['market', 'stock_code'])['volume'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    df['vol_ratio'] = df['volume'] / (df['vol_mean_5'] + 1e-6)
    
    # 3. 横截面特征 (同一天、同一市场内)
    df['market_ret_rank'] = df.groupby(['market', 'date'])['pct_change'].rank(pct=True)
    df['market_vol_rank'] = df.groupby(['market', 'date'])['volume'].rank(pct=True)
    
    # 4. 日历特征
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # 清理 NaN (由于 shift 和 rolling 产生的)
    df = df.fillna(0)
    
    # 定义特征列
    feature_cols = [
        'open', 'close', 'high', 'low', 'volume', 'pct_change',
        'price_zscore', 'ret_lag1', 'ret_mean_5', 
        'vol_ratio', 'market_ret_rank', 'market_vol_rank',
        'day_of_week', 'month', 'ohlc_conflict'
    ]
    
    return df, feature_cols
