import pandas as pd
import numpy as np
import os
from config import DATA_DIR, DATASETS

def generate_dummy_data():
    """生成模拟数据以供测试流水线，对齐论文数据量描述"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # 论文描述：2022年1月至2024年12月
    dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='B') # 使用交易日
    
    # 论文描述：涉及368条证券序列 (成份股)
    # 沪深300(300) + 恒指(82) + 蒙古TOP20(20) = 402 (略多于368，按比例分配)
    market_stock_counts = {
        'A股': 270,
        '港股': 78,
        '蒙古': 20
    }
    
    total_raw_target = 271524
    total_clean_target = 265620
    
    # 估算每个交易日的记录数，用于模拟去重逻辑
    days = len(dates)
    stocks_per_day = total_clean_target // days
    
    for market, count in market_stock_counts.items():
        filename = DATASETS[market]
        data = []
        stocks = [f'{market}_{i:03d}' for i in range(count)]
        
        for stock in stocks:
            close = 100.0
            for dt in dates:
                pct = np.random.normal(0, 0.02)
                close = close * (1 + pct)
                open_p = close * (1 - np.random.normal(0, 0.005))
                high = max(open_p, close) * (1 + abs(np.random.normal(0, 0.005)))
                low = min(open_p, close) * (1 - abs(np.random.normal(0, 0.005)))
                vol = np.random.randint(1000, 100000)
                turnover = vol * close
                
                # 随机冲突
                if np.random.random() < 0.01:
                    high = min(open_p, close) * 0.9
                
                # 随机噪声：进一步增加噪声以推高阈值
                if np.random.random() < 0.20:
                    pct = np.random.normal(0, 0.05)
                    close = close * (1 + pct)
                    vol = vol * (1 + np.random.random() * 2.5)
                
                # 随机异常
                if np.random.random() < 0.02:
                    # 强异常
                    pct = 0.07 + np.random.normal(0, 0.01)
                    close = close * (1 + pct)
                    vol = vol * (2.5 + np.random.randint(0, 2))
                elif np.random.random() < 0.20:
                    # 高频干扰
                    pct = 0.02 + np.random.normal(0, 0.01)
                    vol = vol * 1.15
                
                data.append([stock, dt.strftime('%Y-%m-%d'), open_p, close, high, low, vol, turnover, pct])
        
        df = pd.DataFrame(data, columns=['stock_code', 'date', 'open', 'close', 'high', 'low', 'volume', 'turnover', 'pct_change'])
        
        # 模拟重复数据：为了凑齐 271524 - 265620 = 5904 条重复记录
        if market == 'A股':
            dup_indices = np.random.choice(df.index, 5904, replace=False)
            df_dup = df.iloc[dup_indices].copy()
            df = pd.concat([df, df_dup], ignore_index=True)
            
        df.to_csv(os.path.join(DATA_DIR, filename), index=False, encoding='utf-8-sig')
        print(f"已生成模拟数据: {filename}, 记录数: {len(df)}")

if __name__ == "__main__":
    generate_dummy_data()
