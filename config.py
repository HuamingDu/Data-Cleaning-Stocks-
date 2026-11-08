import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 确保输出目录存在
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# SQLite 数据库配置
DB_CONFIG = {
    'db_path': os.path.join(BASE_DIR, 'cross_fin_clean.db')
}

# 输入数据集文件名
DATASETS = {
    'A股': '实验数据_A 股_沪深 300_202201_202412.csv',
    '港股': '实验数据_港股_恒生指数_202201_202412.csv',
    '蒙古': '实验数据_蒙古_MSE_TOP20_202201_202412.csv'
}

# 银标签生成参数
ANOMALY_RULES_THRESHOLD = 2  # 至少命中 2 条规则判定为异常

# 时序划分比例
SPLIT_RATIO = {
    'train': 0.7,
    'val': 0.1,
    'test': 0.2
}

# Matplotlib 设置
import matplotlib
matplotlib.use('Agg')
