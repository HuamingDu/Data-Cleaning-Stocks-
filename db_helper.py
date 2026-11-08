import sqlite3
import pandas as pd
from sqlalchemy import create_engine, text
from config import DB_CONFIG
import datetime
import os

def get_engine():
    """获取 SQLAlchemy 引擎 (SQLite)"""
    url = f"sqlite:///{DB_CONFIG['db_path']}"
    return create_engine(url)

def init_db():
    """初始化数据库并建表"""
    # SQLite 自动处理数据库文件的创建
    engine = get_engine()
    with engine.connect() as conn:
        # 1. raw_data
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS raw_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stock_code TEXT,
                date DATE,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume REAL,
                turnover REAL,
                pct_change REAL,
                market TEXT
            )
        """))
        
        # 3. label_stat_table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS label_stat_table (
                market TEXT,
                rule_name TEXT,
                hit_count INTEGER,
                hit_rate REAL
            )
        """))

        # 4. dataset_split_stat
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS dataset_split_stat (
                set_type TEXT,
                sample_count INTEGER,
                anomaly_count INTEGER,
                anomaly_rate REAL
            )
        """))

        # 5. method_compare_result
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS method_compare_result (
                method TEXT,
                accuracy REAL,
                precision_val REAL,
                recall REAL,
                f1_score REAL
            )
        """))

        # 6. market_f1_result
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_f1_result (
                market TEXT,
                f1_score REAL
            )
        """))

        # 7. leave_one_out_result
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS leave_one_out_result (
                target_market TEXT,
                local_f1 REAL,
                transfer_f1 REAL,
                drop_val REAL
            )
        """))

        # 8. ablation_result
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ablation_result (
                feature_set TEXT,
                f1_score REAL,
                f1_drop REAL
            )
        """))

        # 9. clean_quality_compare
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clean_quality_compare (
                market TEXT,
                metric TEXT,
                before_val REAL,
                after_val REAL
            )
        """))

        # 10. confusion_matrix_table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS confusion_matrix_table (
                exp_name TEXT,
                tp INTEGER,
                tn INTEGER,
                fp INTEGER,
                fn INTEGER
            )
        """))

        # 11. clean_log
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS clean_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                log_type TEXT,
                stock_code TEXT,
                date DATE,
                field TEXT,
                old_value TEXT,
                new_value TEXT,
                message TEXT
            )
        """))
        conn.commit()

def insert_log(log_type, stock_code, date, field, old_value, new_value, message):
    """记录清洗日志"""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO clean_log (timestamp, log_type, stock_code, date, field, old_value, new_value, message)
            VALUES (:ts, :lt, :sc, :dt, :fd, :ov, :nv, :msg)
        """), {
            'ts': datetime.datetime.now(),
            'lt': log_type,
            'sc': stock_code,
            'dt': date,
            'fd': field,
            'ov': str(old_value),
            'nv': str(new_value),
            'msg': message
        })
        conn.commit()

def save_df_to_db(df, table_name, if_exists='replace'):
    """保存 DataFrame 到数据库"""
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
