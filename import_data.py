# -*- coding: utf-8 -*-
"""
自动导入CSV数据到MySQL数据库的stocks_history表
用法：
    python import_data.py [csv文件路径]  # 默认导入 stock_history-2.csv
"""
import sys
import pandas as pd
from sqlalchemy import create_engine
from config import DB_CONFIG

# 获取命令行参数
if len(sys.argv) > 1:
    csv_path = sys.argv[1]
else:
    csv_path = 'stock_history-2.csv'

# 读取CSV数据
try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except Exception as e:
    print(f'读取CSV文件失败: {e}')
    sys.exit(1)

# 构建数据库连接
try:
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{DB_CONFIG['database']}?charset={DB_CONFIG['charset']}"
    )
except Exception as e:
    print(f'数据库连接失败: {e}')
    sys.exit(1)

# 导入数据到stocks_history表
try:
    df.to_sql('stocks_history', engine, if_exists='append', index=False)
    print(f'成功导入 {len(df)} 条数据到 stocks_history 表！')
except Exception as e:
    print(f'数据导入失败: {e}')
    sys.exit(1) 