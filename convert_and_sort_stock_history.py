import pandas as pd
import csv

# 读取原始Excel文件
input_file = 'stock_history.xlsx'
output_file = 'stock_history-2.csv'

# 读取所有sheet
xls = pd.ExcelFile(input_file)

# 用于存储所有数据的DataFrame列表
df_list = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name)
    # 将trade_date字段转换为YYYY-MM-DD格式
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d')
    # 按trade_date升序排序
    df = df.sort_values('trade_date')
    # 如果没有stock_name列，添加sheet名为stock_name
    if 'stock_name' not in df.columns:
        df['stock_name'] = sheet_name
    df_list.append(df)

# 合并所有sheet的数据
all_df = pd.concat(df_list, ignore_index=True)

# 保存为CSV文件，UTF-8编码，包含表头
all_df.to_csv(output_file, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

# 中文注释：
# 1. 本脚本将trade_date字段格式从20200101改为2020-01-01
# 2. 并按trade_date升序排序，所有sheet合并为一个csv文件stocks_history-2.csv
# 3. 保留stock_name列，便于区分股票 