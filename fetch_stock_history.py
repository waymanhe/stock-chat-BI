import tushare as ts
import pandas as pd
from datetime import datetime
# 新增：导入Tushare Token
from config import TUSHARE_TOKEN

# 设置tushare的token，这里需要你替换为你自己的token
TS_TOKEN = TUSHARE_TOKEN
ts.set_token(TS_TOKEN)
pro = ts.pro_api()

# 股票代码及名称映射 自己找一些感兴趣的股票
stocks = {
    '贵州茅台': '600519.SH',
    '五粮液': '000858.SZ',
    '国泰君安': '601211.SH',
    '中芯国际': '688981.SH',
    '新亚强': '603155.SH',
    '地素时尚': '603587.SH',
    '恒生国企ETF': '159850.SZ',
}

# 设置起止日期
start_date = '20200101'
end_date = datetime.now().strftime('%Y%m%d')

# 用于存储所有股票的历史数据
df_list = []

for name, code in stocks.items():
    # 获取日线行情
    df = pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
    # 按日期升序排列
    df = df.sort_values('trade_date')
    # 增加股票中文名列
    df['stock_name'] = name
    df_list.append(df)

# 合并所有股票数据
all_df = pd.concat(df_list, ignore_index=True)

# 保存为单sheet的Excel文件
all_df.to_excel('stock_history.xlsx', index=False)

# 中文注释：
# 1. 本脚本将所有股票数据合并为一个sheet，并增加stock_name列
# 2. 运行后生成的stock_history.xlsx为单sheet，包含所有股票数据
# 3. 若sheet名过长或包含特殊字符，可能会报错，可适当修改sheet名 