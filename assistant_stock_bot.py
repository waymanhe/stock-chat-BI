import os
import asyncio
from typing import Optional
import dashscope
from qwen_agent.agents import Assistant
from qwen_agent.gui import WebUI
import pandas as pd
from sqlalchemy import create_engine
from qwen_agent.tools.base import BaseTool, register_tool
import matplotlib.pyplot as plt
import io
import base64
import time
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from prophet import Prophet
# 新增：导入配置文件
from config import DB_CONFIG, DASHSCOPE_API_KEY

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 配置 DashScope
# 优先从config.py获取API Key
# =====================
dashscope.api_key = DASHSCOPE_API_KEY or os.getenv('DASHSCOPE_API_KEY', '')
dashscope.timeout = 30  # 设置超时时间为 30 秒

# 读取faq.txt内容，作为知识库tips
with open(os.path.join(os.path.dirname(__file__), 'faq.txt'), encoding='utf-8') as f:
    faq_tips = f.read()

# 获取当前系统时间
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ====== 股票助手 system prompt 和函数描述 ======
system_prompt = f"""我是股票查询助手，当前系统时间为：{current_time}
以下是关于股票历史价格表 stocks_history 的字段信息，我可能会编写对应的SQL，对数据进行查询
-- 股票历史价格表
CREATE TABLE stocks_history (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    ts_code VARCHAR(16) NOT NULL COMMENT '股票代码',
    trade_date DATE NOT NULL COMMENT '交易日期',
    `open` DECIMAL(12,2) COMMENT '开盘价',
    high DECIMAL(12,2) COMMENT '最高价',
    low DECIMAL(12,2) COMMENT '最低价',
    `close` DECIMAL(12,2) COMMENT '收盘价',
    pre_close DECIMAL(12,2) COMMENT '昨收价',
    `change` DECIMAL(12,2) COMMENT '涨跌额',
    pct_chg DECIMAL(8,4) COMMENT '涨跌幅',
    vol DECIMAL(20,2) COMMENT '成交量(手)',
    amount DECIMAL(20,2) COMMENT '成交额(元)',
    stock_name VARCHAR(32) COMMENT '股票中文名'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票历史价格表（合并所有股票）';

我将回答用户关于股票历史行情、涨跌幅、成交量、个股走势等相关问题。

每当 exc_sql 工具返回 markdown 表格和图片时，你必须原样输出工具返回的全部内容（包括图片 markdown），不要只总结表格，也不要省略图片。这样用户才能直接看到表格和图片。

每当 arima_stock 工具使用前，请先使用 exc_sql 工具获取到正确的 ts_code。

请遵循以下流程：
    1. **名称识别阶段**
    - 当用户输入包含中文股票名称（如"贵州茅台"）时，必须优先生成包含`stock_name`字段的WHERE条件
    - 示例1：用户输入"查询贵州茅台2024年收盘价" → 生成`SELECT * FROM stocks_history WHERE stock_name='贵州茅台' AND trade_date BETWEEN '2024-01-01' AND '2024-12-31'`

    2. **代码转换规则**  
    - 系统会自动将`stock_name`条件转换为`ts_code`（如"贵州茅台" → "600519.SH"），你无需手动处理代码转换
    - 禁止直接使用中文名称作为`ts_code`（错误示例：`WHERE ts_code='贵州茅台'`）
    - 若用户同时提供代码和名称，优先使用代码（示例：`WHERE ts_code='600519.SH' OR stock_name='贵州茅台'`）

    3. **SQL生成规范**  
    - 字段选择：必须包含`ts_code`字段以便验证转换结果
    - 时间范围：若未指定日期，默认查询最近30天数据（示例：`AND trade_date >= CURDATE() - INTERVAL 30 DAY`）
    - 性能优化：避免全表扫描，优先使用索引字段（如`trade_date`）

    4. **错误处理机制**  
    - 若名称转换失败（如"未找到股票名称"），需返回明确错误提示并建议用户核对名称
    - 若存在名称歧义（如"平安"可能指"中国平安"或"平安银行"），需主动要求用户澄清
    【重要说明】当用户或LLM输入股票中文名称时，系统会自动查表获取正确的ts_code（股票代码），并用于后续SQL查询和分析。你可以直接用中文名提问，无需记忆股票代码。

    【参数决策说明】
    - 当用户问题为"对比分析"或只需结构化文本总结时，请将 need_visualize 设为 False；
    - 当用户问题为单只股票走势、统计、图表展示等时，请将 need_visualize 设为 True。

【知识库Tips】
{faq_tips}
"""

# ====== 数据库连接配置 ======
# DB_CONFIG 已从 config.py 导入

# ====== exc_sql 工具类实现 ======
@register_tool('exc_sql')
class ExcSQLTool(BaseTool):
    """
    SQL查询工具，执行传入的SQL语句并返回结果，并自动进行可视化。
    """
    description = '对于生成的SQL，进行SQL查询，并自动可视化。need_visualize参数为False时不进行可视化，适合对比分析等场景。'
    parameters = [{
        'name': 'sql_input',
        'type': 'string',
        'description': '生成的SQL语句',
        'required': True
    }, {
        'name': 'need_visualize',
        'type': 'boolean',
        'description': '是否需要可视化和统计信息，默认True。对比分析等场景可设为False。',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        args = json.loads(params)
        sql_input = args['sql_input']
        # 是否需要可视化，默认True
        need_visualize = args.get('need_visualize', True)
        # 数据库名 stock_history
        database = args.get('database', DB_CONFIG['database'])
        # 使用 DB_CONFIG 构建数据库连接字符串
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{database}?charset={DB_CONFIG['charset']}",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        try:
            df = pd.read_sql(sql_input, engine)
            # 提取前5行和后5行，合并去重
            if len(df) > 10:
                md = pd.concat([df.head(5), df.tail(5)]).drop_duplicates().to_markdown(index=False)
            else:
                md = df.to_markdown(index=False)
            # 只有一行，或不需要可视化时，只返回表格
            if len(df) == 1 or not need_visualize:
                return md
            # 增加 describe 信息，方便AI总结
            describe_md = df.describe().to_markdown()
            # 自动创建图片保存目录
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'stock_bar_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            # 生成图表
            generate_chart_png(df, save_path)
            img_path = os.path.join('image_show', filename)
            img_md = f'![股票行情柱状图]({img_path})'
            # 返回顺序：表格、describe、图片
            return f"{md}\n\n**数据描述（describe）**\n\n{describe_md}\n\n{img_md}"
        except Exception as e:
            return f"SQL执行或可视化出错: {str(e)}"

# ========== 通用可视化函数 ========== 
def generate_chart_png(df_sql, save_path):
    # 获取所有列名
    columns = df_sql.columns
    # 数据总行数
    n = len(df_sql)
    # 横坐标序列
    x = np.arange(n)
    # 获取object类型的列
    object_columns = df_sql.select_dtypes(include='O').columns.tolist()
    if columns[0] in object_columns:
        object_columns.remove(columns[0])
    num_columns = df_sql.select_dtypes(exclude='O').columns.tolist()

    # 判断数据量，决定用柱状图还是折线图
    use_line = n > 20

    # 横坐标标签采样，避免过多重叠
    if n > 20:
        # 等距采样10个点
        idxs = np.linspace(0, n-1, 10, dtype=int)
        xticks = df_sql[columns[0]].iloc[idxs]
        xticks_pos = idxs
    else:
        xticks = df_sql[columns[0]]
        xticks_pos = x

    plt.figure(figsize=(10, 6))
    if use_line:
        # 折线图
        for column in columns[1:]:
            if np.issubdtype(df_sql[column].dtype, np.number):
                plt.plot(x, df_sql[column], label=column)
        plt.xticks(xticks_pos, xticks, rotation=45)
        plt.title("股票行情折线图（大数据量自动切换）")
    else:
        # 柱状图/堆积柱状图
        if len(object_columns) > 0:
            # 透视为堆积柱状图
            pivot_df = df_sql.pivot_table(index=columns[0], columns=object_columns, values=num_columns, fill_value=0)
            bottoms = None
            for col in pivot_df.columns:
                plt.bar(pivot_df.index, pivot_df[col], bottom=bottoms, label=str(col))
                if bottoms is None:
                    bottoms = pivot_df[col].copy()
                else:
                    bottoms += pivot_df[col]
            plt.title("股票行情堆积柱状图")
        else:
            bottom = np.zeros(n)
            for column in columns[1:]:
                plt.bar(x, df_sql[column], bottom=bottom, label=column)
                bottom += df_sql[column]
            plt.xticks(xticks_pos, xticks, rotation=45)
            plt.title("股票行情柱状图")
    plt.legend()
    plt.xlabel(columns[0])
    plt.ylabel("数值")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ====== 初始化股票助手服务 ======
def init_agent_service():
    """初始化股票助手服务"""
    llm_cfg = {
        'model': 'qwen-turbo-2025-04-28',
        'timeout': 30,
        'retry_count': 3,
    }

    # MCP 工具配置
    tools = [
        {
            "mcpServers": {
                "tavily-mcp": {
                    "command": "npx",
                    "args": ["-y", "tavily-mcp@0.1.4"],
                    "env": {
                        "TAVILY_API_KEY": "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"
                    },
                    "disabled": False,
                    "autoApprove": []
                }
            }
        },
        'exc_sql',
        'arima_stock',
        'prophet_analysis',
        'boll_detection'
    ]

    try:
        bot = Assistant(
            llm=llm_cfg,
            name='股票查询助手',
            description='股票历史行情查询与分析',
            system_message=system_prompt,
            function_list=tools,
            files = ['./faq.txt']
        )
        print("助手初始化成功！")
        return bot
    except Exception as e:
        print(f"助手初始化失败: {str(e)}")
        raise

def app_tui():
    """终端交互模式，支持连续对话和实时响应"""
    try:
        bot = init_agent_service()
        messages = []
        while True:
            try:
                query = input('请输入你的股票问题: ')
                file = input('文件路径（如无可回车跳过）: ').strip()
                if not query:
                    print('问题不能为空！')
                    continue
                if not file:
                    messages.append({'role': 'user', 'content': query})
                else:
                    messages.append({'role': 'user', 'content': [{'text': query}, {'file': file}]})
                print("正在处理您的请求...")
                response = []
                for response in bot.run(messages):
                    print('助手回复:', response)
                messages.extend(response)
            except Exception as e:
                print(f"处理请求时出错: {str(e)}")
                print("请重试或输入新的问题")
    except Exception as e:
        print(f"启动终端模式失败: {str(e)}")

def app_gui():
    """图形界面模式，提供 Web 图形界面"""
    try:
        print("正在启动 Web 界面...")
        bot = init_agent_service()
        chatbot_config = {
            'prompt.suggestions': [
                '查询2024年全年贵州茅台的收盘价走势',
                '统计2024年4月国泰君安的日均成交量',
                '对比2024年中芯国际和贵州茅台的涨跌幅',
            ]
        }
        print("Web 界面准备就绪，正在启动服务...")
        WebUI(
            bot,
            chatbot_config=chatbot_config
        ).run()
    except Exception as e:
        print(f"启动 Web 界面失败: {str(e)}")
        print("请检查网络连接和 API Key 配置")

@register_tool('arima_stock')
class ArimaStockTool(BaseTool):
    """
    使用ARIMA模型预测未来N天的股票收盘价。
    """
    description = '输入ts_code和预测天数n，预测未来n天的收盘价（基于最近一年历史数据，ARIMA(5,1,5)建模）'
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': '股票代码（必填）',
        'required': True
    }, {
        'name': 'n',
        'type': 'integer',
        'description': '预测天数，默认5',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        import pandas as pd
        from sqlalchemy import create_engine
        args = json.loads(params)
        ts_code = args['ts_code']
        n = int(args.get('n', 5))
        # 数据库名 stock_history
        database = args.get('database', DB_CONFIG['database'])
        # 使用 DB_CONFIG 构建数据库连接字符串
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{database}?charset={DB_CONFIG['charset']}",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        # 获取最近一年的历史数据
        today = datetime.today().date()
        last_year = today - timedelta(days=365)
        sql = f"""
            SELECT trade_date, close FROM stocks_history
            WHERE ts_code='{ts_code}' AND trade_date >= '{last_year}' AND trade_date <= '{today}'
            ORDER BY trade_date
        """
        try:
            df = pd.read_sql(sql, engine)
            if len(df) < 30:
                return '历史数据不足，无法建模预测。'
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            close_series = df['close'].astype(float)
            # ARIMA建模
            model = ARIMA(close_series, order=(5,1,5))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=n)
            # 生成预测日期
            last_date = df['trade_date'].max()
            pred_dates = [last_date + timedelta(days=i) for i in range(1, n+1)]
            pred_df = pd.DataFrame({'预测日期': pred_dates, '预测收盘价': forecast})
            # 结果表格
            md = pred_df.to_markdown(index=False)
            # 可视化
            plt.figure(figsize=(10,6))
            plt.plot(df['trade_date'], close_series, label='历史收盘价')
            plt.plot(pred_df['预测日期'], pred_df['预测收盘价'], 'ro--', label='预测收盘价')
            plt.xlabel('日期')
            plt.ylabel('收盘价')
            plt.title(f'{ts_code} 未来{n}天ARIMA预测')
            plt.legend()
            plt.tight_layout()
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'arima_pred_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path)
            plt.close()
            img_path = os.path.join('image_show', filename)
            img_md = f'![ARIMA预测图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f"ARIMA预测出错: {str(e)}"

@register_tool('prophet_analysis')
class ProphetAnalysisTool(BaseTool):
    """
    使用Prophet对指定股票收盘价进行周期性分析，输出趋势（trend）、周（weekly）、年（yearly）成分，并可视化。
    """
    description = '输入ts_code和可选时间范围，对股票收盘价进行Prophet周期性分析，输出趋势、周、年成分及可视化。'
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': '股票代码（必填）',
        'required': True
    }, {
        'name': 'start_date',
        'type': 'string',
        'description': '起始日期，格式YYYY-MM-DD，默认最近一年',
        'required': False
    }, {
        'name': 'end_date',
        'type': 'string',
        'description': '结束日期，格式YYYY-MM-DD，默认今天',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        import pandas as pd
        from sqlalchemy import create_engine
        from prophet import Prophet
        args = json.loads(params)
        ts_code = args['ts_code']
        today = datetime.today().date()
        # 处理时间范围
        end_date = args.get('end_date', str(today))
        start_date = args.get('start_date')
        if not start_date:
            start_date = str((datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=365)))
        # 数据库名 stock_history
        database = args.get('database', DB_CONFIG['database'])
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{database}?charset={DB_CONFIG['charset']}",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        sql = f"""
            SELECT trade_date, close FROM stocks_history
            WHERE ts_code='{ts_code}' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        try:
            df = pd.read_sql(sql, engine)
            if len(df) < 30:
                return '历史数据不足，无法进行Prophet周期性分析。'
            df['ds'] = pd.to_datetime(df['trade_date'])
            df['y'] = df['close'].astype(float)
            # Prophet建模
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(df[['ds', 'y']])
            future = m.make_future_dataframe(periods=0)  # 只分析历史
            forecast = m.predict(future)
            # 生成趋势、周、年分解图
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'prophet_decomp_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            fig = m.plot_components(forecast)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
            # 结果表格（展示最后5天的趋势、周、年分量）
            show_cols = ['ds', 'trend', 'weekly', 'yearly', 'yhat']
            show_df = forecast[show_cols].tail(5)
            md = show_df.to_markdown(index=False)
            img_path = os.path.join('image_show', filename)
            img_md = f'![Prophet周期分解图]({img_path})'
            return f"{md}\n\n{img_md}"
        except Exception as e:
            return f"Prophet周期性分析出错: {str(e)}"

@register_tool('boll_detection')
class BollDetectionTool(BaseTool):
    """
    使用布林带（Bollinger Bands）检测股票超买超卖点，默认20日+2σ，支持自定义时间范围。
    """
    description = '输入ts_code和可选时间范围，检测股票布林带超买（>上轨）和超卖（<下轨）日期，默认20日+2σ，最近一年。'
    parameters = [{
        'name': 'ts_code',
        'type': 'string',
        'description': '股票代码（必填）',
        'required': True
    }, {
        'name': 'start_date',
        'type': 'string',
        'description': '起始日期，格式YYYY-MM-DD，默认最近一年',
        'required': False
    }, {
        'name': 'end_date',
        'type': 'string',
        'description': '结束日期，格式YYYY-MM-DD，默认今天',
        'required': False
    }]

    def call(self, params: str, **kwargs) -> str:
        import json
        import matplotlib.pyplot as plt
        import io, os, time
        import numpy as np
        import pandas as pd
        from sqlalchemy import create_engine
        args = json.loads(params)
        ts_code = args['ts_code']
        today = datetime.today().date()
        end_date = args.get('end_date', str(today))
        start_date = args.get('start_date')
        if not start_date:
            start_date = str((datetime.strptime(end_date, '%Y-%m-%d').date() - timedelta(days=365)))
        database = args.get('database', DB_CONFIG['database'])
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:3306/{database}?charset={DB_CONFIG['charset']}",
            connect_args={'connect_timeout': 10}, pool_size=10, max_overflow=20
        )
        sql = f"""
            SELECT trade_date, close FROM stocks_history
            WHERE ts_code='{ts_code}' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        try:
            df = pd.read_sql(sql, engine)
            if len(df) < 30:
                return '历史数据不足，无法进行布林带检测。'
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            # 计算布林带
            window = 20
            df['MA20'] = df['close'].rolling(window).mean()
            df['STD20'] = df['close'].rolling(window).std()
            df['Upper'] = df['MA20'] + 2 * df['STD20']
            df['Lower'] = df['MA20'] - 2 * df['STD20']
            # 检测超买/超卖
            overbought = df[df['close'] > df['Upper']][['trade_date', 'close', 'Upper']]
            oversold = df[df['close'] < df['Lower']][['trade_date', 'close', 'Lower']]
            # 结果表格
            ob_md = overbought.tail(5).to_markdown(index=False) if not overbought.empty else '无超买点'
            os_md = oversold.tail(5).to_markdown(index=False) if not oversold.empty else '无超卖点'
            # 可视化
            save_dir = os.path.join(os.path.dirname(__file__), 'image_show')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'boll_{ts_code}_{int(time.time()*1000)}.png'
            save_path = os.path.join(save_dir, filename)
            plt.figure(figsize=(12,6))
            plt.plot(df['trade_date'], df['close'], label='收盘价')
            plt.plot(df['trade_date'], df['MA20'], label='MA20')
            plt.plot(df['trade_date'], df['Upper'], label='上轨', linestyle='--')
            plt.plot(df['trade_date'], df['Lower'], label='下轨', linestyle='--')
            plt.scatter(overbought['trade_date'], overbought['close'], color='red', label='超买', marker='^')
            plt.scatter(oversold['trade_date'], oversold['close'], color='blue', label='超卖', marker='v')
            plt.xlabel('日期')
            plt.ylabel('收盘价')
            plt.title(f'{ts_code} 布林带异常点检测')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            img_path = os.path.join('image_show', filename)
            img_md = f'![布林带异常点检测]({img_path})'
            return f"**超买点（收盘价>上轨）**\n{ob_md}\n\n**超卖点（收盘价<下轨）**\n{os_md}\n\n{img_md}"
        except Exception as e:
            return f"布林带检测出错: {str(e)}"

if __name__ == '__main__':
    app_gui()  # 默认启动 Web 图形界面 