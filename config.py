# -*- coding: utf-8 -*-
"""
配置文件：集中管理敏感信息和全局配置
请勿将本文件上传到公开仓库（建议添加到 .gitignore）
"""

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',      # 数据库主机地址
    'user': 'root',           # 数据库用户名
    'password': 'xxxxx',   # 数据库密码
    'database': 'xxxxxx',      # 数据库名
    'charset': 'utf8mb4'
}

# DashScope API Key
DASHSCOPE_API_KEY = 'xxxxxxxx'  # 请修改为你的API Key 

# Tushare Token
TUSHARE_TOKEN = 'xxxxxxxxxx'  # 请修改为你的Tushare Token