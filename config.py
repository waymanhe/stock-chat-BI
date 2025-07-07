# -*- coding: utf-8 -*-
"""
配置文件：集中管理敏感信息和全局配置
请勿将本文件上传到公开仓库（建议添加到 .gitignore）
"""

# 数据库连接配置
DB_CONFIG = {
    'host': 'localhost',      # 数据库主机地址
    'user': 'root',           # 数据库用户名
    'password': 'He030201',   # 数据库密码
    'database': 'stock_history',      # 数据库名
    'charset': 'utf8mb4'
}

# DashScope API Key
DASHSCOPE_API_KEY = 'sk-c74fb20105a44b038d7872a41c6f40e4'  # 请修改为你的API Key 

# Tushare Token
TUSHARE_TOKEN = '21c3db75e9c07629aa214e66a6738b2e0e02e9e7dd29aed9d7d87250'  # 请修改为你的Tushare Token