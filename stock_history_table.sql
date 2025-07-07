-- 合并后股票历史价格表建表语句
-- 中文字段注释已添加

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