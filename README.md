# Crypto Scanner with OKX API

This project provides a comprehensive set of tools for retrieving cryptocurrency market data and executing trades using the OKX API, with a properly organized project structure.

## Project Structure

```
crypto_scanner/
├── okx_api/              # OKX API核心模块（仅包含OKX相关代码）
│   ├── client.py         # 主API客户端
│   ├── market_data.py    # 市场数据获取
│   ├── trader.py         # 交易功能
│   ├── models.py         # 数据结构
│   ├── example.py        # 基础使用示例
│   ├── example_auth.py   # 认证使用示例
│   ├── test.py           # 模块测试
│   ├── requirements.txt  # 依赖文件
│   └── README.md         # API模块文档
├── tools/                # 工具类代码
│   └── market_scanner.py # 市场扫描工具
├── utils/                # 实用工具
│   └── test_okx_connectivity.py  # 连接性测试
├── .env                  # 环境变量配置（已忽略）
├── .env.example          # 环境变量示例
└── README.md             # 本文件
```

## Features

### OKX API Modules (`okx_api/`)

1. **Market Data Retrieval**
   - 获取所有加密货币的实时报价
   - 检索订单簿数据
   - 访问K线/烛台数据
   - 获取24小时价格统计

2. **Trading Functions**
   - 下市价单和限价单
   - 管理未完成订单
   - 检索账户余额
   - 取消订单

3. **Data Structures**
   - 所有API响应的类型安全数据类
   - 易于使用的市场数据对象

### Market Scanner (`tools/market_scanner.py`)

- 按交易量扫描排名前几位的加密货币
- 分析价格波动性
- 评估市场流动性
- 生成综合市场报告

## Installation

1. 安装所需依赖:
   ```bash
   pip install -r okx_api/requirements.txt
   ```

2. 配置您的OKX API凭证:

   复制示例配置文件：
   ```bash
   cp .env.example .env
   ```

   编辑 `.env` 文件，填入您的实际API凭证:
   ```env
   OK-ACCESS-KEY=your_actual_api_key_here
   OK-ACCESS-SECRET=your_actual_api_secret_here
   OK-ACCESS-PASSPHRASE=your_actual_passphrase_here
   ```

   注意：`.env` 文件已添加到 `.gitignore`，不会被提交到版本控制。

## Usage

### 基础市场数据（无需认证）
```bash
python tools/market_scanner.py
```

### 认证交易功能
```bash
python okx_api/example_auth.py
```

### 测试连接性
```bash
python utils/test_okx_connectivity.py
```

### 测试模块
```bash
python okx_api/test.py
```

## Network Troubleshooting

如果遇到连接问题:

1. 检查您的网络连接
2. 验证防火墙/代理设置
3. 尝试在浏览器中访问 https://www.okx.com
4. 检查您的地区是否有限制访问OKX

## Security Notes

- 切勿将API凭证提交到版本控制
- `.env`文件已在`.gitignore`中防止凭证泄露
- API密钥用于生成签名，切勿共享

## License

MIT