#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用环境变量的OKX API客户端示例
"""

import os
import sys
from dotenv import load_dotenv

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载 .env 文件中的环境变量
load_dotenv()

def main():
    """演示如何使用环境变量"""
    print("Loading environment variables example")
    print("=" * 30)

    # 从环境变量中获取OKX API凭证
    api_key = os.getenv('OK-ACCESS-KEY')
    api_secret = os.getenv('OK-ACCESS-SECRET')
    passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

    print(f"API Key: {api_key}")
    print(f"API Secret: {'*' * len(api_secret) if api_secret else 'Not found'}")
    print(f"Passphrase: {'*' * len(passphrase) if passphrase else 'Not found'}")

    # 检查是否所有必需的环境变量都已设置
    if api_key and api_secret and passphrase:
        print("\n[SUCCESS] 所有环境变量已正确设置")

        # 现在可以初始化OKX客户端
        try:
            from okx_api.client import OKXClient
            from okx_api.trader import Trader

            # 初始化客户端
            client = OKXClient(
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase
            )

            print("[SUCCESS] OKX客户端初始化成功")

            # 可以进行认证操作
            trader = Trader(client)
            print("[SUCCESS] 交易模块初始化成功")

        except ImportError as e:
            print(f"[ERROR] 导入错误: {e}")
        except Exception as e:
            print(f"[ERROR] 初始化错误: {e}")

    else:
        print("\n[WARNING] 缺少必要的环境变量")
        print("请确保 .env 文件包含以下变量:")
        print("  OK-ACCESS-KEY=your_api_key_here")
        print("  OK-ACCESS-SECRET=your_api_secret_here")
        print("  OK-ACCESS-PASSPHRASE=your_passphrase_here")

if __name__ == "__main__":
    main()