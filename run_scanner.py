#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crypto Scanner 启动脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """运行市场扫描工具"""
    try:
        # 动态导入并运行市场扫描工具
        import tools.market_scanner
        tools.market_scanner.main()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所有依赖: pip install -r okx_api/requirements.txt")
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    main()