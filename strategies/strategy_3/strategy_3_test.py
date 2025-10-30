#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/22/25 9:52 PM
@File       : strategy_3_test.py
@Description: 长下影线策略测试
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from strategies.strategy_3.strategy_3 import LongShadowStrategy
from strategies.strategy_3.shared_config import load_config_from_file, get_user_input, print_final_config
from utils.logger import logger


def test_strategy_3():
    """测试策略3"""
    logger.info("长下影线策略测试")
    logger.info("=" * 50)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='长下影线策略测试')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 加载用户指定的配置文件作为默认值
        default_config = load_config_from_file(args.config)
        if not default_config:
            logger.error("配置文件加载失败，使用系统默认值")
            default_config = {}
    else:
        # 加载默认配置文件作为用户输入的默认值
        config_path = os.path.join(os.path.dirname(__file__), 'configs/default.json')
        default_config = load_config_from_file(config_path)
        if not default_config:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 咨询用户输入
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 初始化客户端和策略
    client = OKXClient()
    strategy = LongShadowStrategy(client)
    
    # 设置参数
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    min_volume_ccy = config.get('min_volume_ccy', 100000)
    volume_factor = config.get('volume_factor', 1.2)
    use_volume = config.get('use_volume', True)
    
    # 测试策略
    try:
        signal = strategy.calculate_long_shadow_signal(
            symbol, bar, min_volume_ccy, volume_factor, use_volume, mock_position=0
        )
        
        details = strategy.get_strategy_details(
            symbol, bar, min_volume_ccy, volume_factor, use_volume, mock_position=0
        )
        
        if details:
            logger.info(f"\n{symbol} 策略详情:")
            logger.info(f"  当前价格: ${details['current_price']:.4f}")
            logger.info(f"  倒数第二根K线:")
            logger.info(f"    开盘: ${details['prev2_open']:.4f}")
            logger.info(f"    最高: ${details['prev2_high']:.4f}")
            logger.info(f"    最低: ${details['prev2_low']:.4f}")
            logger.info(f"    收盘: ${details['prev2_close']:.4f}")
            logger.info(f"  长下影线条件: {details['long_shadow_condition']}")
            logger.info(f"  长上影线条件: {details['short_shadow_condition']}")
            logger.info(f"  多单入场条件: {details['long_entry_condition']}")
            logger.info(f"  空单入场条件: {details['short_entry_condition']}")
            logger.info(f"  成交量条件: {details['volume_condition_met']}")
            logger.info(f"  信号: {signal} ({'买入多单' if signal == 1 else '卖出空单' if signal == -1 else '持有'})")
        else:
            logger.error(f"无法获取 {symbol} 的策略详情")
            
    except Exception as e:
        logger.error(f"测试策略时出错: {e}")


def test_multiple_symbols():
    """测试多个交易对"""
    logger.info("\n多交易对测试")
    logger.info("=" * 50)
    
    client = OKXClient()
    strategy = LongShadowStrategy(client)
    
    test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT']
    
    for symbol in test_symbols:
        try:
            signal = strategy.calculate_long_shadow_signal(
                symbol, '1m', 100000, 1.2, True, mock_position=0
            )
            
            details = strategy.get_strategy_details(
                symbol, '1m', 100000, 1.2, True, mock_position=0
            )
            
            if details:
                logger.info(f"{symbol}:")
                logger.info(f"  价格: ${details['current_price']:.4f}")
                logger.info(f"  长下影线: {details['long_shadow_condition']}")
                logger.info(f"  长上影线: {details['short_shadow_condition']}")
                logger.info(f"  成交量条件: {details['volume_condition_met']}")
                logger.info(f"  信号: {signal} ({'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'})")
            else:
                logger.warning(f"无法获取 {symbol} 的策略详情")
                
        except Exception as e:
            logger.error(f"测试 {symbol} 时出错: {e}")


if __name__ == "__main__":
    test_strategy_3()
    test_multiple_symbols()