#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : shared_config.py
@Description: Shared configuration functions for strategy 2
"""

import json
import os
from utils.logger import logger


def load_config_from_file(config_path: str) -> dict:
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从配置文件 {config_path} 加载配置成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def get_user_input(default_config: dict = None) -> dict:
    """获取用户输入参数，使用配置文件的默认值"""
    logger.info("高频策略参数配置")
    logger.info("=" * 50)

    # 使用配置文件的默认值
    if default_config is None:
        default_config = {}

    default_symbol = default_config.get('symbol', 'BTC-USDT')
    default_bar = default_config.get('bar', '1m')
    default_consecutive_bars = default_config.get('consecutive_bars', 2)
    default_atr_period = default_config.get('atr_period', 14)
    default_atr_threshold = default_config.get('atr_threshold', 0.8)
    default_trailing_stop_pct = default_config.get('trailing_stop_pct', 0.8)
    default_trade = default_config.get('trade', False)
    default_trade_amount = default_config.get('trade_amount', 10.0)
    default_trade_mode = default_config.get('trade_mode', 3)
    default_leverage = default_config.get('leverage', 3)
    default_use_volume = default_config.get('use_volume', True)
    default_volume_factor = default_config.get('volume_factor', 1.2)
    default_breakout_stop_bars = default_config.get('breakout_stop_bars', 2)
    default_params = default_config.get('params', {})

    try:
        symbol = input(f"请输入交易对 (默认 {default_symbol}): ").strip() or default_symbol
        bar = input(f"请输入K线周期 (默认 {default_bar}): ").strip() or default_bar
        
        consecutive_bars_input = input(f"请输入连续K线数量 (默认 {default_consecutive_bars}): ").strip()
        consecutive_bars = int(consecutive_bars_input) if consecutive_bars_input else default_consecutive_bars
        
        atr_period_input = input(f"请输入ATR周期 (默认 {default_atr_period}): ").strip()
        atr_period = int(atr_period_input) if atr_period_input else default_atr_period
        
        atr_threshold_input = input(f"请输入ATR阈值 (默认 {default_atr_threshold}): ").strip()
        atr_threshold = float(atr_threshold_input) if atr_threshold_input else default_atr_threshold
        
        trailing_stop_input = input(f"请输入移动止损百分比 (默认 {default_trailing_stop_pct}%): ").strip()
        trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else default_trailing_stop_pct

        breakout_stop_input = input(f"请输入突破止损K线数量 (默认 {default_breakout_stop_bars}): ").strip()
        breakout_stop_bars = int(breakout_stop_input) if breakout_stop_input else default_breakout_stop_bars

        trade_input = input(f"是否真实交易 (y/n, 默认 {'y' if default_trade else 'n'}): ").strip().lower()
        if trade_input:
            trade = trade_input == 'y' or trade_input == 'yes'
        else:
            trade = default_trade

        # 成交量条件
        use_volume_input = input(f"是否使用成交量条件 (y/n, 默认 {'y' if default_use_volume else 'n'}): ").strip().lower()
        if use_volume_input:
            use_volume = use_volume_input == 'y' or use_volume_input == 'yes'
        else:
            use_volume = default_use_volume

        volume_factor = default_volume_factor
        if use_volume:
            volume_factor_input = input(f"请输入成交量放大倍数 (默认 {default_volume_factor}): ").strip()
            volume_factor = float(volume_factor_input) if volume_factor_input else default_volume_factor

        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

        if trade:
            trade_amount_input = input(f"请输入每次交易的USDT金额 (默认 {default_trade_amount}): ").strip()
            trade_amount = float(trade_amount_input) if trade_amount_input else default_trade_amount

            logger.info("交易模式选择:")
            logger.info("1. 现货模式")
            logger.info("2. 全仓杠杆模式")
            logger.info("3. 逐仓杠杆模式")

            trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
            default_trade_mode_name = trade_mode_names.get(default_trade_mode, "逐仓杠杆")

            trade_mode_input = input(f"请选择交易模式 (1/2/3, 默认 {default_trade_mode}): ").strip()
            trade_mode = int(trade_mode_input) if trade_mode_input in ['1', '2', '3'] else default_trade_mode

            if trade_mode in [2, 3]:
                leverage_input = input(f"请输入杠杆倍数 (默认 {default_leverage}): ").strip()
                leverage = int(leverage_input) if leverage_input else default_leverage

    except (EOFError, KeyboardInterrupt):
        logger.info("\n用户取消输入，使用默认配置")
        # 使用默认配置
        symbol = default_symbol
        bar = default_bar
        consecutive_bars = default_consecutive_bars
        atr_period = default_atr_period
        atr_threshold = default_atr_threshold
        trailing_stop_pct = default_trailing_stop_pct
        breakout_stop_bars = default_breakout_stop_bars
        trade = default_trade
        use_volume = default_use_volume
        volume_factor = default_volume_factor
        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

    return {
        'symbol': symbol,
        'bar': bar,
        'consecutive_bars': consecutive_bars,
        'atr_period': atr_period,
        'atr_threshold': atr_threshold,
        'trailing_stop_pct': trailing_stop_pct,
        'breakout_stop_bars': breakout_stop_bars,
        'trade': trade,
        'trade_amount': trade_amount,
        'trade_mode': trade_mode,
        'leverage': leverage,
        'use_volume': use_volume,
        'volume_factor': volume_factor,
        'params': {
            'volume_factor': volume_factor,
            'use_volume': use_volume
        }
    }


def print_final_config(config: dict):
    """打印最终使用的参数"""
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    consecutive_bars = config.get('consecutive_bars', 2)
    atr_period = config.get('atr_period', 14)
    atr_threshold = config.get('atr_threshold', 0.8)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.8)
    breakout_stop_bars = config.get('breakout_stop_bars', 2)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)
    use_volume = config.get('use_volume', True)
    volume_factor = config.get('volume_factor', 1.2)

    logger.info("\n" + "=" * 50)
    logger.info("本次运行使用的最终参数:")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  K线周期: {bar}")
    logger.info(f"  连续K线: {consecutive_bars}")
    logger.info(f"  ATR周期: {atr_period}")
    logger.info(f"  ATR阈值: {atr_threshold}")
    logger.info(f"  移动止损: {trailing_stop_pct}%")
    logger.info(f"  突破止损K线: {breakout_stop_bars}")
    logger.info(f"  真实交易: {'是' if trade else '否'}")
    logger.info(f"  使用成交量: {'是' if use_volume else '否'}")
    if use_volume:
        logger.info(f"  成交量倍数: {volume_factor}")
    
    if trade:
        trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
        logger.info(f"  交易模式: {trade_mode_names.get(trade_mode, '未知')}")
        logger.info(f"  每次交易金额: {trade_amount} USDT")
        if trade_mode in [2, 3]:
            logger.info(f"  杠杆倍数: {leverage}x")
    
    logger.info("=" * 50 + "\n")