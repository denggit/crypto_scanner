#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : shared_config.py
@Description: Shared configuration functions for strategy 1
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
    logger.info("EMA交叉策略参数配置")
    logger.info("=" * 50)

    # 使用配置文件的默认值
    if default_config is None:
        default_config = {}

    default_symbol = default_config.get('symbol', 'BTC-USDT')
    default_bar = default_config.get('bar', '1m')
    default_short_ma = default_config.get('short_ma', 5)
    default_long_ma = default_config.get('long_ma', 20)
    default_mode = default_config.get('mode', 'loose')
    default_trailing_stop_pct = default_config.get('trailing_stop_pct', 1.0)
    default_trade = default_config.get('trade', False)
    default_trade_amount = default_config.get('trade_amount', 10.0)
    default_trade_mode = default_config.get('trade_mode', 3)
    default_leverage = default_config.get('leverage', 3)
    default_assist_cond = default_config.get('assist_cond', 'volume')
    default_volatility_exit = default_config.get('volatility_exit', False)
    default_volatility_threshold = default_config.get('volatility_threshold', 0.5)
    default_params = default_config.get('params', {})

    try:
        symbol = input(f"请输入交易对 (默认 {default_symbol}): ").strip() or default_symbol
        bar = input(f"请输入K线周期 (默认 {default_bar}): ").strip() or default_bar
        
        short_ma_input = input(f"请输入短EMA周期 (默认 {default_short_ma}): ").strip()
        short_ma = int(short_ma_input) if short_ma_input else default_short_ma
        
        long_ma_input = input(f"请输入长EMA周期 (默认 {default_long_ma}): ").strip()
        long_ma = int(long_ma_input) if long_ma_input else default_long_ma
        
        mode = input(f"请输入模式 (strict/loose, 默认 {default_mode}): ").strip() or default_mode
        trailing_stop_input = input(f"请输入移动止损百分比 (默认 {default_trailing_stop_pct}%): ").strip()
        trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else default_trailing_stop_pct

        trade_input = input(f"是否真实交易 (y/n, 默认 {'y' if default_trade else 'n'}): ").strip().lower()
        if trade_input:
            trade = trade_input == 'y' or trade_input == 'yes'
        else:
            trade = default_trade

        # 波动率退出条件
        volatility_exit_input = input(f"是否启用波动率退出条件 (y/n, 默认 {'y' if default_volatility_exit else 'n'}): ").strip().lower()
        if volatility_exit_input:
            volatility_exit = volatility_exit_input == 'y' or volatility_exit_input == 'yes'
        else:
            volatility_exit = default_volatility_exit

        volatility_threshold = default_volatility_threshold
        if volatility_exit:
            volatility_threshold_input = input(f"请输入波动率阈值 (默认 {default_volatility_threshold}): ").strip()
            volatility_threshold = float(volatility_threshold_input) if volatility_threshold_input else default_volatility_threshold

        logger.info("辅助条件选择:")
        logger.info("1. 成交量放大 (volume)")
        logger.info("2. RSI条件 (rsi)")
        logger.info("3. 无辅助条件 (none)")

        assist_cond_map = {'1': 'volume', '2': 'rsi', '3': None}
        default_assist_input = '1' if default_assist_cond == 'volume' else '2' if default_assist_cond == 'rsi' else '3'

        assist_input = input(f"请选择辅助条件 (1/2/3, 默认 {default_assist_input}): ").strip()
        assist_cond = assist_cond_map.get(assist_input, default_assist_cond)

        # 根据选择的辅助条件动态获取对应的技术指标参数
        params = default_params.copy()  # 复制默认参数
        
        if assist_cond == 'volume':
            # 成交量辅助条件：询问成交量相关参数
            default_vol_multiplier = default_params.get('vol_multiplier', 1.2)
            default_confirmation_pct = default_params.get('confirmation_pct', 0.2)
            
            vol_multiplier_input = input(f"请输入成交量放大倍数 (默认 {default_vol_multiplier}): ").strip()
            params['vol_multiplier'] = float(vol_multiplier_input) if vol_multiplier_input else default_vol_multiplier
            
            confirmation_pct_input = input(f"请输入确认突破百分比 (默认 {default_confirmation_pct}%): ").strip()
            params['confirmation_pct'] = float(confirmation_pct_input) if confirmation_pct_input else default_confirmation_pct
        
        elif assist_cond == 'rsi':
            # RSI辅助条件：询问RSI相关参数
            default_rsi_period = default_params.get('rsi_period', 9)
            rsi_period_input = input(f"请输入RSI周期 (默认 {default_rsi_period}): ").strip()
            params['rsi_period'] = int(rsi_period_input) if rsi_period_input else default_rsi_period

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
        short_ma = default_short_ma
        long_ma = default_long_ma
        mode = default_mode
        trailing_stop_pct = default_trailing_stop_pct
        trade = default_trade
        assist_cond = default_assist_cond
        volatility_exit = default_volatility_exit
        volatility_threshold = default_volatility_threshold
        params = default_params.copy()
        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

    return {
        'symbol': symbol,
        'bar': bar,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'mode': mode,
        'trailing_stop_pct': trailing_stop_pct,
        'trade': trade,
        'trade_amount': trade_amount,
        'trade_mode': trade_mode,
        'leverage': leverage,
        'assist_cond': assist_cond,
        'volatility_exit': volatility_exit,
        'volatility_threshold': volatility_threshold,
        'params': params
    }


def print_final_config(config: dict):
    """打印最终使用的参数"""
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    short_ma = config.get('short_ma', 5)
    long_ma = config.get('long_ma', 20)
    mode = config.get('mode', 'loose')
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)
    assist_cond = config.get('assist_cond', 'volume')
    volatility_exit = config.get('volatility_exit', False)
    volatility_threshold = config.get('volatility_threshold', 0.5)
    params = config.get('params', {})

    logger.info("\n" + "=" * 50)
    logger.info("本次运行使用的最终参数:")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  K线周期: {bar}")
    logger.info(f"  EMA参数: {short_ma}/{long_ma}")
    logger.info(f"  模式: {mode}")
    logger.info(f"  移动止损: {trailing_stop_pct}%")
    logger.info(f"  真实交易: {'是' if trade else '否'}")
    logger.info(f"  辅助条件: {assist_cond if assist_cond else '无'}")
    logger.info(f"  波动率退出: {'是' if volatility_exit else '否'}")
    if volatility_exit:
        logger.info(f"  波动率阈值: {volatility_threshold}")
    
    if assist_cond == 'volume':
        logger.info(f"  成交量倍数: {params.get('vol_multiplier', 1.2)}")
        logger.info(f"  确认百分比: {params.get('confirmation_pct', 0.2)}%")
    elif assist_cond == 'rsi':
        logger.info(f"  RSI周期: {params.get('rsi_period', 9)}")
    
    if trade:
        trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
        logger.info(f"  交易模式: {trade_mode_names.get(trade_mode, '未知')}")
        logger.info(f"  每次交易金额: {trade_amount} USDT")
        if trade_mode in [2, 3]:
            logger.info(f"  杠杆倍数: {leverage}x")
    
    logger.info("=" * 50 + "\n")