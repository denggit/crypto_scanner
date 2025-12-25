#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : shared_config.py
@Description: Shared configuration functions for strategy 6
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
    logger.info("VCB策略参数配置")
    logger.info("=" * 50)

    # 使用配置文件的默认值
    if default_config is None:
        default_config = {}

    default_symbol = default_config.get('symbol', 'BTC-USDT')
    default_bar = default_config.get('bar', '1m')
    default_atr_short_period = default_config.get('atr_short_period', 10)
    default_atr_mid_period = default_config.get('atr_mid_period', 60)
    default_atr_ratio_threshold = default_config.get('atr_ratio_threshold', 0.5)
    default_bb_period = default_config.get('bb_period', 20)
    default_bb_std = default_config.get('bb_std', 2)
    default_bb_width_ratio = default_config.get('bb_width_ratio', 0.7)
    default_volume_period = default_config.get('volume_period', 20)
    default_volume_multiplier = default_config.get('volume_multiplier', 1.0)
    default_ttl_bars = default_config.get('ttl_bars', 30)
    default_trailing_stop_pct = default_config.get('trailing_stop_pct', 1.0)
    default_stop_loss_atr_multiplier = default_config.get('stop_loss_atr_multiplier', 0.8)
    default_take_profit_r = default_config.get('take_profit_r', 2.0)
    default_trade = default_config.get('trade', False)
    default_trade_amount = default_config.get('trade_amount', 10.0)
    default_trade_mode = default_config.get('trade_mode', 3)
    default_leverage = default_config.get('leverage', 3)

    try:
        symbol = input(f"请输入交易对 (默认 {default_symbol}): ").strip() or default_symbol
        bar = input(f"请输入K线周期 (默认 {default_bar}): ").strip() or default_bar
        
        atr_short_input = input(f"请输入短期ATR周期 (默认 {default_atr_short_period}): ").strip()
        atr_short_period = int(atr_short_input) if atr_short_input else default_atr_short_period
        
        atr_mid_input = input(f"请输入中期ATR周期 (默认 {default_atr_mid_period}): ").strip()
        atr_mid_period = int(atr_mid_input) if atr_mid_input else default_atr_mid_period
        
        atr_ratio_input = input(f"请输入ATR比率阈值 (默认 {default_atr_ratio_threshold}): ").strip()
        atr_ratio_threshold = float(atr_ratio_input) if atr_ratio_input else default_atr_ratio_threshold
        
        bb_period_input = input(f"请输入布林带周期 (默认 {default_bb_period}): ").strip()
        bb_period = int(bb_period_input) if bb_period_input else default_bb_period
        
        bb_std_input = input(f"请输入布林带标准差倍数 (默认 {default_bb_std}): ").strip()
        bb_std = int(bb_std_input) if bb_std_input else default_bb_std
        
        bb_width_input = input(f"请输入布林带宽度收缩比率 (默认 {default_bb_width_ratio}): ").strip()
        bb_width_ratio = float(bb_width_input) if bb_width_input else default_bb_width_ratio
        
        volume_period_input = input(f"请输入成交量均线周期 (默认 {default_volume_period}): ").strip()
        volume_period = int(volume_period_input) if volume_period_input else default_volume_period
        
        volume_multiplier_input = input(f"请输入成交量放大倍数 (默认 {default_volume_multiplier}): ").strip()
        volume_multiplier = float(volume_multiplier_input) if volume_multiplier_input else default_volume_multiplier
        
        ttl_bars_input = input(f"请输入压缩事件TTL（K线数量） (默认 {default_ttl_bars}): ").strip()
        ttl_bars = int(ttl_bars_input) if ttl_bars_input else default_ttl_bars
        
        trailing_stop_input = input(f"请输入移动止损百分比 (默认 {default_trailing_stop_pct}%): ").strip()
        trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else default_trailing_stop_pct
        
        stop_loss_atr_input = input(f"请输入止损ATR倍数 (默认 {default_stop_loss_atr_multiplier}): ").strip()
        stop_loss_atr_multiplier = float(stop_loss_atr_input) if stop_loss_atr_input else default_stop_loss_atr_multiplier
        
        take_profit_r_input = input(f"请输入止盈R倍数 (默认 {default_take_profit_r}): ").strip()
        take_profit_r = float(take_profit_r_input) if take_profit_r_input else default_take_profit_r

        trade_input = input(f"是否真实交易 (y/n, 默认 {'y' if default_trade else 'n'}): ").strip().lower()
        if trade_input:
            trade = trade_input == 'y' or trade_input == 'yes'
        else:
            trade = default_trade

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
        atr_short_period = default_atr_short_period
        atr_mid_period = default_atr_mid_period
        atr_ratio_threshold = default_atr_ratio_threshold
        bb_period = default_bb_period
        bb_std = default_bb_std
        bb_width_ratio = default_bb_width_ratio
        volume_period = default_volume_period
        volume_multiplier = default_volume_multiplier
        ttl_bars = default_ttl_bars
        trailing_stop_pct = default_trailing_stop_pct
        stop_loss_atr_multiplier = default_stop_loss_atr_multiplier
        take_profit_r = default_take_profit_r
        trade = default_trade
        trade_amount = default_trade_amount
        trade_mode = default_trade_mode
        leverage = default_leverage

    return {
        'symbol': symbol,
        'bar': bar,
        'atr_short_period': atr_short_period,
        'atr_mid_period': atr_mid_period,
        'atr_ratio_threshold': atr_ratio_threshold,
        'bb_period': bb_period,
        'bb_std': bb_std,
        'bb_width_ratio': bb_width_ratio,
        'volume_period': volume_period,
        'volume_multiplier': volume_multiplier,
        'ttl_bars': ttl_bars,
        'trailing_stop_pct': trailing_stop_pct,
        'stop_loss_atr_multiplier': stop_loss_atr_multiplier,
        'take_profit_r': take_profit_r,
        'trade': trade,
        'trade_amount': trade_amount,
        'trade_mode': trade_mode,
        'leverage': leverage
    }


def print_final_config(config: dict):
    """打印最终使用的参数"""
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    atr_short_period = config.get('atr_short_period', 10)
    atr_mid_period = config.get('atr_mid_period', 60)
    atr_ratio_threshold = config.get('atr_ratio_threshold', 0.5)
    bb_period = config.get('bb_period', 20)
    bb_std = config.get('bb_std', 2)
    bb_width_ratio = config.get('bb_width_ratio', 0.7)
    volume_period = config.get('volume_period', 20)
    volume_multiplier = config.get('volume_multiplier', 1.0)
    ttl_bars = config.get('ttl_bars', 30)
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    stop_loss_atr_multiplier = config.get('stop_loss_atr_multiplier', 0.8)
    take_profit_r = config.get('take_profit_r', 2.0)
    trade = config.get('trade', False)
    trade_amount = config.get('trade_amount', 10.0)
    trade_mode = config.get('trade_mode', 3)
    leverage = config.get('leverage', 3)

    logger.info("\n" + "=" * 50)
    logger.info("本次运行使用的最终参数:")
    logger.info(f"  交易对: {symbol}")
    logger.info(f"  K线周期: {bar}")
    logger.info(f"  ATR参数: {atr_short_period}/{atr_mid_period}")
    logger.info(f"  ATR比率阈值: {atr_ratio_threshold}")
    logger.info(f"  布林带参数: {bb_period}, {bb_std}")
    logger.info(f"  布林带宽度收缩比率: {bb_width_ratio}")
    logger.info(f"  成交量均线周期: {volume_period}")
    logger.info(f"  成交量放大倍数: {volume_multiplier}")
    logger.info(f"  压缩事件TTL: {ttl_bars}根K线")
    logger.info(f"  移动止损: {trailing_stop_pct}%")
    logger.info(f"  止损ATR倍数: {stop_loss_atr_multiplier}")
    logger.info(f"  止盈R倍数: {take_profit_r}")
    logger.info(f"  真实交易: {'是' if trade else '否'}")
    
    if trade:
        trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
        logger.info(f"  交易模式: {trade_mode_names.get(trade_mode, '未知')}")
        logger.info(f"  每次交易金额: {trade_amount} USDT")
        if trade_mode in [2, 3]:
            logger.info(f"  杠杆倍数: {leverage}x")
    
    logger.info("=" * 50 + "\n")

