#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : position_manager.py
@Description: 仓位管理模块 - 实现硬止损、主动止盈、失败退出逻辑
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from apis.okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import atr, bollinger_bands
from utils.logger import logger


# 主流币列表
MAJOR_COINS = [
    'BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE',
    'AVAX', 'TON', 'TRX', 'DOT', 'MATIC', 'LTC', 'BCH'
]


class PositionManager:
    """
    仓位管理器
    
    实现三层平仓逻辑：
    1. 硬止损（结构性错误，必须走）
    2. 主动止盈（结构兑现）
    3. 失败退出（时间成本控制）
    """
    
    def __init__(self, market_data_retriever: MarketDataRetriever,
                 bar: str = '1m',
                 atr_mid_period: int = 60,
                 stop_loss_atr_multiplier: float = 0.8,
                 take_profit_mode: str = 'r_multiple',  # 'r_multiple', 'bb_middle', 'bb_opposite', 'atr_trailing'
                 take_profit_r: float = 2.0,
                 take_profit_r_major: float = 1.5,  # 主流币R倍数
                 take_profit_r_alt: float = 2.5,  # 山寨币R倍数
                 failure_exit_bars: int = 10,  # 失败退出K线数量
                 failure_exit_atr_threshold: float = 1.2,  # 失败退出ATR阈值（突破后ATR需扩展到此倍数）
                 break_even_r: float = 1.0):  # Break-even触发R倍数
        """
        初始化仓位管理器
        
        Args:
            market_data_retriever: 市场数据获取器
            bar: K线周期
            atr_mid_period: 中期ATR周期
            stop_loss_atr_multiplier: 止损ATR倍数
            take_profit_mode: 止盈模式
            take_profit_r: 止盈R倍数（默认）
            take_profit_r_major: 主流币止盈R倍数
            take_profit_r_alt: 山寨币止盈R倍数
            failure_exit_bars: 失败退出K线数量
            failure_exit_atr_threshold: 失败退出ATR阈值
            break_even_r: Break-even触发R倍数
        """
        self.market_data_retriever = market_data_retriever
        self.bar = bar
        self.atr_mid_period = atr_mid_period
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_mode = take_profit_mode
        self.take_profit_r = take_profit_r
        self.take_profit_r_major = take_profit_r_major
        self.take_profit_r_alt = take_profit_r_alt
        self.failure_exit_bars = failure_exit_bars
        self.failure_exit_atr_threshold = failure_exit_atr_threshold
        self.break_even_r = break_even_r
    
    def is_major_coin(self, symbol: str) -> bool:
        """
        判断是否为主流币
        
        Args:
            symbol: 交易对符号（如 BTC-USDT）
            
        Returns:
            bool: True表示主流币
        """
        base_coin = symbol.split('-')[0]
        return base_coin in MAJOR_COINS
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, position: int,
                           compression_event=None) -> float:
        """
        计算硬止损价格
        
        硬止损逻辑：
        - 基于压缩区间另一侧
        - ATR(60) 的兜底
        
        Args:
            symbol: 交易对符号
            entry_price: 入场价格
            position: 持仓方向（1=做多, -1=做空）
            compression_event: 压缩事件（可选）
            
        Returns:
            float: 止损价格
        """
        try:
            # 获取ATR数据
            limit = self.atr_mid_period + 5
            df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
            
            if df is None or len(df) < self.atr_mid_period:
                # 如果无法获取数据，使用固定百分比止损
                if position == 1:
                    return entry_price * 0.98  # 2%止损
                else:
                    return entry_price * 1.02  # 2%止损
            
            from tools.technical_indicators import atr
            atr_mid = atr(df, self.atr_mid_period)
            
            if len(atr_mid) < 1 or pd.isna(atr_mid.iloc[-1]) or atr_mid.iloc[-1] == 0:
                # 使用固定百分比止损
                if position == 1:
                    return entry_price * 0.98
                else:
                    return entry_price * 1.02
            
            current_atr = atr_mid.iloc[-1]
            atr_stop = current_atr * self.stop_loss_atr_multiplier
            
            # 计算基于压缩区间的止损
            if compression_event:
                if position == 1:
                    # 做多：止损在压缩区间下轨
                    compression_stop = compression_event.bb_lower
                    # 取较大值（更宽松的止损）
                    stop_loss = max(compression_stop, entry_price - atr_stop)
                else:
                    # 做空：止损在压缩区间上轨
                    compression_stop = compression_event.bb_upper
                    # 取较小值（更宽松的止损）
                    stop_loss = min(compression_stop, entry_price + atr_stop)
            else:
                # 没有压缩事件，使用ATR止损
                if position == 1:
                    stop_loss = entry_price - atr_stop
                else:
                    stop_loss = entry_price + atr_stop
            
            return stop_loss
            
        except Exception as e:
            logger.error(f"计算止损时出错 {symbol}: {e}")
            # 使用固定百分比止损
            if position == 1:
                return entry_price * 0.98
            else:
                return entry_price * 1.02
    
    def calculate_take_profit(self, symbol: str, entry_price: float, stop_loss: float,
                             position: int, compression_event=None) -> float:
        """
        计算止盈价格
        
        支持三种止盈模式：
        1. R倍止盈
        2. 布林中轨/对侧轨道止盈
        3. ATR跟踪止盈
        
        Args:
            symbol: 交易对符号
            entry_price: 入场价格
            stop_loss: 止损价格
            position: 持仓方向（1=做多, -1=做空）
            compression_event: 压缩事件（可选）
            
        Returns:
            float: 止盈价格
        """
        try:
            # 计算风险（R）
            risk = abs(entry_price - stop_loss)
            if risk == 0:
                risk = entry_price * 0.02  # 默认2%风险
            
            # 根据币种类型选择R倍数
            if self.is_major_coin(symbol):
                take_profit_r = self.take_profit_r_major
            else:
                take_profit_r = self.take_profit_r_alt
            
            if self.take_profit_mode == 'r_multiple':
                # R倍止盈
                if position == 1:
                    return entry_price + risk * take_profit_r
                else:
                    return entry_price - risk * take_profit_r
            
            elif self.take_profit_mode == 'bb_middle':
                # 布林中轨止盈
                if compression_event:
                    if position == 1:
                        return compression_event.bb_middle
                    else:
                        return compression_event.bb_middle
                else:
                    # 如果没有压缩事件，使用R倍止盈
                    if position == 1:
                        return entry_price + risk * take_profit_r
                    else:
                        return entry_price - risk * take_profit_r
            
            elif self.take_profit_mode == 'bb_opposite':
                # 对侧轨道止盈
                if compression_event:
                    if position == 1:
                        # 做多：止盈在对侧（上轨）
                        return compression_event.bb_upper
                    else:
                        # 做空：止盈在对侧（下轨）
                        return compression_event.bb_lower
                else:
                    # 如果没有压缩事件，使用R倍止盈
                    if position == 1:
                        return entry_price + risk * take_profit_r
                    else:
                        return entry_price - risk * take_profit_r
            
            elif self.take_profit_mode == 'atr_trailing':
                # ATR跟踪止盈（需要实时计算，这里返回初始止盈）
                # 实际跟踪逻辑在 check_take_profit 中实现
                if position == 1:
                    return entry_price + risk * take_profit_r
                else:
                    return entry_price - risk * take_profit_r
            
            else:
                # 默认使用R倍止盈
                if position == 1:
                    return entry_price + risk * take_profit_r
                else:
                    return entry_price - risk * take_profit_r
                    
        except Exception as e:
            logger.error(f"计算止盈时出错 {symbol}: {e}")
            # 使用默认R倍止盈
            risk = abs(entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
            if position == 1:
                return entry_price + risk * self.take_profit_r
            else:
                return entry_price - risk * self.take_profit_r
    
    def check_hard_stop_loss(self, symbol: str, current_price: float, 
                            position: int, stop_loss: float) -> Tuple[bool, str]:
        """
        检查硬止损
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
            position: 持仓方向（1=做多, -1=做空）
            stop_loss: 止损价格
            
        Returns:
            Tuple[bool, str]: (是否触发, 触发原因)
        """
        if stop_loss <= 0:
            return False, ""
        
        if position == 1:
            # 做多：价格跌破止损
            if current_price <= stop_loss:
                return True, "HARD_STOP_LOSS"
        elif position == -1:
            # 做空：价格涨破止损
            if current_price >= stop_loss:
                return True, "HARD_STOP_LOSS"
        
        return False, ""
    
    def check_take_profit(self, symbol: str, current_price: float, position: int,
                         entry_price: float, stop_loss: float, take_profit: float,
                         compression_event=None) -> Tuple[bool, str, Optional[float]]:
        """
        检查主动止盈
        
        支持多种止盈模式：
        - R倍止盈：固定R倍数
        - 布林中轨止盈：价格回到布林中轨
        - 对侧轨道止盈：价格到达对侧轨道
        - ATR跟踪止盈：ATR扩展后跟踪
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
            position: 持仓方向（1=做多, -1=做空）
            entry_price: 入场价格
            stop_loss: 止损价格
            take_profit: 初始止盈价格
            compression_event: 压缩事件（可选）
            
        Returns:
            Tuple[bool, str, Optional[float]]: (是否触发, 触发原因, 新的止盈价格)
        """
        try:
            if self.take_profit_mode == 'r_multiple':
                # R倍止盈：简单价格比较
                if position == 1:
                    if current_price >= take_profit:
                        return True, "TAKE_PROFIT_R", None
                else:
                    if current_price <= take_profit:
                        return True, "TAKE_PROFIT_R", None
            
            elif self.take_profit_mode == 'bb_middle':
                # 布林中轨止盈
                if compression_event:
                    bb_middle = compression_event.bb_middle
                    if position == 1:
                        # 做多：价格回到中轨下方
                        if current_price <= bb_middle:
                            return True, "TAKE_PROFIT_BB_MIDDLE", None
                    else:
                        # 做空：价格回到中轨上方
                        if current_price >= bb_middle:
                            return True, "TAKE_PROFIT_BB_MIDDLE", None
                else:
                    # 没有压缩事件，使用R倍止盈
                    if position == 1:
                        if current_price >= take_profit:
                            return True, "TAKE_PROFIT_R", None
                    else:
                        if current_price <= take_profit:
                            return True, "TAKE_PROFIT_R", None
            
            elif self.take_profit_mode == 'bb_opposite':
                # 对侧轨道止盈
                if compression_event:
                    if position == 1:
                        # 做多：到达上轨
                        if current_price >= compression_event.bb_upper:
                            return True, "TAKE_PROFIT_BB_OPPOSITE", None
                    else:
                        # 做空：到达下轨
                        if current_price <= compression_event.bb_lower:
                            return True, "TAKE_PROFIT_BB_OPPOSITE", None
                else:
                    # 没有压缩事件，使用R倍止盈
                    if position == 1:
                        if current_price >= take_profit:
                            return True, "TAKE_PROFIT_R", None
                    else:
                        if current_price <= take_profit:
                            return True, "TAKE_PROFIT_R", None
            
            elif self.take_profit_mode == 'atr_trailing':
                # ATR跟踪止盈
                limit = self.atr_mid_period + 5
                df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
                
                if df is None or len(df) < self.atr_mid_period:
                    # 数据不足，使用固定止盈
                    if position == 1:
                        if current_price >= take_profit:
                            return True, "TAKE_PROFIT_R", None
                    else:
                        if current_price <= take_profit:
                            return True, "TAKE_PROFIT_R", None
                    return False, "", None
                
                from tools.technical_indicators import atr
                atr_short = atr(df, 10)
                atr_mid = atr(df, self.atr_mid_period)
                
                if len(atr_short) < 1 or len(atr_mid) < 1:
                    return False, "", None
                
                current_atr_short = atr_short.iloc[-1]
                current_atr_mid = atr_mid.iloc[-1]
                
                if pd.isna(current_atr_short) or pd.isna(current_atr_mid):
                    return False, "", None
                
                # 获取入场时的ATR（需要从持仓信息中获取）
                # 这里简化处理，使用当前ATR判断
                # 如果ATR扩展超过阈值，触发止盈
                if current_atr_short > current_atr_mid * 1.5:
                    return True, "TAKE_PROFIT_ATR_TRAILING", None
                
                # ATR跟踪：动态调整止盈价格
                risk = abs(entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
                if self.is_major_coin(symbol):
                    take_profit_r = self.take_profit_r_major
                else:
                    take_profit_r = self.take_profit_r_alt
                
                # 计算新的止盈价格（基于当前价格和ATR）
                if position == 1:
                    new_take_profit = current_price - current_atr_short * 0.5  # 回撤0.5倍ATR
                    # 止盈价格只能上移，不能下移
                    if new_take_profit > take_profit:
                        return False, "", new_take_profit
                    elif current_price >= take_profit:
                        return True, "TAKE_PROFIT_ATR_TRAILING", None
                else:
                    new_take_profit = current_price + current_atr_short * 0.5
                    if new_take_profit < take_profit:
                        return False, "", new_take_profit
                    elif current_price <= take_profit:
                        return True, "TAKE_PROFIT_ATR_TRAILING", None
            
            return False, "", None
            
        except Exception as e:
            logger.error(f"检查止盈时出错 {symbol}: {e}")
            return False, "", None
    
    def check_failure_exit(self, symbol: str, entry_time: datetime, 
                          entry_atr: float) -> Tuple[bool, str]:
        """
        检查失败退出
        
        失败退出逻辑：
        如果突破后 X 根 K 线内无法扩展波动率 → 主动平仓
        
        Args:
            symbol: 交易对符号
            entry_time: 入场时间
            entry_atr: 入场时的ATR(10)值
            
        Returns:
            Tuple[bool, str]: (是否触发, 触发原因)
        """
        try:
            # 计算从入场到现在经过了多少根K线
            time_diff = datetime.now() - entry_time
            
            if self.bar == '1m':
                bars_elapsed = int(time_diff.total_seconds() / 60)
            elif self.bar == '5m':
                bars_elapsed = int(time_diff.total_seconds() / 300)
            else:
                bars_elapsed = int(time_diff.total_seconds() / 60)  # 默认1m
            
            # 如果还没到检查时间，直接返回
            if bars_elapsed < self.failure_exit_bars:
                return False, ""
            
            # 获取当前ATR
            limit = self.atr_mid_period + 5
            df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
            
            if df is None or len(df) < 10:
                return False, ""
            
            from tools.technical_indicators import atr
            atr_short = atr(df, 10)
            
            if len(atr_short) < 1 or pd.isna(atr_short.iloc[-1]):
                return False, ""
            
            current_atr = atr_short.iloc[-1]
            
            # 检查ATR是否扩展
            if entry_atr > 0:
                atr_expansion_ratio = current_atr / entry_atr
                
                # 如果ATR没有扩展到阈值，说明突破失败
                if atr_expansion_ratio < self.failure_exit_atr_threshold:
                    return True, "FAILURE_EXIT"
            
            return False, ""
            
        except Exception as e:
            logger.error(f"检查失败退出时出错 {symbol}: {e}")
            return False, ""
    
    def check_break_even(self, symbol: str, current_price: float, position: int,
                        entry_price: float, stop_loss: float) -> Tuple[bool, float]:
        """
        检查Break-even条件
        
        当浮盈 ≥ 1R 时，将止损移到入场价（或+手续费）
        
        Args:
            symbol: 交易对符号
            current_price: 当前价格
            position: 持仓方向（1=做多, -1=做空）
            entry_price: 入场价格
            stop_loss: 当前止损价格
            
        Returns:
            Tuple[bool, float]: (是否触发, 新的止损价格)
        """
        try:
            # 计算风险（R）
            risk = abs(entry_price - stop_loss) if stop_loss > 0 else entry_price * 0.02
            
            # 计算当前浮盈
            if position == 1:
                # 做多
                unrealized_pnl = current_price - entry_price
            else:
                # 做空
                unrealized_pnl = entry_price - current_price
            
            # 检查是否达到Break-even条件
            if unrealized_pnl >= risk * self.break_even_r:
                # 将止损移到入场价（或略高于入场价，考虑手续费）
                new_stop_loss = entry_price * 1.001 if position == 1 else entry_price * 0.999
                return True, new_stop_loss
            
            return False, stop_loss
            
        except Exception as e:
            logger.error(f"检查Break-even时出错 {symbol}: {e}")
            return False, stop_loss

