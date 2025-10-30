#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/22/25 9:52 PM
@File       : strategy_3.py
@Description: 长下影线策略 - 基于长下影线形态 + 成交量过滤
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import sma
from utils.logger import logger


class LongShadowStrategy:
    """长下影线策略类"""
    
    def __init__(self, client: OKXClient):
        self.client = client
        self.market_data_retriever = MarketDataRetriever(client)

    def calculate_long_shadow_signal(self, symbol: str, bar: str = '1m', 
                                   min_volume_ccy: float = 100000,
                                   volume_factor: float = 1.2,
                                   use_volume: bool = True,
                                   mock_position: int = 0) -> int:
        """
        计算长下影线策略信号
        
        信号规则：
        1. 无持仓时，倒数第二根K线是长下影线且成交量条件满足且倒数第一根K线满足入场条件时，返回信号1（买入）
        2. 无持仓时，倒数第二根K线是长上影线且成交量条件满足且倒数第一根K线满足入场条件时，返回信号-1（卖出）
        3. 持多仓时，当前K线收盘价 < 前一根K线的(high+low)/2，返回信号0（平多）
        4. 持空仓时，当前K线收盘价 > 前一根K线的(high+low)/2，返回信号0（平空）
        5. 否则返回信号0（持有）
        
        Args:
            symbol: 交易对符号 (e.g., 'BTC-USDT')
            bar: K线时间间隔 (default: 1m)
            min_volume_ccy: 最小24小时交易量 (default: 100000)
            volume_factor: 成交量放大倍数 (default: 1.2)
            use_volume: 是否使用成交量条件 (default: True)
            mock_position: 模拟持仓状态 (0:无持仓, 1:多仓, -1:空仓)
            
        Returns:
            int: 信号值 (1=买入, -1=卖出, 0=持有)
        """
        try:
            # 获取足够的K线数据
            limit = 12  # 需要至少12根K线用于计算
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return 0
                
            # 获取最新数据
            current_close = df['c'].iloc[-1] if 'c' in df.columns else df['close'].iloc[-1]
            current_open = df['o'].iloc[-1] if 'o' in df.columns else df['open'].iloc[-1]
            current_high = df['h'].iloc[-1] if 'h' in df.columns else df['high'].iloc[-1]
            current_low = df['l'].iloc[-1] if 'l' in df.columns else df['low'].iloc[-1]
            current_volume = df['vol'].iloc[-1] if 'vol' in df.columns else df['volume'].iloc[-1]

            # 获取倒数第二根K线数据（长下影线检测）
            prev2_close = df['c'].iloc[-2] if 'c' in df.columns else df['close'].iloc[-2]
            prev2_open = df['o'].iloc[-2] if 'o' in df.columns else df['open'].iloc[-2]
            prev2_high = df['h'].iloc[-2] if 'h' in df.columns else df['high'].iloc[-2]
            prev2_low = df['l'].iloc[-2] if 'l' in df.columns else df['low'].iloc[-2]
            prev2_volume = df['vol'].iloc[-2] if 'vol' in df.columns else df['volume'].iloc[-2]

            # 计算成交量条件
            volume_condition_met = False
            if use_volume and len(df) >= 11:  # 前10根K线平均成交量
                volumes = df['vol'] if 'vol' in df.columns else df['volume']
                avg_volume = volumes.iloc[-11:-1].mean()
                volume_condition_met = prev2_volume > avg_volume * volume_factor

            # 检查长下影线条件（多单入场）
            long_shadow_condition = self._check_long_lower_shadow(prev2_open, prev2_high, prev2_low, prev2_close)
            
            # 检查长上影线条件（空单入场）
            short_shadow_condition = self._check_long_upper_shadow(prev2_open, prev2_high, prev2_low, prev2_close)

            # 检查倒数第一根K线入场条件
            long_entry_condition = self._check_long_entry_condition(current_open, current_high, current_low, current_close)
            short_entry_condition = self._check_short_entry_condition(current_open, current_high, current_low, current_close)

            # 开多条件
            if mock_position == 0 and long_shadow_condition and long_entry_condition:
                if not use_volume or volume_condition_met:
                    return 1

            # 开空条件
            elif mock_position == 0 and short_shadow_condition and short_entry_condition:
                if not use_volume or volume_condition_met:
                    return -1

            # 平仓条件
            elif mock_position != 0:
                # 获取前一根K线数据
                prev_close = df['c'].iloc[-2] if 'c' in df.columns else df['close'].iloc[-2]
                prev_high = df['h'].iloc[-2] if 'h' in df.columns else df['high'].iloc[-2]
                prev_low = df['l'].iloc[-2] if 'l' in df.columns else df['low'].iloc[-2]
                
                # 计算前一根K线的(high+low)/2
                prev_mid_price = (prev_high + prev_low) / 2
                
                if mock_position == 1 and current_close < prev_mid_price:
                    return 0  # 平多仓
                elif mock_position == -1 and current_close > prev_mid_price:
                    return 0  # 平空仓

            # 持有信号
            return 0
                
        except Exception as e:
            logger.error(f"Error calculating long shadow signal for {symbol}: {e}")
            return 0

    def _check_long_lower_shadow(self, open_price: float, high: float, low: float, close: float) -> bool:
        """
        检查长下影线条件
        
        条件：下影线长度 > 整根K线长度的1/2
        """
        candle_length = high - low
        if candle_length == 0:
            return False
            
        lower_shadow = min(open_price, close) - low
        lower_shadow_ratio = lower_shadow / candle_length
        
        return lower_shadow_ratio > 1/2

    def _check_long_upper_shadow(self, open_price: float, high: float, low: float, close: float) -> bool:
        """
        检查长上影线条件
        
        条件：上影线长度 > 整根K线长度的1/2
        """
        candle_length = high - low
        if candle_length == 0:
            return False
            
        upper_shadow = high - max(open_price, close)
        upper_shadow_ratio = upper_shadow / candle_length
        
        return upper_shadow_ratio > 1/2

    def _check_long_entry_condition(self, open_price: float, high: float, low: float, close: float) -> bool:
        """
        检查多单入场条件
        
        条件：
        1. 收盘价 > 开盘价
        2. 上影线长度 < 整根K线长度的1/5
        """
        if close <= open_price:
            return False
            
        candle_length = high - low
        if candle_length == 0:
            return False
            
        upper_shadow = high - close
        upper_shadow_ratio = upper_shadow / candle_length
        
        return upper_shadow_ratio < 1/5

    def _check_short_entry_condition(self, open_price: float, high: float, low: float, close: float) -> bool:
        """
        检查空单入场条件
        
        条件：
        1. 收盘价 < 开盘价
        2. 下影线长度 < 整根K线长度的1/5
        """
        if close >= open_price:
            return False
            
        candle_length = high - low
        if candle_length == 0:
            return False
            
        lower_shadow = open_price - low
        lower_shadow_ratio = lower_shadow / candle_length
        
        return lower_shadow_ratio < 1/5

    def calculate_trailing_stop(self, current_price: float, position: int, 
                               highest_price: float, lowest_price: float, 
                               trailing_stop_pct: float = 0.8) -> Dict[str, Any]:
        """
        计算移动止损
        
        Args:
            current_price: 当前价格
            position: 持仓方向 (1:多仓, -1:空仓)
            highest_price: 最高价记录
            lowest_price: 最低价记录
            trailing_stop_pct: 移动止损百分比
            
        Returns:
            Dict: 包含止损信息和是否触发的字典
        """
        stop_triggered = False
        stop_price = 0.0
        
        if position == 1:  # 多仓
            # 更新最高价
            if current_price > highest_price:
                highest_price = current_price
            
            # 计算止损价
            stop_price = highest_price * (1 - trailing_stop_pct / 100.0)
            
            # 检查是否触发止损
            if current_price <= stop_price:
                stop_triggered = True
                
        elif position == -1:  # 空仓
            # 更新最低价
            if current_price < lowest_price:
                lowest_price = current_price
            
            # 计算止损价
            stop_price = lowest_price * (1 + trailing_stop_pct / 100.0)
            
            # 检查是否触发止损
            if current_price >= stop_price:
                stop_triggered = True

        return {
            'stop_triggered': stop_triggered,
            'stop_price': stop_price,
            'highest_price': highest_price,
            'lowest_price': lowest_price
        }

    def calculate_take_profit(self, current_price: float, position: int, 
                            entry_price: float, take_profit_pct: float = 2.0) -> Dict[str, Any]:
        """
        计算止盈
        
        Args:
            current_price: 当前价格
            position: 持仓方向 (1:多仓, -1:空仓)
            entry_price: 入场价格
            take_profit_pct: 止盈百分比
            
        Returns:
            Dict: 包含止盈信息和是否触发的字典
        """
        take_profit_triggered = False
        take_profit_price = 0.0
        
        if position == 1:  # 多仓
            # 计算止盈价
            take_profit_price = entry_price * (1 + take_profit_pct / 100.0)
            
            # 检查是否触发止盈
            if current_price >= take_profit_price:
                take_profit_triggered = True
                
        elif position == -1:  # 空仓
            # 计算止盈价
            take_profit_price = entry_price * (1 - take_profit_pct / 100.0)
            
            # 检查是否触发止盈
            if current_price <= take_profit_price:
                take_profit_triggered = True

        return {
            'take_profit_triggered': take_profit_triggered,
            'take_profit_price': take_profit_price
        }

    def get_strategy_details(self, symbol: str, bar: str = '1m', 
                           min_volume_ccy: float = 100000,
                           volume_factor: float = 1.2,
                           use_volume: bool = True,
                           mock_position: int = 0) -> dict:
        """
        获取策略详细信息，用于调试和分析
        
        Args:
            symbol: 交易对符号
            bar: K线时间间隔
            min_volume_ccy: 最小24小时交易量
            volume_factor: 成交量放大倍数
            use_volume: 是否使用成交量条件
            mock_position: 模拟持仓状态 (0:无持仓, 1:多仓, -1:空仓)
            
        Returns:
            dict: 包含策略计算详情的字典
        """
        try:
            # 获取足够的K线数据
            limit = 12
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return {}
                
            # 获取最新数据
            current_close = df['c'].iloc[-1] if 'c' in df.columns else df['close'].iloc[-1]
            current_open = df['o'].iloc[-1] if 'o' in df.columns else df['open'].iloc[-1]
            current_high = df['h'].iloc[-1] if 'h' in df.columns else df['high'].iloc[-1]
            current_low = df['l'].iloc[-1] if 'l' in df.columns else df['low'].iloc[-1]
            current_volume = df['vol'].iloc[-1] if 'vol' in df.columns else df['volume'].iloc[-1]

            # 获取倒数第二根K线数据
            prev2_close = df['c'].iloc[-2] if 'c' in df.columns else df['close'].iloc[-2]
            prev2_open = df['o'].iloc[-2] if 'o' in df.columns else df['open'].iloc[-2]
            prev2_high = df['h'].iloc[-2] if 'h' in df.columns else df['high'].iloc[-2]
            prev2_low = df['l'].iloc[-2] if 'l' in df.columns else df['low'].iloc[-2]
            prev2_volume = df['vol'].iloc[-2] if 'vol' in df.columns else df['volume'].iloc[-2]

            # 计算成交量条件
            volume_condition_met = False
            avg_volume = 0
            if use_volume and len(df) >= 11:
                volumes = df['vol'] if 'vol' in df.columns else df['volume']
                avg_volume = volumes.iloc[-11:-1].mean()
                volume_condition_met = prev2_volume > avg_volume * volume_factor

            # 检查各种条件
            long_shadow_condition = self._check_long_lower_shadow(prev2_open, prev2_high, prev2_low, prev2_close)
            short_shadow_condition = self._check_long_upper_shadow(prev2_open, prev2_high, prev2_low, prev2_close)
            long_entry_condition = self._check_long_entry_condition(current_open, current_high, current_low, current_close)
            short_entry_condition = self._check_short_entry_condition(current_open, current_high, current_low, current_close)

            # 计算信号
            signal = self.calculate_long_shadow_signal(symbol, bar, min_volume_ccy, volume_factor, use_volume, mock_position)

            return {
                'symbol': symbol,
                'current_price': float(current_close),
                'current_open': float(current_open),
                'current_high': float(current_high),
                'current_low': float(current_low),
                'prev2_open': float(prev2_open),
                'prev2_high': float(prev2_high),
                'prev2_low': float(prev2_low),
                'prev2_close': float(prev2_close),
                'volume_condition_met': volume_condition_met,
                'long_shadow_condition': long_shadow_condition,
                'short_shadow_condition': short_shadow_condition,
                'long_entry_condition': long_entry_condition,
                'short_entry_condition': short_entry_condition,
                'current_volume': float(current_volume),
                'prev2_volume': float(prev2_volume),
                'avg_volume': float(avg_volume),
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy details for {symbol}: {e}")
            return {}


def main():
    """测试函数"""
    logger.info("Long Shadow Strategy Test")
    logger.info("=" * 50)
    
    # 初始化客户端
    client = OKXClient()
    strategy = LongShadowStrategy(client)
    
    # 测试几个常见的交易对
    test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
    
    for symbol in test_symbols:
        try:
            signal = strategy.calculate_long_shadow_signal(symbol, bar='1m', min_volume_ccy=100000, 
                                                          volume_factor=1.2, use_volume=True, mock_position=0)
            details = strategy.get_strategy_details(symbol, bar='1m', min_volume_ccy=100000, 
                                                  volume_factor=1.2, use_volume=True, mock_position=0)
            
            if details:
                logger.info(f"\n{symbol}:")
                logger.info(f"  Price: ${details['current_price']:.4f}")
                logger.info(f"  Long Shadow: {details['long_shadow_condition']}")
                logger.info(f"  Short Shadow: {details['short_shadow_condition']}")
                logger.info(f"  Long Entry: {details['long_entry_condition']}")
                logger.info(f"  Short Entry: {details['short_entry_condition']}")
                logger.info(f"  Volume Expansion: {details['volume_condition_met']}")
                logger.info(f"  Signal: {signal} ({'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'})")
        except Exception as e:
            logger.error(f"Error testing {symbol}: {e}")


if __name__ == "__main__":
    main()