#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 12:50 PM
@File       : strategy_2.py
@Description: 1分钟高频短线策略 - 基于K线连续突破确认 + ATR过滤
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import atr, sma
from utils.logger import logger


class HighFrequencyStrategy:
    """1分钟高频短线策略类"""
    
    def __init__(self, client: OKXClient):
        self.client = client
        self.market_data_retriever = MarketDataRetriever(client)

    def calculate_high_frequency_signal(self, symbol: str, bar: str = '1m', 
                                      consecutive_bars: int = 2,
                                      atr_period: int = 14,
                                      atr_threshold: float = 0.8,
                                      volume_factor: float = 1.2,
                                      use_volume: bool = True,
                                      mock_position: int = 0,
                                      breakout_stop_bars: int = 2) -> int:
        """
        计算高频策略信号
        
        信号规则：
        1. 无持仓时，连续n根K线向上突破且ATR条件满足且成交量条件满足时，返回信号1（买入）
        2. 无持仓时，连续n根K线向下突破且ATR条件满足且成交量条件满足时，返回信号-1（卖出）
        3. 持多仓时，连续2根K线向下突破时，返回信号0（平多）
        4. 持空仓时，连续2根K线向上突破时，返回信号0（平空）
        5. 否则返回信号0（持有）
        
        Args:
            symbol: 交易对符号 (e.g., 'BTC-USDT')
            bar: K线时间间隔 (default: 1m)
            consecutive_bars: 连续K线数量 (default: 2)
            atr_period: ATR周期 (default: 14)
            atr_threshold: ATR阈值 (default: 0.8)
            volume_factor: 成交量放大倍数 (default: 1.2)
            use_volume: 是否使用成交量条件 (default: True)
            mock_position: 模拟持仓状态 (0:无持仓, 1:多仓, -1:空仓)
            
        Returns:
            int: 信号值 (1=买入, -1=卖出, 0=持有)
        """
        try:
            # 获取足够的K线数据
            limit = max(consecutive_bars + 1, atr_period + 1, 21)  # 21用于成交量计算
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return 0
                
            # 获取最新数据
            current_close = df['c'].iloc[-1] if 'c' in df.columns else df['close'].iloc[-1]
            current_high = df['h'].iloc[-1] if 'h' in df.columns else df['high'].iloc[-1]
            current_low = df['l'].iloc[-1] if 'l' in df.columns else df['low'].iloc[-1]
            current_volume = df['vol'].iloc[-1] if 'vol' in df.columns else df['volume'].iloc[-1]

            # 计算典型价格 (high + low + close) / 3
            typical_prices = (df['h'] + df['l'] + df['c']) / 3 if 'h' in df.columns else (df['high'] + df['low'] + df['close']) / 3

            # 计算ATR
            atr_values = atr(df, atr_period)
            current_atr = atr_values.iloc[-1]
            
            # 计算ATR均值
            atr_mean = atr_values.iloc[-atr_period:].mean() if len(atr_values) >= atr_period else current_atr

            # 计算成交量条件
            volume_condition_met = False
            if use_volume and len(df) >= 21:  # 前20根K线平均成交量
                volumes = df['vol'] if 'vol' in df.columns else df['volume']
                avg_volume = volumes.iloc[-21:-1].mean()
                volume_condition_met = current_volume > avg_volume * volume_factor

            # 检查连续突破条件
            long_breakout = self._check_consecutive_breakout(df, typical_prices, consecutive_bars, direction='up')
            short_breakout = self._check_consecutive_breakout(df, typical_prices, consecutive_bars, direction='down')

            # ATR过滤条件
            atr_condition_met = current_atr > atr_mean * atr_threshold

            # 开多条件
            if mock_position == 0 and long_breakout and atr_condition_met:
                if not use_volume or volume_condition_met:
                    return 1

            # 开空条件
            elif mock_position == 0 and short_breakout and atr_condition_met:
                if not use_volume or volume_condition_met:
                    return -1

            # 平仓条件 (连续breakout_stop_bars根K线反向突破)
            elif mock_position != 0:
                if mock_position == 1 and self._check_consecutive_breakout(df, typical_prices, breakout_stop_bars, direction='down'):
                    return 0
                elif mock_position == -1 and self._check_consecutive_breakout(df, typical_prices, breakout_stop_bars, direction='up'):
                    return 0

            # 持有信号
            return 0
                
        except Exception as e:
            logger.error(f"Error calculating high frequency signal for {symbol}: {e}")
            return 0

    def _check_consecutive_breakout(self, df: pd.DataFrame, typical_prices: pd.Series, 
                                   consecutive_bars: int, direction: str) -> bool:
        """
        检查连续突破条件
        
        Args:
            df: K线数据
            typical_prices: 典型价格序列
            consecutive_bars: 连续K线数量
            direction: 突破方向 ('up' 或 'down')
            
        Returns:
            bool: 是否满足连续突破条件
        """
        if len(df) < consecutive_bars + 1:
            return False

        # 检查最近consecutive_bars根K线是否连续突破
        for i in range(consecutive_bars):
            current_idx = -1 - i
            prev_idx = -2 - i
            
            if direction == 'up':
                # 向上突破: 当前close > 前一根typical price
                if df['close'].iloc[current_idx] <= typical_prices.iloc[prev_idx]:
                    return False
            else:
                # 向下突破: 当前close < 前一根typical price
                if df['close'].iloc[current_idx] >= typical_prices.iloc[prev_idx]:
                    return False

        return True

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

    def get_strategy_details(self, symbol: str, bar: str = '1m', 
                           consecutive_bars: int = 2,
                           atr_period: int = 14,
                           atr_threshold: float = 0.8,
                           volume_factor: float = 1.2,
                           use_volume: bool = True,
                           mock_position: int = 0,
                           breakout_stop_bars: int = 2) -> dict:
        """
        获取策略详细信息，用于调试和分析
        
        Args:
            symbol: 交易对符号
            bar: K线时间间隔
            consecutive_bars: 连续K线数量 (default: 2)
            atr_period: ATR周期 (default: 14)
            atr_threshold: ATR阈值 (default: 0.8)
            volume_factor: 成交量放大倍数 (default: 1.2)
            use_volume: 是否使用成交量条件 (default: True)
            mock_position: 模拟持仓状态 (0:无持仓, 1:多仓, -1:空仓)
            
        Returns:
            dict: 包含策略计算详情的字典
        """
        try:
            # 获取足够的K线数据
            limit = max(consecutive_bars + 1, atr_period + 1, 21)
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return {}
                
            # 获取最新数据
            current_close = df['c'].iloc[-1] if 'c' in df.columns else df['close'].iloc[-1]
            current_high = df['h'].iloc[-1] if 'h' in df.columns else df['high'].iloc[-1]
            current_low = df['l'].iloc[-1] if 'l' in df.columns else df['low'].iloc[-1]
            current_volume = df['vol'].iloc[-1] if 'vol' in df.columns else df['volume'].iloc[-1]

            # 计算典型价格 (high + low + close) / 3
            typical_prices = (df['h'] + df['l'] + df['c']) / 3 if 'h' in df.columns else (df['high'] + df['low'] + df['close']) / 3
            current_typical = typical_prices.iloc[-1]
            prev_typical = typical_prices.iloc[-2] if len(typical_prices) > 1 else current_typical

            # 计算ATR
            atr_values = atr(df, atr_period)
            current_atr = atr_values.iloc[-1]
            
            # 计算ATR均值
            atr_mean = atr_values.iloc[-atr_period:].mean() if len(atr_values) >= atr_period else current_atr

            # 计算成交量条件
            volume_condition_met = False
            avg_volume = 0
            if use_volume and len(df) >= 21:
                volumes = df['vol'] if 'vol' in df.columns else df['volume']
                avg_volume = volumes.iloc[-21:-1].mean()
                volume_condition_met = current_volume > avg_volume * volume_factor

            # 检查连续突破条件
            long_breakout = self._check_consecutive_breakout(df, typical_prices, consecutive_bars, direction='up')
            short_breakout = self._check_consecutive_breakout(df, typical_prices, consecutive_bars, direction='down')

            # ATR过滤条件
            atr_condition_met = current_atr > atr_mean * atr_threshold

            # 计算信号
            signal = self.calculate_high_frequency_signal(symbol, bar, consecutive_bars, atr_period, 
                                                         atr_threshold, volume_factor, use_volume, mock_position, breakout_stop_bars)

            return {
                'symbol': symbol,
                'current_price': float(current_close),
                'current_typical': float(current_typical),
                'prev_typical': float(prev_typical),
                'atr': float(current_atr),
                'atr_mean': float(atr_mean),
                'atr_condition_met': atr_condition_met,
                'volume_condition_met': volume_condition_met,
                'long_breakout': long_breakout,
                'short_breakout': short_breakout,
                'consecutive_bars': consecutive_bars,
                'breakout_stop_bars': breakout_stop_bars,
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'signal': signal
            }
            
        except Exception as e:
            logger.error(f"Error getting strategy details for {symbol}: {e}")
            return {}


def main():
    """测试函数"""
    logger.info("High Frequency Strategy Test")
    logger.info("=" * 50)
    
    # 初始化客户端
    client = OKXClient()
    strategy = HighFrequencyStrategy(client)
    
    # 测试几个常见的交易对
    test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
    
    for symbol in test_symbols:
        try:
            signal = strategy.calculate_high_frequency_signal(symbol, bar='1m', consecutive_bars=2, 
                                                             atr_period=14, atr_threshold=0.8, 
                                                             volume_factor=1.2, use_volume=True, mock_position=0)
            details = strategy.get_strategy_details(symbol, bar='1m', consecutive_bars=2, 
                                                  atr_period=14, atr_threshold=0.8, 
                                                  volume_factor=1.2, use_volume=True, mock_position=0)
            
            if details:
                logger.info(f"\n{symbol}:")
                logger.info(f"  Price: ${details['current_price']:.4f}")
                logger.info(f"  Typical Price: ${details['current_typical']:.4f}")
                logger.info(f"  ATR: {details['atr']:.4f}")
                logger.info(f"  ATR Mean: {details['atr_mean']:.4f}")
                logger.info(f"  ATR Condition: {details['atr_condition_met']}")
                logger.info(f"  Volume Expansion: {details['volume_condition_met']}")
                logger.info(f"  Long Breakout: {details['long_breakout']}")
                logger.info(f"  Short Breakout: {details['short_breakout']}")
                logger.info(f"  Signal: {signal} ({'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'})")
        except Exception as e:
            logger.error(f"Error testing {symbol}: {e}")


if __name__ == "__main__":
    main()