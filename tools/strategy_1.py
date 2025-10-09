#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/8/2025 5:08 PM
@File       : strategy_1.py
@Description: EMA交叉策略 - 基于EMA5和EMA20的交叉信号
"""

import os
import sys
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import ema


class EMACrossoverStrategy:
    """EMA交叉策略类"""
    
    def __init__(self, client: OKXClient):
        self.client = client
        self.market_data_retriever = MarketDataRetriever(client)

    def calculate_ema_crossover_signal(self, symbol: str, bar: str = '1m', 
                                     short_ma: int = 5, long_ma: int = 20,
                                     vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                                     mode: str = 'strict') -> int:
        """
        计算EMA交叉信号
        
        信号规则：
        1. 当成交量放大，且EMA20斜率大于0，EMA5上穿EMA20且K线收盘价大于EMA5时，返回信号1（买入）
        2. 当EMA20斜率小于0，EMA5下穿EMA20且K线收盘价小于EMA5时，返回信号-1（卖出）
        3. 否则返回信号0（持有）
        
        Args:
            symbol: 交易对符号 (e.g., 'BTC-USDT')
            bar: K线时间间隔 (default: 1m)
            short_ma: 短期EMA周期 (default: 5)
            long_ma: 长期EMA周期 (default: 20)
            vol_multiplier: 成交量放大倍数 (default: 1.2)
            confirmation_pct: 确认突破百分比 (default: 0.2)
            mode: 模式 ('strict' or 'loose', default: 'strict')
                  strict mode: 要求当前K线放量（趋势稳健）
                  loose mode: 允许前一根放量（更激进，适合山寨）
            
        Returns:
            int: 信号值 (1=买入, -1=卖出, 0=持有)
        """
        try:
            # 获取足够的K线数据来计算EMA和成交量
            limit = long_ma + 5  # 需要额外数据来计算斜率和交叉
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return 0
                
            # 获取价格和成交量数据
            closes = df['c'] if 'c' in df.columns else df['close']
            volumes = df['vol'] if 'vol' in df.columns else df['volume']
            
            if len(closes) < limit or len(volumes) < limit:
                return 0
                
            # 计算EMA
            ema_short = ema(closes, short_ma)
            ema_long = ema(closes, long_ma)
            
            if len(ema_short) < 2 or len(ema_long) < 2:
                return 0
                
            # 获取最新值和前一个值
            current_close = closes.iloc[-1]
            prev_close = closes.iloc[-2]
            
            current_ema_short = ema_short.iloc[-1]
            prev_ema_short = ema_short.iloc[-2]
            
            current_ema_long = ema_long.iloc[-1]
            prev_ema_long = ema_long.iloc[-2]
            
            # 计算EMA20斜率（使用快速斜率和慢速斜率的组合作为爆发感应器）
            if len(ema_long) >= 5:
                # 快速反应（最近两根K）
                ema20_slope_fast = (ema_long.iloc[-1] - ema_long.iloc[-2]) / ema_long.iloc[-2] if ema_long.iloc[-2] != 0 else 0
                # 稳定趋势（最近五根K）
                ema20_slope_slow = (ema_long.iloc[-1] - ema_long.iloc[-5]) / (5 * ema_long.iloc[-5]) if ema_long.iloc[-5] != 0 else 0
                # 组合判断逻辑，稍偏重趋势平滑的权重
                ema20_slope = (ema20_slope_fast + ema20_slope_slow * 2) / 3
            else:
                ema20_slope = (current_ema_long - prev_ema_long) / prev_ema_long if prev_ema_long != 0 else 0
            
            # 计算成交量条件（根据周期选择不同的成交量判断方法）
            current_volume = volumes.iloc[-1]
            volume_ratio = 0  # 初始化volume_ratio
            
            # 根据模式选择成交量判断方式
            if mode == 'loose' and len(volumes) >= 2:
                # loose模式：允许前一根K线放量
                prev_volume = volumes.iloc[-2]
                if len(volumes) >= 11:  # 需要足够的数据来计算前一根的EMA或平均成交量
                    # 对于小周期（1m~5m），使用EMA成交量
                    if bar in ['1m', '3m', '5m']:
                        vol_ema = volumes.ewm(span=10).mean()
                        current_volume_ratio = current_volume / vol_ema.iloc[-1] if vol_ema.iloc[-1] != 0 else 0
                        prev_volume_ratio = prev_volume / vol_ema.iloc[-2] if vol_ema.iloc[-2] != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                    # 对于大周期（15m+），使用平均成交量
                    elif bar in ['15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D']:
                        avg_volume = volumes.iloc[-11:-1].mean()  # 前11根K线的平均成交量（排除最后一根）
                        current_volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        prev_volume_ratio = prev_volume / avg_volume if avg_volume != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                    else:
                        # 默认情况使用平均成交量
                        avg_volume = volumes.iloc[-11:-1].mean()
                        current_volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        prev_volume_ratio = prev_volume / avg_volume if avg_volume != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                else:
                    volume_expansion = True  # 数据不足时默认满足成交量条件
            else:
                # strict模式：要求当前K线放量（默认模式）
                if len(volumes) >= 10:
                    # 对于小周期（1m~5m），使用EMA成交量
                    if bar in ['1m', '3m', '5m']:
                        vol_ema = volumes.ewm(span=10).mean()
                        volume_ratio = current_volume / vol_ema.iloc[-1] if vol_ema.iloc[-1] != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                    # 对于大周期（15m+），使用平均成交量
                    elif bar in ['15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D']:
                        avg_volume = volumes.iloc[-10:-1].mean()  # 前10根K线的平均成交量（排除最后一根）
                        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                    else:
                        # 默认情况使用平均成交量
                        avg_volume = volumes.iloc[-10:-1].mean()
                        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                else:
                    volume_expansion = True  # 数据不足时默认满足成交量条件
            
            # 判断EMA5上穿EMA20 (EMA5从下向上穿过EMA20)
            ema_bullish_crossover = prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long
            
            # 判断EMA5下穿EMA20 (EMA5从上向下穿过EMA20)
            ema_bearish_crossover = prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long
            
            # 确认机制：必须有效突破confirmation_pct%以上才算真穿，减少震荡区假信号
            confirmation_factor = confirmation_pct / 100.0
            confirmed_bullish = ema_bullish_crossover and current_close > current_ema_long * (1 + confirmation_factor)
            confirmed_bearish = ema_bearish_crossover and current_close < current_ema_long * (1 - confirmation_factor)
            
            # 判断收盘价与EMA5的关系
            price_above_ema5 = current_close > current_ema_short
            price_below_ema5 = current_close < current_ema_short
            
            # 生成信号
            # 买入信号：成交量放大，且EMA20斜率大于0，EMA5确认上穿EMA20且K线收盘价大于EMA5
            if volume_expansion and ema20_slope > 0 and confirmed_bullish and price_above_ema5:
                return 1
                
            # 卖出信号：EMA20斜率小于0，EMA5确认下穿EMA20且K线收盘价小于EMA5
            elif ema20_slope < 0 and confirmed_bearish and price_below_ema5:
                return -1
                
            # 持有信号
            else:
                return 0
                
        except Exception as e:
            print(f"Error calculating EMA crossover signal for {symbol}: {e}")
            return 0

    def get_strategy_details(self, symbol: str, bar: str = '1m', 
                           short_ma: int = 5, long_ma: int = 20,
                           vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                           mode: str = 'strict') -> dict:
        """
        获取策略详细信息，用于调试和分析
        
        Args:
            symbol: 交易对符号
            bar: K线时间间隔
            short_ma: 短期EMA周期
            long_ma: 长期EMA周期
            vol_multiplier: 成交量放大倍数 (default: 1.2)
            confirmation_pct: 确认突破百分比 (default: 0.2)
            mode: 模式 ('strict' or 'loose', default: 'strict')
                  strict mode: 要求当前K线放量（趋势稳健）
                  loose mode: 允许前一根放量（更激进，适合山寨）
            
        Returns:
            dict: 包含策略计算详情的字典
        """
        try:
            # 获取足够的K线数据来计算EMA和成交量
            limit = long_ma + 5
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return {}
                
            # 获取价格和成交量数据
            closes = df['c'] if 'c' in df.columns else df['close']
            volumes = df['vol'] if 'vol' in df.columns else df['volume']
            
            # 计算EMA
            ema_short = ema(closes, short_ma)
            ema_long = ema(closes, long_ma)
            
            # 获取最新值和前一个值
            current_close = closes.iloc[-1]
            prev_close = closes.iloc[-2]
            
            current_ema_short = ema_short.iloc[-1] if len(ema_short) > 0 else 0
            prev_ema_short = ema_short.iloc[-2] if len(ema_short) > 1 else 0
            
            current_ema_long = ema_long.iloc[-1] if len(ema_long) > 0 else 0
            prev_ema_long = ema_long.iloc[-2] if len(ema_long) > 1 else 0
            
            # 计算EMA20斜率（使用快速斜率和慢速斜率的组合作为爆发感应器）
            if len(ema_long) >= 5:
                # 快速反应（最近两根K）
                ema20_slope_fast = (ema_long.iloc[-1] - ema_long.iloc[-2]) / ema_long.iloc[-2] if ema_long.iloc[-2] != 0 else 0
                # 稳定趋势（最近五根K）
                ema20_slope_slow = (ema_long.iloc[-1] - ema_long.iloc[-5]) / (5 * ema_long.iloc[-5]) if ema_long.iloc[-5] != 0 else 0
                # 组合判断逻辑，稍偏重趋势平滑的权重
                ema20_slope = (ema20_slope_fast + ema20_slope_slow * 2) / 3
            else:
                ema20_slope = (current_ema_long - prev_ema_long) / prev_ema_long if prev_ema_long != 0 else 0
            
            # 计算成交量条件（根据周期选择不同的成交量判断方法）
            current_volume = volumes.iloc[-1] if len(volumes) > 0 else 0
            volume_ratio = 0  # 初始化volume_ratio
            
            # 根据模式选择成交量判断方式
            if mode == 'loose' and len(volumes) >= 2:
                # loose模式：允许前一根K线放量
                prev_volume = volumes.iloc[-2]
                if len(volumes) >= 11:  # 需要足够的数据来计算前一根的EMA或平均成交量
                    # 对于小周期（1m~5m），使用EMA成交量
                    if bar in ['1m', '3m', '5m']:
                        vol_ema = volumes.ewm(span=10).mean()
                        current_volume_ratio = current_volume / vol_ema.iloc[-1] if vol_ema.iloc[-1] != 0 else 0
                        prev_volume_ratio = prev_volume / vol_ema.iloc[-2] if vol_ema.iloc[-2] != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                    # 对于大周期（15m+），使用平均成交量
                    elif bar in ['15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D']:
                        avg_volume = volumes.iloc[-11:-1].mean()  # 前11根K线的平均成交量（排除最后一根）
                        current_volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        prev_volume_ratio = prev_volume / avg_volume if avg_volume != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                    else:
                        # 默认情况使用平均成交量
                        avg_volume = volumes.iloc[-11:-1].mean()
                        current_volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        prev_volume_ratio = prev_volume / avg_volume if avg_volume != 0 else 0
                        volume_ratio = max(current_volume_ratio, prev_volume_ratio)
                        volume_expansion = volume_ratio > vol_multiplier
                else:
                    volume_expansion = True  # 数据不足时默认满足成交量条件
            else:
                # strict模式：要求当前K线放量（默认模式）
                if len(volumes) >= 10:
                    # 对于小周期（1m~5m），使用EMA成交量
                    if bar in ['1m', '3m', '5m']:
                        vol_ema = volumes.ewm(span=10).mean()
                        volume_ratio = current_volume / vol_ema.iloc[-1] if vol_ema.iloc[-1] != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                    # 对于大周期（15m+），使用平均成交量
                    elif bar in ['15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D']:
                        avg_volume = volumes.iloc[-10:-1].mean()  # 前10根K线的平均成交量（排除最后一根）
                        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                    else:
                        # 默认情况使用平均成交量
                        avg_volume = volumes.iloc[-10:-1].mean()
                        volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
                        volume_expansion = volume_ratio > vol_multiplier
                else:
                    volume_expansion = True  # 数据不足时默认满足成交量条件
            
            # 判断交叉
            ema_bullish_crossover = prev_ema_short <= prev_ema_long and current_ema_short > current_ema_long
            ema_bearish_crossover = prev_ema_short >= prev_ema_long and current_ema_short < current_ema_long
            
            # 确认机制：必须有效突破confirmation_pct%以上才算真穿，减少震荡区假信号
            confirmation_factor = confirmation_pct / 100.0
            confirmed_bullish = ema_bullish_crossover and current_close > current_ema_long * (1 + confirmation_factor)
            confirmed_bearish = ema_bearish_crossover and current_close < current_ema_long * (1 - confirmation_factor)
            
            # 判断价格与EMA关系
            price_above_ema5 = current_close > current_ema_short
            price_below_ema5 = current_close < current_ema_short
            
            # 计算信号
            signal = self.calculate_ema_crossover_signal(symbol, bar, short_ma, long_ma, vol_multiplier, confirmation_pct, mode)
            
            # 计算平均成交量用于返回（如果未定义则设为0）
            avg_volume = volumes.iloc[-10:-1].mean() if len(volumes) >= 10 else 0
            
            return {
                'symbol': symbol,
                'current_price': float(current_close),
                'ema5': float(current_ema_short),
                'ema20': float(current_ema_long),
                'ema20_slope': ema20_slope,
                'ema20_slope_fast': ema20_slope_fast if 'ema20_slope_fast' in locals() else 0,
                'ema20_slope_slow': ema20_slope_slow if 'ema20_slope_slow' in locals() else 0,
                'volume_expansion': volume_expansion,
                'volume_ratio': float(volume_ratio),
                'confirmation_pct': confirmation_pct,
                'confirmed_bullish': confirmed_bullish,
                'confirmed_bearish': confirmed_bearish,
                'signal_strength': float(abs(ema20_slope) * volume_ratio),  # 动量信号强度
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'ema_bullish_crossover': ema_bullish_crossover,
                'ema_bearish_crossover': ema_bearish_crossover,
                'price_above_ema5': price_above_ema5,
                'price_below_ema5': price_below_ema5,
                'signal': signal
            }
            
        except Exception as e:
            print(f"Error getting strategy details for {symbol}: {e}")
            return {}


def main():
    """测试函数"""
    print("EMA Crossover Strategy Test")
    print("=" * 50)
    
    # 初始化客户端
    client = OKXClient()
    strategy = EMACrossoverStrategy(client)
    
    # 测试几个常见的交易对
    test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
    
    for symbol in test_symbols:
        try:
            signal = strategy.calculate_ema_crossover_signal(symbol, bar='1m', short_ma=5, long_ma=20, vol_multiplier=1.2, confirmation_pct=0.2, mode='strict')
            details = strategy.get_strategy_details(symbol, bar='1m', short_ma=5, long_ma=20, vol_multiplier=1.2, confirmation_pct=0.2, mode='strict')
            
            if details:
                print(f"\n{symbol}:")
                print(f"  Price: ${details['current_price']:.4f}")
                print(f"  EMA5: {details['ema5']:.4f}")
                print(f"  EMA20: {details['ema20']:.4f}")
                print(f"  EMA20 Slope: {details['ema20_slope']*100:.2f}%")
                print(f"  Volume Expansion: {details['volume_expansion']}")
                print(f"  Bullish Crossover: {details['ema_bullish_crossover']}")
                print(f"  Bearish Crossover: {details['ema_bearish_crossover']}")
                print(f"  Price Above EMA5: {details['price_above_ema5']}")
                print(f"  Signal: {signal} ({'BUY' if signal == 1 else 'SELL' if signal == -1 else 'HOLD'})")
        except Exception as e:
            print(f"Error testing {symbol}: {e}")


if __name__ == "__main__":
    main()