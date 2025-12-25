#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : strategy_6.py
@Description: Volatility Compression Breakout (VCB) 策略 - 波动率压缩突破策略
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import get_okx_client, OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import atr, bollinger_bands, sma, ema
from utils.logger import logger


class CompressionEvent:
    """压缩事件类，用于管理压缩状态"""
    
    def __init__(self, symbol: str, start_time: datetime, atr_ratio: float, 
                 bb_width: float, bb_upper: float, bb_lower: float, 
                 bb_middle: float, ttl_bars: int = 30):
        """
        初始化压缩事件
        
        Args:
            symbol: 交易对符号
            start_time: 压缩开始时间
            atr_ratio: ATR比率 (ATR_short / ATR_mid)
            bb_width: 布林带宽度
            bb_upper: 布林带上轨
            bb_lower: 布林带下轨
            bb_middle: 布林带中轨
            ttl_bars: TTL（存活时间，以K线数量为单位）
        """
        self.symbol = symbol
        self.start_time = start_time
        self.atr_ratio = atr_ratio
        self.bb_width = bb_width
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_middle = bb_middle
        self.ttl_bars = ttl_bars
        self.bar_count = 0  # 已存活的K线数量
    
    def is_expired(self, current_bar_count: int) -> bool:
        """
        检查压缩事件是否过期
        
        Args:
            current_bar_count: 当前K线数量（从开始时间算起）
            
        Returns:
            bool: True表示已过期
        """
        return current_bar_count >= self.ttl_bars
    
    def is_invalid(self, current_atr_ratio: float, threshold: float = 0.7) -> bool:
        """
        检查压缩事件是否失效（ATR比率回升）
        
        Args:
            current_atr_ratio: 当前ATR比率
            threshold: 失效阈值（默认0.7）
            
        Returns:
            bool: True表示已失效
        """
        return current_atr_ratio > threshold


class VCBStrategy:
    """Volatility Compression Breakout 策略类"""
    
    def __init__(self, client: OKXClient):
        """
        初始化VCB策略
        
        Args:
            client: OKX客户端实例
        """
        self.client = client
        self.market_data_retriever = MarketDataRetriever(client)
        
        # 压缩池：symbol -> CompressionEvent
        self.compression_pool: Dict[str, CompressionEvent] = {}
    
    def detect_compression(self, symbol: str, bar: str = '1m',
                          atr_short_period: int = 10, atr_mid_period: int = 60,
                          atr_ratio_threshold: float = 0.5,
                          bb_period: int = 20, bb_std: int = 2,
                          bb_width_ratio: float = 0.7,
                          ttl_bars: int = 30) -> Optional[CompressionEvent]:
        """
        检测波动率压缩
        
        压缩判定条件：
        1. ATR(short) < threshold × ATR(mid)
        2. 布林带宽度显著收缩
        
        Args:
            symbol: 交易对符号
            bar: K线周期（默认1m）
            atr_short_period: 短期ATR周期（默认10）
            atr_mid_period: 中期ATR周期（默认60）
            atr_ratio_threshold: ATR比率阈值（默认0.5）
            bb_period: 布林带周期（默认20）
            bb_std: 布林带标准差倍数（默认2）
            bb_width_ratio: 布林带宽度收缩比率（默认0.7）
            ttl_bars: 压缩事件TTL（K线数量，默认30）
            
        Returns:
            CompressionEvent: 如果检测到压缩，返回压缩事件；否则返回None
        """
        try:
            # 获取足够的K线数据
            limit = max(atr_mid_period, bb_period) + 20
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return None
            
            # 获取价格数据
            closes = df['c'] if 'c' in df.columns else df['close']
            highs = df['h'] if 'h' in df.columns else df['high']
            lows = df['l'] if 'l' in df.columns else df['low']
            
            # 计算ATR
            atr_short = atr(df, atr_short_period)
            atr_mid = atr(df, atr_mid_period)
            
            if len(atr_short) < 1 or len(atr_mid) < 1:
                return None
            
            # 获取最新值
            current_atr_short = atr_short.iloc[-1]
            current_atr_mid = atr_mid.iloc[-1]
            
            if pd.isna(current_atr_short) or pd.isna(current_atr_mid) or current_atr_mid == 0:
                return None
            
            # 计算ATR比率
            atr_ratio = current_atr_short / current_atr_mid
            
            # 条件1：ATR相对收缩
            if atr_ratio >= atr_ratio_threshold:
                return None
            
            # 计算布林带
            bb_upper, bb_middle, bb_lower = bollinger_bands(closes, bb_period, bb_std)
            
            if len(bb_upper) < 1 or len(bb_middle) < 1 or len(bb_lower) < 1:
                return None
            
            # 获取最新值
            current_bb_upper = bb_upper.iloc[-1]
            current_bb_middle = bb_middle.iloc[-1]
            current_bb_lower = bb_lower.iloc[-1]
            
            if pd.isna(current_bb_upper) or pd.isna(current_bb_middle) or pd.isna(current_bb_lower):
                return None
            
            # 计算布林带宽度
            bb_width = (current_bb_upper - current_bb_lower) / current_bb_middle if current_bb_middle != 0 else 0
            
            # 计算60根K线的平均布林带宽度（用于相对比较）
            if len(bb_upper) >= 60:
                bb_width_60_mean = ((bb_upper.iloc[-60:] - bb_lower.iloc[-60:]) / bb_middle.iloc[-60:]).mean()
            else:
                bb_width_60_mean = bb_width
            
            # 条件2：布林带宽度显著收缩
            if bb_width >= bb_width_ratio * bb_width_60_mean:
                return None
            
            # 如果该币种已有压缩事件，检查是否需要更新
            if symbol in self.compression_pool:
                existing_event = self.compression_pool[symbol]
                # 如果新压缩的ATR比率更低（压缩更明显），则更新
                if atr_ratio < existing_event.atr_ratio:
                    logger.info(f"更新 {symbol} 的压缩事件: ATR比率 {existing_event.atr_ratio:.4f} -> {atr_ratio:.4f}")
                    self.compression_pool[symbol] = CompressionEvent(
                        symbol=symbol,
                        start_time=datetime.now(),
                        atr_ratio=atr_ratio,
                        bb_width=bb_width,
                        bb_upper=current_bb_upper,
                        bb_lower=current_bb_lower,
                        bb_middle=current_bb_middle,
                        ttl_bars=ttl_bars
                    )
                    return self.compression_pool[symbol]
                else:
                    # 保持现有事件
                    return None
            
            # 创建新的压缩事件
            compression_event = CompressionEvent(
                symbol=symbol,
                start_time=datetime.now(),
                atr_ratio=atr_ratio,
                bb_width=bb_width,
                bb_upper=current_bb_upper,
                bb_lower=current_bb_lower,
                bb_middle=current_bb_middle,
                ttl_bars=ttl_bars
            )
            
            self.compression_pool[symbol] = compression_event
            logger.info(f"检测到 {symbol} 的压缩事件: ATR比率={atr_ratio:.4f}, BB宽度={bb_width:.4f}")
            
            return compression_event
            
        except Exception as e:
            logger.error(f"检测压缩时出错 {symbol}: {e}")
            return None
    
    def detect_breakout(self, symbol: str, bar: str = '1m',
                       volume_period: int = 20, volume_multiplier: float = 1.0) -> Tuple[int, Dict]:
        """
        检测突破信号
        
        突破条件：
        1. 价格突破布林带（上轨做多，下轨做空）
        2. 成交量放大（Volume > multiplier × MA(Volume, period)）
        
        Args:
            symbol: 交易对符号
            bar: K线周期（默认1m）
            volume_period: 成交量均线周期（默认20）
            volume_multiplier: 成交量放大倍数（默认1.0，即只要大于均线即可）
            
        Returns:
            Tuple[int, Dict]: (信号值, 详情字典)
                - 信号值: 1=做多, -1=做空, 0=无信号
                - 详情字典: 包含突破相关的详细信息
        """
        try:
            # 检查该币种是否在压缩池中
            if symbol not in self.compression_pool:
                return 0, {}
            
            compression_event = self.compression_pool[symbol]
            
            # 获取K线数据
            limit = volume_period + 5
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return 0, {}
            
            # 获取价格和成交量数据
            closes = df['c'] if 'c' in df.columns else df['close']
            volumes = df['vol'] if 'vol' in df.columns else df['volume']
            
            if len(closes) < 1 or len(volumes) < volume_period:
                return 0, {}
            
            current_close = closes.iloc[-1]
            current_volume = volumes.iloc[-1]
            
            # 计算成交量均线
            volume_ma = sma(volumes, volume_period)
            
            if len(volume_ma) < 1 or pd.isna(volume_ma.iloc[-1]):
                return 0, {}
            
            avg_volume = volume_ma.iloc[-1]
            
            # 检查成交量放大
            volume_expansion = current_volume > volume_multiplier * avg_volume
            
            # 获取压缩事件中的布林带边界
            bb_upper = compression_event.bb_upper
            bb_lower = compression_event.bb_lower
            bb_middle = compression_event.bb_middle
            
            # 检测突破
            signal = 0
            details = {
                'current_price': float(current_close),
                'bb_upper': float(bb_upper),
                'bb_lower': float(bb_lower),
                'bb_middle': float(bb_middle),
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'volume_expansion': volume_expansion,
                'atr_ratio': compression_event.atr_ratio,
                'bb_width': compression_event.bb_width
            }
            
            # 做多突破：价格突破上轨且成交量放大
            if current_close > bb_upper and volume_expansion:
                signal = 1
                details['breakout_type'] = 'long'
                details['breakout_price'] = float(current_close)
                logger.info(f"{symbol} 检测到做多突破: 价格={current_close:.4f}, 上轨={bb_upper:.4f}, "
                          f"成交量比率={current_volume/avg_volume:.2f}")
            
            # 做空突破：价格突破下轨且成交量放大
            elif current_close < bb_lower and volume_expansion:
                signal = -1
                details['breakout_type'] = 'short'
                details['breakout_price'] = float(current_close)
                logger.info(f"{symbol} 检测到做空突破: 价格={current_close:.4f}, 下轨={bb_lower:.4f}, "
                          f"成交量比率={current_volume/avg_volume:.2f}")
            
            # 如果检测到突破，从压缩池中移除该事件
            if signal != 0:
                del self.compression_pool[symbol]
                logger.info(f"{symbol} 突破后已从压缩池移除")
            
            return signal, details
            
        except Exception as e:
            logger.error(f"检测突破时出错 {symbol}: {e}")
            return 0, {}
    
    def cleanup_compression_pool(self, symbol: str = None, bar: str = '1m',
                                 atr_short_period: int = 10, atr_mid_period: int = 60):
        """
        清理压缩池中的过期或失效事件
        
        Args:
            symbol: 指定清理的币种（None表示清理所有币种）
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
        """
        try:
            symbols_to_check = [symbol] if symbol else list(self.compression_pool.keys())
            
            for sym in symbols_to_check:
                if sym not in self.compression_pool:
                    continue
                
                event = self.compression_pool[sym]
                
                # 检查是否过期（需要获取当前K线数量）
                # 简化处理：使用时间差估算
                time_diff = datetime.now() - event.start_time
                if bar == '1m':
                    estimated_bars = int(time_diff.total_seconds() / 60)
                elif bar == '5m':
                    estimated_bars = int(time_diff.total_seconds() / 300)
                else:
                    estimated_bars = int(time_diff.total_seconds() / 60)  # 默认1m
                
                if event.is_expired(estimated_bars):
                    logger.info(f"{sym} 压缩事件已过期（存活{estimated_bars}根K线），从池中移除")
                    del self.compression_pool[sym]
                    continue
                
                # 检查是否失效（ATR比率回升）
                try:
                    limit = max(atr_mid_period, 20) + 5
                    df = self.market_data_retriever.get_kline(sym, bar, limit)
                    
                    if df is not None and len(df) >= limit:
                        atr_short = atr(df, atr_short_period)
                        atr_mid = atr(df, atr_mid_period)
                        
                        if len(atr_short) > 0 and len(atr_mid) > 0:
                            current_atr_short = atr_short.iloc[-1]
                            current_atr_mid = atr_mid.iloc[-1]
                            
                            if not pd.isna(current_atr_short) and not pd.isna(current_atr_mid) and current_atr_mid != 0:
                                current_atr_ratio = current_atr_short / current_atr_mid
                                
                                if event.is_invalid(current_atr_ratio):
                                    logger.info(f"{sym} 压缩事件已失效（ATR比率回升至{current_atr_ratio:.4f}），从池中移除")
                                    del self.compression_pool[sym]
                except Exception as e:
                    logger.warning(f"检查 {sym} 压缩事件失效状态时出错: {e}")
                    
        except Exception as e:
            logger.error(f"清理压缩池时出错: {e}")
    
    def get_strategy_details(self, symbol: str, bar: str = '1m',
                            atr_short_period: int = 10, atr_mid_period: int = 60,
                            bb_period: int = 20, bb_std: int = 2) -> Dict:
        """
        获取策略详细信息，用于调试和分析
        
        Args:
            symbol: 交易对符号
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            
        Returns:
            dict: 包含策略计算详情的字典
        """
        try:
            limit = max(atr_mid_period, bb_period) + 5
            df = self.market_data_retriever.get_kline(symbol, bar, limit)
            
            if df is None or len(df) < limit:
                return {}
            
            closes = df['c'] if 'c' in df.columns else df['close']
            volumes = df['vol'] if 'vol' in df.columns else df['volume']
            
            # 计算ATR
            atr_short = atr(df, atr_short_period)
            atr_mid = atr(df, atr_mid_period)
            
            # 计算布林带
            bb_upper, bb_middle, bb_lower = bollinger_bands(closes, bb_period, bb_std)
            
            current_close = closes.iloc[-1]
            current_atr_short = atr_short.iloc[-1] if len(atr_short) > 0 else 0
            current_atr_mid = atr_mid.iloc[-1] if len(atr_mid) > 0 else 0
            atr_ratio = current_atr_short / current_atr_mid if current_atr_mid != 0 else 0
            
            current_bb_upper = bb_upper.iloc[-1] if len(bb_upper) > 0 else 0
            current_bb_middle = bb_middle.iloc[-1] if len(bb_middle) > 0 else 0
            current_bb_lower = bb_lower.iloc[-1] if len(bb_lower) > 0 else 0
            bb_width = (current_bb_upper - current_bb_lower) / current_bb_middle if current_bb_middle != 0 else 0
            
            # 检查是否在压缩池中
            in_pool = symbol in self.compression_pool
            compression_info = {}
            if in_pool:
                event = self.compression_pool[symbol]
                compression_info = {
                    'compression_start_time': event.start_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'compression_atr_ratio': event.atr_ratio,
                    'compression_bb_width': event.bb_width
                }
            
            return {
                'symbol': symbol,
                'current_price': float(current_close),
                'atr_short': float(current_atr_short),
                'atr_mid': float(current_atr_mid),
                'atr_ratio': float(atr_ratio),
                'bb_upper': float(current_bb_upper),
                'bb_middle': float(current_bb_middle),
                'bb_lower': float(current_bb_lower),
                'bb_width': float(bb_width),
                'in_compression_pool': in_pool,
                **compression_info
            }
            
        except Exception as e:
            logger.error(f"获取策略详情时出错 {symbol}: {e}")
            return {}
    
    def get_compression_pool_size(self) -> int:
        """获取压缩池大小"""
        return len(self.compression_pool)
    
    def get_compression_pool_symbols(self) -> list:
        """获取压缩池中的所有币种"""
        return list(self.compression_pool.keys())


def main():
    """测试函数"""
    logger.info("VCB Strategy Test")
    logger.info("=" * 50)
    
    # 初始化客户端
    client = get_okx_client()
    strategy = VCBStrategy(client)
    
    # 测试几个常见的交易对
    test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
    
    for symbol in test_symbols:
        try:
            # 检测压缩
            compression = strategy.detect_compression(symbol, bar='1m')
            if compression:
                logger.info(f"\n{symbol} 检测到压缩:")
                logger.info(f"  ATR比率: {compression.atr_ratio:.4f}")
                logger.info(f"  布林带宽度: {compression.bb_width:.4f}")
            
            # 检测突破
            signal, details = strategy.detect_breakout(symbol, bar='1m')
            if signal != 0:
                logger.info(f"\n{symbol} 检测到突破信号: {signal}")
                logger.info(f"  突破类型: {details.get('breakout_type', 'N/A')}")
                logger.info(f"  突破价格: {details.get('breakout_price', 0):.4f}")
            
            # 获取详情
            details = strategy.get_strategy_details(symbol, bar='1m')
            if details:
                logger.info(f"\n{symbol} 策略详情:")
                logger.info(f"  价格: ${details['current_price']:.4f}")
                logger.info(f"  ATR比率: {details['atr_ratio']:.4f}")
                logger.info(f"  布林带宽度: {details['bb_width']:.4f}")
                logger.info(f"  在压缩池中: {details['in_compression_pool']}")
                
        except Exception as e:
            logger.error(f"测试 {symbol} 时出错: {e}")


if __name__ == "__main__":
    main()

