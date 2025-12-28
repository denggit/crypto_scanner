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
from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import get_okx_client, OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from tools.technical_indicators import atr, bollinger_bands, sma
from utils.logger import logger


class CompressionEvent:
    """压缩事件类，用于管理压缩状态"""

    def __init__(self, symbol: str, start_time: datetime, atr_ratio: float,
                 bb_width: float, bb_upper: float, bb_lower: float,
                 bb_middle: float, compression_low: float = None,
                 compression_high: float = None, ttl_bars: int = 30,
                 timeframe: str = '1m', compression_score: float = 0.0,
                 metrics: dict = None, breakout_levels: dict = None,
                 invalidation_levels: dict = None):
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
            compression_low: 压缩区间最低价（用于结构验证）
            compression_high: 压缩区间最高价（用于结构验证）
            ttl_bars: TTL（存活时间，以K线数量为单位）
            timeframe: 时间框架（默认1m）
            compression_score: 压缩评分（0-100）
            metrics: 压缩指标字典
            breakout_levels: 突破水平字典 {'up': 做多突破价, 'down': 做空突破价}
            invalidation_levels: 失效水平字典 {'up': 做多失效价, 'down': 做空失效价}
        """
        self.symbol = symbol
        self.start_time = start_time
        self.atr_ratio = atr_ratio
        self.bb_width = bb_width
        self.bb_upper = bb_upper
        self.bb_lower = bb_lower
        self.bb_middle = bb_middle
        self.compression_low = compression_low if compression_low is not None else bb_lower
        self.compression_high = compression_high if compression_high is not None else bb_upper
        self.ttl_bars = ttl_bars
        self.bar_count = 0  # 已存活的K线数量

        # v2新增属性
        self.timeframe = timeframe
        self.compression_score = compression_score
        self.initial_compression_score = compression_score  # v2.1新增：记录初始评分（用于评分锁定）
        self.score_locked = False  # v2.1新增：评分是否锁定
        self.metrics = metrics or {}
        self.breakout_levels = breakout_levels or {}
        self.invalidation_levels = invalidation_levels or {}
        # v2.1新增：临界保护线（用于快速判断是否在临界区）
        self.pre_breakout_upper = compression_high * 0.995 if compression_high else None  # 上轨 - 0.5%
        self.pre_breakout_lower = compression_low * 1.005 if compression_low else None  # 下轨 + 0.5%

    def is_expired(self, current_bar_count: int) -> bool:
        """
        检查压缩事件是否过期
        
        Args:
            current_bar_count: 当前K线数量（从开始时间算起）
            
        Returns:
            bool: True表示已过期
        """
        return current_bar_count >= self.ttl_bars

    def is_invalid(self, current_atr_ratio: float = None, threshold: float = 0.7,
                   min_compression_score: float = 60.0, current_price: float = None) -> bool:
        """
        检查压缩事件是否失效（ATR比率回升或压缩评分过低）

        Args:
            current_atr_ratio: 当前ATR比率（可选）
            threshold: ATR比率失效阈值（默认0.7）
            min_compression_score: 最小压缩评分阈值（默认60.0）
            current_price: 当前价格（可选，用于v2.1评分锁定机制）

        Returns:
            bool: True表示已失效
        """
        # v2.1新增：评分锁定机制
        # 如果价格未破坏结构（在压缩区间内），且初始评分≥70，则评分锁定
        if current_price is not None and self.initial_compression_score >= 70.0:
            # 检查价格是否在压缩区间内（未破坏结构）
            if self.compression_low <= current_price <= self.compression_high:
                # 价格在压缩区间内，评分锁定（不因评分下降而失效）
                if not self.score_locked:
                    self.score_locked = True
                    logger.debug(f"{self.symbol} 评分锁定：价格在压缩区间内，初始评分={self.initial_compression_score:.2f}")
                # 评分锁定后，只检查ATR比率，不检查评分
                if current_atr_ratio is not None:
                    return current_atr_ratio > threshold
                return False
            else:
                # 价格已破坏结构，评分锁定失效
                if self.score_locked:
                    self.score_locked = False
                    logger.debug(f"{self.symbol} 评分锁定失效：价格已破坏结构")

        # v2新增：压缩评分过低则失效（但如果在临界区会被豁免，见cleanup_compression_pool）
        if self.compression_score < min_compression_score:
            return True

        # 原有逻辑：ATR比率回升
        if current_atr_ratio is not None:
            return current_atr_ratio > threshold

        return False

    def is_in_pre_breakout_zone(self, current_price: float) -> bool:
        """
        检查价格是否在临界突破区（v2.1新增）

        Args:
            current_price: 当前价格

        Returns:
            bool: True表示在临界突破区
        """
        if self.pre_breakout_upper is None or self.pre_breakout_lower is None:
            return False
        
        # 根据白皮书：上临界线 = 上轨 * 0.995，下临界线 = 下轨 * 1.005
        # 价格在 [上临界线, 上轨] 或 [下轨, 下临界线] 范围内
        return (self.pre_breakout_upper <= current_price <= self.compression_high) or \
               (self.compression_low <= current_price <= self.pre_breakout_lower)


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

    def _calculate_compression_score(self, df, atr_short, atr_mid, bb_width,
                                     bb_width_history, closes, volumes, highs, lows,
                                     score_weight_atr: float = 0.3,
                                     score_weight_duration: float = 0.25,
                                     score_weight_volume: float = 0.2,
                                     score_weight_range: float = 0.15,
                                     score_weight_ma: float = 0.1) -> dict:
        """
        计算压缩评分（v2新增）

        Args:
            df: DataFrame，包含价格和成交量数据
            atr_short: 短期ATR序列
            atr_mid: 中期ATR序列
            bb_width: 当前布林带宽度
            bb_width_history: 历史布林带宽度序列
            closes: 收盘价序列
            volumes: 成交量序列
            highs: 最高价序列
            lows: 最低价序列

        Returns:
            dict: 包含压缩评分和各项指标得分的字典
        """
        try:
            # 1. ATR相对压缩得分 (30%)
            if len(atr_mid) > 0 and atr_mid.iloc[-1] != 0:
                atr_ratio = atr_short.iloc[-1] / atr_mid.iloc[-1]
                # ATR比率越低得分越高，使用反向线性映射：0.3->100分，0.7->0分
                atr_score = max(0, min(100, 250 - 250 * atr_ratio))  # 0.3=100, 0.7=0
            else:
                atr_score = 0
                atr_ratio = 1.0

            # 2. 压缩持续时间得分 (25%)
            # 计算最近20根K线中ATR比率低于0.6的比例
            lookback = min(20, len(atr_short), len(atr_mid))
            if lookback > 0:
                atr_ratios = []
                for i in range(1, lookback + 1):
                    if atr_mid.iloc[-i] != 0:
                        atr_ratios.append(atr_short.iloc[-i] / atr_mid.iloc[-i])
                if atr_ratios:
                    low_atr_ratio_count = sum(1 for ratio in atr_ratios if ratio < 0.6)
                    duration_score = (low_atr_ratio_count / lookback) * 100
                else:
                    duration_score = 0
            else:
                duration_score = 0

            # 3. 成交量健康度得分 (20%)
            # 检查成交量是否稳定，而不是萎缩
            if len(volumes) >= 20:
                recent_volumes = volumes.iloc[-10:]  # 最近10根K线
                avg_recent_volume = recent_volumes.mean()
                avg_volume_20 = volumes.iloc[-20:].mean()
                if avg_volume_20 > 0:
                    volume_ratio = avg_recent_volume / avg_volume_20
                    # 0.8-1.2之间得分最高，过高或过低都扣分
                    if 0.8 <= volume_ratio <= 1.2:
                        volume_score = 100
                    elif 0.6 <= volume_ratio < 0.8 or 1.2 < volume_ratio <= 1.5:
                        volume_score = 60
                    else:
                        volume_score = 20
                else:
                    volume_score = 0
            else:
                volume_score = 0

            # 4. 价格区间收敛得分 (15%)
            # 计算最近10根K线的价格范围与20根K线的价格范围之比
            if len(highs) >= 20 and len(lows) >= 20:
                recent_high = highs.iloc[-10:].max()
                recent_low = lows.iloc[-10:].min()
                recent_range = recent_high - recent_low

                prev_high = highs.iloc[-20:-10].max()
                prev_low = lows.iloc[-20:-10].min()
                prev_range = prev_high - prev_low

                if prev_range > 0:
                    range_ratio = recent_range / prev_range
                    # 范围越小得分越高：0.5=100分，1.0=0分
                    range_score = max(0, min(100, 200 - 200 * range_ratio))
                else:
                    range_score = 50
            else:
                range_score = 0

            # 5. 均线聚合度得分 (10%)
            # 检查短期均线（5, 10, 20）是否聚合
            if len(closes) >= 20:
                from tools.technical_indicators import sma
                ma5 = sma(closes, 5)
                ma10 = sma(closes, 10)
                ma20 = sma(closes, 20)

                if len(ma5) > 0 and len(ma10) > 0 and len(ma20) > 0:
                    ma5_val = ma5.iloc[-1]
                    ma10_val = ma10.iloc[-1]
                    ma20_val = ma20.iloc[-1]

                    if ma20_val > 0:
                        # 计算均线之间的最大差距百分比
                        max_diff = max(abs(ma5_val - ma10_val), abs(ma5_val - ma20_val), abs(ma10_val - ma20_val))
                        diff_pct = max_diff / ma20_val

                        # 差距越小得分越高：0.5%=100分，2%=0分
                        ma_score = max(0, min(100, 133 - 66.7 * diff_pct * 100))  # 转换为百分比
                    else:
                        ma_score = 50
                else:
                    ma_score = 0
            else:
                ma_score = 0

            # 计算加权总分（使用配置的权重）
            total_score = (
                    atr_score * score_weight_atr +
                    duration_score * score_weight_duration +
                    volume_score * score_weight_volume +
                    range_score * score_weight_range +
                    ma_score * score_weight_ma
            )

            return {
                'total_score': round(total_score, 2),
                'atr_score': round(atr_score, 2),
                'duration_score': round(duration_score, 2),
                'volume_score': round(volume_score, 2),
                'range_score': round(range_score, 2),
                'ma_score': round(ma_score, 2),
                'atr_ratio': atr_ratio
            }

        except Exception as e:
            logger.error(f"计算压缩评分时出错: {e}")
            return {
                'total_score': 0,
                'atr_score': 0,
                'duration_score': 0,
                'volume_score': 0,
                'range_score': 0,
                'ma_score': 0,
                'atr_ratio': 1.0
            }

    def _validate_15m_compression(self, df_15m: pd.DataFrame,
                                  price_deviation_threshold: float = 2.0,
                                  atr_relative_threshold: float = 1.5,
                                  amplitude_ratio_threshold: float = 0.4) -> Tuple[bool, dict]:
        """
        15分钟周期压缩验证（根据白皮书附录）

        Args:
            df_15m: 15分钟K线数据

        Returns:
            Tuple[bool, dict]: (是否通过验证, 验证详情)
        """
        try:
            from tools.technical_indicators import atr, sma

            # 获取价格数据
            closes_15m = df_15m['c'] if 'c' in df_15m.columns else df_15m['close']
            highs_15m = df_15m['h'] if 'h' in df_15m.columns else df_15m['high']
            lows_15m = df_15m['l'] if 'l' in df_15m.columns else df_15m['low']

            current_close = closes_15m.iloc[-1]

            # 1. 趋势方向确认
            # 计算20周期均线（5小时均线）和50周期均线（12.5小时均线）
            ma20_15m = sma(closes_15m, 20)
            ma50_15m = sma(closes_15m, 50)

            if len(ma20_15m) < 1:
                return False, {"error": "MA20计算失败"}

            current_ma20 = ma20_15m.iloc[-1]

            # 价格偏离度 = abs(收盘价 - MA20) / MA20 * 100
            price_deviation = abs(current_close - current_ma20) / current_ma20 * 100 if current_ma20 != 0 else 100

            # 趋势过滤条件：价格偏离度 < threshold
            trend_too_strong = price_deviation > price_deviation_threshold

            # 2. 波动率背景确认
            # 计算ATR(14) - 最近3.5小时ATR
            atr_14_15m = atr(df_15m, 14)

            if len(atr_14_15m) < 1:
                return False, {"error": "ATR计算失败"}

            current_atr_14 = atr_14_15m.iloc[-1]
            atr_relative_level = current_atr_14 / current_close * 100 if current_close != 0 else 100

            # 波动率过高：ATR相对水平 > threshold
            volatility_too_high = atr_relative_level > atr_relative_threshold

            # 3. 价格结构确认
            # 最近5根15分钟K线（75分钟）的最大振幅
            lookback_5 = min(5, len(highs_15m))
            recent_high_5 = float(highs_15m.iloc[-lookback_5:].max())
            recent_low_5 = float(lows_15m.iloc[-lookback_5:].min())
            amplitude_75min = (recent_high_5 - recent_low_5) / recent_low_5 * 100 if recent_low_5 != 0 else 100

            # 最近20根15分钟K线（5小时）的最大振幅
            lookback_20 = min(20, len(highs_15m))
            recent_high_20 = float(highs_15m.iloc[-lookback_20:].max())
            recent_low_20 = float(lows_15m.iloc[-lookback_20:].min())
            amplitude_5h = (recent_high_20 - recent_low_20) / recent_low_20 * 100 if recent_low_20 != 0 else 100

            # 振幅比率 = 75分钟振幅 / 5小时振幅
            amplitude_ratio = amplitude_75min / amplitude_5h if amplitude_5h != 0 else 1.0

            # 结构支持压缩：振幅比率 < threshold
            structure_supports_compression = amplitude_ratio < amplitude_ratio_threshold

            # 验证条件（全部需要满足）
            validation_passed = (not trend_too_strong) and (not volatility_too_high) and structure_supports_compression

            # 计算支持度评分（0-100）
            support_score = 0

            # 趋势支持度（40分）
            if price_deviation < 1.0:
                support_score += 40
            elif price_deviation < 1.5:
                support_score += 30
            elif price_deviation < 2.0:
                support_score += 20

            # 波动率支持度（30分）
            if atr_relative_level < 1.0:
                support_score += 30
            elif atr_relative_level < 1.3:
                support_score += 20
            elif atr_relative_level < 1.5:
                support_score += 10

            # 结构支持度（30分）
            if amplitude_ratio < 0.3:
                support_score += 30
            elif amplitude_ratio < 0.4:
                support_score += 20
            elif amplitude_ratio < 0.5:
                support_score += 10

            details = {
                'validation_passed': validation_passed,
                'support_score': support_score,
                'price_deviation': round(price_deviation, 2),
                'atr_relative_level': round(atr_relative_level, 2),
                'amplitude_ratio': round(amplitude_ratio, 2),
                'trend_too_strong': trend_too_strong,
                'volatility_too_high': volatility_too_high,
                'structure_supports_compression': structure_supports_compression,
                'amplitude_75min': round(amplitude_75min, 2),
                'amplitude_5h': round(amplitude_5h, 2)
            }

            return validation_passed, details

        except Exception as e:
            logger.error(f"15分钟周期验证出错: {e}")
            return False, {"error": str(e)}

    def detect_compression(self, symbol: str,
                           atr_short_period: int = 10, atr_mid_period: int = 60,
                           atr_ratio_threshold: float = 0.5,
                           bb_period: int = 20, bb_std: int = 2,
                           bb_width_ratio: float = 0.7,
                           ttl_bars: int = 30,
                           compression_score_threshold: float = 70.0,
                           validation_price_deviation_threshold: float = 2.0,
                           validation_atr_relative_threshold: float = 1.5,
                           validation_amplitude_ratio_threshold: float = 0.4,
                           breakout_threshold: float = 0.002,
                           breakout_invalidation_threshold: float = 0.03,
                           score_weight_atr: float = 0.3,
                           score_weight_duration: float = 0.25,
                           score_weight_volume: float = 0.2,
                           score_weight_range: float = 0.15,
                           score_weight_ma: float = 0.1) -> Optional[CompressionEvent]:
        """
        检测波动率压缩（v2多时间框架版本）

        压缩判定条件：
        1. Level 1 (5分钟周期): ATR相对压缩 + 布林带宽度收缩 + 压缩评分>=threshold
        2. Level 2 (15分钟周期): 趋势强度验证 + 波动率背景确认 + 价格结构确认

        Args:
            symbol: 交易对符号
            atr_short_period: 短期ATR周期（默认10）
            atr_mid_period: 中期ATR周期（默认60）
            atr_ratio_threshold: ATR比率阈值（默认0.5）
            bb_period: 布林带周期（默认20）
            bb_std: 布林带标准差倍数（默认2）
            bb_width_ratio: 布林带宽度收缩比率（默认0.7）
            ttl_bars: 压缩事件TTL（K线数量，默认30）
            compression_score_threshold: 压缩评分阈值（默认70.0，只有评分>=此值才创建压缩事件）
            validation_price_deviation_threshold: 15分钟验证价格偏离度阈值（默认2.0%）
            validation_atr_relative_threshold: 15分钟验证ATR相对水平阈值（默认1.5%）
            validation_amplitude_ratio_threshold: 15分钟验证振幅比率阈值（默认0.4）
            breakout_threshold: 突破幅度阈值（默认0.002，即0.2%）
            breakout_invalidation_threshold: 失效水平阈值（默认0.03，即3%）
            score_weight_atr: 压缩评分ATR权重（默认0.3）
            score_weight_duration: 压缩评分持续时间权重（默认0.25）
            score_weight_volume: 压缩评分成交量权重（默认0.2）
            score_weight_range: 压缩评分价格区间权重（默认0.15）
            score_weight_ma: 压缩评分均线权重（默认0.1）

        Returns:
            CompressionEvent: 如果检测到压缩，返回压缩事件；否则返回None
        """
        try:
            # ==================== Level 1: 5分钟周期压缩检测 ====================
            # 获取5分钟K线数据
            limit_5m = max(atr_mid_period, bb_period) + 20
            df_5m = self.market_data_retriever.get_kline(symbol, '5m', limit_5m)

            if df_5m is None or len(df_5m) < limit_5m:
                return None

            # 获取价格和成交量数据
            closes_5m = df_5m['c'] if 'c' in df_5m.columns else df_5m['close']
            highs_5m = df_5m['h'] if 'h' in df_5m.columns else df_5m['high']
            lows_5m = df_5m['l'] if 'l' in df_5m.columns else df_5m['low']
            volumes_5m = df_5m['vol'] if 'vol' in df_5m.columns else df_5m['volume']

            # 计算ATR
            atr_short_5m = atr(df_5m, atr_short_period)
            atr_mid_5m = atr(df_5m, atr_mid_period)

            if len(atr_short_5m) < 1 or len(atr_mid_5m) < 1:
                return None

            # 获取最新值
            current_atr_short_5m = atr_short_5m.iloc[-1]
            current_atr_mid_5m = atr_mid_5m.iloc[-1]

            if pd.isna(current_atr_short_5m) or pd.isna(current_atr_mid_5m) or current_atr_mid_5m == 0:
                return None

            # 计算ATR比率
            atr_ratio_5m = current_atr_short_5m / current_atr_mid_5m

            # 条件1：ATR相对收缩
            if atr_ratio_5m >= atr_ratio_threshold:
                return None

            # 计算布林带
            bb_upper_5m, bb_middle_5m, bb_lower_5m = bollinger_bands(closes_5m, bb_period, bb_std)

            if len(bb_upper_5m) < 1 or len(bb_middle_5m) < 1 or len(bb_lower_5m) < 1:
                return None

            # 获取最新值
            current_bb_upper_5m = bb_upper_5m.iloc[-1]
            current_bb_middle_5m = bb_middle_5m.iloc[-1]
            current_bb_lower_5m = bb_lower_5m.iloc[-1]

            if pd.isna(current_bb_upper_5m) or pd.isna(current_bb_middle_5m) or pd.isna(current_bb_lower_5m):
                return None

            # 计算布林带宽度
            bb_width_5m = (current_bb_upper_5m - current_bb_lower_5m) / current_bb_middle_5m if current_bb_middle_5m != 0 else 0

            # 计算60根K线的平均布林带宽度（用于相对比较）
            if len(bb_upper_5m) >= 60:
                bb_width_60_mean_5m = ((bb_upper_5m.iloc[-60:] - bb_lower_5m.iloc[-60:]) / bb_middle_5m.iloc[-60:]).mean()
            else:
                bb_width_60_mean_5m = bb_width_5m

            # 条件2：布林带宽度显著收缩
            if bb_width_5m >= bb_width_ratio * bb_width_60_mean_5m:
                return None

            # v2新增：计算压缩评分
            # 获取布林带宽度历史序列
            if len(bb_upper_5m) >= 60:
                bb_width_history_5m = ((bb_upper_5m.iloc[-60:] - bb_lower_5m.iloc[-60:]) / bb_middle_5m.iloc[-60:])
            else:
                bb_width_history_5m = pd.Series([bb_width_5m])

            # 计算压缩评分
            score_result = self._calculate_compression_score(
                df=df_5m,
                atr_short=atr_short_5m,
                atr_mid=atr_mid_5m,
                bb_width=bb_width_5m,
                bb_width_history=bb_width_history_5m,
                closes=closes_5m,
                volumes=volumes_5m,
                highs=highs_5m,
                lows=lows_5m,
                score_weight_atr=score_weight_atr,
                score_weight_duration=score_weight_duration,
                score_weight_volume=score_weight_volume,
                score_weight_range=score_weight_range,
                score_weight_ma=score_weight_ma
            )

            compression_score = score_result['total_score']
            atr_ratio_5m = score_result['atr_ratio']  # 使用评分计算中的atr_ratio（更准确）

            # v2新增：只有压缩评分>=threshold才认为是高质量压缩
            if compression_score < compression_score_threshold:
                logger.debug(f"{symbol} 压缩评分不足: {compression_score:.2f} < {compression_score_threshold}，忽略")
                return None

            # ==================== Level 2: 15分钟周期验证 ====================
            # 获取15分钟K线数据进行验证
            limit_15m = 100  # 至少100根15分钟K线（约25小时）
            df_15m = self.market_data_retriever.get_kline(symbol, '15m', limit_15m)

            if df_15m is None or len(df_15m) < limit_15m:
                logger.debug(f"{symbol} 15分钟数据不足，跳过验证")
                return None

            # 15分钟验证逻辑（根据白皮书附录）
            validation_passed, validation_details = self._validate_15m_compression(
                df_15m,
                price_deviation_threshold=validation_price_deviation_threshold,
                atr_relative_threshold=validation_atr_relative_threshold,
                amplitude_ratio_threshold=validation_amplitude_ratio_threshold
            )
            if not validation_passed:
                logger.debug(f"{symbol} 15分钟周期验证未通过: {validation_details}")
                return None

            logger.info(f"{symbol} 15分钟周期验证通过: {validation_details}")

            # 计算压缩区间的高低点（使用最近20根5分钟K线的最高价和最低价）
            compression_period = min(20, len(highs_5m))
            compression_high = float(highs_5m.iloc[-compression_period:].max())
            compression_low = float(lows_5m.iloc[-compression_period:].min())

            # v2.1新增：计算突破水平和失效水平（使用配置参数）
            # 突破水平：压缩区间边界 ± breakout_threshold（v2.1从1%降低到0.2%）
            breakout_up = compression_high * (1 + breakout_threshold)
            breakout_down = compression_low * (1 - breakout_threshold)
            # 失效水平：压缩区间边界 ± breakout_invalidation_threshold
            invalidation_up = compression_high * (1 + breakout_invalidation_threshold)
            invalidation_down = compression_low * (1 - breakout_invalidation_threshold)
            # v2.1新增：临界保护线（根据白皮书：上临界线=上轨*0.995，下临界线=下轨*1.005）
            pre_breakout_upper = compression_high * 0.995  # 上轨 - 0.5%
            pre_breakout_lower = compression_low * 1.005  # 下轨 + 0.5%

            # 如果该币种已有压缩事件，检查是否需要更新
            if symbol in self.compression_pool:
                existing_event = self.compression_pool[symbol]
                # v2修改：如果新压缩的评分更高，则更新
                if compression_score > existing_event.compression_score:
                    logger.info(
                        f"更新 {symbol} 的压缩事件: 评分 {existing_event.compression_score:.2f} -> {compression_score:.2f}")
                    # v2.1：更新时也更新临界保护线
                    event = CompressionEvent(
                        symbol=symbol,
                        start_time=datetime.now(),
                        atr_ratio=atr_ratio_5m,
                        bb_width=bb_width_5m,
                        bb_upper=current_bb_upper_5m,
                        bb_lower=current_bb_lower_5m,
                        bb_middle=current_bb_middle_5m,
                        compression_low=compression_low,
                        compression_high=compression_high,
                        ttl_bars=ttl_bars,
                        timeframe='5m',
                        compression_score=compression_score,
                        metrics={
                            'atr_score': score_result['atr_score'],
                            'duration_score': score_result['duration_score'],
                            'volume_score': score_result['volume_score'],
                            'range_score': score_result['range_score'],
                            'ma_score': score_result['ma_score']
                        },
                        breakout_levels={'up': breakout_up, 'down': breakout_down},
                        invalidation_levels={'up': invalidation_up, 'down': invalidation_down}
                    )
                    # v2.1：手动设置临界保护线
                    event.pre_breakout_upper = pre_breakout_upper
                    event.pre_breakout_lower = pre_breakout_lower
                    self.compression_pool[symbol] = event
                    return self.compression_pool[symbol]
                else:
                    # 保持现有事件
                    return None

            # 创建新的压缩事件
            compression_event = CompressionEvent(
                symbol=symbol,
                start_time=datetime.now(),
                atr_ratio=atr_ratio_5m,
                bb_width=bb_width_5m,
                bb_upper=current_bb_upper_5m,
                bb_lower=current_bb_lower_5m,
                bb_middle=current_bb_middle_5m,
                compression_low=compression_low,
                compression_high=compression_high,
                ttl_bars=ttl_bars,
                timeframe='5m',
                compression_score=compression_score,
                metrics={
                    'atr_score': score_result['atr_score'],
                    'duration_score': score_result['duration_score'],
                    'volume_score': score_result['volume_score'],
                    'range_score': score_result['range_score'],
                    'ma_score': score_result['ma_score']
                },
                breakout_levels={'up': breakout_up, 'down': breakout_down},
                invalidation_levels={'up': invalidation_up, 'down': invalidation_down}
            )
            # v2.1：手动设置临界保护线
            compression_event.pre_breakout_upper = pre_breakout_upper
            compression_event.pre_breakout_lower = pre_breakout_lower

            self.compression_pool[symbol] = compression_event
            logger.info(
                f"检测到 {symbol} 的压缩事件: 评分={compression_score:.2f}, ATR比率={atr_ratio_5m:.4f}, BB宽度={bb_width_5m:.4f}")

            return compression_event

        except Exception as e:
            logger.error(f"检测压缩时出错 {symbol}: {e}")
            return None

    def _evaluate_breakout_quality(self, df, current_close, current_volume,
                                   atr_14, highs, lows, closes, volumes,
                                   body_atr_multiplier: float = 0.4,
                                   shadow_ratio: float = 0.5,
                                   volume_min_multiplier: float = 1.2,  # v2.1从1.5降低到1.2
                                   new_high_low_lookback: int = 10) -> dict:
        """
        评估突破质量（v2.1新增）

        根据v2.1白皮书，突破K线必须满足以下条件中的至少3条：
        1. 实体 ≥ 0.4 × ATR(14)
        2. 影线短（假突破过滤，影线<50%实体，v2.1从30%放宽到50%）
        3. 成交量显著高于近期低点（v2.1从1.5倍降低到1.2倍）
        4. 创局部新高/新低（动量）

        Args:
            df: 包含当前K线的DataFrame
            current_close: 当前收盘价
            current_volume: 当前成交量
            atr_14: ATR(14)序列
            highs: 最高价序列
            lows: 最低价序列
            closes: 收盘价序列
            volumes: 成交量序列

        Returns:
            dict: 包含各项条件评估结果和质量得分
        """
        try:
            # 获取当前K线的开盘价
            current_open = df['o'].iloc[-1] if 'o' in df.columns else closes.iloc[-2]

            # 1. 实体大小条件
            candle_body = abs(current_close - current_open)
            atr_14_value = atr_14.iloc[-1] if len(atr_14) > 0 else 0
            condition1 = candle_body >= body_atr_multiplier * atr_14_value if atr_14_value > 0 else False

            # 2. 影线长度条件
            if current_close > current_open:  # 阳线
                upper_shadow = highs.iloc[-1] - current_close
                lower_shadow = current_open - lows.iloc[-1]
            else:  # 阴线
                upper_shadow = highs.iloc[-1] - current_open
                lower_shadow = current_close - lows.iloc[-1]

            max_shadow = max(upper_shadow, lower_shadow)
            condition2 = max_shadow < shadow_ratio * candle_body if candle_body > 0 else True

            # 3. 成交量条件
            # 检查成交量是否显著高于最近N根K线的最低成交量
            lookback = min(new_high_low_lookback, len(volumes))
            if lookback > 0:
                min_recent_volume = volumes.iloc[-lookback:].min()
                condition3 = current_volume > volume_min_multiplier * min_recent_volume if min_recent_volume > 0 else False
            else:
                condition3 = False

            # 4. 创新高/新低条件
            lookback_high_low = min(new_high_low_lookback, len(highs), len(lows))
            if lookback_high_low > 0:
                # 检查是否创10根K线新高或新低
                is_new_high = current_close == highs.iloc[-lookback_high_low:].max()
                is_new_low = current_close == lows.iloc[-lookback_high_low:].min()
                condition4 = is_new_high or is_new_low
            else:
                condition4 = False

            # 计算满足的条件数量
            conditions_met = sum([condition1, condition2, condition3, condition4])
            quality_pass = conditions_met >= 3

            return {
                'condition1': condition1,
                'condition2': condition2,
                'condition3': condition3,
                'condition4': condition4,
                'conditions_met': conditions_met,
                'quality_pass': quality_pass,
                'candle_body': candle_body,
                'atr_14_value': atr_14_value,
                'max_shadow': max_shadow
            }

        except Exception as e:
            logger.error(f"评估突破质量时出错: {e}")
            return {
                'condition1': False,
                'condition2': False,
                'condition3': False,
                'condition4': False,
                'conditions_met': 0,
                'quality_pass': False,
                'candle_body': 0,
                'atr_14_value': 0,
                'max_shadow': 0
            }

    def detect_breakout(self, symbol: str,
                        volume_period: int = 20, volume_multiplier: float = 1.2,
                        breakout_threshold: float = 0.002,  # v2.1从0.01(1%)降低到0.002(0.2%)
                        breakout_body_atr_multiplier: float = 0.4,
                        breakout_shadow_ratio: float = 0.5,
                        breakout_volume_min_multiplier: float = 1.2,
                        breakout_new_high_low_lookback: int = 10) -> Tuple[int, Dict]:
        """
        检测突破信号（v2.1多时间框架版本）

        突破条件（使用1分钟K线数据）：
        1. 价格突破压缩区间边界（上轨+breakout_threshold做多，下轨-breakout_threshold做空，v2.1从1%降低到0.2%）
        2. 成交量放大（Volume ≥ volume_multiplier × MA(Volume, period)，v2.1从1.5倍降低到1.2倍）
        3. 突破质量评分≥3/4个条件

        Args:
            symbol: 交易对符号
            volume_period: 成交量均线周期（默认20）
            volume_multiplier: 成交量放大倍数（默认1.2，v2.1从1.5降低）
            breakout_threshold: 突破幅度阈值（默认0.002即0.2%，v2.1从0.01即1%降低）
            breakout_body_atr_multiplier: 突破实体ATR倍数（默认0.4，实体≥此值×ATR(14)）
            breakout_shadow_ratio: 突破影线比率（默认0.5即50%，v2.1从0.3即30%放宽）
            breakout_volume_min_multiplier: 突破成交量最小倍数（默认1.2，v2.1从1.5降低）
            breakout_new_high_low_lookback: 创新高/新低回看周期（默认10根K线）

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

            # 获取K线数据（需要更多数据用于突破质量评估）
            limit = max(volume_period + 5, 30)  # 至少30根用于ATR(14)计算
            df = self.market_data_retriever.get_kline(symbol, '1m', limit)

            if df is None or len(df) < limit:
                return 0, {}

            # 获取价格和成交量数据
            closes = df['c'] if 'c' in df.columns else df['close']
            volumes = df['vol'] if 'vol' in df.columns else df['volume']
            highs = df['h'] if 'h' in df.columns else df['high']
            lows = df['l'] if 'l' in df.columns else df['low']

            # 计算ATR(14)用于突破质量评估
            from tools.technical_indicators import atr
            atr_14 = atr(df, 14)

            if len(closes) < 1 or len(volumes) < volume_period:
                return 0, {}

            current_close = closes.iloc[-1]
            current_volume = volumes.iloc[-1]

            # 计算成交量均线
            volume_ma = sma(volumes, volume_period)

            if len(volume_ma) < 1 or pd.isna(volume_ma.iloc[-1]):
                return 0, {}

            avg_volume = volume_ma.iloc[-1]

            # v2.1修改：成交量需要≥volume_multiplier×20均量（v2.1从1.5倍降低到1.2倍，允许温和放量启动）
            volume_expansion = current_volume >= volume_multiplier * avg_volume

            # v2.1修改：使用压缩事件的突破水平而不是布林带边界
            # 如果压缩事件中没有突破水平，使用配置的breakout_threshold计算（v2.1从1%降低到0.2%）
            if compression_event.breakout_levels:
                breakout_up = compression_event.breakout_levels.get('up')
                breakout_down = compression_event.breakout_levels.get('down')
                # 如果breakout_levels存在但值为None，使用配置的threshold计算
                if breakout_up is None:
                    breakout_up = compression_event.compression_high * (1 + breakout_threshold)
                if breakout_down is None:
                    breakout_down = compression_event.compression_low * (1 - breakout_threshold)
            else:
                # 向后兼容：如果没有配置，使用配置的breakout_threshold（v2.1从1%降低到0.2%）
                breakout_up = compression_event.compression_high * (1 + breakout_threshold)
                breakout_down = compression_event.compression_low * (1 - breakout_threshold)

            # 检测突破
            signal = 0
            details = {
                'current_price': float(current_close),
                'breakout_up': float(breakout_up),
                'breakout_down': float(breakout_down),
                'compression_high': float(compression_event.compression_high),
                'compression_low': float(compression_event.compression_low),
                'current_volume': float(current_volume),
                'avg_volume': float(avg_volume),
                'volume_expansion': volume_expansion,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
                'atr_ratio': compression_event.atr_ratio,
                'compression_score': compression_event.compression_score,
                'compression_event': compression_event  # 添加压缩事件，用于结构验证和止损计算
            }

            # 评估突破质量
            quality_result = self._evaluate_breakout_quality(
                df=df,
                current_close=current_close,
                current_volume=current_volume,
                atr_14=atr_14,
                highs=highs,
                lows=lows,
                closes=closes,
                volumes=volumes,
                body_atr_multiplier=breakout_body_atr_multiplier,
                shadow_ratio=breakout_shadow_ratio,
                volume_min_multiplier=breakout_volume_min_multiplier,
                new_high_low_lookback=breakout_new_high_low_lookback
            )
            details['breakout_quality'] = quality_result

            # 第一层过滤：价格突破且成交量达标
            long_breakout_price = current_close > breakout_up and volume_expansion
            short_breakout_price = current_close < breakout_down and volume_expansion

            # 第二层过滤：突破质量
            quality_pass = quality_result['quality_pass']

            # 做多突破
            if long_breakout_price and quality_pass:
                signal = 1
                details['breakout_type'] = 'long'
                details['breakout_price'] = float(current_close)
                logger.info(f"{symbol} 检测到高质量做多突破: 价格={current_close:.4f}, 突破水平={breakout_up:.4f}, "
                            f"成交量比率={current_volume / avg_volume:.2f}, 质量得分={quality_result['conditions_met']}/4, 压缩评分={compression_event.compression_score:.1f}")

            # 做空突破
            elif short_breakout_price and quality_pass:
                signal = -1
                details['breakout_type'] = 'short'
                details['breakout_price'] = float(current_close)
                logger.info(f"{symbol} 检测到高质量做空突破: 价格={current_close:.4f}, 突破水平={breakout_down:.4f}, "
                            f"成交量比率={current_volume / avg_volume:.2f}, 质量得分={quality_result['conditions_met']}/4, 压缩评分={compression_event.compression_score:.1f}")

            # 如果检测到突破，从压缩池中移除该事件（但保留在details中供后续使用）
            if signal != 0:
                del self.compression_pool[symbol]
                logger.info(f"{symbol} 突破后已从压缩池移除")
            elif long_breakout_price or short_breakout_price:
                # 有价格突破但质量不过关，记录日志
                logger.debug(f"{symbol} 有价格突破但质量不过关: 价格={current_close:.4f}, "
                             f"质量得分={quality_result['conditions_met']}/4, 需要≥3")

            return signal, details

        except Exception as e:
            logger.error(f"检测突破时出错 {symbol}: {e}")
            return 0, {}

    def cleanup_compression_pool(self, symbol: str = None,
                                 atr_short_period: int = 10, atr_mid_period: int = 60,
                                 compression_score_min: float = 60.0,
                                 atr_ratio_invalidation_threshold: float = 0.7,
                                 pre_breakout_protection_zone: float = 0.005):
        """
        清理压缩池中的过期或失效事件（v2多时间框架版本）

        Args:
            symbol: 指定清理的币种（None表示清理所有币种）
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            compression_score_min: 压缩评分最低阈值（低于此值将被移除，除非在临界突破区）
            atr_ratio_invalidation_threshold: ATR比率失效阈值（高于此值认为压缩失效）
            pre_breakout_protection_zone: 临界突破保护区（价格距离突破边界在此范围内时，豁免失效判定）

        注意：压缩事件基于5分钟时间框架，因此清理时使用5分钟K线数据
        """
        try:
            symbols_to_check = [symbol] if symbol else list(self.compression_pool.keys())

            for sym in symbols_to_check:
                if sym not in self.compression_pool:
                    continue

                event = self.compression_pool[sym]

                # 检查是否过期（需要获取当前K线数量）
                # 简化处理：使用时间差估算（压缩事件基于5分钟时间框架）
                time_diff = datetime.now() - event.start_time
                estimated_bars = int(time_diff.total_seconds() / 300)  # 5分钟K线

                if event.is_expired(estimated_bars):
                    logger.info(f"{sym} 压缩事件已过期（存活{estimated_bars}根K线），从池中移除")
                    del self.compression_pool[sym]
                    continue

                # v2.1新增：检查是否在临界突破区（防止黎明前失效）
                # 根据白皮书：上临界线 = 上轨 * 0.995，下临界线 = 下轨 * 1.005
                is_pre_breakout = False
                current_price = None
                try:
                    # 获取当前价格判断是否在临界区
                    ticker = self.market_data_retriever.get_ticker_by_symbol(sym)
                    if ticker and ticker.last:
                        current_price = float(ticker.last)
                        # v2.1：使用临界保护线直接判断（更准确）
                        is_pre_breakout = event.is_in_pre_breakout_zone(current_price)
                except:
                    pass

                # 检查压缩评分是否过低（<threshold）
                # 但如果在临界突破区，则豁免（防止黎明前失效）
                if event.compression_score < compression_score_min and not is_pre_breakout:
                    logger.info(f"{sym} 压缩事件评分过低（{event.compression_score:.2f} < {compression_score_min}），从池中移除")
                    del self.compression_pool[sym]
                    continue
                elif event.compression_score < compression_score_min and is_pre_breakout:
                    logger.info(f"{sym} 压缩事件评分过低但处于临界突破区，触发保护机制，暂不移除")

                # 检查是否失效（ATR比率回升）
                try:
                    limit = max(atr_mid_period, 20) + 5
                    df = self.market_data_retriever.get_kline(sym, '5m', limit)

                    if df is not None and len(df) >= limit:
                        atr_short = atr(df, atr_short_period)
                        atr_mid = atr(df, atr_mid_period)

                        if len(atr_short) > 0 and len(atr_mid) > 0:
                            current_atr_short = atr_short.iloc[-1]
                            current_atr_mid = atr_mid.iloc[-1]

                            if not pd.isna(current_atr_short) and not pd.isna(current_atr_mid) and current_atr_mid != 0:
                                current_atr_ratio = current_atr_short / current_atr_mid

                                # v2.1：使用更新后的is_invalid方法，传入当前价格以支持评分锁定机制
                                # 但如果在临界突破区，则豁免（防止黎明前失效）
                                if event.is_invalid(current_atr_ratio, threshold=atr_ratio_invalidation_threshold,
                                                   current_price=current_price):
                                    if is_pre_breakout:
                                        logger.info(f"{sym} 压缩事件ATR比率回升但处于临界突破区，触发保护机制，暂不移除")
                                    else:
                                        logger.info(
                                            f"{sym} 压缩事件已失效（ATR比率回升至{current_atr_ratio:.4f}），从池中移除")
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
            compression = strategy.detect_compression(symbol)
            if compression:
                logger.info(f"\n{symbol} 检测到压缩:")
                logger.info(f"  ATR比率: {compression.atr_ratio:.4f}")
                logger.info(f"  布林带宽度: {compression.bb_width:.4f}")
                logger.info(f"  压缩评分: {compression.compression_score:.1f}")

            # 检测突破
            signal, details = strategy.detect_breakout(symbol)
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
