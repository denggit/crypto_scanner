#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : monitor.py
@Description: Strategy 6 specific monitor implementation
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_monitor import BaseMonitor
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_6_vcb_v02_v02.strategy_6 import VCBStrategy
from utils.logger import logger


class Strategy6Monitor(BaseMonitor):
    """VCB Strategy Monitor"""

    def __init__(self, symbol: str, bar: str = '1m',
                 atr_short_period: int = 10, atr_mid_period: int = 60,
                 atr_ratio_threshold: float = 0.5,
                 bb_period: int = 20, bb_std: int = 2,
                 bb_width_ratio: float = 0.7,
                 volume_period: int = 20, volume_multiplier: float = 1.0,
                 ttl_bars: int = 30,
                 trailing_stop_pct: float = 1.0,
                 stop_loss_atr_multiplier: float = 0.8,
                 take_profit_r: float = 2.0, **params):
        """
        初始化 Strategy 6 Monitor
        
        Args:
            symbol: 交易对符号
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            atr_ratio_threshold: ATR比率阈值
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            bb_width_ratio: 布林带宽度收缩比率
            volume_period: 成交量均线周期
            volume_multiplier: 成交量放大倍数
            ttl_bars: 压缩事件TTL（K线数量）
            trailing_stop_pct: 移动止损百分比
            stop_loss_atr_multiplier: 止损ATR倍数
            take_profit_r: 止盈R倍数
            **params: 其他参数
        """
        super().__init__(symbol, bar, data_dir="monitor_data", file_prefix="strategy_monitor")

        self.atr_short_period = atr_short_period
        self.atr_mid_period = atr_mid_period
        self.atr_ratio_threshold = atr_ratio_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_ratio = bb_width_ratio
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.ttl_bars = ttl_bars
        self.trailing_stop_pct = trailing_stop_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_r = take_profit_r

        self.client = OKXClient()
        self.strategy = VCBStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)

        # 止损和止盈价格
        self.stop_loss_price = 0.0
        self.take_profit_price = 0.0

    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'atr_ratio', 'bb_width',
            'bb_upper', 'bb_lower', 'volume_expansion', 'signal', 'action',
            'position', 'entry_price', 'exit_price', 'return_rate',
            'stop_loss', 'take_profit'
        ]

    def _calculate_stop_loss_take_profit(self, entry_price: float, signal: int,
                                         compression_event=None):
        """
        计算止损和止盈价格
        
        Args:
            entry_price: 入场价格
            signal: 信号方向（1=做多, -1=做空）
            compression_event: 压缩事件（可选，用于计算基于压缩区间的止损）
            
        Returns:
            Tuple[float, float]: (止损价格, 止盈价格)
        """
        import pandas as pd

        try:
            # 获取ATR用于计算止损
            limit = self.atr_mid_period + 5
            df = self.market_data_retriever.get_kline(self.symbol, self.bar, limit)

            if df is None or len(df) < limit:
                # 如果无法获取数据，使用固定百分比止损
                if signal == 1:
                    stop_loss = entry_price * (1 - self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 + self.take_profit_r * self.trailing_stop_pct / 100.0)
                else:
                    stop_loss = entry_price * (1 + self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 - self.take_profit_r * self.trailing_stop_pct / 100.0)
                return stop_loss, take_profit

            from tools.technical_indicators import atr
            atr_mid = atr(df, self.atr_mid_period)

            if len(atr_mid) < 1:
                # 使用固定百分比止损
                if signal == 1:
                    stop_loss = entry_price * (1 - self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 + self.take_profit_r * self.trailing_stop_pct / 100.0)
                else:
                    stop_loss = entry_price * (1 + self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 - self.take_profit_r * self.trailing_stop_pct / 100.0)
                return stop_loss, take_profit

            current_atr = atr_mid.iloc[-1]

            if pd.isna(current_atr) or current_atr == 0:
                # 使用固定百分比止损
                if signal == 1:
                    stop_loss = entry_price * (1 - self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 + self.take_profit_r * self.trailing_stop_pct / 100.0)
                else:
                    stop_loss = entry_price * (1 + self.trailing_stop_pct / 100.0)
                    take_profit = entry_price * (1 - self.take_profit_r * self.trailing_stop_pct / 100.0)
                return stop_loss, take_profit

            # 计算基于ATR的止损
            atr_stop_loss = current_atr * self.stop_loss_atr_multiplier

            # 如果有压缩事件，使用压缩区间的另一侧作为止损参考
            if compression_event:
                if signal == 1:
                    # 做多：止损在压缩区间下轨或ATR止损中取较大值
                    compression_stop = compression_event.bb_lower
                    stop_loss = max(compression_stop, entry_price - atr_stop_loss)
                else:
                    # 做空：止损在压缩区间上轨或ATR止损中取较小值
                    compression_stop = compression_event.bb_upper
                    stop_loss = min(compression_stop, entry_price + atr_stop_loss)
            else:
                # 没有压缩事件，使用ATR止损
                if signal == 1:
                    stop_loss = entry_price - atr_stop_loss
                else:
                    stop_loss = entry_price + atr_stop_loss

            # 计算止盈（基于R倍数）
            risk = abs(entry_price - stop_loss)
            if signal == 1:
                take_profit = entry_price + risk * self.take_profit_r
            else:
                take_profit = entry_price - risk * self.take_profit_r

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"计算止损止盈时出错: {e}")
            # 使用固定百分比止损
            if signal == 1:
                stop_loss = entry_price * (1 - self.trailing_stop_pct / 100.0)
                take_profit = entry_price * (1 + self.take_profit_r * self.trailing_stop_pct / 100.0)
            else:
                stop_loss = entry_price * (1 + self.trailing_stop_pct / 100.0)
                take_profit = entry_price * (1 - self.take_profit_r * self.trailing_stop_pct / 100.0)
            return stop_loss, take_profit

    def _check_trailing_stop(self, price: float) -> bool:
        """Check trailing stop condition"""
        if self.position == 1:
            if price > self.highest_price:
                self.highest_price = price
            stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif self.position == -1:
            if price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = price
            stop_price = self.lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False

    def _check_stop_loss_take_profit(self, price: float) -> Tuple[bool, str]:
        """
        检查止损和止盈条件
        
        Returns:
            Tuple[bool, str]: (是否触发, 触发类型)
        """
        if self.position == 0:
            return False, ""

        if self.position == 1:
            # 做多：检查止损和止盈
            if self.stop_loss_price > 0 and price <= self.stop_loss_price:
                return True, "STOP_LOSS"
            if self.take_profit_price > 0 and price >= self.take_profit_price:
                return True, "TAKE_PROFIT"
        elif self.position == -1:
            # 做空：检查止损和止盈
            if self.stop_loss_price > 0 and price >= self.stop_loss_price:
                return True, "STOP_LOSS"
            if self.take_profit_price > 0 and price <= self.take_profit_price:
                return True, "TAKE_PROFIT"

        return False, ""

    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade based on signal"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0

        trailing_stop_triggered = self._check_trailing_stop(price)
        stop_loss_triggered, stop_type = self._check_stop_loss_take_profit(price)

        if self.position == 0:
            if signal == 1:
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.lowest_price = 0.0

                # 计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]

                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, 1, compression_event
                )

                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                self.position = -1
                self.entry_price = price
                self.lowest_price = price
                self.highest_price = 0.0

                # 计算止损和止盈
                compression_event = None
                if self.symbol in self.strategy.compression_pool:
                    compression_event = self.strategy.compression_pool[self.symbol]

                self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                    price, -1, compression_event
                )

                action = "SHORT_OPEN"
                self.trade_count += 1
        else:
            if self.position == 1:
                if stop_loss_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = f"LONG_CLOSE_{stop_type}"
                    self.position = 0
                    self.highest_price = 0.0
                    self.stop_loss_price = 0.0
                    self.take_profit_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.highest_price = 0.0
                    self.stop_loss_price = 0.0
                    self.take_profit_price = 0.0
                    self.trade_count += 1
                elif signal == -1:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_SHORT_OPEN"
                    self.position = -1
                    self.entry_price = price
                    self.highest_price = 0.0
                    self.lowest_price = price

                    # 重新计算止损和止盈
                    compression_event = None
                    if self.symbol in self.strategy.compression_pool:
                        compression_event = self.strategy.compression_pool[self.symbol]

                    self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                        price, -1, compression_event
                    )
                    self.trade_count += 1
            elif self.position == -1:
                if stop_loss_triggered:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = f"SHORT_CLOSE_{stop_type}"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.stop_loss_price = 0.0
                    self.take_profit_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.stop_loss_price = 0.0
                    self.take_profit_price = 0.0
                    self.trade_count += 1
                elif signal == 1:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_LONG_OPEN"
                    self.position = 1
                    self.entry_price = price
                    self.lowest_price = 0.0
                    self.highest_price = price

                    # 重新计算止损和止盈
                    compression_event = None
                    if self.symbol in self.strategy.compression_pool:
                        compression_event = self.strategy.compression_pool[self.symbol]

                    self.stop_loss_price, self.take_profit_price = self._calculate_stop_loss_take_profit(
                        price, 1, compression_event
                    )
                    self.trade_count += 1

        if action != "HOLD":
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'price': price,
                'atr_ratio': details.get('atr_ratio', 0),
                'bb_width': details.get('bb_width', 0),
                'bb_upper': details.get('bb_upper', 0),
                'bb_lower': details.get('bb_lower', 0),
                'volume_expansion': details.get('volume_expansion', False),
                'signal': signal,
                'action': action,
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate,
                'stop_loss': self.stop_loss_price,
                'take_profit': self.take_profit_price
            }

            self._enqueue_write(record)

            logger.info(f"[模拟交易] [{record['timestamp']}] {self.symbol} {action}: "
                        f"价格={price:.4f}, 信号={signal}, 收益率={return_rate * 100:.2f}%")

    def run(self):
        """Run monitoring loop"""
        logger.info(f"开始模拟监控 {self.symbol} 的VCB策略...")
        logger.info(f"策略参数: ATR({self.atr_short_period}/{self.atr_mid_period}), "
                    f"ATR比率阈值={self.atr_ratio_threshold}, "
                    f"BB({self.bb_period}, {self.bb_std}), "
                    f"成交量倍数={self.volume_multiplier}, "
                    f"TTL={self.ttl_bars}根K线, "
                    f"移动止损={self.trailing_stop_pct}%")

        try:
            while True:
                self._wait_for_next_bar()

                try:
                    # 清理压缩池
                    self.strategy.cleanup_compression_pool(
                        symbol=self.symbol,
                        atr_short_period=self.atr_short_period,
                        atr_mid_period=self.atr_mid_period
                    )

                    # 检测压缩
                    compression = self.strategy.detect_compression(
                        symbol=self.symbol,
                        atr_short_period=self.atr_short_period,
                        atr_mid_period=self.atr_mid_period,
                        atr_ratio_threshold=self.atr_ratio_threshold,
                        bb_period=self.bb_period,
                        bb_std=self.bb_std,
                        bb_width_ratio=self.bb_width_ratio,
                        ttl_bars=self.ttl_bars
                    )

                    # 检测突破
                    signal, details = self.strategy.detect_breakout(
                        symbol=self.symbol,
                        volume_period=self.volume_period,
                        volume_multiplier=self.volume_multiplier
                    )

                    price = details.get('current_price', 0)

                    if price > 0:
                        self.execute_trade(signal, price, details)

                        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                                    f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                                    f"仓位: {self.position}, 交易次数: {self.trade_count}, "
                                    f"压缩池大小: {self.strategy.get_compression_pool_size()}")
                    else:
                        logger.warning(f"无法获取 {self.symbol} 的价格数据")

                except Exception as e:
                    logger.error(f"计算信号时出错: {e}")
                    import time
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("监控已停止")
            logger.info(f"总交易次数: {self.trade_count}")
            logger.info(f"记录文件: {self.csv_file}")
            logger.info(f"备份文件: {self.backup_file}")
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")
