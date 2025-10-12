#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : monitor.py
@Description: Strategy 2 specific monitor implementation
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_monitor import BaseMonitor
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_2.strategy_2 import HighFrequencyStrategy
from utils.logger import logger


class Strategy2Monitor(BaseMonitor):
    """High Frequency Strategy Monitor"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 consecutive_bars: int = 2, atr_period: int = 14,
                 atr_threshold: float = 0.8, trailing_stop_pct: float = 0.8, **params):
        """
        Initialize Strategy 2 Monitor
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            consecutive_bars: Number of consecutive bars for breakout
            atr_period: ATR period
            atr_threshold: ATR threshold multiplier
            trailing_stop_pct: Trailing stop percentage
            **params: Additional parameters
                  volume_factor: Volume multiplier (default: 1.2)
                  use_volume: Whether to use volume condition (default: True)
        """
        super().__init__(symbol, bar, data_dir="monitor_data", file_prefix="strategy2_monitor")
        
        self.consecutive_bars = consecutive_bars
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.trailing_stop_pct = trailing_stop_pct
        
        # 设置默认参数并合并用户提供的参数
        self.params = {
            'volume_factor': 1.2,
            'use_volume': True
        }
        self.params.update(params)
        
        self.client = OKXClient()
        self.strategy = HighFrequencyStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
    
    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'typical_price', 'atr', 'atr_mean',
            'atr_condition', 'volume_condition', 'long_breakout', 'short_breakout',
            'signal', 'action', 'position', 'entry_price', 'exit_price', 'return_rate'
        ]
    
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
    
    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade based on signal"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        
        trailing_stop_triggered = self._check_trailing_stop(price)
        
        if self.position == 0:
            if signal == 1:
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.lowest_price = 0.0
                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                self.position = -1
                self.entry_price = price
                self.lowest_price = price
                self.highest_price = 0.0
                action = "SHORT_OPEN"
                self.trade_count += 1
        else:
            if self.position == 1:
                if trailing_stop_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
                elif signal == -1:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_SHORT_OPEN"
                    self.position = -1
                    self.entry_price = price
                    self.highest_price = 0.0
                    self.lowest_price = price
                    self.trade_count += 1
            elif self.position == -1:
                if trailing_stop_triggered:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
                elif signal == 1:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_LONG_OPEN"
                    self.position = 1
                    self.entry_price = price
                    self.lowest_price = 0.0
                    self.highest_price = price
                    self.trade_count += 1
        
        if action != "HOLD":
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'price': price,
                'typical_price': details.get('current_typical', 0),
                'atr': details.get('atr', 0),
                'atr_mean': details.get('atr_mean', 0),
                'atr_condition': details.get('atr_condition_met', False),
                'volume_condition': details.get('volume_condition_met', False),
                'long_breakout': details.get('long_breakout', False),
                'short_breakout': details.get('short_breakout', False),
                'signal': signal,
                'action': action,
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate
            }
            
            self._enqueue_write(record)
            
            logger.info(f"[模拟交易] [{record['timestamp']}] {self.symbol} {action}: "
                  f"价格={price:.4f}, 信号={signal}, 收益率={return_rate*100:.2f}%")
    
    def run(self):
        """Run monitoring loop"""
        logger.info(f"开始模拟监控 {self.symbol} 的高频策略...")
        logger.info(f"策略参数: 连续K线={self.consecutive_bars}, "
              f"ATR周期={self.atr_period}, ATR阈值={self.atr_threshold}, "
              f"成交量倍数={self.params.get('volume_factor', 1.2)}, "
              f"移动止损={self.trailing_stop_pct}%")
        
        try:
            while True:
                self._wait_for_next_bar()
                
                try:
                    signal = self.strategy.calculate_high_frequency_signal(
                        self.symbol, self.bar, self.consecutive_bars, self.atr_period,
                        self.atr_threshold, self.params.get('volume_factor', 1.2),
                        self.params.get('use_volume', True), self.position
                    )
                    
                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.consecutive_bars, self.atr_period,
                        self.atr_threshold, self.params.get('volume_factor', 1.2),
                        self.params.get('use_volume', True), self.position
                    )
                    
                    price = details.get('current_price', 0)
                    
                    if price > 0:
                        self.execute_trade(signal, price, details)
                        
                        logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                              f"仓位: {self.position}, 交易次数: {self.trade_count}")
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
