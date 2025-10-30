#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : monitor.py
@Description: Strategy 3 specific monitor implementation
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_monitor import BaseMonitor
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_3.strategy_3 import LongShadowStrategy
from utils.logger import logger


class Strategy3Monitor(BaseMonitor):
    """Long Shadow Strategy Monitor"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 min_volume_ccy: float = 1000000, volume_factor: float = 1.2,
                 trailing_stop_pct: float = 0.0, take_profit_pct: float = 0.0, **params):
        """
        Initialize Strategy 3 Monitor
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            min_volume_ccy: Minimum 24h volume in USDT
            volume_factor: Volume multiplier
            trailing_stop_pct: Trailing stop percentage
            take_profit_pct: Take profit percentage
            **params: Additional parameters
                  use_volume: Whether to use volume condition (default: True)
        """
        super().__init__(symbol, bar, data_dir="monitor_data", file_prefix="strategy3_monitor")
        
        self.min_volume_ccy = min_volume_ccy
        self.volume_factor = volume_factor
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        
        # 设置默认参数并合并用户提供的参数
        self.params = {
            'use_volume': True
        }
        self.params.update(params)
        
        self.client = OKXClient()
        self.strategy = LongShadowStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
    
    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'current_open', 'current_high', 'current_low',
            'prev2_open', 'prev2_high', 'prev2_low', 'prev2_close',
            'volume_condition', 'long_shadow_condition', 'short_shadow_condition',
            'long_entry_condition', 'short_entry_condition', 'signal', 'action',
            'position', 'entry_price', 'exit_price', 'return_rate'
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
    
    def _check_take_profit(self, price: float) -> bool:
        """Check take profit condition"""
        if self.position == 1:
            take_profit_price = self.entry_price * (1 + self.take_profit_pct / 100.0)
            if price >= take_profit_price:
                return True
        elif self.position == -1:
            take_profit_price = self.entry_price * (1 - self.take_profit_pct / 100.0)
            if price <= take_profit_price:
                return True
        return False
    
    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade based on signal"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        
        trailing_stop_triggered = self._check_trailing_stop(price)
        take_profit_triggered = self._check_take_profit(price)
        
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
                if take_profit_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_TAKE_PROFIT"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
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
                elif signal == 0:  # 策略平多信号
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_STRATEGY"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
            elif self.position == -1:
                if take_profit_triggered:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_TAKE_PROFIT"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
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
                elif signal == 0:  # 策略平空信号
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_STRATEGY"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
        
        if action != "HOLD":
            record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'price': price,
                'current_open': details.get('current_open', 0),
                'current_high': details.get('current_high', 0),
                'current_low': details.get('current_low', 0),
                'prev2_open': details.get('prev2_open', 0),
                'prev2_high': details.get('prev2_high', 0),
                'prev2_low': details.get('prev2_low', 0),
                'prev2_close': details.get('prev2_close', 0),
                'volume_condition': details.get('volume_condition_met', False),
                'long_shadow_condition': details.get('long_shadow_condition', False),
                'short_shadow_condition': details.get('short_shadow_condition', False),
                'long_entry_condition': details.get('long_entry_condition', False),
                'short_entry_condition': details.get('short_entry_condition', False),
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
        logger.info(f"开始模拟监控 {self.symbol} 的长下影线策略...")
        logger.info(f"策略参数: 最小24小时交易量={self.min_volume_ccy}, "
              f"成交量倍数={self.volume_factor}, "
              f"移动止损={self.trailing_stop_pct}%, "
              f"止盈={self.take_profit_pct}%")
        
        try:
            while True:
                self._wait_for_next_bar()
                
                try:
                    signal = self.strategy.calculate_long_shadow_signal(
                        self.symbol, self.bar, self.min_volume_ccy, self.volume_factor,
                        self.params.get('use_volume', True), self.position
                    )
                    
                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.min_volume_ccy, self.volume_factor,
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