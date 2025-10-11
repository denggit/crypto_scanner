#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : monitor.py
@Description: Strategy 1 specific monitor implementation
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_monitor import BaseMonitor
from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_1.strategy_1 import EMACrossoverStrategy
from utils.logger import logger


class Strategy1Monitor(BaseMonitor):
    """EMA Crossover Strategy Monitor"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0):
        """
        Initialize Strategy 1 Monitor
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            short_ma: Short EMA period
            long_ma: Long EMA period
            vol_multiplier: Volume multiplier
            confirmation_pct: Confirmation percentage
            mode: Mode ('strict' or 'loose')
            trailing_stop_pct: Trailing stop percentage
        """
        super().__init__(symbol, bar, data_dir="monitor_data", file_prefix="strategy_monitor")
        
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.vol_multiplier = vol_multiplier
        self.confirmation_pct = confirmation_pct
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        
        self.client = OKXClient()
        self.strategy = EMACrossoverStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
    
    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'ema5', 'ema20', 'ema20_slope',
            'volume_expansion', 'volume_ratio', 'signal', 'action',
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
                'ema5': details.get('ema5', 0),
                'ema20': details.get('ema20', 0),
                'ema20_slope': details.get('ema20_slope', 0),
                'volume_expansion': details.get('volume_expansion', False),
                'volume_ratio': details.get('volume_ratio', 0),
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
        logger.info(f"开始模拟监控 {self.symbol} 的EMA交叉策略...")
        logger.info(f"策略参数: EMA{self.short_ma}/EMA{self.long_ma}, "
              f"成交量倍数={self.vol_multiplier}, 确认百分比={self.confirmation_pct}%, "
              f"模式={self.mode}, 移动止损={self.trailing_stop_pct}%")
        
        try:
            while True:
                self._wait_for_next_bar()
                
                try:
                    signal = self.strategy.calculate_ema_crossover_signal(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.vol_multiplier, self.confirmation_pct, self.mode
                    )
                    
                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.vol_multiplier, self.confirmation_pct, self.mode
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
