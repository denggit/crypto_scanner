#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/10/2025
@File       : strategy_1_monitor.py
@Description: EMA交叉策略实盘模拟监控系统
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategies.strategy_1.methods.monitor import Strategy1Monitor
from strategies.strategy_1.methods.trader import Strategy1Trader
from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from strategies.strategy_1.strategy_1 import EMACrossoverStrategy


class StrategyMonitor(Strategy1Monitor):
    """Strategy Monitor with Trading Support"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0,
                 trade: bool = False, trade_amount: float = 10.0,
                 trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Strategy Monitor with optional trading support
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            short_ma: Short EMA period
            long_ma: Long EMA period
            vol_multiplier: Volume multiplier
            confirmation_pct: Confirmation percentage
            mode: Mode ('strict' or 'loose')
            trailing_stop_pct: Trailing stop percentage
            trade: Whether to execute real trades
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        self.trade = trade
        self.trade_amount = trade_amount
        
        if trade:
            data_dir = "trade_data"
            file_prefix = "trade"
        else:
            data_dir = "../../tools/monitor_data"
            file_prefix = "strategy_monitor"
        
        self.symbol = symbol
        self.bar = bar
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.vol_multiplier = vol_multiplier
        self.confirmation_pct = confirmation_pct
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        
        from sdk.base_monitor import BaseMonitor
        BaseMonitor.__init__(self, symbol, bar, data_dir, file_prefix)
        
        self.client = OKXClient()
        self.strategy = EMACrossoverStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        
        if self.trade:
            self.trader = Strategy1Trader(self.client, trade_amount, trade_mode, leverage)
    
    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade with optional real trading support"""
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
            
            mode_text = "[真实交易]" if self.trade else "[模拟交易]"
            print(f"{mode_text} [{record['timestamp']}] {self.symbol} {action}: "
                  f"价格={price:.4f}, 信号={signal}, 收益率={return_rate*100:.2f}%")
            
            if self.trade:
                self.trader.execute_trade(action, self.symbol, price)
    
    def run(self):
        """Run monitoring loop with optional trading support"""
        mode_text = "真实交易" if self.trade else "模拟监控"
        print(f"开始{mode_text} {self.symbol} 的EMA交叉策略...")
        print(f"策略参数: EMA{self.short_ma}/EMA{self.long_ma}, "
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
                        
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                              f"仓位: {self.position}, 交易次数: {self.trade_count}")
                    else:
                        print(f"无法获取 {self.symbol} 的价格数据")
                
                except Exception as e:
                    print(f"计算信号时出错: {e}")
                    import time
                    time.sleep(5)
                
        except KeyboardInterrupt:
            print("\n监控已停止")
            print(f"总交易次数: {self.trade_count}")
            print(f"记录文件: {self.csv_file}")
            print(f"备份文件: {self.backup_file}")
        except Exception as e:
            print(f"监控过程中出错: {e}")


def main():
    """主函数"""
    print("EMA交叉策略实盘模拟监控系统")
    print("=" * 50)
    
    symbol = input("请输入交易对 (默认 BTC-USDT): ").strip() or "BTC-USDT"
    bar = input("请输入K线周期 (默认 1m): ").strip() or "1m"
    mode = input("请输入模式 (strict/loose, 默认 strict): ").strip() or "strict"
    trailing_stop_input = input("请输入移动止损百分比 (默认 1.0%): ").strip()
    trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else 1.0
    trade_input = input("是否真实交易 (y/n, 默认 n): ").strip().lower()
    trade = trade_input == 'y' or trade_input == 'yes'
    
    trade_amount = 10.0
    trade_mode = 3
    leverage = 3
    
    if trade:
        trade_amount_input = input("请输入每次交易的USDT金额 (默认 10.0): ").strip()
        trade_amount = float(trade_amount_input) if trade_amount_input else 10.0
        
        print("\n交易模式选择:")
        print("1. 现货模式")
        print("2. 全仓杠杆模式")
        print("3. 逐仓杠杆模式 (默认)")
        trade_mode_input = input("请选择交易模式 (1/2/3, 默认 3): ").strip()
        trade_mode = int(trade_mode_input) if trade_mode_input in ['1', '2', '3'] else 3
        
        if trade_mode in [2, 3]:
            leverage_input = input("请输入杠杆倍数 (默认 3): ").strip()
            leverage = int(leverage_input) if leverage_input else 3
    
    monitor = StrategyMonitor(
        symbol=symbol,
        bar=bar,
        short_ma=5,
        long_ma=20,
        vol_multiplier=1.2,
        confirmation_pct=0.2,
        mode=mode,
        trailing_stop_pct=trailing_stop_pct,
        trade=trade,
        trade_amount=trade_amount,
        trade_mode=trade_mode,
        leverage=leverage
    )
    
    monitor.run()


if __name__ == "__main__":
    main()
