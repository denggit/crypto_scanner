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
from utils.logger import logger


class StrategyMonitor(Strategy1Monitor):
    """Strategy Monitor with Unified Logic for Mock and Real Trading"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0,
                 trade: bool = False, trade_amount: float = 10.0,
                 trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Strategy Monitor with unified mock/real trading logic
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            short_ma: Short EMA period
            long_ma: Long EMA period
            vol_multiplier: Volume multiplier
            confirmation_pct: Confirmation percentage
            mode: Mode ('strict' or 'loose')
            trailing_stop_pct: Trailing stop percentage
            trade: Whether to execute real trades (True) or mock trades (False)
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        self.trade_amount = trade_amount
        self.trade_mode_setting = trade_mode
        self.leverage = leverage
        
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.vol_multiplier = vol_multiplier
        self.confirmation_pct = confirmation_pct
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        
        from sdk.base_monitor import BaseMonitor
        BaseMonitor.__init__(self, symbol, bar, trade_mode=trade)
        
        self.client = OKXClient()
        self.strategy = EMACrossoverStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        
        self.trader = Strategy1Trader(self.client, trade_amount, trade_mode, leverage)
    
    def get_csv_headers(self) -> list:
        """Get CSV headers for recording data"""
        return [
            'timestamp', 'symbol', 'price', 'ema5', 'ema20', 'ema20_slope',
            'volume_expansion', 'volume_ratio', 'signal', 'action',
            'position', 'entry_price', 'exit_price', 'return_rate'
        ]
    
    def get_trader(self):
        """Get trader instance"""
        return self.trader
    
    def _check_trailing_stop(self, price: float) -> bool:
        """Check trailing stop condition for mock position"""
        if self.mock_position == 1:
            # 持多仓：更新最高价，检查是否跌破止损价
            if price > self.mock_highest_price:
                self.mock_highest_price = price
            stop_price = self.mock_highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif self.mock_position == -1:
            # 持空仓：更新最低价，检查是否涨破止损价
            if price < self.mock_lowest_price:
                self.mock_lowest_price = price
            stop_price = self.mock_lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False
    
    def execute_trade(self, signal: int, price: float, details: dict):
        """Unified trading logic for both mock and real trading"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        
        trailing_stop_triggered = self._check_trailing_stop(price)
        
        if self.mock_position == 0:
            if signal == 1:
                self.mock_position = 1
                self.mock_entry_price = price
                self.mock_highest_price = price  # 持多仓时记录最高价
                self.mock_lowest_price = price   # 持多仓时也记录最低价用于参考
                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                self.mock_position = -1
                self.mock_entry_price = price
                self.mock_lowest_price = price   # 持空仓时记录最低价
                self.mock_highest_price = price  # 持空仓时也记录最高价用于参考
                action = "SHORT_OPEN"
                self.trade_count += 1
        else:
            if self.mock_position == 1:
                if trailing_stop_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                    action = "LONG_CLOSE_TRAILING_STOP"
                    self.mock_position = 0
                    self.mock_highest_price = 0.0
                    self.trade_count += 1
                elif signal == -1:
                    exit_price = price
                    return_rate = (exit_price - self.mock_entry_price) / self.mock_entry_price
                    action = "LONG_CLOSE_SHORT_OPEN"
                    self.mock_position = -1
                    self.mock_entry_price = price
                    self.mock_highest_price = price  # 重新初始化
                    self.mock_lowest_price = price   # 重新初始化
                    self.trade_count += 1
            elif self.mock_position == -1:
                if trailing_stop_triggered:
                    exit_price = price
                    return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                    action = "SHORT_CLOSE_TRAILING_STOP"
                    self.mock_position = 0
                    self.mock_lowest_price = 0.0
                    self.trade_count += 1
                elif signal == 1:
                    exit_price = price
                    return_rate = (self.mock_entry_price - exit_price) / self.mock_entry_price
                    action = "SHORT_CLOSE_LONG_OPEN"
                    self.mock_position = 1
                    self.mock_entry_price = price
                    self.mock_lowest_price = price   # 重新初始化
                    self.mock_highest_price = price  # 重新初始化
                    self.trade_count += 1
        
        if action != "HOLD":
            # 记录模拟交易数据（始终记录）
            mock_record = {
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
                'position': self.mock_position,
                'entry_price': self.mock_entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate
            }
            self._enqueue_mock_write(mock_record)
            
            # 如果是真实交易模式，执行真实交易并记录真实交易数据
            if self.trade_mode:
                # 执行真实交易
                trade_result = self.trader.execute_trade(action, self.symbol, price)
                
                # 记录真实交易数据
                if trade_result:
                    real_record = {
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'symbol': self.symbol,
                        'action': action,
                        'order_id': trade_result.ordId if hasattr(trade_result, 'ordId') else '',
                        'order_price': trade_result.px if hasattr(trade_result, 'px') else price,
                        'order_size': trade_result.sz if hasattr(trade_result, 'sz') else self.trade_amount,
                        'order_type': trade_result.ordType if hasattr(trade_result, 'ordType') else 'market',
                        'order_side': trade_result.side if hasattr(trade_result, 'side') else '',
                        'order_state': trade_result.state if hasattr(trade_result, 'state') else '',
                        'filled_size': trade_result.accFillSz if hasattr(trade_result, 'accFillSz') else 0,
                        'avg_fill_price': trade_result.avgPx if hasattr(trade_result, 'avgPx') else price,
                        'fee': trade_result.fee if hasattr(trade_result, 'fee') else 0,
                        'trade_timestamp': trade_result.ts if hasattr(trade_result, 'ts') else int(datetime.now().timestamp() * 1000)
                    }
                    self._enqueue_real_write(real_record)
            
            mode_prefix = self.get_mode_prefix()
            logger.info(f"{mode_prefix} [{mock_record['timestamp']}] {self.symbol} {action}: "
                  f"价格={price:.4f}, 信号={signal}, 收益率={return_rate*100:.2f}%")
    
    def run(self):
        """Run monitoring loop with unified mock/real trading logic"""
        mode_text = "真实交易" if self.trade_mode else "模拟监控"
        logger.info(f"开始{mode_text} {self.symbol} 的EMA交叉策略...")
        logger.info(f"策略参数: EMA{self.short_ma}/EMA{self.long_ma}, "
              f"成交量倍数={self.vol_multiplier}, 确认百分比={self.confirmation_pct}%, "
              f"模式={self.mode}, 移动止损={self.trailing_stop_pct}%")
        
        if self.trade_mode:
            trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
            logger.info(f"交易模式: {trade_mode_names.get(self.trade_mode_setting, '未知')}, "
                  f"每次交易金额: {self.trade_amount} USDT")
            if self.trade_mode_setting in [2, 3]:
                logger.info(f"杠杆倍数: {self.leverage}x")
        
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
                              f"模拟仓位: {self.mock_position}, 交易次数: {self.trade_count}")
                    else:
                        logger.warning(f"无法获取 {self.symbol} 的价格数据")
                
                except Exception as e:
                    logger.error(f"计算信号时出错: {e}")
                    import time
                    time.sleep(5)
                
        except KeyboardInterrupt:
            logger.info("监控已停止")
            logger.info(f"总交易次数: {self.trade_count}")
            logger.info(f"模拟记录文件: {self.mock_csv_file}")
            logger.info(f"模拟备份文件: {self.mock_backup_file}")
            if self.trade_mode:
                logger.info(f"真实记录文件: {self.real_csv_file}")
                logger.info(f"真实备份文件: {self.real_backup_file}")
        except Exception as e:
            logger.error(f"监控过程中出错: {e}")


def main():
    """主函数"""
    logger.info("EMA交叉策略实盘模拟监控系统")
    logger.info("=" * 50)
    
    symbol = input("请输入交易对 (默认 BTC-USDT): ").strip() or "BTC-USDT"
    bar = input("请输入K线周期 (默认 1m): ").strip() or "1m"
    mode = input("请输入模式 (strict/loose, 默认 loose): ").strip() or "loose"
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
        
        logger.info("交易模式选择:")
        logger.info("1. 现货模式")
        logger.info("2. 全仓杠杆模式")
        logger.info("3. 逐仓杠杆模式 (默认)")
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
