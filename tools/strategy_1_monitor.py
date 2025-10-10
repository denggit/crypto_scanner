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
import time
import pandas as pd
import threading
from datetime import datetime
from typing import Optional, Dict, Any
import copy

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from tools.strategy_1 import EMACrossoverStrategy


class StrategyMonitor:
    """EMA交叉策略实盘模拟监控器"""
    
    def __init__(self, symbol: str, bar: str = '1m', 
                 short_ma: int = 5, long_ma: int = 20,
                 vol_multiplier: float = 1.2, confirmation_pct: float = 0.2,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0):
        """
        初始化监控器
        
        Args:
            symbol: 交易对符号
            bar: K线时间间隔
            short_ma: 短期EMA周期
            long_ma: 长期EMA周期
            vol_multiplier: 成交量放大倍数
            confirmation_pct: 确认突破百分比
            mode: 模式 ('strict' or 'loose')
            trailing_stop_pct: 移动止损百分比 (default: 1.0%)
        """
        self.symbol = symbol
        self.bar = bar
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.vol_multiplier = vol_multiplier
        self.confirmation_pct = confirmation_pct
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        
        # 初始化客户端和策略
        self.client = OKXClient()
        self.strategy = EMACrossoverStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        
        # 交易状态
        self.position = 0  # 0: 无仓位, 1: 多仓, -1: 空仓
        self.entry_price = 0.0  # 入场价格
        self.trade_count = 0  # 交易次数
        self.highest_price = 0.0  # 持仓期间最高价（多仓用）
        self.lowest_price = 0.0  # 持仓期间最低价（空仓用）
        
        # 创建数据目录
        self.data_dir = "monitor_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        # 固定记录文件名（带目录）
        self.csv_file = os.path.join(self.data_dir, f"strategy_monitor_{symbol.replace('-', '_')}.csv")
        self.backup_file = os.path.join(self.data_dir, f"strategy_monitor_{symbol.replace('-', '_')}_backup.csv")
        
        # 初始化CSV文件
        self._init_csv_file()
        
        # 从CSV恢复状态
        self._restore_state()
        
        # 写入队列和线程
        self.write_queue = []
        self.write_lock = threading.Lock()
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()
        
        print(f"策略监控器已初始化: {symbol}")
        print(f"记录文件: {self.csv_file}")
        print(f"备份文件: {self.backup_file}")
    
    def _init_csv_file(self):
        """初始化CSV文件（如果不存在）"""
        headers = [
            'timestamp', 'symbol', 'price', 'ema5', 'ema20', 'ema20_slope', 
            'volume_expansion', 'volume_ratio', 'signal', 'action', 
            'position', 'entry_price', 'exit_price', 'return_rate'
        ]
        
        # 检查主文件是否存在
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file, index=False)
            print(f"创建新的记录文件: {self.csv_file}")
        else:
            print(f"使用现有记录文件: {self.csv_file}")
        
        # 检查备份文件是否存在
        if not os.path.exists(self.backup_file):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.backup_file, index=False)
            print(f"创建新的备份文件: {self.backup_file}")
        else:
            print(f"使用现有备份文件: {self.backup_file}")
    
    def _restore_state(self):
        """从CSV文件恢复最新的持仓状态"""
        try:
            if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
                df = pd.read_csv(self.csv_file)
                if not df.empty:
                    last_record = df.iloc[-1]
                    self.position = int(last_record['position'])
                    self.entry_price = float(last_record['entry_price'])
                    self.trade_count = len(df[df['action'] != 'HOLD'])
                    # 恢复止损价格
                    if self.position == 1:
                        self.highest_price = self.entry_price
                        self.lowest_price = 0.0
                    elif self.position == -1:
                        self.lowest_price = self.entry_price
                        self.highest_price = 0.0
                    else:
                        self.highest_price = 0.0
                        self.lowest_price = 0.0
                    print(f"状态已从文件恢复: position={self.position}, entry_price={self.entry_price:.4f}, trade_count={self.trade_count}")
                else:
                    print("CSV文件为空，使用初始状态")
            else:
                print("未找到历史记录，使用初始状态")
        except Exception as e:
            print(f"恢复状态失败，使用初始状态: {e}")
    
    def _background_writer(self):
        """后台写入线程"""
        while True:
            try:
                # 检查是否有数据需要写入
                if self.write_queue:
                    with self.write_lock:
                        data_to_write = self.write_queue.copy()
                        self.write_queue.clear()
                    
                    # 写入主文件
                    if data_to_write:
                        df = pd.DataFrame(data_to_write)
                        df.to_csv(self.csv_file, mode='a', header=False, index=False)
                        
                        # 写入备份文件
                        df.to_csv(self.backup_file, mode='a', header=False, index=False)
                
                time.sleep(1)  # 每秒检查一次
            except Exception as e:
                print(f"后台写入线程错误: {e}")
                time.sleep(5)
    
    def _enqueue_write(self, data: Dict[str, Any]):
        """将数据加入写入队列"""
        with self.write_lock:
            self.write_queue.append(data)
    
    def _get_next_bar_timestamp(self) -> int:
        """获取下一个K线的时间戳"""
        # 获取当前时间
        now = datetime.now()
        
        if self.bar == '1m':
            # 下一分钟的开始时间
            next_minute = now.replace(second=0, microsecond=0)
            if next_minute <= now:
                next_minute = next_minute.replace(minute=next_minute.minute + 1)
            return int(next_minute.timestamp() * 1000)
        elif self.bar == '5m':
            # 下一个5分钟的开始时间
            next_5min = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
            if next_5min <= now:
                next_5min = next_5min.replace(minute=next_5min.minute + 5)
            return int(next_5min.timestamp() * 1000)
        elif self.bar == '15m':
            # 下一个15分钟的开始时间
            next_15min = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            if next_15min <= now:
                next_15min = next_15min.replace(minute=next_15min.minute + 15)
            return int(next_15min.timestamp() * 1000)
        else:
            # 默认1分钟
            next_minute = now.replace(second=0, microsecond=0)
            if next_minute <= now:
                next_minute = next_minute.replace(minute=next_minute.minute + 1)
            return int(next_minute.timestamp() * 1000)
    
    def _wait_for_next_bar(self):
        """等待到下一个K线时间点"""
        next_timestamp = self._get_next_bar_timestamp()
        current_timestamp = int(time.time() * 1000)
        
        # 计算等待时间
        wait_time = (next_timestamp - current_timestamp) / 1000.0
        
        if wait_time > 0:
            print(f"等待 {wait_time:.1f} 秒到下一个K线时间点...")
            time.sleep(wait_time)
    
    def _check_trailing_stop(self, price: float) -> bool:
        """检查移动止损条件"""
        if self.position == 1:
            # 多仓：更新最高价，检查是否触发止损
            if price > self.highest_price:
                self.highest_price = price
            stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif self.position == -1:
            # 空仓：更新最低价，检查是否触发止损
            if price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = price
            stop_price = self.lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False
    
    def _execute_trade(self, signal: int, price: float, details: dict):
        """执行交易"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        
        # 记录原始仓位
        old_position = self.position
        
        # 检查移动止损
        trailing_stop_triggered = self._check_trailing_stop(price)
        
        if self.position == 0:
            # 无仓位状态
            if signal == 1:
                # 开多仓
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.lowest_price = 0.0
                action = "LONG_OPEN"
                self.trade_count += 1
            elif signal == -1:
                # 开空仓
                self.position = -1
                self.entry_price = price
                self.lowest_price = price
                self.highest_price = 0.0
                action = "SHORT_OPEN"
                self.trade_count += 1
        else:
            # 有仓位状态
            if self.position == 1:
                # 多仓状态
                if trailing_stop_triggered:
                    # 移动止损触发，平多仓
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
                elif signal == -1:
                    # 平多仓，开空仓
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_SHORT_OPEN"
                    self.position = -1
                    self.entry_price = price
                    self.highest_price = 0.0
                    self.lowest_price = price
                    self.trade_count += 1
            elif self.position == -1:
                # 空仓状态
                if trailing_stop_triggered:
                    # 移动止损触发，平空仓
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_TRAILING_STOP"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
                elif signal == 1:
                    # 平空仓，开多仓
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_LONG_OPEN"
                    self.position = 1
                    self.entry_price = price
                    self.lowest_price = 0.0
                    self.highest_price = price
                    self.trade_count += 1
        
        # 只记录非HOLD操作
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
            
            # 加入写入队列
            self._enqueue_write(record)
            
            # 打印交易信息
            print(f"[{record['timestamp']}] {self.symbol} {action}: "
                  f"价格={price:.4f}, 信号={signal}, 收益率={return_rate*100:.2f}%")
    
    def run_simulation(self):
        """运行模拟监控"""
        print(f"开始监控 {self.symbol} 的EMA交叉策略...")
        print(f"策略参数: EMA{self.short_ma}/EMA{self.long_ma}, "
              f"成交量倍数={self.vol_multiplier}, 确认百分比={self.confirmation_pct}%, "
              f"模式={self.mode}, 移动止损={self.trailing_stop_pct}%")
        
        try:
            while True:
                # 等待到下一个K线时间点
                self._wait_for_next_bar()
                
                try:
                    # 获取策略信号
                    signal = self.strategy.calculate_ema_crossover_signal(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.vol_multiplier, self.confirmation_pct, self.mode
                    )
                    
                    # 获取详细信息
                    details = self.strategy.get_strategy_details(
                        self.symbol, self.bar, self.short_ma, self.long_ma,
                        self.vol_multiplier, self.confirmation_pct, self.mode
                    )
                    
                    # 获取当前价格
                    price = details.get('current_price', 0)
                    
                    if price > 0:
                        # 执行交易
                        self._execute_trade(signal, price, details)
                        
                        # 打印当前状态
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                              f"{self.symbol} 价格: {price:.4f}, 信号: {signal}, "
                              f"仓位: {self.position}, 交易次数: {self.trade_count}")
                    else:
                        print(f"无法获取 {self.symbol} 的价格数据")
                
                except Exception as e:
                    print(f"计算信号时出错: {e}")
                    time.sleep(5)  # 出错时等待5秒
                
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
    
    # 配置参数
    symbol = input("请输入交易对 (默认 BTC-USDT): ").strip() or "BTC-USDT"
    bar = input("请输入K线周期 (默认 1m): ").strip() or "1m"
    mode = input("请输入模式 (strict/loose, 默认 strict): ").strip() or "strict"
    trailing_stop_input = input("请输入移动止损百分比 (默认 1.0%): ").strip()
    trailing_stop_pct = float(trailing_stop_input) if trailing_stop_input else 1.0
    
    # 创建监控器
    monitor = StrategyMonitor(
        symbol=symbol,
        bar=bar,
        short_ma=5,
        long_ma=20,
        vol_multiplier=1.2,
        confirmation_pct=0.2,
        mode=mode,
        trailing_stop_pct=trailing_stop_pct
    )
    
    # 运行模拟
    monitor.run_simulation()


if __name__ == "__main__":
    main()