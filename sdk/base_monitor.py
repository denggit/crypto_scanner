#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : base_monitor.py
@Description: Base Monitor class for strategy monitoring
"""

import os
import time
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseMonitor(ABC):
    """Base class for strategy monitoring"""
    
    def __init__(self, symbol: str, bar: str = '1m', trade_mode: bool = False):
        """
        Initialize Base Monitor
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            trade_mode: True for real trading, False for mock trading
        """
        self.symbol = symbol
        self.bar = bar
        self.trade_mode = trade_mode
        
        # 模拟交易数据记录
        self.mock_data_dir = os.path.join("tracking_data", "monitor_data")
        self.mock_file_prefix = "monitor"
        
        # 真实交易数据记录（仅在trade_mode=True时使用）
        if trade_mode:
            self.real_data_dir = os.path.join("tracking_data", "trade_data")
            self.real_file_prefix = "trade"
        
        self.mock_position = 0
        self.mock_entry_price = 0.0
        self.mock_highest_price = 0.0
        self.mock_lowest_price = 0.0
        self.trade_count = 0
        
        # 确保 tracking_data 根目录存在
        if not os.path.exists("tracking_data"):
            os.makedirs("tracking_data")
        
        # 初始化模拟交易数据记录
        if not os.path.exists(self.mock_data_dir):
            os.makedirs(self.mock_data_dir)
        
        self.mock_csv_file = os.path.join(self.mock_data_dir, f"strategy_{self.mock_file_prefix}_{symbol.replace('-', '_')}.csv")
        self.mock_backup_file = os.path.join(self.mock_data_dir, f"strategy_{self.mock_file_prefix}_{symbol.replace('-', '_')}_backup.csv")
        
        # 初始化真实交易数据记录（仅在trade_mode=True时）
        if trade_mode:
            if not os.path.exists(self.real_data_dir):
                os.makedirs(self.real_data_dir)
            self.real_csv_file = os.path.join(self.real_data_dir, f"strategy_{self.real_file_prefix}_{symbol.replace('-', '_')}.csv")
            self.real_backup_file = os.path.join(self.real_data_dir, f"strategy_{self.real_file_prefix}_{symbol.replace('-', '_')}_backup.csv")
        
        self._init_csv_files()
        self._restore_state()
        
        # 使用两个独立的写入队列
        self.mock_write_queue = []
        self.real_write_queue = []
        self.write_lock = threading.Lock()
        
        # 启动后台写入线程
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()
        
        print(f"监控器已初始化: {symbol}")
        print(f"模拟记录文件: {self.mock_csv_file}")
        if trade_mode:
            print(f"真实记录文件: {self.real_csv_file}")
    
    def _init_csv_files(self):
        """Initialize CSV files if not exists"""
        mock_headers = self.get_csv_headers()
        
        # 初始化模拟交易CSV文件
        if not os.path.exists(self.mock_csv_file):
            df = pd.DataFrame(columns=mock_headers)
            df.to_csv(self.mock_csv_file, index=False)
            print(f"创建新的模拟记录文件: {self.mock_csv_file}")
        else:
            print(f"使用现有模拟记录文件: {self.mock_csv_file}")
        
        if not os.path.exists(self.mock_backup_file):
            df = pd.DataFrame(columns=mock_headers)
            df.to_csv(self.mock_backup_file, index=False)
            print(f"创建新的模拟备份文件: {self.mock_backup_file}")
        else:
            print(f"使用现有模拟备份文件: {self.mock_backup_file}")
        
        # 初始化真实交易CSV文件（仅在trade_mode=True时）
        if self.trade_mode:
            real_headers = self.get_real_csv_headers()
            if not os.path.exists(self.real_csv_file):
                df = pd.DataFrame(columns=real_headers)
                df.to_csv(self.real_csv_file, index=False)
                print(f"创建新的真实记录文件: {self.real_csv_file}")
            else:
                print(f"使用现有真实记录文件: {self.real_csv_file}")
            
            if not os.path.exists(self.real_backup_file):
                df = pd.DataFrame(columns=real_headers)
                df.to_csv(self.real_backup_file, index=False)
                print(f"创建新的真实备份文件: {self.real_backup_file}")
            else:
                print(f"使用现有真实备份文件: {self.real_backup_file}")
    
    def _restore_state(self):
        """Restore latest mock position state from CSV file"""
        try:
            # 从模拟交易文件恢复状态
            if os.path.exists(self.mock_csv_file) and os.path.getsize(self.mock_csv_file) > 0:
                df = pd.read_csv(self.mock_csv_file)
                if not df.empty:
                    last_record = df.iloc[-1]
                    self.mock_position = int(last_record['position'])
                    self.mock_entry_price = float(last_record['entry_price'])
                    self.trade_count = len(df[df['action'] != 'HOLD'])
                    
                    if self.mock_position == 1:
                        self.mock_highest_price = self.mock_entry_price
                        self.mock_lowest_price = self.mock_entry_price
                    elif self.mock_position == -1:
                        self.mock_lowest_price = self.mock_entry_price
                        self.mock_highest_price = self.mock_entry_price
                    else:
                        self.mock_highest_price = 0.0
                        self.mock_lowest_price = 0.0
                    print(f"模拟状态已从文件恢复: position={self.mock_position}, entry_price={self.mock_entry_price:.4f}, trade_count={self.trade_count}")
                else:
                    print("模拟CSV文件为空，使用初始状态")
            else:
                print("未找到模拟历史记录，使用初始状态")
        except Exception as e:
            print(f"恢复状态失败，使用初始状态: {e}")
    
    def _background_writer(self):
        """Background writer thread"""
        while True:
            try:
                # 处理模拟交易数据写入
                if self.mock_write_queue:
                    with self.write_lock:
                        mock_data_to_write = self.mock_write_queue.copy()
                        self.mock_write_queue.clear()
                    
                    if mock_data_to_write:
                        df = pd.DataFrame(mock_data_to_write)
                        df.to_csv(self.mock_csv_file, mode='a', header=False, index=False)
                        df.to_csv(self.mock_backup_file, mode='a', header=False, index=False)
                
                # 处理真实交易数据写入（仅在trade_mode=True时）
                if self.trade_mode and self.real_write_queue:
                    with self.write_lock:
                        real_data_to_write = self.real_write_queue.copy()
                        self.real_write_queue.clear()
                    
                    if real_data_to_write:
                        df = pd.DataFrame(real_data_to_write)
                        df.to_csv(self.real_csv_file, mode='a', header=False, index=False)
                        df.to_csv(self.real_backup_file, mode='a', header=False, index=False)
                
                time.sleep(1)
            except Exception as e:
                print(f"后台写入线程错误: {e}")
                time.sleep(5)
    
    def _enqueue_mock_write(self, data: Dict[str, Any]):
        """Enqueue mock trading data for writing"""
        with self.write_lock:
            self.mock_write_queue.append(data)
    
    def _enqueue_real_write(self, data: Dict[str, Any]):
        """Enqueue real trading data for writing (only when trade_mode=True)"""
        if self.trade_mode:
            with self.write_lock:
                self.real_write_queue.append(data)
        else:
            print("警告: 尝试记录真实交易数据，但trade_mode=False")
    
    def _get_next_bar_timestamp(self) -> int:
        """Get next K-line timestamp"""
        now = datetime.now()
        
        if self.bar == '1m':
            next_minute = now.replace(second=0, microsecond=0)
            if next_minute <= now:
                next_minute = next_minute + timedelta(minutes=1)
            return int(next_minute.timestamp() * 1000)
        elif self.bar == '5m':
            next_5min = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
            if next_5min <= now:
                next_5min = next_5min + timedelta(minutes=5)
            return int(next_5min.timestamp() * 1000)
        elif self.bar == '15m':
            next_15min = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            if next_15min <= now:
                next_15min = next_15min + timedelta(minutes=15)
            return int(next_15min.timestamp() * 1000)
        else:
            next_minute = now.replace(second=0, microsecond=0)
            if next_minute <= now:
                next_minute = next_minute + timedelta(minutes=1)
            return int(next_minute.timestamp() * 1000)
    
    def _wait_for_next_bar(self):
        """Wait for next K-line time point"""
        next_timestamp = self._get_next_bar_timestamp()
        current_timestamp = int(time.time() * 1000)
        
        wait_time = (next_timestamp - current_timestamp) / 1000.0
        
        if wait_time > 0:
            print(f"等待 {wait_time:.1f} 秒到下一个K线时间点...")
            time.sleep(wait_time)
    
    @abstractmethod
    def get_csv_headers(self) -> list:
        """Get CSV headers for mock trading data"""
        pass
    
    def get_real_csv_headers(self) -> list:
        """Get CSV headers for real trading data"""
        return [
            'timestamp', 'symbol', 'action', 'order_id', 'order_price',
            'order_size', 'order_type', 'order_side', 'order_state',
            'filled_size', 'avg_fill_price', 'fee', 'trade_timestamp'
        ]
    
    def _check_trailing_stop(self, price: float) -> bool:
        """Check if trailing stop is triggered (implemented by subclass if needed)"""
        return False
    
    def get_mode_prefix(self) -> str:
        """Get log prefix based on trade mode"""
        return "[真实交易]" if self.trade_mode else "[模拟交易]"
    
    @abstractmethod
    def get_trader(self):
        """Get trader instance (implemented by subclass)"""
        pass
    
    @abstractmethod
    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade based on signal"""
        pass
    
    @abstractmethod
    def run(self):
        """Run monitoring loop"""
        pass
