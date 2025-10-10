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
    
    def __init__(self, symbol: str, bar: str = '1m', data_dir: str = "monitor_data", file_prefix: str = "monitor"):
        """
        Initialize Base Monitor
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            data_dir: Data directory for storing records
            file_prefix: Prefix for CSV files
        """
        self.symbol = symbol
        self.bar = bar
        self.data_dir = data_dir
        self.file_prefix = file_prefix
        
        self.position = 0
        self.entry_price = 0.0
        self.trade_count = 0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.csv_file = os.path.join(self.data_dir, f"strategy_{file_prefix}_{symbol.replace('-', '_')}.csv")
        self.backup_file = os.path.join(self.data_dir, f"strategy_{file_prefix}_{symbol.replace('-', '_')}_backup.csv")
        
        self._init_csv_file()
        self._restore_state()
        
        self.write_queue = []
        self.write_lock = threading.Lock()
        self.writer_thread = threading.Thread(target=self._background_writer, daemon=True)
        self.writer_thread.start()
        
        print(f"监控器已初始化: {symbol}")
        print(f"记录文件: {self.csv_file}")
        print(f"备份文件: {self.backup_file}")
    
    def _init_csv_file(self):
        """Initialize CSV file if not exists"""
        headers = self.get_csv_headers()
        
        if not os.path.exists(self.csv_file):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.csv_file, index=False)
            print(f"创建新的记录文件: {self.csv_file}")
        else:
            print(f"使用现有记录文件: {self.csv_file}")
        
        if not os.path.exists(self.backup_file):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.backup_file, index=False)
            print(f"创建新的备份文件: {self.backup_file}")
        else:
            print(f"使用现有备份文件: {self.backup_file}")
    
    def _restore_state(self):
        """Restore latest position state from CSV file"""
        try:
            if os.path.exists(self.csv_file) and os.path.getsize(self.csv_file) > 0:
                df = pd.read_csv(self.csv_file)
                if not df.empty:
                    last_record = df.iloc[-1]
                    self.position = int(last_record['position'])
                    self.entry_price = float(last_record['entry_price'])
                    self.trade_count = len(df[df['action'] != 'HOLD'])
                    
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
        """Background writer thread"""
        while True:
            try:
                if self.write_queue:
                    with self.write_lock:
                        data_to_write = self.write_queue.copy()
                        self.write_queue.clear()
                    
                    if data_to_write:
                        df = pd.DataFrame(data_to_write)
                        df.to_csv(self.csv_file, mode='a', header=False, index=False)
                        df.to_csv(self.backup_file, mode='a', header=False, index=False)
                
                time.sleep(1)
            except Exception as e:
                print(f"后台写入线程错误: {e}")
                time.sleep(5)
    
    def _enqueue_write(self, data: Dict[str, Any]):
        """Enqueue data for writing"""
        with self.write_lock:
            self.write_queue.append(data)
    
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
        """Get CSV headers for recording data"""
        pass
    
    @abstractmethod
    def execute_trade(self, signal: int, price: float, details: dict):
        """Execute trade based on signal"""
        pass
    
    @abstractmethod
    def run(self):
        """Run monitoring loop"""
        pass
