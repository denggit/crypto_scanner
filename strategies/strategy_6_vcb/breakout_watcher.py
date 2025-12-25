#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : breakout_watcher.py
@Description: Breakout Watcher（消费者）- 监控压缩池中的币种，检测突破
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_6_vcb.strategy_6 import VCBStrategy
from utils.logger import logger


class BreakoutWatcher:
    """
    突破监控器（消费者）
    
    职责：
    1. 只监控压缩池中的币种
    2. 高频检查突破信号（每根K线或tick）
    3. 一旦突破 → 发信号 → 删除事件
    """
    
    def __init__(self, client: OKXClient, strategy: VCBStrategy):
        """
        初始化突破监控器
        
        Args:
            client: OKX客户端
            strategy: VCB策略实例（共享压缩池）
        """
        self.client = client
        self.strategy = strategy
        self.market_data_retriever = MarketDataRetriever(client)
        
        # 突破信号回调函数
        self.breakout_callbacks = []
        
        # 统计信息
        self.breakout_count = 0
        self.last_breakout_time = None
    
    def watch_compression_pool(self, bar: str = '1m',
                              volume_period: int = 20,
                              volume_multiplier: float = 1.0) -> List[Dict]:
        """
        监控压缩池中的所有币种，检测突破
        
        Args:
            bar: K线周期
            volume_period: 成交量均线周期
            volume_multiplier: 成交量放大倍数
            
        Returns:
            list: 突破信号列表，每个元素包含 {symbol, signal, details}
        """
        try:
            # 获取压缩池中的所有币种
            symbols_in_pool = self.strategy.get_compression_pool_symbols()
            
            if not symbols_in_pool:
                return []
            
            logger.debug(f"监控压缩池中的 {len(symbols_in_pool)} 个币种: {symbols_in_pool}")
            
            # 检测每个币种的突破
            breakouts = []
            
            for symbol in symbols_in_pool:
                try:
                    signal, details = self.strategy.detect_breakout(
                        symbol=symbol,
                        bar=bar,
                        volume_period=volume_period,
                        volume_multiplier=volume_multiplier
                    )
                    
                    if signal != 0:
                        # 发现突破
                        breakout_info = {
                            'symbol': symbol,
                            'signal': signal,  # 1=做多, -1=做空
                            'details': details,
                            'timestamp': datetime.now()
                        }
                        breakouts.append(breakout_info)
                        
                        # 更新统计
                        self.breakout_count += 1
                        self.last_breakout_time = datetime.now()
                        
                        logger.info(f"🚀 突破信号: {symbol} {'做多' if signal == 1 else '做空'} "
                                  f"价格={details.get('current_price', 0):.4f}, "
                                  f"成交量比率={details.get('current_volume', 0) / details.get('avg_volume', 1):.2f}")
                        
                        # 触发回调
                        self._trigger_callbacks(breakout_info)
                        
                except Exception as e:
                    logger.warning(f"检测 {symbol} 突破时出错: {e}")
                    continue
            
            return breakouts
            
        except Exception as e:
            logger.error(f"监控压缩池时出错: {e}")
            return []
    
    def register_breakout_callback(self, callback):
        """
        注册突破信号回调函数
        
        Args:
            callback: 回调函数，接收参数 (symbol, signal, details)
        """
        self.breakout_callbacks.append(callback)
    
    def _trigger_callbacks(self, breakout_info: Dict):
        """触发所有注册的回调函数"""
        for callback in self.breakout_callbacks:
            try:
                callback(
                    breakout_info['symbol'],
                    breakout_info['signal'],
                    breakout_info['details']
                )
            except Exception as e:
                logger.error(f"执行突破回调时出错: {e}")
    
    def get_watch_stats(self) -> Dict:
        """获取监控统计信息"""
        return {
            'breakout_count': self.breakout_count,
            'last_breakout_time': self.last_breakout_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_breakout_time else None,
            'current_pool_size': self.strategy.get_compression_pool_size(),
            'symbols_in_pool': self.strategy.get_compression_pool_symbols()
        }

