#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : volatility_exit.py
@Description: 波动率退出条件共享逻辑
"""

import numpy as np
from utils.logger import logger


class VolatilityExit:
    """波动率退出条件类"""
    
    def __init__(self, short_ma: int = 5, volatility_threshold: float = 0.5):
        """
        初始化波动率退出条件
        
        Args:
            short_ma: 短EMA周期，用于确定波动率计算窗口
            volatility_threshold: 波动率阈值 (0.5表示50%)
        """
        self.short_ma = short_ma
        self.volatility_threshold = volatility_threshold
        self.price_history = []
    
    def update_price_history(self, price: float):
        """
        更新价格历史记录
        
        Args:
            price: 当前价格
        """
        self.price_history.append(price)
        # 保持最近足够的数据用于波动率计算
        max_history_length = 2 * self.short_ma + 10
        if len(self.price_history) > max_history_length:
            self.price_history = self.price_history[-max_history_length:]
    
    def check_volatility_exit(self, position: int) -> bool:
        """
        检查波动率退出条件
        
        Args:
            position: 当前仓位 (1: 多仓, -1: 空仓, 0: 无仓位)
            
        Returns:
            bool: 是否触发波动率退出
        """
        if position == 0:
            return False
        
        if len(self.price_history) < 2 * self.short_ma:
            return False
        
        try:
            # 获取最近的价格数据
            recent_prices = self.price_history[-2 * self.short_ma:]
            
            # 计算当前窗口的波动率
            current_window = recent_prices[-self.short_ma:]
            current_high = max(current_window)
            current_low = min(current_window)
            current_avg = np.mean(current_window)
            
            if current_avg > 0:
                current_range = (current_high - current_low) / current_avg
            else:
                return False
            
            # 计算前一个窗口的波动率
            prev_window = recent_prices[-2 * self.short_ma:-self.short_ma]
            prev_high = max(prev_window)
            prev_low = min(prev_window)
            prev_avg = np.mean(prev_window)
            
            if prev_avg > 0:
                prev_range = (prev_high - prev_low) / prev_avg
            else:
                return False
            
            # 检查波动率是否显著减小
            if prev_range > 0 and current_range < prev_range * self.volatility_threshold:
                logger.info(f"波动率退出触发: 当前波动率={current_range:.4f}, 前波动率={prev_range:.4f}, 阈值={self.volatility_threshold}")
                return True
                
        except Exception as e:
            logger.warning(f"计算波动率退出条件时出错: {e}")
            
        return False


def check_volatility_exit_static(df, current_index: int, short_ma: int, volatility_threshold: float) -> bool:
    """
    静态方法：检查波动率退出条件（用于回测等场景）
    
    Args:
        df: 包含价格数据的DataFrame
        current_index: 当前数据索引
        short_ma: 短EMA周期
        volatility_threshold: 波动率阈值
        
    Returns:
        bool: 是否触发波动率退出
    """
    if current_index < 2 * short_ma:
        return False
    
    try:
        # 获取价格数据
        highs = df['h'] if 'h' in df.columns else df['high']
        lows = df['l'] if 'l' in df.columns else df['low']
        closes = df['c'] if 'c' in df.columns else df['close']
        
        # 计算当前窗口的波动率
        current_start = current_index - short_ma + 1
        current_end = current_index + 1
        current_highs = highs.iloc[current_start:current_end]
        current_lows = lows.iloc[current_start:current_end]
        current_closes = closes.iloc[current_start:current_end]
        
        current_ranges = (current_highs - current_lows) / current_closes
        avg_range = np.mean(current_ranges)
        
        # 计算前一个窗口的波动率
        prev_start = current_index - 2 * short_ma + 1
        prev_end = current_index - short_ma + 1
        prev_highs = highs.iloc[prev_start:prev_end]
        prev_lows = lows.iloc[prev_start:prev_end]
        prev_closes = closes.iloc[prev_start:prev_end]
        
        prev_ranges = (prev_highs - prev_lows) / prev_closes
        prev_avg_range = np.mean(prev_ranges)
        
        # 检查波动率是否显著减小
        if prev_avg_range > 0 and avg_range < prev_avg_range * volatility_threshold:
            return True
            
    except Exception as e:
        logger.warning(f"计算波动率退出条件时出错: {e}")
        
    return False