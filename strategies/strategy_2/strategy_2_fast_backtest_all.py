#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 2:42 PM
@File       : strategy_2_fast_backtest_all.py
@Description: 高频短线策略批量快速回测系统 - 自动扫描高交易量币种并批量回测
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import concurrent.futures
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_2.strategy_2 import HighFrequencyStrategy
from strategies.strategy_2.shared_config import load_config_from_file, get_user_input, print_final_config
from tools.market_scanner import CryptoScanner
from utils.logger import logger


class BatchFastBacktest:
    """批量快速回测类"""
    
    def __init__(self, bar: str = '1m',
                 consecutive_bars: int = 2, atr_period: int = 14,
                 atr_threshold: float = 0.8, trailing_stop_pct: float = 0.8,
                 volume_factor: float = 1.2, use_volume: bool = True,
                 breakout_stop_bars: int = 2,
                 buy_fee_rate: float = 0.0005, sell_fee_rate: float = 0.0005):
        """
        Initialize Batch Fast Backtest
        
        Args:
            bar: K-line time interval
            consecutive_bars: Number of consecutive bars for breakout
            atr_period: ATR period
            atr_threshold: ATR threshold multiplier
            trailing_stop_pct: Trailing stop percentage
            volume_factor: Volume expansion factor
            use_volume: Whether to use volume condition
            breakout_stop_bars: Number of consecutive bars for breakout stop
        """
        self.bar = bar
        self.consecutive_bars = consecutive_bars
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.trailing_stop_pct = trailing_stop_pct
        self.volume_factor = volume_factor
        self.use_volume = use_volume
        self.breakout_stop_bars = breakout_stop_bars
        
        # 手续费参数
        self.buy_fee_rate = buy_fee_rate  # 买入手续费率 0.05%
        self.sell_fee_rate = sell_fee_rate  # 卖出手续费率 0.05%
        
        self.client = OKXClient()
        self.strategy = HighFrequencyStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        self.scanner = CryptoScanner(self.client)
        
        # 批量回测结果
        self.batch_results = []
        
        # 缓存技术指标计算
        self._atr_cache = {}
        self._volume_cache = {}
    
    def run_single_backtest(self, symbol: str, limit: int = 300):
        """运行单个币种的快速回测"""
        logger.info(f"开始回测 {symbol}...")
        
        try:
            # 获取历史数据
            df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
            if df is None or len(df) == 0 or len(df) < limit:
                logger.warning(f"{symbol}: 无法获取足够的历史数据，实际获取: {len(df) if df is not None else 0} 根K线")
                return None
            
            # 回测状态
            position = 0  # 0: 无仓位, 1: 多仓, -1: 空仓
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            trade_count = 0
            total_fee = 0.0
            close_trades = []
            
            # 批量计算所有K线的信号和详细信息
            signals, details_list, typical_prices = self._calculate_signals_in_bulk(df, symbol)
            
            # 按时间顺序处理每根K线
            for i in range(len(df)):
                if i < max(self.atr_period, self.consecutive_bars + 1, 21):  # 确保有足够的数据计算指标
                    continue
                    
                signal = signals[i]
                details = details_list[i]
                price = details.get('current_price', 0)
                
                if price > 0:
                    # 执行交易逻辑
                    trade_result = self._execute_trade_logic(
                        signal, price, details, position, entry_price, 
                        highest_price, lowest_price, trade_count, total_fee, close_trades,
                        df, typical_prices, i
                    )
                    
                    position = trade_result['position']
                    entry_price = trade_result['entry_price']
                    highest_price = trade_result['highest_price']
                    lowest_price = trade_result['lowest_price']
                    trade_count = trade_result['trade_count']
                    total_fee = trade_result['total_fee']
                    
                    # 检查移动止损
                    if position != 0:
                        self._check_trailing_stop(price, position, highest_price, lowest_price)
            
            # 计算回测指标
            report = self._generate_single_report(symbol, close_trades, trade_count, total_fee)
            return report
            
        except Exception as e:
            logger.error(f"{symbol}: 回测过程中出错: {e}")
            return None
    
    def _calculate_signals_in_bulk(self, df: pd.DataFrame, symbol: str):
        """批量计算所有K线的信号和详细信息"""
        signals = []
        details_list = []
        
        # 获取价格和成交量数据
        closes = df['c'] if 'c' in df.columns else df['close']
        highs = df['h'] if 'h' in df.columns else df['high']
        lows = df['l'] if 'l' in df.columns else df['low']
        volumes = df['vol'] if 'vol' in df.columns else df['volume']
        
        # 计算典型价格 (high + low + close) / 3
        typical_prices = (highs + lows + closes) / 3
        
        # 计算ATR - 使用缓存
        from tools.technical_indicators import atr
        cache_key = f"{symbol}_{self.bar}_{self.atr_period}"
        if cache_key in self._atr_cache:
            atr_values = self._atr_cache[cache_key]
        else:
            atr_values = atr(df, self.atr_period)
            self._atr_cache[cache_key] = atr_values
        
        # 计算成交量条件
        volume_expansions = []
        volume_ratios = []
        
        for i in range(len(volumes)):
            if i < 21:  # 前20根K线平均成交量
                volume_expansions.append(False)
                volume_ratios.append(0)
                continue
            
            current_volume = volumes.iloc[i]
            avg_volume = volumes.iloc[i-20:i].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_ratios.append(volume_ratio)
            volume_expansions.append(volume_ratio >= self.volume_factor)
        
        # 批量计算信号
        for i in range(len(df)):
            if i < max(self.atr_period, self.consecutive_bars + 1, 21):
                signals.append(0)
                details_list.append({})
                continue
                
            current_close = closes.iloc[i]
            current_high = highs.iloc[i]
            current_low = lows.iloc[i]
            current_volume = volumes.iloc[i]
            current_typical = typical_prices.iloc[i]
            
            # 计算ATR条件
            current_atr = atr_values.iloc[i]
            atr_mean = atr_values.iloc[i-self.atr_period:i].mean() if i >= self.atr_period else current_atr
            atr_condition_met = current_atr > atr_mean * self.atr_threshold
            
            # 计算成交量条件
            volume_condition_met = False
            if self.use_volume and i >= 21:
                volume_condition_met = volume_expansions[i]
            
            # 检查连续突破条件
            long_breakout = self._check_consecutive_breakout(df, typical_prices, i, self.consecutive_bars, direction='up')
            short_breakout = self._check_consecutive_breakout(df, typical_prices, i, self.consecutive_bars, direction='down')
            
            # 计算技术指标信号（不包含仓位逻辑）
            signal = 0
            
            # 开多技术条件
            if long_breakout and atr_condition_met:
                if not self.use_volume or volume_condition_met:
                    signal = 1
            
            # 开空技术条件
            elif short_breakout and atr_condition_met:
                if not self.use_volume or volume_condition_met:
                    signal = -1
            
            # 构建详细信息
            details = {
                'current_price': float(current_close),
                'current_typical': float(current_typical),
                'atr': float(current_atr),
                'atr_mean': float(atr_mean),
                'atr_condition_met': atr_condition_met,
                'volume_condition_met': volume_condition_met,
                'long_breakout': long_breakout,
                'short_breakout': short_breakout,
                'current_volume': float(current_volume),
                'volume_ratio': volume_ratios[i] if i < len(volume_ratios) else 0
            }
            
            signals.append(signal)
            details_list.append(details)
        
        return signals, details_list, typical_prices
    
    def _execute_trade_logic(self, signal: int, price: float, details: dict,
                           position: int, entry_price: float, highest_price: float, 
                           lowest_price: float, trade_count: int, total_fee: float, close_trades: list,
                           df: pd.DataFrame, typical_prices: pd.Series, current_idx: int):
        """执行交易逻辑"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        trade_fee = 0.0
        trade_type = ""
        exit_reason = ""
        
        # 检查移动止损
        trailing_stop_triggered = self._check_trailing_stop(price, position, highest_price, lowest_price)
        
        # 检查平仓条件 (连续breakout_stop_bars根K线反向突破)
        close_signal = 0
        if position == 1 and self._check_consecutive_breakout(df, typical_prices, current_idx, self.breakout_stop_bars, direction='down'):
            close_signal = -1
        elif position == -1 and self._check_consecutive_breakout(df, typical_prices, current_idx, self.breakout_stop_bars, direction='up'):
            close_signal = 1
        
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                highest_price = price
                lowest_price = price
                action = "LONG_OPEN"
                trade_type = "LONG"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                trade_count += 1
            elif signal == -1:
                position = -1
                entry_price = price
                lowest_price = price
                highest_price = price
                action = "SHORT_OPEN"
                trade_type = "SHORT"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                trade_count += 1
        elif position == 1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "LONG_CLOSE_TRAILING_STOP"
                exit_reason = "TRAILING_STOP"
                trade_type = "LONG"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                position = 0
                highest_price = 0.0
                trade_count += 1
            elif close_signal == -1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "LONG_CLOSE_BREAKOUT"
                exit_reason = "BREAKOUT"
                trade_type = "LONG"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                position = 0
                highest_price = 0.0
                trade_count += 1
            elif signal == -1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "LONG_CLOSE_SHORT_OPEN"
                exit_reason = "REVERSE_SIGNAL"
                trade_type = "LONG"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                position = -1
                entry_price = price
                highest_price = price
                lowest_price = price
                trade_count += 1
        elif position == -1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "SHORT_CLOSE_TRAILING_STOP"
                exit_reason = "TRAILING_STOP"
                trade_type = "SHORT"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                position = 0
                lowest_price = 0.0
                trade_count += 1
            elif close_signal == 1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "SHORT_CLOSE_BREAKOUT"
                exit_reason = "BREAKOUT"
                trade_type = "SHORT"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                position = 0
                lowest_price = 0.0
                trade_count += 1
            elif signal == 1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "SHORT_CLOSE_LONG_OPEN"
                exit_reason = "REVERSE_SIGNAL"
                trade_type = "SHORT"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                position = 1
                entry_price = price
                lowest_price = price
                highest_price = price
                trade_count += 1
        
        # 记录平仓交易（包含详细统计信息）
        if action != "HOLD" and return_rate != 0:
            # 计算持仓时间，确保不会出现负数
            holding_bars = 0
            if hasattr(self, '_last_entry_idx'):
                holding_bars = max(0, current_idx - self._last_entry_idx)
            
            close_trades.append({
                'return_rate': return_rate,
                'exit_price': exit_price,
                'action': action,
                'trade_type': trade_type,
                'exit_reason': exit_reason,
                'entry_price': entry_price,
                'position_holding_bars': holding_bars,
                'atr_condition': details.get('atr_condition_met', False),
                'volume_condition': details.get('volume_condition_met', False),
                'long_breakout': details.get('long_breakout', False),
                'short_breakout': details.get('short_breakout', False),
                'current_price': price,
                'highest_price': highest_price,
                'lowest_price': lowest_price
            })
            
            # 平仓后重置开仓索引
            if action.endswith("_CLOSE"):
                if hasattr(self, '_last_entry_idx'):
                    delattr(self, '_last_entry_idx')
            
        # 记录开仓索引
        if action in ["LONG_OPEN", "SHORT_OPEN"]:
            self._last_entry_idx = current_idx
        
        return {
            'position': position,
            'entry_price': entry_price,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
            'trade_count': trade_count,
            'total_fee': total_fee
        }
    
    def _check_trailing_stop(self, price: float, position: int, highest_price: float, lowest_price: float) -> bool:
        """检查移动止损条件"""
        if position == 1:
            # 持多仓：更新最高价，检查是否跌破止损价
            if price > highest_price:
                highest_price = price
            stop_price = highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif position == -1:
            # 持空仓：更新最低价，检查是否涨破止损价
            if price < lowest_price:
                lowest_price = price
            stop_price = lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False
    
    def _check_consecutive_breakout(self, df: pd.DataFrame, typical_prices: pd.Series, current_idx: int, 
                                   consecutive_bars: int, direction: str) -> bool:
        """
        检查连续突破条件
        
        Args:
            df: K线数据
            typical_prices: 典型价格序列
            current_idx: 当前K线索引
            consecutive_bars: 连续K线数量
            direction: 突破方向 ('up' 或 'down')
            
        Returns:
            bool: 是否满足连续突破条件
        """
        if current_idx < consecutive_bars:
            return False
        
        # 检查最近consecutive_bars根K线是否连续突破
        for i in range(consecutive_bars):
            idx = current_idx - i
            prev_idx = current_idx - i - 1
            
            if direction == 'up':
                # 向上突破: 当前close > 前一根typical price
                if df['close'].iloc[idx] <= typical_prices.iloc[prev_idx]:
                    return False
            else:
                # 向下突破: 当前close < 前一根typical price
                if df['close'].iloc[idx] >= typical_prices.iloc[prev_idx]:
                    return False
        
        return True
    
    def _calculate_return_rate(self, entry_price: float, exit_price: float, position: int) -> float:
        """计算考虑手续费后的净收益率"""
        if position == 1:  # 多仓
            entry_cost = entry_price * (1 + self.buy_fee_rate)
            exit_net_value = exit_price * (1 - self.sell_fee_rate)
            return_rate = (exit_net_value - entry_cost) / entry_cost
        elif position == -1:  # 空仓
            entry_cost = entry_price * (1 + self.sell_fee_rate)
            exit_net_value = exit_price * (1 - self.buy_fee_rate)
            return_rate = (entry_cost - exit_net_value) / entry_cost
        else:
            return_rate = 0.0
        
        return return_rate
    
    def _generate_single_report(self, symbol: str, close_trades: list, trade_count: int, total_fee: float):
        """生成单个币种的回测报告"""
        # 过滤出平仓交易（只有平仓交易才有收益率）
        close_trades_df = pd.DataFrame(close_trades) if close_trades else pd.DataFrame()
        
        # 计算回测指标（只基于平仓交易）
        total_return = close_trades_df['return_rate'].sum() * 100 if len(close_trades_df) > 0 else 0  # 转换为百分比
        win_trades = close_trades_df[close_trades_df['return_rate'] > 0] if len(close_trades_df) > 0 else pd.DataFrame()
        loss_trades = close_trades_df[close_trades_df['return_rate'] < 0] if len(close_trades_df) > 0 else pd.DataFrame()
        
        win_rate = len(win_trades) / len(close_trades_df) * 100 if len(close_trades_df) > 0 else 0
        avg_win = win_trades['return_rate'].mean() * 100 if len(win_trades) > 0 else 0
        avg_loss = loss_trades['return_rate'].mean() * 100 if len(loss_trades) > 0 else 0
        profit_factor = abs(win_trades['return_rate'].sum() / loss_trades['return_rate'].sum()) if len(loss_trades) > 0 and loss_trades['return_rate'].sum() != 0 else float('inf')
        
        # 计算夏普比率（简化版，假设无风险利率为0）
        returns = close_trades_df['return_rate'].dropna() if len(close_trades_df) > 0 else pd.Series()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() != 0 else 0
        
        # 最大回撤
        if len(close_trades_df) > 0:
            cumulative_returns = (1 + close_trades_df['return_rate']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0
        
        # 详细统计信息
        detailed_stats = {
            # 交易类型统计
            'long_trades_count': 0,
            'short_trades_count': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'long_avg_return': 0,
            'short_avg_return': 0,
            
            # 止损原因统计
            'trailing_stop_count': 0,
            'breakout_stop_count': 0,
            'reverse_signal_count': 0,
            'trailing_stop_win_rate': 0,
            'breakout_stop_win_rate': 0,
            'reverse_signal_win_rate': 0,
            
            # 止损占比分析
            'trailing_stop_ratio': 0,
            'breakout_stop_ratio': 0,
            'reverse_signal_ratio': 0,
            
            # 亏损原因分析
            'loss_trailing_stop_ratio': 0,
            'loss_breakout_ratio': 0,
            'loss_reverse_signal_ratio': 0,
            
            # 止损收益贡献分析
            'trailing_stop_return_pct': 0,
            'breakout_stop_return_pct': 0,
            'reverse_signal_return_pct': 0,
            'trailing_stop_return_ratio': 0,
            'breakout_stop_return_ratio': 0,
            'reverse_signal_return_ratio': 0,
            
            # 条件统计
            'atr_condition_count': 0,
            'volume_condition_count': 0,
            'atr_condition_win_rate': 0,
            'volume_condition_win_rate': 0,
            
            # 持仓时间统计
            'avg_holding_bars': 0,
            'max_holding_bars': 0,
            'min_holding_bars': 0,
            
            # 亏损分析
            'avg_loss_amount': 0,
            'max_loss_amount': 0,
            'loss_trades_count': 0,
            'profit_trades_count': 0
        }
        
        if len(close_trades_df) > 0:
            # 交易类型统计
            long_trades = close_trades_df[close_trades_df['trade_type'] == 'LONG']
            short_trades = close_trades_df[close_trades_df['trade_type'] == 'SHORT']
            
            # 止损原因统计
            trailing_stop_trades = close_trades_df[close_trades_df['exit_reason'] == 'TRAILING_STOP']
            breakout_trades = close_trades_df[close_trades_df['exit_reason'] == 'BREAKOUT']
            reverse_signal_trades = close_trades_df[close_trades_df['exit_reason'] == 'REVERSE_SIGNAL']
            
            # 条件统计
            atr_condition_trades = close_trades_df[close_trades_df['atr_condition'] == True]
            volume_condition_trades = close_trades_df[close_trades_df['volume_condition'] == True]
            
            # 计算止损占比
            total_stop_trades = len(trailing_stop_trades) + len(breakout_trades) + len(reverse_signal_trades)
            trailing_stop_ratio = len(trailing_stop_trades) / total_stop_trades * 100 if total_stop_trades > 0 else 0
            breakout_stop_ratio = len(breakout_trades) / total_stop_trades * 100 if total_stop_trades > 0 else 0
            reverse_signal_ratio = len(reverse_signal_trades) / total_stop_trades * 100 if total_stop_trades > 0 else 0
            
            # 计算亏损原因分析
            loss_by_trailing_stop = len(trailing_stop_trades[trailing_stop_trades['return_rate'] < 0]) if len(trailing_stop_trades) > 0 else 0
            loss_by_breakout = len(breakout_trades[breakout_trades['return_rate'] < 0]) if len(breakout_trades) > 0 else 0
            loss_by_reverse_signal = len(reverse_signal_trades[reverse_signal_trades['return_rate'] < 0]) if len(reverse_signal_trades) > 0 else 0
            
            total_loss_trades = loss_by_trailing_stop + loss_by_breakout + loss_by_reverse_signal
            loss_trailing_stop_ratio = loss_by_trailing_stop / total_loss_trades * 100 if total_loss_trades > 0 else 0
            loss_breakout_ratio = loss_by_breakout / total_loss_trades * 100 if total_loss_trades > 0 else 0
            loss_reverse_signal_ratio = loss_by_reverse_signal / total_loss_trades * 100 if total_loss_trades > 0 else 0
            
            # 计算止损收益贡献分析
            trailing_stop_return = trailing_stop_trades['return_rate'].sum() * 100 if len(trailing_stop_trades) > 0 else 0
            breakout_stop_return = breakout_trades['return_rate'].sum() * 100 if len(breakout_trades) > 0 else 0
            reverse_signal_return = reverse_signal_trades['return_rate'].sum() * 100 if len(reverse_signal_trades) > 0 else 0
            
            # 计算止损收益占比
            total_stop_return = trailing_stop_return + breakout_stop_return + reverse_signal_return
            trailing_stop_return_ratio = trailing_stop_return / total_stop_return * 100 if total_stop_return != 0 else 0
            breakout_stop_return_ratio = breakout_stop_return / total_stop_return * 100 if total_stop_return != 0 else 0
            reverse_signal_return_ratio = reverse_signal_return / total_stop_return * 100 if total_stop_return != 0 else 0
            
            # 计算各种止损方式的平均收益率
            trailing_stop_avg_return = trailing_stop_trades['return_rate'].mean() * 100 if len(trailing_stop_trades) > 0 else 0
            breakout_stop_avg_return = breakout_trades['return_rate'].mean() * 100 if len(breakout_trades) > 0 else 0
            reverse_signal_avg_return = reverse_signal_trades['return_rate'].mean() * 100 if len(reverse_signal_trades) > 0 else 0
            
            # 计算各种止损导致最终收益的百分比
            total_final_return = close_trades_df['return_rate'].sum() * 100 if len(close_trades_df) > 0 else 0
            trailing_stop_final_return_pct = trailing_stop_return / total_final_return * 100 if total_final_return != 0 else 0
            breakout_stop_final_return_pct = breakout_stop_return / total_final_return * 100 if total_final_return != 0 else 0
            reverse_signal_final_return_pct = reverse_signal_return / total_final_return * 100 if total_final_return != 0 else 0
            
            detailed_stats = {
                # 交易类型统计
                'long_trades_count': len(long_trades),
                'short_trades_count': len(short_trades),
                'long_win_rate': len(long_trades[long_trades['return_rate'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
                'short_win_rate': len(short_trades[short_trades['return_rate'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
                'long_avg_return': long_trades['return_rate'].mean() * 100 if len(long_trades) > 0 else 0,
                'short_avg_return': short_trades['return_rate'].mean() * 100 if len(short_trades) > 0 else 0,
                
                # 止损原因统计
                'trailing_stop_count': len(trailing_stop_trades),
                'breakout_stop_count': len(breakout_trades),
                'reverse_signal_count': len(reverse_signal_trades),
                'trailing_stop_win_rate': len(trailing_stop_trades[trailing_stop_trades['return_rate'] > 0]) / len(trailing_stop_trades) * 100 if len(trailing_stop_trades) > 0 else 0,
                'breakout_stop_win_rate': len(breakout_trades[breakout_trades['return_rate'] > 0]) / len(breakout_trades) * 100 if len(breakout_trades) > 0 else 0,
                'reverse_signal_win_rate': len(reverse_signal_trades[reverse_signal_trades['return_rate'] > 0]) / len(reverse_signal_trades) * 100 if len(reverse_signal_trades) > 0 else 0,
                
                # 止损占比分析
                'trailing_stop_ratio': trailing_stop_ratio,
                'breakout_stop_ratio': breakout_stop_ratio,
                'reverse_signal_ratio': reverse_signal_ratio,
                
                # 亏损原因分析
                'loss_trailing_stop_ratio': loss_trailing_stop_ratio,
                'loss_breakout_ratio': loss_breakout_ratio,
                'loss_reverse_signal_ratio': loss_reverse_signal_ratio,
                
                # 止损收益贡献分析
                'trailing_stop_return_pct': trailing_stop_return,
                'breakout_stop_return_pct': breakout_stop_return,
                'reverse_signal_return_pct': reverse_signal_return,
                'trailing_stop_return_ratio': trailing_stop_return_ratio,
                'breakout_stop_return_ratio': breakout_stop_return_ratio,
                'reverse_signal_return_ratio': reverse_signal_return_ratio,
                
                # 各种止损方式的平均收益率
                'trailing_stop_avg_return': trailing_stop_avg_return,
                'breakout_stop_avg_return': breakout_stop_avg_return,
                'reverse_signal_avg_return': reverse_signal_avg_return,
                
                # 各种止损导致最终收益的百分比
                'trailing_stop_final_return_pct': trailing_stop_final_return_pct,
                'breakout_stop_final_return_pct': breakout_stop_final_return_pct,
                'reverse_signal_final_return_pct': reverse_signal_final_return_pct,
                
                # 条件统计
                'atr_condition_count': len(atr_condition_trades),
                'volume_condition_count': len(volume_condition_trades),
                'atr_condition_win_rate': len(atr_condition_trades[atr_condition_trades['return_rate'] > 0]) / len(atr_condition_trades) * 100 if len(atr_condition_trades) > 0 else 0,
                'volume_condition_win_rate': len(volume_condition_trades[volume_condition_trades['return_rate'] > 0]) / len(volume_condition_trades) * 100 if len(volume_condition_trades) > 0 else 0,
                
                # 持仓时间统计
                'avg_holding_bars': close_trades_df['position_holding_bars'].mean() if len(close_trades_df) > 0 else 0,
                'max_holding_bars': close_trades_df['position_holding_bars'].max() if len(close_trades_df) > 0 else 0,
                'min_holding_bars': close_trades_df['position_holding_bars'].min() if len(close_trades_df) > 0 else 0,
                
                # 亏损分析
                'avg_loss_amount': loss_trades['return_rate'].mean() * 100 if len(loss_trades) > 0 else 0,
                'max_loss_amount': loss_trades['return_rate'].min() * 100 if len(loss_trades) > 0 else 0,
                'loss_trades_count': len(loss_trades),
                'profit_trades_count': len(win_trades)
            }
        
        report = {
            'symbol': symbol,
            'total_trades': trade_count,
            'total_return_pct': total_return,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'close_trades_count': len(close_trades_df),
            **detailed_stats
        }
        
        return report
    
    def _run_single_backtest_with_retry(self, symbol: str, limit: int, max_retries: int = 3) -> Dict[str, Any]:
        """带重试机制的单币种回测"""
        for attempt in range(max_retries):
            try:
                return self.run_single_backtest(symbol, limit)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"{symbol}: 重试{max_retries}次后仍然失败: {e}")
                    raise
                logger.warning(f"{symbol}: 第{attempt + 1}次尝试失败: {e}, 重试...")
                import time
                time.sleep(1)  # 重试前等待1秒
        return None
    
    def run_batch_backtest(self, min_vol_ccy: float = 100000000, limit: int = 300, max_workers: int = 5):
        """运行批量回测"""
        logger.info(f"开始批量回测，筛选24小时交易量 >= {min_vol_ccy:,.0f} USDT的币种")
        
        try:
            # 获取高交易量币种
            symbols = self.scanner._get_volume_filtered_symbols('USDT', min_vol_ccy, use_cache=True, inst_type="SWAP")
            
            if not symbols:
                logger.error(f"未找到24小时交易量 >= {min_vol_ccy:,.0f} USDT的币种")
                return []
            
            logger.info(f"找到 {len(symbols)} 个符合条件的币种")
            
            # 使用线程池并行处理
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_symbol = {
                    executor.submit(self._run_single_backtest_with_retry, symbol, limit): symbol 
                    for symbol in symbols
                }
                
                # 收集结果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        report = future.result()
                        if report:
                            results.append(report)
                        completed += 1
                        logger.info(f"进度: {completed}/{len(symbols)} - {symbol} 完成")
                    except Exception as e:
                        logger.error(f"{symbol}: 回测失败: {e}")
            
            # 按收益率排序
            results.sort(key=lambda x: x['total_return_pct'], reverse=True)
            
            self.batch_results = results
            return results
            
        except Exception as e:
            logger.error(f"批量回测过程中出错: {e}")
            return []


def print_batch_report(results: list, config: dict):
    """打印批量回测报告"""
    logger.info("\n" + "=" * 80)
    logger.info("高频短线策略批量快速回测报告")
    logger.info("=" * 80)
    
    # 打印策略参数
    logger.info("策略参数:")
    logger.info(f"  K线周期: {config.get('bar', '1m')}")
    logger.info(f"  连续K线: {config.get('consecutive_bars', 2)}")
    logger.info(f"  ATR周期: {config.get('atr_period', 14)}")
    logger.info(f"  ATR阈值: {config.get('atr_threshold', 0.8)}")
    logger.info(f"  移动止损: {config.get('trailing_stop_pct', 0.8)}%")
    logger.info(f"  成交量倍数: {config.get('volume_factor', 1.2)}")
    logger.info(f"  使用成交量: {'是' if config.get('use_volume', True) else '否'}")
    logger.info(f"  突破止损K线: {config.get('breakout_stop_bars', 2)}")
    
    logger.info("-" * 80)
    logger.info("批量回测结果排行榜 (按收益率排序):")
    logger.info("-" * 80)
    
    if not results:
        logger.info("  没有有效的回测结果")
        return
    
    # 打印表头
    logger.info(f"{'排名':<4} {'交易对':<12} {'收益率(%)':<10} {'胜率(%)':<8} {'盈亏比':<8} {'夏普比率':<10} {'最大回撤(%)':<12} {'交易次数':<8}")
    logger.info("-" * 80)
    
    # 打印结果
    for i, result in enumerate(results[:50], 1):  # 只显示前50名
        symbol = result['symbol']
        total_return = result['total_return_pct']
        win_rate = result['win_rate_pct']
        profit_factor = result['profit_factor']
        sharpe_ratio = result['sharpe_ratio']
        max_drawdown = result['max_drawdown_pct']
        total_trades = result['total_trades']
        
        # 格式化输出
        logger.info(f"{i:<4} {symbol:<12} {total_return:>9.2f} {win_rate:>7.1f} {profit_factor:>7.1f} {sharpe_ratio:>9.2f} {max_drawdown:>11.2f} {total_trades:>8}")
    
    # 统计信息
    logger.info("-" * 80)
    logger.info("统计信息:")
    logger.info(f"  总测试币种数: {len(results)}")
    logger.info(f"  平均收益率: {np.mean([r['total_return_pct'] for r in results]):.2f}%")
    logger.info(f"  最高收益率: {max([r['total_return_pct'] for r in results]):.2f}%")
    logger.info(f"  最低收益率: {min([r['total_return_pct'] for r in results]):.2f}%")
    logger.info(f"  正收益币种数: {len([r for r in results if r['total_return_pct'] > 0])}")
    logger.info(f"  负收益币种数: {len([r for r in results if r['total_return_pct'] < 0])}")
    
    # 详细性能分析
    logger.info("-" * 80)
    logger.info("详细性能分析:")
    
    # 计算详细统计
    valid_results = [r for r in results if r.get('close_trades_count', 0) > 0]
    if valid_results:
        # 交易类型统计
        avg_long_trades = np.mean([r.get('long_trades_count', 0) for r in valid_results])
        avg_short_trades = np.mean([r.get('short_trades_count', 0) for r in valid_results])
        avg_long_win_rate = np.mean([r.get('long_win_rate', 0) for r in valid_results])
        avg_short_win_rate = np.mean([r.get('short_win_rate', 0) for r in valid_results])
        avg_long_return = np.mean([r.get('long_avg_return', 0) for r in valid_results])
        avg_short_return = np.mean([r.get('short_avg_return', 0) for r in valid_results])
        
        # 止损原因统计
        avg_trailing_stop_count = np.mean([r.get('trailing_stop_count', 0) for r in valid_results])
        avg_breakout_stop_count = np.mean([r.get('breakout_stop_count', 0) for r in valid_results])
        avg_reverse_signal_count = np.mean([r.get('reverse_signal_count', 0) for r in valid_results])
        avg_trailing_stop_ratio = np.mean([r.get('trailing_stop_ratio', 0) for r in valid_results])
        avg_breakout_stop_ratio = np.mean([r.get('breakout_stop_ratio', 0) for r in valid_results])
        avg_reverse_signal_ratio = np.mean([r.get('reverse_signal_ratio', 0) for r in valid_results])
        
        # 止损胜率统计
        avg_trailing_stop_win_rate = np.mean([r.get('trailing_stop_win_rate', 0) for r in valid_results])
        avg_breakout_stop_win_rate = np.mean([r.get('breakout_stop_win_rate', 0) for r in valid_results])
        avg_reverse_signal_win_rate = np.mean([r.get('reverse_signal_win_rate', 0) for r in valid_results])
        
        # 亏损原因分析
        avg_loss_trailing_stop_ratio = np.mean([r.get('loss_trailing_stop_ratio', 0) for r in valid_results])
        avg_loss_breakout_ratio = np.mean([r.get('loss_breakout_ratio', 0) for r in valid_results])
        avg_loss_reverse_signal_ratio = np.mean([r.get('loss_reverse_signal_ratio', 0) for r in valid_results])
        
        # 条件统计
        avg_atr_condition_count = np.mean([r.get('atr_condition_count', 0) for r in valid_results])
        avg_volume_condition_count = np.mean([r.get('volume_condition_count', 0) for r in valid_results])
        avg_atr_condition_win_rate = np.mean([r.get('atr_condition_win_rate', 0) for r in valid_results])
        avg_volume_condition_win_rate = np.mean([r.get('volume_condition_win_rate', 0) for r in valid_results])
        
        # 持仓时间统计
        avg_holding_bars = np.mean([r.get('avg_holding_bars', 0) for r in valid_results])
        max_holding_bars = np.max([r.get('max_holding_bars', 0) for r in valid_results])
        min_holding_bars = np.min([r.get('min_holding_bars', 0) for r in valid_results])
        
        # 亏损分析
        avg_loss_amount = np.mean([r.get('avg_loss_amount', 0) for r in valid_results])
        max_loss_amount = np.max([r.get('max_loss_amount', 0) for r in valid_results])
        avg_loss_trades_count = np.mean([r.get('loss_trades_count', 0) for r in valid_results])
        avg_profit_trades_count = np.mean([r.get('profit_trades_count', 0) for r in valid_results])
        
        # 止损收益贡献分析
        avg_trailing_stop_return_pct = np.mean([r.get('trailing_stop_return_pct', 0) for r in valid_results])
        avg_breakout_stop_return_pct = np.mean([r.get('breakout_stop_return_pct', 0) for r in valid_results])
        avg_reverse_signal_return_pct = np.mean([r.get('reverse_signal_return_pct', 0) for r in valid_results])
        avg_trailing_stop_return_ratio = np.mean([r.get('trailing_stop_return_ratio', 0) for r in valid_results])
        avg_breakout_stop_return_ratio = np.mean([r.get('breakout_stop_return_ratio', 0) for r in valid_results])
        avg_reverse_signal_return_ratio = np.mean([r.get('reverse_signal_return_ratio', 0) for r in valid_results])
        
        # 各种止损导致最终收益的百分比
        avg_trailing_stop_final_return_pct = np.mean([r.get('trailing_stop_final_return_pct', 0) for r in valid_results])
        avg_breakout_stop_final_return_pct = np.mean([r.get('breakout_stop_final_return_pct', 0) for r in valid_results])
        avg_reverse_signal_final_return_pct = np.mean([r.get('reverse_signal_final_return_pct', 0) for r in valid_results])
        
        # 各种止损方式的平均收益率
        avg_trailing_stop_avg_return = np.mean([r.get('trailing_stop_avg_return', 0) for r in valid_results])
        avg_breakout_stop_avg_return = np.mean([r.get('breakout_stop_avg_return', 0) for r in valid_results])
        avg_reverse_signal_avg_return = np.mean([r.get('reverse_signal_avg_return', 0) for r in valid_results])
        
        logger.info(f"  平均多仓交易数: {avg_long_trades:.1f}")
        logger.info(f"  平均空仓交易数: {avg_short_trades:.1f}")
        logger.info(f"  平均多仓胜率: {avg_long_win_rate:.1f}%")
        logger.info(f"  平均空仓胜率: {avg_short_win_rate:.1f}%")
        logger.info(f"  平均多仓收益率: {avg_long_return:.2f}%")
        logger.info(f"  平均空仓收益率: {avg_short_return:.2f}%")
        logger.info(f"  平均持仓时间: {avg_holding_bars:.1f} 根K线")
        logger.info(f"  最长持仓时间: {max_holding_bars:.1f} 根K线")
        logger.info(f"  最短持仓时间: {min_holding_bars:.1f} 根K线")
        
        logger.info("-" * 80)
        logger.info("止损原因分析:")
        logger.info(f"  移动止损次数: {avg_trailing_stop_count:.1f} ({avg_trailing_stop_ratio:.1f}%)")
        logger.info(f"  突破止损次数: {avg_breakout_stop_count:.1f} ({avg_breakout_stop_ratio:.1f}%)")
        logger.info(f"  反向信号止损: {avg_reverse_signal_count:.1f} ({avg_reverse_signal_ratio:.1f}%)")
        logger.info(f"  移动止损胜率: {avg_trailing_stop_win_rate:.1f}%")
        logger.info(f"  突破止损胜率: {avg_breakout_stop_win_rate:.1f}%")
        logger.info(f"  反向信号胜率: {avg_reverse_signal_win_rate:.1f}%")
        
        logger.info("-" * 80)
        logger.info("亏损原因分析:")
        logger.info(f"  移动止损导致亏损: {avg_loss_trailing_stop_ratio:.1f}%")
        logger.info(f"  突破止损导致亏损: {avg_loss_breakout_ratio:.1f}%")
        logger.info(f"  反向信号导致亏损: {avg_loss_reverse_signal_ratio:.1f}%")
        logger.info(f"  平均亏损金额: {avg_loss_amount:.2f}%")
        logger.info(f"  最大亏损金额: {max_loss_amount:.2f}%")
        logger.info(f"  平均亏损交易数: {avg_loss_trades_count:.1f}")
        logger.info(f"  平均盈利交易数: {avg_profit_trades_count:.1f}")
        
        logger.info("-" * 80)
        logger.info("条件有效性分析:")
        logger.info(f"  ATR条件触发次数: {avg_atr_condition_count:.1f}")
        logger.info(f"  成交量条件触发次数: {avg_volume_condition_count:.1f}")
        logger.info(f"  ATR条件胜率: {avg_atr_condition_win_rate:.1f}%")
        logger.info(f"  成交量条件胜率: {avg_volume_condition_win_rate:.1f}%")
        
        logger.info("-" * 80)
        logger.info("止损收益贡献分析:")
        logger.info(f"  移动止损贡献收益: {avg_trailing_stop_return_pct:.2f}% ({avg_trailing_stop_return_ratio:.1f}%)")
        logger.info(f"  突破止损贡献收益: {avg_breakout_stop_return_pct:.2f}% ({avg_breakout_stop_return_ratio:.1f}%)")
        logger.info(f"  反向信号贡献收益: {avg_reverse_signal_return_pct:.2f}% ({avg_reverse_signal_return_ratio:.1f}%)")
        
        logger.info("-" * 80)
        logger.info("各种止损方式的平均收益率:")
        logger.info(f"  移动止损平均收益率: {avg_trailing_stop_avg_return:.2f}%")
        logger.info(f"  突破止损平均收益率: {avg_breakout_stop_avg_return:.2f}%")
        logger.info(f"  反向信号平均收益率: {avg_reverse_signal_avg_return:.2f}%")
    
    # 优化建议
    logger.info("-" * 80)
    logger.info("优化建议:")
    
    if valid_results:
        # 基于统计数据的优化建议
        avg_total_return = np.mean([r['total_return_pct'] for r in valid_results])
        avg_win_rate = np.mean([r['win_rate_pct'] for r in valid_results])
        avg_max_drawdown = np.mean([r['max_drawdown_pct'] for r in valid_results])
        
        if avg_total_return < 0:
            logger.info("  ⚠️  策略整体亏损，建议:")
            logger.info("     - 检查连续K线数量是否合适")
            logger.info("     - 调整ATR阈值过滤条件")
            logger.info("     - 考虑增加移动止损比例")
        elif avg_win_rate < 50:
            logger.info("  ⚠️  胜率较低但可能盈利，建议:")
            logger.info("     - 关注盈亏比而非胜率")
            logger.info("     - 检查止损设置是否过于严格")
        elif avg_max_drawdown < -10:
            logger.info("  ⚠️  回撤较大，建议:")
            logger.info("     - 增加移动止损比例")
            logger.info("     - 减少连续K线数量以降低风险")
        else:
            logger.info("  ✅  策略表现良好，可以:")
            logger.info("     - 考虑实盘测试")
            logger.info("     - 进一步优化参数提升收益")
        
        # 基于止损分析的优化建议
        if avg_trailing_stop_count > avg_breakout_stop_count:
            logger.info("  📊  移动止损触发较多，建议:")
            logger.info("     - 适当降低移动止损比例")
            logger.info("     - 检查是否持仓时间过短")
        elif avg_breakout_stop_count > avg_trailing_stop_count:
            logger.info("  📊  突破止损触发较多，建议:")
            logger.info("     - 增加连续K线数量以提高信号质量")
            logger.info("     - 检查ATR阈值是否合适")
        
        # 基于交易方向的优化建议
        if avg_long_return > avg_short_return:
            logger.info("  📈  多仓表现优于空仓，建议:")
            logger.info("     - 考虑增加多仓权重")
            logger.info("     - 优化空仓入场条件")
        elif avg_short_return > avg_long_return:
            logger.info("  📉  空仓表现优于多仓，建议:")
            logger.info("     - 考虑增加空仓权重")
            logger.info("     - 优化多仓入场条件")
        
        # 基于条件有效性的优化建议
        if avg_atr_condition_win_rate > avg_volume_condition_win_rate:
            logger.info("  🔍  ATR条件有效性更高，建议:")
            logger.info("     - 保持或加强ATR过滤")
            logger.info("     - 考虑优化成交量条件参数")
        elif avg_volume_condition_win_rate > avg_atr_condition_win_rate:
            logger.info("  🔍  成交量条件有效性更高，建议:")
            logger.info("     - 保持或加强成交量过滤")
            logger.info("     - 考虑优化ATR条件参数")
        
        # 基于亏损原因的优化建议
        if avg_loss_trailing_stop_ratio > 50:
            logger.info("  💔  移动止损是主要亏损来源，建议:")
            logger.info("     - 降低移动止损比例")
            logger.info("     - 增加持仓时间")
        elif avg_loss_breakout_ratio > 50:
            logger.info("  💔  突破止损是主要亏损来源，建议:")
            logger.info("     - 增加连续K线数量")
            logger.info("     - 调整ATR阈值")
        elif avg_loss_reverse_signal_ratio > 50:
            logger.info("  💔  反向信号是主要亏损来源，建议:")
            logger.info("     - 减少反向开仓频率")
            logger.info("     - 增加信号确认条件")
    
    logger.info("=" * 80 + "\n")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='高频短线策略批量快速回测系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    parser.add_argument('--limit', type=int, help='回测数据量', default=300)
    parser.add_argument('--min_vol', type=float, help='最小交易量(USDT)', default=20000000)
    parser.add_argument('--workers', type=int, help='并行工作线程数', default=5)
    parser.add_argument('--top_n', type=int, help='只测试前N个高交易量币种', default=None)
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        # 加载用户指定的配置文件作为默认值
        default_config = load_config_from_file(args.config)
        if not default_config:
            logger.error("配置文件加载失败，使用系统默认值")
            default_config = {}
    else:
        # 加载默认配置文件作为用户输入的默认值
        config_path = os.path.join(os.path.dirname(__file__), 'configs/btc_usdt_swap.json')
        default_config = load_config_from_file(config_path)
        if not default_config:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 咨询用户输入
    logger.info("高频短线策略批量快速回测系统")
    logger.info("=" * 50)
    logger.info("注意：此系统将自动扫描高交易量币种并批量回测")
    logger.info("=" * 50)
    
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 设置参数
    bar = config.get('bar', '1m')
    consecutive_bars = config.get('consecutive_bars', 2)
    atr_period = config.get('atr_period', 14)
    atr_threshold = config.get('atr_threshold', 0.8)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.8)
    use_volume = config.get('use_volume', True)
    volume_factor = config.get('volume_factor', 1.2)
    breakout_stop_bars = config.get('breakout_stop_bars', 2)
    
    # 创建批量回测实例
    batch_backtest = BatchFastBacktest(
        bar=bar,
        consecutive_bars=consecutive_bars,
        atr_period=atr_period,
        atr_threshold=atr_threshold,
        trailing_stop_pct=trailing_stop_pct,
        volume_factor=volume_factor,
        use_volume=use_volume,
        breakout_stop_bars=breakout_stop_bars
    )
    
    # 运行批量回测
    results = batch_backtest.run_batch_backtest(min_vol_ccy=args.min_vol, limit=args.limit)
    
    if results:
        # 打印批量报告
        print_batch_report(results, config)
        
        # 保存结果到Excel
        save_batch_results_to_excel(results, config)
    else:
        logger.error("批量回测失败，请检查配置和数据")


def save_batch_results_to_excel(results: list, config: dict, output_dir: str = "backtest_results"):
    """保存批量回测结果到Excel文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用北京时间
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    
    # Excel文件名
    excel_filename = f"{output_dir}/batch_backtest_report_{timestamp}.xlsx"
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 创建Excel写入器
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Sheet 1: 回测结果
        df.to_excel(writer, sheet_name='回测结果', index=False)
        
        # Sheet 2: 策略参数
        param_data = {
            '参数': [
                'K线周期', '连续K线', 'ATR周期', 'ATR阈值', 
                '移动止损(%)', '成交量倍数', '使用成交量', '突破止损K线'
            ],
            '数值': [
                config.get('bar', '1m'), config.get('consecutive_bars', 2), config.get('atr_period', 14),
                config.get('atr_threshold', 0.8), config.get('trailing_stop_pct', 0.8), 
                config.get('volume_factor', 1.2), '是' if config.get('use_volume', True) else '否',
                config.get('breakout_stop_bars', 2)
            ]
        }
        param_df = pd.DataFrame(param_data)
        param_df.to_excel(writer, sheet_name='策略参数', index=False)
        
        # Sheet 3: 统计信息
        if len(results) > 0:
            valid_results = [r for r in results if r.get('close_trades_count', 0) > 0]
            if valid_results:
                stats_data = {
                    '统计指标': [
                        '总测试币种数', '平均收益率(%)', '最高收益率(%)', '最低收益率(%)',
                        '正收益币种数', '负收益币种数', '平均胜率(%)', '平均盈亏比',
                        '平均夏普比率', '平均最大回撤(%)',
                        '平均多仓交易数', '平均空仓交易数', '平均多仓胜率(%)', '平均空仓胜率(%)',
                        '平均多仓收益率(%)', '平均空仓收益率(%)',
                        '平均持仓时间(K线)', '最长持仓时间(K线)', '最短持仓时间(K线)',
                        '移动止损次数', '突破止损次数', '反向信号止损次数',
                        '移动止损占比(%)', '突破止损占比(%)', '反向信号止损占比(%)',
                        '移动止损胜率(%)', '突破止损胜率(%)', '反向信号胜率(%)',
                        '移动止损亏损占比(%)', '突破止损亏损占比(%)', '反向信号亏损占比(%)',
                        'ATR条件触发次数', '成交量条件触发次数',
                        'ATR条件胜率(%)', '成交量条件胜率(%)',
                        '平均亏损金额(%)', '最大亏损金额(%)',
                        '平均亏损交易数', '平均盈利交易数',
                        '移动止损平均收益率(%)', '突破止损平均收益率(%)', '反向信号平均收益率(%)'
                    ],
                    '数值': [
                        len(results),
                        np.mean([r['total_return_pct'] for r in results]),
                        max([r['total_return_pct'] for r in results]),
                        min([r['total_return_pct'] for r in results]),
                        len([r for r in results if r['total_return_pct'] > 0]),
                        len([r for r in results if r['total_return_pct'] < 0]),
                        np.mean([r['win_rate_pct'] for r in results]),
                        np.mean([r['profit_factor'] for r in results]),
                        np.mean([r['sharpe_ratio'] for r in results]),
                        np.mean([r['max_drawdown_pct'] for r in results]),
                        np.mean([r.get('long_trades_count', 0) for r in valid_results]),
                        np.mean([r.get('short_trades_count', 0) for r in valid_results]),
                        np.mean([r.get('long_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('short_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('long_avg_return', 0) for r in valid_results]),
                        np.mean([r.get('short_avg_return', 0) for r in valid_results]),
                        np.mean([r.get('avg_holding_bars', 0) for r in valid_results]),
                        np.max([r.get('max_holding_bars', 0) for r in valid_results]),
                        np.min([r.get('min_holding_bars', 0) for r in valid_results]),
                        np.mean([r.get('trailing_stop_count', 0) for r in valid_results]),
                        np.mean([r.get('breakout_stop_count', 0) for r in valid_results]),
                        np.mean([r.get('reverse_signal_count', 0) for r in valid_results]),
                        np.mean([r.get('trailing_stop_ratio', 0) for r in valid_results]),
                        np.mean([r.get('breakout_stop_ratio', 0) for r in valid_results]),
                        np.mean([r.get('reverse_signal_ratio', 0) for r in valid_results]),
                        np.mean([r.get('trailing_stop_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('breakout_stop_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('reverse_signal_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('loss_trailing_stop_ratio', 0) for r in valid_results]),
                        np.mean([r.get('loss_breakout_ratio', 0) for r in valid_results]),
                        np.mean([r.get('loss_reverse_signal_ratio', 0) for r in valid_results]),
                        np.mean([r.get('atr_condition_count', 0) for r in valid_results]),
                        np.mean([r.get('volume_condition_count', 0) for r in valid_results]),
                        np.mean([r.get('atr_condition_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('volume_condition_win_rate', 0) for r in valid_results]),
                        np.mean([r.get('avg_loss_amount', 0) for r in valid_results]),
                        np.max([r.get('max_loss_amount', 0) for r in valid_results]),
                        np.mean([r.get('loss_trades_count', 0) for r in valid_results]),
                        np.mean([r.get('profit_trades_count', 0) for r in valid_results]),
                        np.mean([r.get('trailing_stop_avg_return', 0) for r in valid_results]),
                        np.mean([r.get('breakout_stop_avg_return', 0) for r in valid_results]),
                        np.mean([r.get('reverse_signal_avg_return', 0) for r in valid_results])
                    ]
                }
            else:
                stats_data = {
                    '统计指标': [
                        '总测试币种数', '平均收益率(%)', '最高收益率(%)', '最低收益率(%)',
                        '正收益币种数', '负收益币种数', '平均胜率(%)', '平均盈亏比',
                        '平均夏普比率', '平均最大回撤(%)'
                    ],
                    '数值': [
                        len(results),
                        np.mean([r['total_return_pct'] for r in results]),
                        max([r['total_return_pct'] for r in results]),
                        min([r['total_return_pct'] for r in results]),
                        len([r for r in results if r['total_return_pct'] > 0]),
                        len([r for r in results if r['total_return_pct'] < 0]),
                        np.mean([r['win_rate_pct'] for r in results]),
                        np.mean([r['profit_factor'] for r in results]),
                        np.mean([r['sharpe_ratio'] for r in results]),
                        np.mean([r['max_drawdown_pct'] for r in results])
                    ]
                }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
    
    logger.info(f"批量回测报告已保存到: {excel_filename}")
    return excel_filename


if __name__ == "__main__":
    main()