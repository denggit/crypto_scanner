#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/23/25  
@File       : strategy_3_fast_backtest_all.py
@Description: Batch fast backtest for Long Shadow Strategy across multiple symbols - 真正的快速回测
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json
import concurrent.futures
from typing import List, Dict, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_3.strategy_3 import LongShadowStrategy
from strategies.strategy_3.shared_config import load_config_from_file, get_user_input, print_final_config
from tools.market_scanner import CryptoScanner
from utils.logger import logger


class BatchFastBacktest:
    """批量快速回测类 - 真正的快速回测实现"""
    
    def __init__(self, bar: str = '1m',
                 min_volume_ccy: float = 1000000, volume_factor: float = 1.2,
                 trailing_stop_pct: float = 0.0, take_profit_pct: float = 0.0,
                 use_volume: bool = True, trade_amount: float = 10.0,
                 max_workers: int = 5):
        """
        Initialize Batch Fast Backtest
        
        Args:
            bar: K-line time interval
            min_volume_ccy: Minimum 24h volume in USDT
            volume_factor: Volume multiplier
            trailing_stop_pct: Trailing stop percentage
            take_profit_pct: Take profit percentage
            use_volume: Whether to use volume condition
            trade_amount: USDT amount for each trade
            max_workers: Maximum number of parallel workers
        """
        self.bar = bar
        self.min_volume_ccy = min_volume_ccy
        self.volume_factor = volume_factor
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.use_volume = use_volume
        self.trade_amount = trade_amount
        self.max_workers = max_workers
        
        # Initialize components
        self.client = OKXClient()
        self.market_data_retriever = MarketDataRetriever(self.client)
        self.strategy = LongShadowStrategy(self.client)
        
        # Cache for performance
        self._cached_data = {}
        
        # Results storage
        self.all_results = {}
    
    def _calculate_signals_in_bulk(self, df: pd.DataFrame, symbol: str) -> Tuple[List[int], List[Dict]]:
        """
        批量计算所有K线的信号和详细信息 - 真正的快速回测
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (signals_list, details_list)
        """
        signals = []
        details_list = []
        
        # 获取价格和成交量数据
        closes = df['c'] if 'c' in df.columns else df['close']
        opens = df['o'] if 'o' in df.columns else df['open']
        highs = df['h'] if 'h' in df.columns else df['high']
        lows = df['l'] if 'l' in df.columns else df['low']
        volumes = df['vol'] if 'vol' in df.columns else df['volume']
        
        # 批量计算成交量条件
        volume_expansions = []
        volume_ratios = []
        
        for i in range(len(volumes)):
            if i < 11:  # 前10根K线平均成交量
                volume_expansions.append(False)
                volume_ratios.append(0)
                continue
            
            current_volume = volumes.iloc[i]
            avg_volume = volumes.iloc[i-10:i].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_ratios.append(volume_ratio)
            volume_expansions.append(volume_ratio >= self.volume_factor)
        
        # 批量计算所有信号
        for i in range(len(df)):
            if i < 2:  # 需要至少2根K线用于计算
                signals.append(0)
                details_list.append({})
                continue
                
            try:
                # 获取当前和之前K线数据
                current_open = opens.iloc[i]
                current_high = highs.iloc[i]
                current_low = lows.iloc[i]
                current_close = closes.iloc[i]
                current_volume = volumes.iloc[i]
                
                prev2_open = opens.iloc[i-2]
                prev2_high = highs.iloc[i-2]
                prev2_low = lows.iloc[i-2]
                prev2_close = closes.iloc[i-2]
                prev2_volume = volumes.iloc[i-2]
                
                # 计算成交量条件
                volume_condition_met = False
                if self.use_volume and i >= 11:
                    volume_condition_met = volume_expansions[i-2] if i-2 < len(volume_expansions) else False
                
                # 检查长下影线条件
                candle_length = prev2_high - prev2_low
                long_shadow_condition = False
                if candle_length > 0:
                    lower_shadow = min(prev2_open, prev2_close) - prev2_low
                    lower_shadow_ratio = lower_shadow / candle_length
                    long_shadow_condition = lower_shadow_ratio > 1/2
                
                # 检查长上影线条件
                short_shadow_condition = False
                if candle_length > 0:
                    upper_shadow = prev2_high - max(prev2_open, prev2_close)
                    upper_shadow_ratio = upper_shadow / candle_length
                    short_shadow_condition = upper_shadow_ratio > 1/2
                
                # 检查入场条件
                current_candle_length = current_high - current_low
                
                long_entry_condition = False
                if current_candle_length > 0:
                    long_entry_condition = (current_close > current_open and 
                                          (current_high - current_close) / current_candle_length < 1/5)
                
                short_entry_condition = False
                if current_candle_length > 0:
                    short_entry_condition = (current_close < current_open and 
                                           (current_open - current_low) / current_candle_length < 1/5)
                
                # 确定信号
                signal = 0
                if long_shadow_condition and long_entry_condition:
                    if not self.use_volume or volume_condition_met:
                        signal = 1
                elif short_shadow_condition and short_entry_condition:
                    if not self.use_volume or volume_condition_met:
                        signal = -1
                
                signals.append(signal)
                
                # 保存详细信息
                details_list.append({
                    'current_price': float(current_close),
                    'current_open': float(current_open),
                    'current_high': float(current_high),
                    'current_low': float(current_low),
                    'prev2_open': float(prev2_open),
                    'prev2_high': float(prev2_high),
                    'prev2_low': float(prev2_low),
                    'prev2_close': float(prev2_close),
                    'volume_condition_met': volume_condition_met,
                    'long_shadow_condition': long_shadow_condition,
                    'short_shadow_condition': short_shadow_condition,
                    'long_entry_condition': long_entry_condition,
                    'short_entry_condition': short_entry_condition,
                    'current_volume': float(current_volume),
                    'prev2_volume': float(prev2_volume),
                    'signal': signal
                })
                
            except Exception as e:
                signals.append(0)
                details_list.append({})
        
        return signals, details_list
    
    def _run_single_backtest(self, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """
        运行单个交易对的快速回测
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Tuple of (symbol, results_dict)
        """
        try:
            logger.info(f"开始快速回测 {symbol}...")
            
            # 获取历史数据
            df = self.market_data_retriever.get_kline(symbol, self.bar, 1000)
            if df is None or len(df) < 12:
                logger.warning(f"{symbol} 数据不足，跳过")
                return symbol, {}
            
            # 批量计算信号
            signals, details_list = self._calculate_signals_in_bulk(df, symbol)
            
            # 执行回测
            position = 0
            entry_price = 0.0
            highest_price = 0.0
            lowest_price = 0.0
            trade_count = 0
            total_return = 0.0
            trades = []
            
            for i, (signal, details) in enumerate(zip(signals, details_list)):
                if not details:
                    continue
                    
                price = details['current_price']
                timestamp = df.index[i] if hasattr(df.index[i], 'strftime') else str(i)
                
                # 执行交易逻辑
                action = "HOLD"
                exit_price = 0.0
                return_rate = 0.0
                
                # 检查止损止盈
                trailing_stop_triggered = False
                take_profit_triggered = False
                
                if position == 1:  # 多仓
                    if price > highest_price:
                        highest_price = price
                    if self.trailing_stop_pct > 0:
                        stop_price = highest_price * (1 - self.trailing_stop_pct / 100.0)
                        trailing_stop_triggered = price <= stop_price
                    if self.take_profit_pct > 0:
                        take_profit_price = entry_price * (1 + self.take_profit_pct / 100.0)
                        take_profit_triggered = price >= take_profit_price
                elif position == -1:  # 空仓
                    if price < lowest_price or lowest_price == 0:
                        lowest_price = price
                    if self.trailing_stop_pct > 0:
                        stop_price = lowest_price * (1 + self.trailing_stop_pct / 100.0)
                        trailing_stop_triggered = price >= stop_price
                    if self.take_profit_pct > 0:
                        take_profit_price = entry_price * (1 - self.take_profit_pct / 100.0)
                        take_profit_triggered = price <= take_profit_price
                
                # 交易执行逻辑
                if position == 0:
                    if signal == 1:
                        position = 1
                        entry_price = price
                        highest_price = price
                        lowest_price = 0.0
                        action = "LONG_OPEN"
                        trade_count += 1
                    elif signal == -1:
                        position = -1
                        entry_price = price
                        lowest_price = price
                        highest_price = 0.0
                        action = "SHORT_OPEN"
                        trade_count += 1
                else:
                    if position == 1:
                        if take_profit_triggered:
                            exit_price = price
                            return_rate = (exit_price - entry_price) / entry_price
                            action = "LONG_CLOSE_TAKE_PROFIT"
                            position = 0
                            highest_price = 0.0
                            trade_count += 1
                        elif trailing_stop_triggered:
                            exit_price = price
                            return_rate = (exit_price - entry_price) / entry_price
                            action = "LONG_CLOSE_TRAILING_STOP"
                            position = 0
                            highest_price = 0.0
                            trade_count += 1
                        elif signal == -1:
                            exit_price = price
                            return_rate = (exit_price - entry_price) / entry_price
                            action = "LONG_CLOSE_SHORT_OPEN"
                            position = -1
                            entry_price = price
                            highest_price = 0.0
                            lowest_price = price
                            trade_count += 1
                        elif signal == 0:  # 策略平多信号
                            exit_price = price
                            return_rate = (exit_price - entry_price) / entry_price
                            action = "LONG_CLOSE_STRATEGY"
                            position = 0
                            highest_price = 0.0
                            trade_count += 1
                    elif position == -1:
                        if take_profit_triggered:
                            exit_price = price
                            return_rate = (entry_price - exit_price) / entry_price
                            action = "SHORT_CLOSE_TAKE_PROFIT"
                            position = 0
                            lowest_price = 0.0
                            trade_count += 1
                        elif trailing_stop_triggered:
                            exit_price = price
                            return_rate = (entry_price - exit_price) / entry_price
                            action = "SHORT_CLOSE_TRAILING_STOP"
                            position = 0
                            lowest_price = 0.0
                            trade_count += 1
                        elif signal == 1:
                            exit_price = price
                            return_rate = (entry_price - exit_price) / entry_price
                            action = "SHORT_CLOSE_LONG_OPEN"
                            position = 1
                            entry_price = price
                            lowest_price = 0.0
                            highest_price = price
                            trade_count += 1
                        elif signal == 0:  # 策略平空信号
                            exit_price = price
                            return_rate = (entry_price - exit_price) / entry_price
                            action = "SHORT_CLOSE_STRATEGY"
                            position = 0
                            lowest_price = 0.0
                            trade_count += 1
                
                if action != "HOLD":
                    trade_record = {
                        'timestamp': str(timestamp),
                        'symbol': symbol,
                        'price': price,
                        'action': action,
                        'position': position,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return_rate': return_rate,
                        'trade_count': trade_count
                    }
                    trades.append(trade_record)
                    total_return += return_rate
            
            # 计算性能指标
            performance = self._calculate_performance_metrics(symbol, trades, total_return, trade_count)
            
            logger.info(f"{symbol} 快速回测完成: {trade_count} 次交易, "
                       f"收益率: {performance['total_return_pct']:.2f}%, "
                       f"胜率: {performance['win_rate']:.2f}%")
            
            return symbol, performance
            
        except Exception as e:
            logger.error(f"回测 {symbol} 时出错: {e}")
            return symbol, {}
    
    def _calculate_performance_metrics(self, symbol: str, trades: List[Dict], 
                                     total_return: float, trade_count: int) -> Dict[str, Any]:
        """计算性能指标"""
        if not trades:
            return {}
        
        # 计算基本指标
        total_return_pct = total_return * 100
        
        # 计算胜率
        winning_trades = [trade for trade in trades if trade['return_rate'] > 0]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # 计算平均盈亏
        if winning_trades:
            avg_win = sum(trade['return_rate'] for trade in winning_trades) / len(winning_trades) * 100
        else:
            avg_win = 0
            
        losing_trades = [trade for trade in trades if trade['return_rate'] < 0]
        if losing_trades:
            avg_loss = sum(trade['return_rate'] for trade in losing_trades) / len(losing_trades) * 100
        else:
            avg_loss = 0
        
        # 计算盈利因子
        total_wins = sum(trade['return_rate'] for trade in winning_trades)
        total_losses = abs(sum(trade['return_rate'] for trade in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # 计算最大回撤
        cumulative_returns = []
        current_return = 0.0
        for trade in trades:
            current_return += trade['return_rate']
            cumulative_returns.append(current_return)
        
        if cumulative_returns:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            'symbol': symbol,
            'total_trades': trade_count,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'trades': trades
        }
    
    def run_batch_backtest(self, symbols: List[str]) -> Dict[str, Any]:
        """
        运行批量快速回测
        
        Args:
            symbols: List of trading pair symbols
            
        Returns:
            Dict with aggregated results
        """
        logger.info(f"开始批量快速回测 {len(symbols)} 个交易对...")
        logger.info(f"参数: 最小24小时交易量={self.min_volume_ccy}, "
                   f"成交量倍数={self.volume_factor}, "
                   f"移动止损={self.trailing_stop_pct}%, "
                   f"止盈={self.take_profit_pct}%")
        
        start_time = datetime.now()
        
        # 并行运行回测
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_symbol = {
                executor.submit(self._run_single_backtest, symbol): symbol 
                for symbol in symbols
            }
            
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, results = future.result()
                    self.all_results[symbol] = results
                except Exception as e:
                    logger.error(f"处理 {symbol} 结果时出错: {e}")
                    self.all_results[symbol] = {}
        
        # 计算聚合统计
        aggregated_results = self._calculate_aggregated_statistics()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"批量快速回测完成! 耗时: {duration:.2f} 秒")
        logger.info(f"成功回测: {aggregated_results['successful_symbols']} 个交易对")
        logger.info(f"平均收益率: {aggregated_results['avg_return_pct']:.2f}%")
        logger.info(f"平均胜率: {aggregated_results['avg_win_rate']:.2f}%")
        
        return aggregated_results
    
    def _calculate_aggregated_statistics(self) -> Dict[str, Any]:
        """计算聚合统计"""
        successful_results = {}
        for symbol, results in self.all_results.items():
            if results and 'total_trades' in results and results['total_trades'] > 0:
                successful_results[symbol] = results
        
        if not successful_results:
            return {
                'successful_symbols': 0,
                'avg_return_pct': 0,
                'avg_win_rate': 0,
                'avg_trades': 0,
                'best_symbol': '',
                'best_return': 0,
                'worst_symbol': '',
                'worst_return': 0
            }
        
        # 计算平均值
        total_return_pct = [r['total_return_pct'] for r in successful_results.values()]
        win_rates = [r['win_rate'] for r in successful_results.values()]
        trade_counts = [r['total_trades'] for r in successful_results.values()]
        
        avg_return_pct = np.mean(total_return_pct)
        avg_win_rate = np.mean(win_rates)
        avg_trades = np.mean(trade_counts)
        
        # 找到最佳和最差表现者
        best_symbol = max(successful_results.keys(), 
                         key=lambda s: successful_results[s]['total_return_pct'])
        worst_symbol = min(successful_results.keys(), 
                          key=lambda s: successful_results[s]['total_return_pct'])
        
        best_return = successful_results[best_symbol]['total_return_pct']
        worst_return = successful_results[worst_symbol]['total_return_pct']
        
        # 计算Sharpe比率（简化版）
        returns_std = np.std(total_return_pct)
        sharpe_ratio = avg_return_pct / returns_std if returns_std > 0 else 0
        
        # 计算成功率（正收益）
        positive_returns = sum(1 for r in total_return_pct if r > 0)
        success_rate = positive_returns / len(total_return_pct) * 100
        
        return {
            'successful_symbols': len(successful_results),
            'total_symbols': len(self.all_results),
            'avg_return_pct': avg_return_pct,
            'avg_win_rate': avg_win_rate,
            'avg_trades': avg_trades,
            'best_symbol': best_symbol,
            'best_return': best_return,
            'worst_symbol': worst_symbol,
            'worst_return': worst_return,
            'sharpe_ratio': sharpe_ratio,
            'success_rate': success_rate,
            'returns_std': returns_std,
            'individual_results': successful_results
        }
    
    def save_results_to_excel(self, filename: str = None):
        """保存回测结果到Excel文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"strategy3_batch_fast_backtest_{timestamp}.xlsx"
        
        # 创建结果目录
        results_dir = "backtest_results"
        os.makedirs(results_dir, exist_ok=True)
        filepath = os.path.join(results_dir, filename)
        
        try:
            # 准备Excel数据
            summary_data = []
            detailed_data = []
            
            # 汇总表
            for symbol, results in self.all_results.items():
                if results and 'total_trades' in results:
                    summary_data.append({
                        'Symbol': symbol,
                        'Total Trades': results['total_trades'],
                        'Total Return (%)': results['total_return_pct'],
                        'Win Rate (%)': results['win_rate'],
                        'Avg Win (%)': results.get('avg_win_pct', 0),
                        'Avg Loss (%)': results.get('avg_loss_pct', 0),
                        'Profit Factor': results.get('profit_factor', 0),
                        'Max Drawdown (%)': results.get('max_drawdown_pct', 0)
                    })
            
            # 详细交易表
            for symbol, results in self.all_results.items():
                if results and 'trades' in results:
                    for trade in results['trades']:
                        detailed_data.append({
                            'Symbol': symbol,
                            'Timestamp': trade['timestamp'],
                            'Price': trade['price'],
                            'Action': trade['action'],
                            'Position': trade['position'],
                            'Entry Price': trade['entry_price'],
                            'Exit Price': trade['exit_price'],
                            'Return Rate (%)': trade['return_rate'] * 100
                        })
            
            # 创建Excel写入器
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # 汇总表
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # 详细交易表
                if detailed_data:
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_df.to_excel(writer, sheet_name='Detailed Trades', index=False)
                
                # 参数表
                params_data = [
                    ['Parameter', 'Value'],
                    ['Bar', self.bar],
                    ['Min Volume CCY', self.min_volume_ccy],
                    ['Volume Factor', self.volume_factor],
                    ['Trailing Stop (%)', self.trailing_stop_pct],
                    ['Take Profit (%)', self.take_profit_pct],
                    ['Use Volume', self.use_volume],
                    ['Trade Amount', self.trade_amount],
                    ['Total Symbols', len(self.all_results)],
                    ['Successful Symbols', len([r for r in self.all_results.values() if r])]
                ]
                params_df = pd.DataFrame(params_data[1:], columns=params_data[0])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            logger.info(f"结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存Excel文件时出错: {e}")


def print_batch_report(results: dict, config: dict):
    """打印批量回测报告"""
    logger.info("\n" + "=" * 80)
    logger.info("长下影线策略批量快速回测报告")
    logger.info("=" * 80)
    
    # 打印策略参数
    logger.info("策略参数:")
    logger.info(f"  K线周期: {config.get('bar', '1m')}")
    logger.info(f"  最小24小时交易量: {config.get('min_volume_ccy', 1000000):,.0f} USDT")
    logger.info(f"  成交量倍数: {config.get('volume_factor', 1.2)}")
    logger.info(f"  移动止损: {config.get('trailing_stop_pct', 0.0)}%")
    logger.info(f"  止盈: {config.get('take_profit_pct', 0.0)}%")
    logger.info(f"  使用成交量: {'是' if config.get('use_volume', True) else '否'}")
    
    logger.info("-" * 80)
    logger.info("批量回测结果汇总:")
    logger.info("-" * 80)
    
    if not results or 'individual_results' not in results:
        logger.info("  没有有效的回测结果")
        return
    
    individual_results = results['individual_results']
    
    # 打印表头
    logger.info(f"{'排名':<4} {'交易对':<12} {'收益率(%)':<10} {'胜率(%)':<8} {'盈亏比':<8} {'夏普比率':<10} {'最大回撤(%)':<12} {'交易次数':<8}")
    logger.info("-" * 80)
    
    # 按收益率排序
    sorted_results = sorted(individual_results.items(), 
                          key=lambda x: x[1]['total_return_pct'], reverse=True)
    
    # 打印结果
    for i, (symbol, result) in enumerate(sorted_results[:50], 1):  # 只显示前50名
        total_return = result['total_return_pct']
        win_rate = result['win_rate']
        profit_factor = result['profit_factor']
        sharpe_ratio = result.get('sharpe_ratio', 0)
        max_drawdown = result['max_drawdown_pct']
        total_trades = result['total_trades']
        
        # 格式化输出
        logger.info(f"{i:<4} {symbol:<12} {total_return:>9.2f} {win_rate:>7.1f} {profit_factor:>7.1f} {sharpe_ratio:>9.2f} {max_drawdown:>11.2f} {total_trades:>8}")
    
    # 统计信息
    logger.info("-" * 80)
    logger.info("统计信息:")
    logger.info(f"  总测试币种数: {results['total_symbols']}")
    logger.info(f"  成功回测币种数: {results['successful_symbols']}")
    logger.info(f"  平均收益率: {results['avg_return_pct']:.2f}%")
    logger.info(f"  最高收益率: {results['best_return']:.2f}% ({results['best_symbol']})")
    logger.info(f"  最低收益率: {results['worst_return']:.2f}% ({results['worst_symbol']})")
    logger.info(f"  平均胜率: {results['avg_win_rate']:.2f}%")
    logger.info(f"  平均交易次数: {results['avg_trades']:.1f}")
    logger.info(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    logger.info(f"  成功率: {results['success_rate']:.1f}%")
    
    logger.info("=" * 80 + "\n")


def get_common_symbols() -> List[str]:
    """获取常见加密货币交易对列表"""
    return [
        'BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT', 'XRP-USDT',
        'SOL-USDT', 'DOT-USDT', 'DOGE-USDT', 'MATIC-USDT', 'LTC-USDT',
        'AVAX-USDT', 'LINK-USDT', 'ATOM-USDT', 'UNI-USDT', 'FIL-USDT',
        'ETC-USDT', 'XLM-USDT', 'ALGO-USDT', 'VET-USDT', 'THETA-USDT'
    ]


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='长下影线策略批量快速回测系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    parser.add_argument('--limit', type=int, help='回测数据量', default=1000)
    parser.add_argument('--min_vol', type=float, help='最小交易量(USDT)', default=1000000)
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
        config_path = os.path.join(os.path.dirname(__file__), 'configs/default.json')
        default_config = load_config_from_file(config_path)
        if not default_config:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 咨询用户输入
    logger.info("长下影线策略批量快速回测系统")
    logger.info("=" * 50)
    logger.info("注意：此系统将自动扫描高交易量币种并批量回测")
    logger.info("=" * 50)
    
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 设置参数
    bar = config.get('bar', '1m')
    min_volume_ccy = config.get('min_volume_ccy', 1000000)
    volume_factor = config.get('volume_factor', 1.2)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.0)
    take_profit_pct = config.get('take_profit_pct', 0.0)
    use_volume = config.get('use_volume', True)
    trade_amount = config.get('trade_amount', 10.0)
    
    # 创建批量回测实例
    batch_backtest = BatchFastBacktest(
        bar=bar,
        min_volume_ccy=min_volume_ccy,
        volume_factor=volume_factor,
        trailing_stop_pct=trailing_stop_pct,
        take_profit_pct=take_profit_pct,
        use_volume=use_volume,
        trade_amount=trade_amount,
        max_workers=args.workers
    )
    
    # 获取高交易量币种
    scanner = CryptoScanner(batch_backtest.client)
    symbols = scanner._get_volume_filtered_symbols('USDT', args.min_vol, use_cache=True, inst_type="SWAP")
    
    if args.top_n:
        symbols = symbols[:args.top_n]
    
    logger.info(f"找到 {len(symbols)} 个符合条件的币种")
    
    if not symbols:
        logger.error(f"未找到24小时交易量 >= {args.min_vol:,.0f} USDT的币种")
        return
    
    # 运行批量回测
    results = batch_backtest.run_batch_backtest(symbols)
    
    if results:
        # 保存结果
        batch_backtest.save_results_to_excel()
        
        # 打印批量报告
        print_batch_report(results, config)
    else:
        logger.error("批量回测失败，请检查配置和数据")


if __name__ == "__main__":
    main()