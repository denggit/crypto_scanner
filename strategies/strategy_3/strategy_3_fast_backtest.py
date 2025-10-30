#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/23/25  
@File       : strategy_3_fast_backtest.py
@Description: Fast backtest for Long Shadow Strategy - 真正的快速回测
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_3.strategy_3 import LongShadowStrategy
from strategies.strategy_3.shared_config import load_config_from_file, get_user_input, print_final_config
from utils.logger import logger


class FastBacktest:
    """快速回测类 - 真正的快速回测实现"""
    
    def __init__(self, symbol: str, bar: str = '1m', 
                 min_volume_ccy: float = 1000000, volume_factor: float = 1.2,
                 trailing_stop_pct: float = 0.0, take_profit_pct: float = 0.0,
                 use_volume: bool = True, trade_amount: float = 10.0):
        """
        Initialize fast backtest
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            min_volume_ccy: Minimum 24h volume in USDT
            volume_factor: Volume multiplier
            trailing_stop_pct: Trailing stop percentage
            take_profit_pct: Take profit percentage
            use_volume: Whether to use volume condition
            trade_amount: USDT amount for each trade
        """
        self.symbol = symbol
        self.bar = bar
        self.min_volume_ccy = min_volume_ccy
        self.volume_factor = volume_factor
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self.use_volume = use_volume
        self.trade_amount = trade_amount
        
        # Initialize components
        self.client = OKXClient()
        self.strategy = LongShadowStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        
        # Backtest state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.trade_count = 0
        self.total_return = 0.0
        self.trades = []
        
    def _calculate_signals_in_bulk(self, df: pd.DataFrame) -> Tuple[List[int], List[Dict]]:
        """
        批量计算所有K线的信号和详细信息 - 真正的快速回测
        
        Args:
            df: DataFrame with OHLCV data
            
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
    
    def _check_trailing_stop(self, price: float) -> bool:
        """检查移动止损条件"""
        if self.position == 1:
            if price > self.highest_price:
                self.highest_price = price
            if self.trailing_stop_pct > 0:
                stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100.0)
                if price <= stop_price:
                    return True
        elif self.position == -1:
            if price < self.lowest_price or self.lowest_price == 0:
                self.lowest_price = price
            if self.trailing_stop_pct > 0:
                stop_price = self.lowest_price * (1 + self.trailing_stop_pct / 100.0)
                if price >= stop_price:
                    return True
        return False
    
    def _check_take_profit(self, price: float) -> bool:
        """检查止盈条件"""
        if self.position == 1:
            if self.take_profit_pct > 0:
                take_profit_price = self.entry_price * (1 + self.take_profit_pct / 100.0)
                if price >= take_profit_price:
                    return True
        elif self.position == -1:
            if self.take_profit_pct > 0:
                take_profit_price = self.entry_price * (1 - self.take_profit_pct / 100.0)
                if price <= take_profit_price:
                    return True
        return False
    
    def execute_trade(self, signal: int, price: float, timestamp: str) -> Dict[str, Any]:
        """执行交易"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        
        trailing_stop_triggered = self._check_trailing_stop(price)
        take_profit_triggered = self._check_take_profit(price)
        
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
                if take_profit_triggered:
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_TAKE_PROFIT"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
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
                elif signal == 0:  # 策略平多信号
                    exit_price = price
                    return_rate = (exit_price - self.entry_price) / self.entry_price
                    action = "LONG_CLOSE_STRATEGY"
                    self.position = 0
                    self.highest_price = 0.0
                    self.trade_count += 1
            elif self.position == -1:
                if take_profit_triggered:
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_TAKE_PROFIT"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
                elif trailing_stop_triggered:
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
                elif signal == 0:  # 策略平空信号
                    exit_price = price
                    return_rate = (self.entry_price - exit_price) / self.entry_price
                    action = "SHORT_CLOSE_STRATEGY"
                    self.position = 0
                    self.lowest_price = 0.0
                    self.trade_count += 1
        
        if action != "HOLD":
            trade_record = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'price': price,
                'action': action,
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate,
                'trade_count': self.trade_count
            }
            self.trades.append(trade_record)
            self.total_return += return_rate
            
            return trade_record
        
        return {}
    
    def run_backtest(self, limit: int = 1000) -> Dict[str, Any]:
        """
        运行快速回测
        
        Args:
            limit: Number of bars to fetch
            
        Returns:
            Dict with backtest results
        """
        logger.info(f"开始快速回测 {self.symbol} 的长下影线策略...")
        logger.info(f"参数: 最小24小时交易量={self.min_volume_ccy}, "
                   f"成交量倍数={self.volume_factor}, "
                   f"移动止损={self.trailing_stop_pct}%, "
                   f"止盈={self.take_profit_pct}%")
        
        try:
            # 获取历史数据
            df = self.market_data_retriever.get_kline(self.symbol, self.bar, limit)
            if df is None or len(df) < 12:
                logger.error(f"无法获取足够的历史数据: {self.symbol}")
                return {}
            
            # 批量计算信号
            signals, details_list = self._calculate_signals_in_bulk(df)
            
            # 重置回测状态
            self.position = 0
            self.entry_price = 0.0
            self.highest_price = 0.0
            self.lowest_price = 0.0
            self.trade_count = 0
            self.total_return = 0.0
            self.trades = []
            
            # 执行交易
            for i, (signal, details) in enumerate(zip(signals, details_list)):
                if not details:
                    continue
                    
                price = details['current_price']
                timestamp = df.index[i] if hasattr(df.index[i], 'strftime') else str(i)
                
                trade_record = self.execute_trade(signal, price, timestamp)
                
                if trade_record:
                    logger.debug(f"交易: {trade_record['action']} 价格: {price:.4f} "
                               f"收益率: {trade_record['return_rate']*100:.2f}%")
            
            # 计算性能指标
            performance = self._calculate_performance_metrics()
            
            logger.info(f"快速回测完成: 总交易次数={self.trade_count}, "
                       f"总收益率={performance['total_return_pct']:.2f}%, "
                       f"胜率={performance['win_rate']:.2f}%")
            
            return performance
            
        except Exception as e:
            logger.error(f"回测过程中出错: {e}")
            return {}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算性能指标"""
        if not self.trades:
            return {}
        
        # 计算基本指标
        total_return_pct = self.total_return * 100
        
        # 计算胜率
        winning_trades = [trade for trade in self.trades if trade['return_rate'] > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        
        # 计算平均盈亏
        if winning_trades:
            avg_win = sum(trade['return_rate'] for trade in winning_trades) / len(winning_trades) * 100
        else:
            avg_win = 0
            
        losing_trades = [trade for trade in self.trades if trade['return_rate'] < 0]
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
        for trade in self.trades:
            current_return += trade['return_rate']
            cumulative_returns.append(current_return)
        
        if cumulative_returns:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = running_max - cumulative_returns
            max_drawdown = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            'symbol': self.symbol,
            'total_trades': self.trade_count,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'trades': self.trades
        }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='长下影线策略快速回测系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    parser.add_argument('--symbol', type=str, help='交易对符号', default='BTC-USDT')
    parser.add_argument('--limit', type=int, help='回测数据量', default=1000)
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
    logger.info("长下影线策略快速回测系统")
    logger.info("=" * 50)
    logger.info("注意：此系统将测试单个交易对的策略表现")
    logger.info("=" * 50)
    
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 设置参数
    symbol = args.symbol
    bar = config.get('bar', '1m')
    min_volume_ccy = config.get('min_volume_ccy', 1000000)
    volume_factor = config.get('volume_factor', 1.2)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.0)
    take_profit_pct = config.get('take_profit_pct', 0.0)
    use_volume = config.get('use_volume', True)
    trade_amount = config.get('trade_amount', 10.0)
    
    # 创建快速回测实例
    backtest = FastBacktest(
        symbol=symbol,
        bar=bar,
        min_volume_ccy=min_volume_ccy,
        volume_factor=volume_factor,
        trailing_stop_pct=trailing_stop_pct,
        take_profit_pct=take_profit_pct,
        use_volume=use_volume,
        trade_amount=trade_amount
    )
    
    # 运行快速回测
    results = backtest.run_backtest(limit=args.limit)
    
    if results:
        logger.info("\n快速回测结果:")
        logger.info(f"  交易对: {results['symbol']}")
        logger.info(f"  总交易次数: {results['total_trades']}")
        logger.info(f"  总收益率: {results['total_return_pct']:.2f}%")
        logger.info(f"  胜率: {results['win_rate']:.2f}%")
        logger.info(f"  平均盈利: {results['avg_win_pct']:.2f}%")
        logger.info(f"  平均亏损: {results['avg_loss_pct']:.2f}%")
        logger.info(f"  盈利因子: {results['profit_factor']:.2f}")
        logger.info(f"  最大回撤: {results['max_drawdown_pct']:.2f}%")
    else:
        logger.error("快速回测失败")


if __name__ == "__main__":
    main()