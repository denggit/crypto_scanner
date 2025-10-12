#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 2:42 PM
@File       : strategy_2_fast_backtest.py
@Description: 高频短线策略快速回测系统
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_2.strategy_2 import HighFrequencyStrategy
from strategies.strategy_2.shared_config import load_config_from_file, get_user_input, print_final_config
from utils.logger import logger


class FastBacktest:
    """快速回测类"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 consecutive_bars: int = 2, atr_period: int = 14,
                 atr_threshold: float = 0.8, trailing_stop_pct: float = 0.8,
                 volume_factor: float = 1.2, use_volume: bool = True,
                 breakout_stop_bars: int = 2,
                 buy_fee_rate: float = 0.0005, sell_fee_rate: float = 0.0005):
        """
        Initialize Fast Backtest
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            consecutive_bars: Number of consecutive bars for breakout
            atr_period: ATR period
            atr_threshold: ATR threshold multiplier
            trailing_stop_pct: Trailing stop percentage
            volume_factor: Volume expansion factor
            use_volume: Whether to use volume condition
            breakout_stop_bars: Number of consecutive bars for breakout stop
        """
        self.symbol = symbol
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
        
        # 回测状态
        self.position = 0  # 0: 无仓位, 1: 多仓, -1: 空仓
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0
        self.trade_count = 0
        self.trade_records = []
        self.total_fee = 0.0  # 累计手续费
        
    def _check_trailing_stop(self, price: float) -> bool:
        """检查移动止损条件"""
        if self.position == 1:
            # 持多仓：更新最高价，检查是否跌破止损价
            if price > self.highest_price:
                self.highest_price = price
            stop_price = self.highest_price * (1 - self.trailing_stop_pct / 100.0)
            if price <= stop_price:
                return True
        elif self.position == -1:
            # 持空仓：更新最低价，检查是否涨破止损价
            if price < self.lowest_price:
                self.lowest_price = price
            stop_price = self.lowest_price * (1 + self.trailing_stop_pct / 100.0)
            if price >= stop_price:
                return True
        return False
    
    def _calculate_return_rate(self, entry_price: float, exit_price: float, position: int) -> float:
        """
        计算考虑手续费后的净收益率
        
        收益率计算逻辑：
        - 开仓成本 = 开仓价格 × (1 + 开仓手续费率)
        - 平仓净值 = 平仓价格 × (1 - 平仓手续费率)
        - 收益率 = (平仓净值 - 开仓成本) / 开仓成本
        
        对于多仓：开仓手续费=买入手续费，平仓手续费=卖出手续费
        对于空仓：开仓手续费=卖出手续费，平仓手续费=买入手续费
        """
        if position == 1:  # 多仓
            # 开仓成本 = 开仓价格 × (1 + 买入手续费率)
            entry_cost = entry_price * (1 + self.buy_fee_rate)
            # 平仓净值 = 平仓价格 × (1 - 卖出手续费率)
            exit_net_value = exit_price * (1 - self.sell_fee_rate)
            return_rate = (exit_net_value - entry_cost) / entry_cost
        elif position == -1:  # 空仓
            # 开仓成本 = 开仓价格 × (1 + 卖出手续费率)
            entry_cost = entry_price * (1 + self.sell_fee_rate)
            # 平仓净值 = 平仓价格 × (1 - 买入手续费率)
            exit_net_value = exit_price * (1 - self.buy_fee_rate)
            return_rate = (entry_cost - exit_net_value) / entry_cost
        else:
            return_rate = 0.0
        
        return return_rate
    
    def _calculate_signals_in_bulk(self, df: pd.DataFrame):
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
        
        # 计算ATR
        from tools.technical_indicators import atr
        atr_values = atr(df, self.atr_period)
        
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
    
    def execute_trade(self, signal: int, price: float, details: dict, timestamp: str, 
                     trailing_stop_triggered: bool = False, close_signal: int = 0):
        """执行交易逻辑"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        trade_fee = 0.0  # 本次交易手续费
        
        if self.position == 0:
            if signal == 1:
                self.position = 1
                self.entry_price = price
                self.highest_price = price
                self.lowest_price = price
                action = "LONG_OPEN"
                trade_fee = price * self.buy_fee_rate  # 开多仓买入手续费
                self.total_fee += trade_fee
                self.trade_count += 1
            elif signal == -1:
                self.position = -1
                self.entry_price = price
                self.lowest_price = price
                self.highest_price = price
                action = "SHORT_OPEN"
                trade_fee = price * self.sell_fee_rate  # 开空仓卖出手续费
                self.total_fee += trade_fee
                self.trade_count += 1
        elif self.position == 1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "LONG_CLOSE_TRAILING_STOP"
                trade_fee = price * self.sell_fee_rate  # 平多仓卖出手续费
                self.total_fee += trade_fee
                self.position = 0
                self.highest_price = 0.0
                self.trade_count += 1
            elif close_signal == -1:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "LONG_CLOSE_BREAKOUT"
                trade_fee = price * self.sell_fee_rate  # 平多仓卖出手续费
                self.total_fee += trade_fee
                self.position = 0
                self.highest_price = 0.0
                self.trade_count += 1
            elif signal == -1:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "LONG_CLOSE_SHORT_OPEN"
                trade_fee = price * self.sell_fee_rate  # 平多仓卖出手续费
                self.total_fee += trade_fee
                self.position = -1
                self.entry_price = price
                self.highest_price = price
                self.lowest_price = price
                self.trade_count += 1
        elif self.position == -1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "SHORT_CLOSE_TRAILING_STOP"
                trade_fee = price * self.buy_fee_rate  # 平空仓买入手续费
                self.total_fee += trade_fee
                self.position = 0
                self.lowest_price = 0.0
                self.trade_count += 1
            elif close_signal == 1:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "SHORT_CLOSE_BREAKOUT"
                trade_fee = price * self.buy_fee_rate  # 平空仓买入手续费
                self.total_fee += trade_fee
                self.position = 0
                self.lowest_price = 0.0
                self.trade_count += 1
            elif signal == 1:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "SHORT_CLOSE_LONG_OPEN"
                trade_fee = price * self.buy_fee_rate  # 平空仓买入手续费
                self.total_fee += trade_fee
                self.position = 1
                self.entry_price = price
                self.lowest_price = price
                self.highest_price = price
                self.trade_count += 1
        
        if action != "HOLD":
            record = {
                'timestamp': timestamp,
                'symbol': self.symbol,
                'price': price,
                'typical_price': details.get('current_typical', 0),
                'atr': details.get('atr', 0),
                'atr_mean': details.get('atr_mean', 0),
                'atr_condition': details.get('atr_condition_met', False),
                'volume_condition': details.get('volume_condition_met', False),
                'long_breakout': details.get('long_breakout', False),
                'short_breakout': details.get('short_breakout', False),
                'signal': signal,
                'action': action,
                'position': self.position,
                'entry_price': self.entry_price,
                'exit_price': exit_price,
                'return_rate': return_rate,
                'trade_fee': trade_fee,
                'total_fee': self.total_fee
            }
            self.trade_records.append(record)
    
    def run_backtest(self, limit: int = 300):
        """运行快速回测"""
        logger.info(f"开始快速回测 {self.symbol} 的高频短线策略...")
        logger.info(f"数据量: {limit} 根K线")
        
        try:
            # 获取历史数据
            df = self.market_data_retriever.get_kline(self.symbol, self.bar, limit)
            if df is None or len(df) == 0 or len(df) < limit:
                logger.error(f"无法获取足够的历史数据，实际获取: {len(df) if df is not None else 0} 根K线")
                return None
            
            logger.info(f"成功获取 {len(df)} 根K线数据")
            
            # 保存K线数据到实例变量中，并转换时间戳为北京时间
            self.kline_data = df.copy()
            if 'timestamp' in self.kline_data.columns:
                self.kline_data['timestamp'] = self.kline_data['timestamp'].apply(self._convert_to_beijing_time)
            
            # 批量计算所有K线的信号和详细信息
            signals, details_list, typical_prices = self._calculate_signals_in_bulk(df)
            
            # 按时间顺序处理每根K线
            for i in range(len(df)):
                if i < max(self.atr_period, self.consecutive_bars + 1, 21):  # 确保有足够的数据计算指标
                    continue
                    
                signal = signals[i]
                details = details_list[i]
                price = details.get('current_price', 0)
                timestamp = self._convert_to_beijing_time(df.iloc[i]['timestamp']) if 'timestamp' in df.columns else str(i)
                
                if price > 0:
                    # 检查移动止损
                    trailing_stop_triggered = self._check_trailing_stop(price)
                    
                    # 检查平仓条件 (连续breakout_stop_bars根K线反向突破)
                    close_signal = 0
                    if self.position == 1 and self._check_consecutive_breakout(df, typical_prices, i, self.breakout_stop_bars, direction='down'):
                        close_signal = -1
                    elif self.position == -1 and self._check_consecutive_breakout(df, typical_prices, i, self.breakout_stop_bars, direction='up'):
                        close_signal = 1
                    
                    # 执行交易
                    self.execute_trade(signal, price, details, timestamp, trailing_stop_triggered, close_signal)
            
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"回测过程中出错: {e}")
            return None
    
    def _convert_to_beijing_time(self, timestamp) -> str:
        """将UTC时间转换为北京时间"""
        try:
            # 处理pandas Timestamp对象
            if hasattr(timestamp, 'tz_localize'):
                # 假设时间戳是UTC时间，先本地化为UTC时区
                utc_time = timestamp.tz_localize(timezone.utc)
                # 转换为北京时间 (UTC+8)
                beijing_time = utc_time.tz_convert(timezone(timedelta(hours=8)))
                return beijing_time.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # 如果是字符串，按原逻辑处理
                utc_time = datetime.strptime(str(timestamp), '%Y-%m-%d %H:%M:%S')
                utc_time = utc_time.replace(tzinfo=timezone.utc)
                beijing_time = utc_time.astimezone(timezone(timedelta(hours=8)))
                return beijing_time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            # 如果转换失败，返回原始时间戳
            return str(timestamp)
    
    def generate_report(self):
        """生成回测报告"""
        # 创建交易记录DataFrame
        trades_df = pd.DataFrame(self.trade_records) if self.trade_records else pd.DataFrame()
        
        # 过滤出平仓交易（只有平仓交易才有收益率）
        close_trades = trades_df[trades_df['return_rate'] != 0] if len(trades_df) > 0 else pd.DataFrame()
        
        # 计算回测指标（只基于平仓交易）
        total_return = close_trades['return_rate'].sum() * 100 if len(close_trades) > 0 else 0  # 转换为百分比
        win_trades = close_trades[close_trades['return_rate'] > 0] if len(close_trades) > 0 else pd.DataFrame()
        loss_trades = close_trades[close_trades['return_rate'] < 0] if len(close_trades) > 0 else pd.DataFrame()
        
        win_rate = len(win_trades) / len(close_trades) * 100 if len(close_trades) > 0 else 0
        avg_win = win_trades['return_rate'].mean() * 100 if len(win_trades) > 0 else 0
        avg_loss = loss_trades['return_rate'].mean() * 100 if len(loss_trades) > 0 else 0
        profit_factor = abs(win_trades['return_rate'].sum() / loss_trades['return_rate'].sum()) if len(loss_trades) > 0 and loss_trades['return_rate'].sum() != 0 else float('inf')
        
        # 计算夏普比率（简化版，假设无风险利率为0）
        returns = close_trades['return_rate'].dropna() if len(close_trades) > 0 else pd.Series()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 and returns.std() != 0 else 0
        
        # 最大回撤
        if len(close_trades) > 0:
            cumulative_returns = (1 + close_trades['return_rate']).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0
        
        # 计算手续费相关指标
        total_fee_pct = self.total_fee / (trades_df['price'].mean() if len(trades_df) > 0 else 1) * 100 if len(trades_df) > 0 else 0
        net_return_pct = total_return - total_fee_pct
        
        report = {
            'symbol': self.symbol,
            'bar': self.bar,
            'consecutive_bars': self.consecutive_bars,
            'atr_period': self.atr_period,
            'atr_threshold': self.atr_threshold,
            'trailing_stop_pct': self.trailing_stop_pct,
            'volume_factor': self.volume_factor,
            'use_volume': self.use_volume,
            'total_trades': len(trades_df),
            'total_return_pct': total_return,
            'total_fee_pct': total_fee_pct,
            'net_return_pct': net_return_pct,
            'win_rate_pct': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_fee': self.total_fee,
            'trades_df': trades_df,
            'kline_data': self.kline_data if hasattr(self, 'kline_data') else pd.DataFrame()
        }
        
        return report


def save_report_to_excel(report: dict, output_dir: str = "backtest_results"):
    """保存回测报告到Excel文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用北京时间
    beijing_time = datetime.now(timezone(timedelta(hours=8)))
    timestamp = beijing_time.strftime('%Y%m%d_%H%M%S')
    symbol = report['symbol'].replace('-', '_')
    
    # Excel文件名
    excel_filename = f"{output_dir}/backtest_report_{symbol}_{timestamp}.xlsx"
    
    # 创建Excel写入器
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        
        # Sheet 1: K线数据
        if len(report['kline_data']) > 0:
            report['kline_data'].to_excel(writer, sheet_name='K线数据', index=False)
        else:
            pd.DataFrame({'信息': ['无K线数据']}).to_excel(writer, sheet_name='K线数据', index=False)
        
        # Sheet 2: 交易记录
        if len(report['trades_df']) > 0:
            report['trades_df'].to_excel(writer, sheet_name='交易记录', index=False)
        else:
            pd.DataFrame({'信息': ['无交易记录']}).to_excel(writer, sheet_name='交易记录', index=False)
        
        # Sheet 3: 整体报告
        summary_data = {
            '指标': [
                '交易对', 'K线周期', '连续K线', 'ATR周期', 'ATR阈值',
                '移动止损', '成交量倍数', '使用成交量',
                '总交易次数', '总收益率(%)', '手续费(%)', '净收益率(%)', '胜率(%)', 
                '平均盈利(%)', '平均亏损(%)', '盈亏比', '夏普比率', '最大回撤(%)'
            ],
            '数值': [
                report['symbol'], report['bar'], report['consecutive_bars'], 
                report['atr_period'], report['atr_threshold'],
                f"{report['trailing_stop_pct']}%", report['volume_factor'], 
                '是' if report['use_volume'] else '否',
                report['total_trades'], f"{report['total_return_pct']:.2f}",
                f"{report['total_fee_pct']:.2f}", f"{report['net_return_pct']:.2f}",
                f"{report['win_rate_pct']:.2f}", f"{report['avg_win_pct']:.2f}",
                f"{report['avg_loss_pct']:.2f}", f"{report['profit_factor']:.2f}",
                f"{report['sharpe_ratio']:.2f}", f"{report['max_drawdown_pct']:.2f}"
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='整体报告', index=False)
        
        # Sheet 4: 参数配置
        config_data = {
            '参数': [
                '交易对', 'K线周期', '连续K线', 'ATR周期', 'ATR阈值',
                '移动止损(%)', '成交量倍数', '使用成交量'
            ],
            '数值': [
                report['symbol'], report['bar'], report['consecutive_bars'], 
                report['atr_period'], report['atr_threshold'],
                report['trailing_stop_pct'], report['volume_factor'], 
                '是' if report['use_volume'] else '否'
            ]
        }
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='参数配置', index=False)
        
        # Sheet 5: 参数调整建议
        suggestions = []
        if report['total_trades'] == 0:
            suggestions = [
                '⚠️ 没有产生任何交易，建议:',
                '  - 检查信号计算逻辑',
                '  - 降低成交量倍数或ATR阈值', 
                '  - 减少连续K线数量'
            ]
        elif report['total_return_pct'] < 0:
            suggestions = [
                '⚠️ 策略亏损，建议:',
                '  - 调整连续K线数量',
                '  - 增加移动止损比例',
                '  - 优化ATR阈值参数'
            ]
        elif report['win_rate_pct'] < 50:
            suggestions = [
                '⚠️ 胜率较低但总体盈利，建议:',
                '  - 关注盈亏比而非胜率',
                '  - 可能适合趋势跟踪策略'
            ]
        elif report['max_drawdown_pct'] < -10:
            suggestions = [
                '⚠️ 回撤较大，建议:',
                '  - 增加移动止损比例',
                '  - 降低仓位或增加过滤条件'
            ]
        else:
            suggestions = [
                '✅ 策略表现良好，可以:',
                '  - 考虑实盘测试',
                '  - 优化参数进一步提升收益'
            ]
        
        suggestions_df = pd.DataFrame({'参数调整建议': suggestions})
        suggestions_df.to_excel(writer, sheet_name='参数调整建议', index=False)
    
    return excel_filename


def print_report(report: dict):
    """打印回测报告"""
    logger.info("\n" + "=" * 60)
    logger.info("高频短线策略快速回测报告")
    logger.info("=" * 60)
    
    logger.info(f"交易对: {report['symbol']}")
    logger.info(f"K线周期: {report['bar']}")
    logger.info(f"连续K线: {report['consecutive_bars']}")
    logger.info(f"ATR周期: {report['atr_period']}")
    logger.info(f"ATR阈值: {report['atr_threshold']}")
    logger.info(f"移动止损: {report['trailing_stop_pct']}%")
    logger.info(f"成交量倍数: {report['volume_factor']}")
    logger.info(f"使用成交量: {'是' if report['use_volume'] else '否'}")
    
    logger.info("-" * 60)
    logger.info("回测结果:")
    logger.info(f"  总交易次数: {report['total_trades']}")
    logger.info(f"  总收益率: {report['total_return_pct']:.2f}%")
    logger.info(f"  手续费: {report['total_fee_pct']:.2f}%")
    logger.info(f"  净收益率: {report['net_return_pct']:.2f}%")
    logger.info(f"  胜率: {report['win_rate_pct']:.2f}%")
    logger.info(f"  平均盈利: {report['avg_win_pct']:.2f}%")
    logger.info(f"  平均亏损: {report['avg_loss_pct']:.2f}%")
    logger.info(f"  盈亏比: {report['profit_factor']:.2f}")
    logger.info(f"  夏普比率: {report['sharpe_ratio']:.2f}")
    logger.info(f"  最大回撤: {report['max_drawdown_pct']:.2f}%")
    
    logger.info("-" * 60)
    logger.info("参数调整建议:")
    
    # 基于回测结果给出参数调整建议
    if report['total_trades'] == 0:
        logger.info("  ⚠️  没有产生任何交易，建议:")
        logger.info("     - 检查信号计算逻辑")
        logger.info("     - 降低成交量倍数或ATR阈值")
        logger.info("     - 减少连续K线数量")
    elif report['total_return_pct'] < 0:
        logger.info("  ⚠️  策略亏损，建议:")
        logger.info("     - 调整连续K线数量")
        logger.info("     - 增加移动止损比例")
        logger.info("     - 优化ATR阈值参数")
    elif report['win_rate_pct'] < 50:
        logger.info("  ⚠️  胜率较低但总体盈利，建议:")
        logger.info("     - 关注盈亏比而非胜率")
        logger.info("     - 可能适合趋势跟踪策略")
    elif report['max_drawdown_pct'] < -10:
        logger.info("  ⚠️  回撤较大，建议:")
        logger.info("     - 增加移动止损比例")
        logger.info("     - 降低仓位或增加过滤条件")
    else:
        logger.info("  ✅  策略表现良好，可以:")
        logger.info("     - 考虑实盘测试")
        logger.info("     - 优化参数进一步提升收益")
    
    logger.info("=" * 60 + "\n")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='高频短线策略快速回测系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    parser.add_argument('--limit', type=int, help='回测数据量', default=300)
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
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 设置参数
    symbol = config.get('symbol', 'BTC-USDT')
    bar = config.get('bar', '1m')
    consecutive_bars = config.get('consecutive_bars', 2)
    atr_period = config.get('atr_period', 14)
    atr_threshold = config.get('atr_threshold', 0.8)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.8)
    use_volume = config.get('use_volume', True)
    volume_factor = config.get('volume_factor', 1.2)
    breakout_stop_bars = config.get('breakout_stop_bars', 2)
    
    # 创建回测实例
    backtest = FastBacktest(
        symbol=symbol,
        bar=bar,
        consecutive_bars=consecutive_bars,
        atr_period=atr_period,
        atr_threshold=atr_threshold,
        trailing_stop_pct=trailing_stop_pct,
        volume_factor=volume_factor,
        use_volume=use_volume,
        breakout_stop_bars=breakout_stop_bars
    )
    
    # 运行回测
    report = backtest.run_backtest(limit=args.limit)
    
    if report:
        # 打印报告
        print_report(report)
        
        # 保存报告到Excel
        excel_file = save_report_to_excel(report)
        logger.info(f"回测报告已保存到: {excel_file}")
    else:
        logger.error("回测失败，请检查配置和数据")


if __name__ == "__main__":
    main()