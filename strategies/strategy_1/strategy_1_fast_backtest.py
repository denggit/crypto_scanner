#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 1:08 AM
@File       : strategy_1_fast_backtest.py
@Description: EMA交叉策略快速回测系统
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
from strategies.strategy_1.strategy_1 import EMACrossoverStrategy
from strategies.strategy_1.shared_config import load_config_from_file, get_user_input, print_final_config
from strategies.strategy_1.methods.volatility_exit import check_volatility_exit_static
from utils.logger import logger


class FastBacktest:
    """快速回测类"""
    
    def __init__(self, symbol: str, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0,
                 assist_cond: str = 'volume', 
                 buy_fee_rate: float = 0.0005, sell_fee_rate: float = 0.0005,
                 volatility_exit: bool = False, volatility_threshold: float = 0.5, **params):
        """
        Initialize Fast Backtest
        
        Args:
            symbol: Trading pair symbol
            bar: K-line time interval
            short_ma: Short EMA period
            long_ma: Long EMA period
            mode: Mode ('strict' or 'loose')
            trailing_stop_pct: Trailing stop percentage
            assist_cond: Assist condition type ('volume', 'rsi', or None)
            volatility_exit: Whether to enable volatility-based exit
            volatility_threshold: Volatility threshold for exit (0.5 means 50% reduction)
            **params: Additional parameters for assist conditions
        """
        self.symbol = symbol
        self.bar = bar
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.mode = mode
        self.trailing_stop_pct = trailing_stop_pct
        self.assist_cond = assist_cond
        self.params = params
        
        # 波动率退出参数
        self.volatility_exit = volatility_exit
        self.volatility_threshold = volatility_threshold
        
        # 手续费参数
        self.buy_fee_rate = buy_fee_rate  # 买入手续费率 0.05%
        self.sell_fee_rate = sell_fee_rate  # 卖出手续费率 0.05%
        
        self.client = OKXClient()
        self.strategy = EMACrossoverStrategy(self.client)
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
    
    def _check_volatility_exit(self, df: pd.DataFrame, current_index: int) -> bool:
        """检查波动率退出条件"""
        if not self.volatility_exit or self.position == 0:
            return False
        
        return check_volatility_exit_static(df, current_index, self.short_ma, self.volatility_threshold)
    
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
    
    def execute_trade(self, signal: int, price: float, details: dict, timestamp: str, df: pd.DataFrame = None, current_index: int = None):
        """执行交易逻辑"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        trade_fee = 0.0  # 本次交易手续费
        
        trailing_stop_triggered = self._check_trailing_stop(price)
        volatility_exit_triggered = False
        
        # 检查波动率退出条件
        if df is not None and current_index is not None:
            volatility_exit_triggered = self._check_volatility_exit(df, current_index)
        
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
            elif volatility_exit_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "LONG_CLOSE_VOLATILITY"
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
            elif volatility_exit_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(self.entry_price, exit_price, self.position)
                action = "SHORT_CLOSE_VOLATILITY"
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
                'return_rate': return_rate,
                'trade_fee': trade_fee,
                'total_fee': self.total_fee
            }
            self.trade_records.append(record)
    
    def run_backtest(self, limit: int = 300):
        """运行快速回测"""
        logger.info(f"开始快速回测 {self.symbol} 的EMA交叉策略...")
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
            signals, details_list = self._calculate_signals_in_bulk(df)
            
            # 按时间顺序处理每根K线
            for i in range(len(df)):
                if i < max(self.long_ma, 10):  # 确保有足够的数据计算指标
                    continue
                    
                signal = signals[i]
                details = details_list[i]
                price = details.get('current_price', 0)
                timestamp = self._convert_to_beijing_time(df.iloc[i]['timestamp']) if 'timestamp' in df.columns else str(i)
                
                if price > 0:
                    # 检查移动止损
                    self._check_trailing_stop(price)
                    # 执行交易
                    self.execute_trade(signal, price, details, timestamp, df, i)
            
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
    
    def _calculate_signals_in_bulk(self, df: pd.DataFrame):
        """批量计算所有K线的信号和详细信息"""
        signals = []
        details_list = []
        
        # 获取价格和成交量数据
        closes = df['c'] if 'c' in df.columns else df['close']
        volumes = df['vol'] if 'vol' in df.columns else df['volume']
        
        # 计算EMA
        from tools.technical_indicators import ema, rsi
        ema_short = ema(closes, self.short_ma)
        ema_long = ema(closes, self.long_ma)
        
        # 计算成交量条件
        vol_window = 10 if self.bar in ['1m', '3m', '5m'] else 7
        volume_ratios = []
        volume_expansions = []
        
        for i in range(len(volumes)):
            if i < vol_window:
                volume_ratios.append(0)
                volume_expansions.append(False)
                continue
            
            current_volume = volumes.iloc[i]
            avg_volume = volumes.iloc[i-vol_window:i].mean()
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            volume_ratios.append(volume_ratio)
            volume_expansions.append(volume_ratio >= self.params.get('vol_multiplier', 1.2))
        
        # 计算EMA20斜率
        ema20_slopes = []
        for i in range(len(ema_long)):
            if i < 1:
                ema20_slopes.append(0)
                continue
            
            current_ema_long = ema_long.iloc[i]
            prev_ema_long = ema_long.iloc[i-1]
            slope = (current_ema_long - prev_ema_long) / prev_ema_long if prev_ema_long != 0 else 0
            ema20_slopes.append(slope)
        
        # 计算RSI条件（如果使用）
        rsi_values = []
        if self.assist_cond == 'rsi':
            rsi_period = self.params.get('rsi_period', 9)
            rsi_values = rsi(closes, rsi_period)
        
        # 批量计算信号
        for i in range(len(df)):
            if i < max(self.long_ma, 10):
                signals.append(0)
                details_list.append({})
                continue
                
            current_close = closes.iloc[i]
            current_ema_short = ema_short.iloc[i]
            prev_ema_short = ema_short.iloc[i-1] if i > 0 else current_ema_short
            current_ema_long = ema_long.iloc[i]
            prev_ema_long = ema_long.iloc[i-1] if i > 0 else current_ema_long
            
            # 计算信号
            signal = 0
            
            # 检查EMA交叉条件
            ema_cross_up = (float(prev_ema_short) <= float(prev_ema_long) and float(current_ema_short) > float(current_ema_long))
            ema_cross_down = (float(prev_ema_short) >= float(prev_ema_long) and float(current_ema_short) < float(current_ema_long))
            
            # 检查辅助条件
            assist_condition_met = False
            if self.assist_cond == 'volume':
                # 成交量条件
                if self.mode == 'strict':
                    # strict模式：要求当前K线放量
                    assist_condition_met = volume_expansions[i]
                else:
                    # loose模式：允许前一根放量
                    assist_condition_met = volume_expansions[i] or (i > 0 and volume_expansions[i-1])
            elif self.assist_cond == 'rsi':
                # RSI条件
                if i < len(rsi_values):
                    current_rsi = rsi_values.iloc[i]
                    rsi_long_entry = self.params.get('rsi_long_entry', 55)
                    rsi_short_entry = self.params.get('rsi_short_entry', 45)
                    if ema_cross_up:
                        assist_condition_met = current_rsi > rsi_long_entry
                    elif ema_cross_down:
                        assist_condition_met = current_rsi < rsi_short_entry
            else:
                # 无辅助条件
                assist_condition_met = True
            
            # 检查EMA20斜率条件
            ema20_slope = ema20_slopes[i] if i < len(ema20_slopes) else 0
            
            # 生成信号
            if ema_cross_up and assist_condition_met and ema20_slope > 0:
                signal = 1
            elif ema_cross_down and assist_condition_met and ema20_slope < 0:
                signal = -1
            
            # 构建详细信息
            details = {
                'current_price': current_close,
                'ema5': current_ema_short,
                'ema20': current_ema_long,
                'ema20_slope': ema20_slope,
                'volume_expansion': volume_expansions[i] if i < len(volume_expansions) else False,
                'volume_ratio': volume_ratios[i] if i < len(volume_ratios) else 0
            }
            
            if self.assist_cond == 'rsi' and i < len(rsi_values):
                details['rsi'] = rsi_values.iloc[i]
            
            signals.append(signal)
            details_list.append(details)
        
        return signals, details_list
    
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
            'short_ma': self.short_ma,
            'long_ma': self.long_ma,
            'mode': self.mode,
            'trailing_stop_pct': self.trailing_stop_pct,
            'assist_cond': self.assist_cond,
            'volatility_exit': self.volatility_exit,
            'volatility_threshold': self.volatility_threshold,
            'params': self.params,
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
                '交易对', 'K线周期', 'EMA参数', '模式', '移动止损', '辅助条件',
                '波动率退出', '波动率阈值',
                '总交易次数', '总收益率(%)', '手续费(%)', '净收益率(%)', '胜率(%)', 
                '平均盈利(%)', '平均亏损(%)', '盈亏比', '夏普比率', '最大回撤(%)'
            ],
            '数值': [
                report['symbol'], report['bar'], f"{report['short_ma']}/{report['long_ma']}",
                report['mode'], f"{report['trailing_stop_pct']}%", report['assist_cond'],
                '是' if report['volatility_exit'] else '否', f"{report['volatility_threshold']:.2f}",
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
                '交易对', 'K线周期', '短EMA周期', '长EMA周期', '模式', 
                '移动止损(%)', '辅助条件', '波动率退出', '波动率阈值',
                '成交量倍数', '确认百分比(%)', 'RSI周期'
            ],
            '数值': [
                report['symbol'], report['bar'], report['short_ma'], report['long_ma'],
                report['mode'], report['trailing_stop_pct'], report['assist_cond'],
                '是' if report['volatility_exit'] else '否', f"{report['volatility_threshold']:.2f}",
                report['params'].get('vol_multiplier', 'N/A'),
                report['params'].get('confirmation_pct', 'N/A'),
                report['params'].get('rsi_period', 'N/A')
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
                '  - 降低成交量倍数或确认百分比', 
                '  - 尝试loose模式'
            ]
        elif report['total_return_pct'] < 0:
            suggestions = [
                '⚠️ 策略亏损，建议:',
                '  - 调整EMA周期组合',
                '  - 增加移动止损比例',
                '  - 优化辅助条件参数'
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
    logger.info("EMA交叉策略快速回测报告")
    logger.info("=" * 60)
    
    logger.info(f"交易对: {report['symbol']}")
    logger.info(f"K线周期: {report['bar']}")
    logger.info(f"EMA参数: {report['short_ma']}/{report['long_ma']}")
    logger.info(f"模式: {report['mode']}")
    logger.info(f"移动止损: {report['trailing_stop_pct']}%")
    logger.info(f"辅助条件: {report['assist_cond']}")
    logger.info(f"波动率退出: {'是' if report['volatility_exit'] else '否'}")
    if report['volatility_exit']:
        logger.info(f"波动率阈值: {report['volatility_threshold']:.2f}")
    
    if report['assist_cond'] == 'volume':
        logger.info(f"成交量倍数: {report['params'].get('vol_multiplier', 1.2)}")
        logger.info(f"确认百分比: {report['params'].get('confirmation_pct', 0.2)}%")
    elif report['assist_cond'] == 'rsi':
        logger.info(f"RSI周期: {report['params'].get('rsi_period', 9)}")
    
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
        logger.info("     - 降低成交量倍数或确认百分比")
        logger.info("     - 尝试loose模式")
    elif report['total_return_pct'] < 0:
        logger.info("  ⚠️  策略亏损，建议:")
        logger.info("     - 调整EMA周期组合")
        logger.info("     - 增加移动止损比例")
        logger.info("     - 优化辅助条件参数")
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
    parser = argparse.ArgumentParser(description='EMA交叉策略快速回测系统')
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
        config_path = os.path.join(os.path.dirname(__file__), 'configs/bnb_usdt_swap.json')
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
    short_ma = config.get('short_ma', 5)
    long_ma = config.get('long_ma', 20)
    mode = config.get('mode', 'loose')
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    assist_cond = config.get('assist_cond', 'volume')
    volatility_exit = config.get('volatility_exit', False)
    volatility_threshold = config.get('volatility_threshold', 0.5)
    params = config.get('params', {})
    
    # 创建回测实例
    backtest = FastBacktest(
        symbol=symbol,
        bar=bar,
        short_ma=short_ma,
        long_ma=long_ma,
        mode=mode,
        trailing_stop_pct=trailing_stop_pct,
        assist_cond=assist_cond,
        volatility_exit=volatility_exit,
        volatility_threshold=volatility_threshold,
        **params
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