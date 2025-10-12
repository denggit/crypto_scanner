#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/25 5:05 AM
@File       : strategy_1_fast_backtest_all.py
@Description: EMA交叉策略批量快速回测系统 - 自动扫描高交易量币种并批量回测
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
from strategies.strategy_1.strategy_1 import EMACrossoverStrategy
from strategies.strategy_1.shared_config import load_config_from_file, get_user_input, print_final_config
from strategies.strategy_1.methods.volatility_exit import check_volatility_exit_static
from tools.market_scanner import CryptoScanner
from utils.logger import logger


class BatchFastBacktest:
    """批量快速回测类"""
    
    def __init__(self, bar: str = '1m',
                 short_ma: int = 5, long_ma: int = 20,
                 mode: str = 'strict', trailing_stop_pct: float = 1.0,
                 assist_cond: str = 'volume', 
                 buy_fee_rate: float = 0.0005, sell_fee_rate: float = 0.0005,
                 volatility_exit: bool = False, volatility_threshold: float = 0.5, **params):
        """
        Initialize Batch Fast Backtest
        
        Args:
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
        self.scanner = CryptoScanner(self.client)
        
        # 批量回测结果
        self.batch_results = []
    
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
            signals, details_list = self._calculate_signals_in_bulk(df, symbol)
            
            # 按时间顺序处理每根K线
            for i in range(len(df)):
                if i < max(self.long_ma, 10):  # 确保有足够的数据计算指标
                    continue
                    
                signal = signals[i]
                details = details_list[i]
                price = details.get('current_price', 0)
                
                if price > 0:
                    # 执行交易逻辑
                    trade_result = self._execute_trade_logic(
                        signal, price, details, position, entry_price, 
                        highest_price, lowest_price, trade_count, total_fee, close_trades
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
    
    def _execute_trade_logic(self, signal: int, price: float, details: dict,
                           position: int, entry_price: float, highest_price: float, 
                           lowest_price: float, trade_count: int, total_fee: float, close_trades: list):
        """执行交易逻辑"""
        action = "HOLD"
        exit_price = 0.0
        return_rate = 0.0
        trade_fee = 0.0
        
        # 检查移动止损
        trailing_stop_triggered = self._check_trailing_stop(price, position, highest_price, lowest_price)
        
        if position == 0:
            if signal == 1:
                position = 1
                entry_price = price
                highest_price = price
                lowest_price = price
                action = "LONG_OPEN"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                trade_count += 1
            elif signal == -1:
                position = -1
                entry_price = price
                lowest_price = price
                highest_price = price
                action = "SHORT_OPEN"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                trade_count += 1
        elif position == 1:
            if trailing_stop_triggered:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "LONG_CLOSE_TRAILING_STOP"
                trade_fee = price * self.sell_fee_rate
                total_fee += trade_fee
                position = 0
                highest_price = 0.0
                trade_count += 1
            elif signal == -1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "LONG_CLOSE_SHORT_OPEN"
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
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                position = 0
                lowest_price = 0.0
                trade_count += 1
            elif signal == 1:
                exit_price = price
                return_rate = self._calculate_return_rate(entry_price, exit_price, position)
                action = "SHORT_CLOSE_LONG_OPEN"
                trade_fee = price * self.buy_fee_rate
                total_fee += trade_fee
                position = 1
                entry_price = price
                lowest_price = price
                highest_price = price
                trade_count += 1
        
        # 记录平仓交易（包含详细统计信息）
        if action != "HOLD" and return_rate != 0:
            close_trades.append({
                'return_rate': return_rate,
                'exit_price': exit_price,
                'action': action,
                'trade_type': 'LONG' if position == 1 else 'SHORT' if position == -1 else 'NONE',
                'exit_reason': 'TRAILING_STOP' if trailing_stop_triggered else 'REVERSE_SIGNAL',
                'entry_price': entry_price,
                'position_holding_bars': 0,
                'atr_condition': False,
                'volume_condition': details.get('volume_expansion', False),
                'long_breakout': False,
                'short_breakout': False,
                'current_price': price,
                'highest_price': highest_price,
                'lowest_price': lowest_price
            })
        
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
            'close_trades_count': len(close_trades_df)
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
    logger.info("EMA交叉策略批量快速回测报告")
    logger.info("=" * 80)
    
    # 打印策略参数
    logger.info("策略参数:")
    logger.info(f"  K线周期: {config.get('bar', '1m')}")
    logger.info(f"  EMA参数: {config.get('short_ma', 5)}/{config.get('long_ma', 20)}")
    logger.info(f"  模式: {config.get('mode', 'strict')}")
    logger.info(f"  移动止损: {config.get('trailing_stop_pct', 1.0)}%")
    logger.info(f"  辅助条件: {config.get('assist_cond', 'volume')}")
    logger.info(f"  波动率退出: {'是' if config.get('volatility_exit', False) else '否'}")
    
    if config.get('assist_cond') == 'volume':
        logger.info(f"  成交量倍数: {config.get('params', {}).get('vol_multiplier', 1.2)}")
        logger.info(f"  确认百分比: {config.get('params', {}).get('confirmation_pct', 0.2)}%")
    elif config.get('assist_cond') == 'rsi':
        logger.info(f"  RSI周期: {config.get('params', {}).get('rsi_period', 9)}")
    
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
    
    logger.info("=" * 80 + "\n")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='EMA交叉策略批量快速回测系统')
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
        config_path = os.path.join(os.path.dirname(__file__), 'configs/bnb_usdt_swap.json')
        default_config = load_config_from_file(config_path)
        if not default_config:
            logger.info("未找到默认配置文件，使用系统默认值")
            default_config = {}
    
    # 咨询用户输入
    logger.info("EMA交叉策略批量快速回测系统")
    logger.info("=" * 50)
    logger.info("注意：此系统将自动扫描高交易量币种并批量回测")
    logger.info("=" * 50)
    
    config = get_user_input(default_config)
    print_final_config(config)
    
    # 设置参数
    bar = config.get('bar', '1m')
    short_ma = config.get('short_ma', 5)
    long_ma = config.get('long_ma', 20)
    mode = config.get('mode', 'loose')
    trailing_stop_pct = config.get('trailing_stop_pct', 1.0)
    assist_cond = config.get('assist_cond', 'volume')
    volatility_exit = config.get('volatility_exit', False)
    volatility_threshold = config.get('volatility_threshold', 0.5)
    params = config.get('params', {})
    
    # 创建批量回测实例
    batch_backtest = BatchFastBacktest(
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
    
    # 运行批量回测
    results = batch_backtest.run_batch_backtest(min_vol_ccy=args.min_vol, limit=args.limit, max_workers=args.workers)
    
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
                'K线周期', '短EMA周期', '长EMA周期', '模式', 
                '移动止损(%)', '辅助条件', '波动率退出', '波动率阈值',
                '成交量倍数', '确认百分比(%)', 'RSI周期'
            ],
            '数值': [
                config.get('bar', '1m'), config.get('short_ma', 5), config.get('long_ma', 20),
                config.get('mode', 'loose'), config.get('trailing_stop_pct', 1.0), 
                config.get('assist_cond', 'volume'),
                '是' if config.get('volatility_exit', False) else '否', 
                f"{config.get('volatility_threshold', 0.5):.2f}",
                config.get('params', {}).get('vol_multiplier', 'N/A'),
                config.get('params', {}).get('confirmation_pct', 'N/A'),
                config.get('params', {}).get('rsi_period', 'N/A')
            ]
        }
        param_df = pd.DataFrame(param_data)
        param_df.to_excel(writer, sheet_name='策略参数', index=False)
        
        # Sheet 3: 统计信息
        if len(results) > 0:
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