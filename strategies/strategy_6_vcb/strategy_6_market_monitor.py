#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : strategy_6_market_monitor.py
@Description: VCB策略市场扫描监控系统 - 扫描整个市场寻找压缩和突破
"""

import argparse
import json
import os
import sys
import time
import threading
import csv
from datetime import datetime, timedelta
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

from strategies.strategy_6_vcb.compression_scanner import CompressionScanner
from strategies.strategy_6_vcb.breakout_watcher import BreakoutWatcher
from strategies.strategy_6_vcb.methods.trader import Strategy6Trader
from strategies.strategy_6_vcb.methods.position_manager import PositionManager
from strategies.strategy_6_vcb.strategy_6 import VCBStrategy
from apis.okx_api.client import OKXClient, get_okx_client
from apis.okx_api.market_data import MarketDataRetriever
from utils.logger import logger


class VCBMarketMonitor:
    """
    VCB市场监控器
    
    架构：
    - CompressionScanner（生产者）：定期扫描市场，发现压缩 → 放入Pool
    - BreakoutWatcher（消费者）：监控Pool中的币种，检测突破 → 执行交易
    """
    
    def __init__(self,
                 # 扫描参数
                 min_vol_ccy: float = 10000000,  # 最小24h交易量（USDT）
                 currency: str = 'USDT',
                 inst_type: str = 'SWAP',
                 scan_interval_minutes: int = 5,  # 扫描间隔（分钟）
                 max_workers: int = 10,
                 
                 # 压缩检测参数
                 bar: str = '1m',
                 atr_short_period: int = 10,
                 atr_mid_period: int = 60,
                 atr_ratio_threshold: float = 0.5,
                 bb_period: int = 20,
                 bb_std: int = 2,
                 bb_width_ratio: float = 0.7,
                 ttl_bars: int = 30,
                 
                 # 突破检测参数
                 volume_period: int = 20,
                 volume_multiplier: float = 1.0,
                 
                 # 交易参数
                 trade: bool = False,
                 trade_amount: float = 10.0,
                 trade_mode: int = 3,
                 leverage: int = 3,
                 
                 # 风险管理参数
                 trailing_stop_pct: float = 1.0,
                 stop_loss_atr_multiplier: float = 0.8,
                 take_profit_r: float = 2.0,
                 take_profit_mode: str = 'r_multiple',  # 'r_multiple', 'bb_middle', 'bb_opposite', 'atr_trailing'
                 take_profit_r_major: float = 1.5,
                 take_profit_r_alt: float = 2.5,
                 failure_exit_bars: int = 10,
                 failure_exit_atr_threshold: float = 1.2,
                 break_even_r: float = 1.0,
                 
                 # 币种过滤参数
                 only_major_coins: bool = False):
        """
        初始化VCB市场监控器
        
        Args:
            min_vol_ccy: 最小24小时交易量（USDT）
            currency: 交易对货币
            inst_type: 合约类型（SPOT/SWAP）
            scan_interval_minutes: 市场扫描间隔（分钟）
            max_workers: 并行扫描线程数
            
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            atr_ratio_threshold: ATR比率阈值
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            bb_width_ratio: 布林带宽度收缩比率
            ttl_bars: 压缩事件TTL（K线数量）
            
            volume_period: 成交量均线周期
            volume_multiplier: 成交量放大倍数
            
            trade: 是否真实交易
            trade_amount: 每次交易金额（USDT）
            trade_mode: 交易模式（1=现货, 2=全仓杠杆, 3=逐仓杠杆）
            leverage: 杠杆倍数
            
            trailing_stop_pct: 移动止损百分比
            stop_loss_atr_multiplier: 止损ATR倍数
            take_profit_r: 止盈R倍数（默认）
            take_profit_mode: 止盈模式
            take_profit_r_major: 主流币止盈R倍数
            take_profit_r_alt: 山寨币止盈R倍数
            failure_exit_bars: 失败退出K线数量
            failure_exit_atr_threshold: 失败退出ATR阈值
            break_even_r: Break-even触发R倍数
            only_major_coins: 是否只做主流币
        """
        # 保存参数
        self.min_vol_ccy = min_vol_ccy
        self.currency = currency
        self.inst_type = inst_type
        self.scan_interval_minutes = scan_interval_minutes
        self.max_workers = max_workers
        self.bar = bar
        self.atr_short_period = atr_short_period
        self.atr_mid_period = atr_mid_period
        self.atr_ratio_threshold = atr_ratio_threshold
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_width_ratio = bb_width_ratio
        self.ttl_bars = ttl_bars
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.trade = trade
        self.trade_amount = trade_amount
        self.trade_mode = trade_mode
        self.leverage = leverage
        self.trailing_stop_pct = trailing_stop_pct
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        self.take_profit_r = take_profit_r
        self.take_profit_mode = take_profit_mode
        self.take_profit_r_major = take_profit_r_major
        self.take_profit_r_alt = take_profit_r_alt
        self.failure_exit_bars = failure_exit_bars
        self.failure_exit_atr_threshold = failure_exit_atr_threshold
        self.break_even_r = break_even_r
        self.only_major_coins = only_major_coins

        self.client = get_okx_client()
        self.strategy = VCBStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)
        
        # 初始化仓位管理器
        self.position_manager = PositionManager(
            market_data_retriever=self.market_data_retriever,
            bar=bar,
            atr_mid_period=atr_mid_period,
            stop_loss_atr_multiplier=stop_loss_atr_multiplier,
            take_profit_mode=take_profit_mode,
            take_profit_r=take_profit_r,
            take_profit_r_major=take_profit_r_major,
            take_profit_r_alt=take_profit_r_alt,
            failure_exit_bars=failure_exit_bars,
            failure_exit_atr_threshold=failure_exit_atr_threshold,
            break_even_r=break_even_r
        )
        
        # 初始化扫描器和监控器
        self.scanner = CompressionScanner(
            client=self.client,
            strategy=self.strategy,
            min_vol_ccy=min_vol_ccy,
            currency=currency,
            inst_type=inst_type,
            max_workers=max_workers,
            only_major_coins=only_major_coins
        )
        
        self.watcher = BreakoutWatcher(
            client=self.client,
            strategy=self.strategy
        )
        
        # 注册突破回调
        self.watcher.register_breakout_callback(self._on_breakout)
        
        # 初始化交易器（如果需要真实交易）
        self.trader = None
        if trade:
            self.trader = Strategy6Trader(self.client, trade_amount, trade_mode, leverage)
        
        # 运行状态
        self.running = False
        self.scan_thread = None
        self.watch_thread = None
        
        # 持仓管理（多币种）
        self.positions: Dict[str, Dict] = {}  # {symbol: {position, entry_price, ...}}
        
        # 交易记录文件
        self.trading_record_file = None
        self.trading_record_lock = threading.Lock()
        self._init_trading_record_file()
    
    def _init_trading_record_file(self):
        """初始化交易记录CSV文件"""
        try:
            # 创建交易记录目录
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            trading_records_dir = os.path.join(project_root, "trading_records", "strategy_6_vcb")
            
            if not os.path.exists(trading_records_dir):
                os.makedirs(trading_records_dir)
                logger.info(f"创建交易记录目录: {trading_records_dir}")
            
            # 生成文件名（启动日期时间）
            start_time = datetime.now()
            filename = start_time.strftime("%Y%m%d_%H%M%S.csv")
            filepath = os.path.join(trading_records_dir, filename)
            
            self.trading_record_file = filepath
            
            # 创建CSV文件并写入表头
            headers = ['时间', '币种', '交易类型', '成交额(USDT)', '杠杆倍数', '平仓盈亏(USDT)']
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            
            logger.info(f"交易记录文件已创建: {filepath}")
            
        except Exception as e:
            logger.error(f"初始化交易记录文件失败: {e}")
            self.trading_record_file = None
    
    def _record_trade(self, symbol: str, trade_type: str, trade_amount: float, 
                     leverage: int, pnl: Optional[float] = None):
        """
        记录交易到CSV文件
        
        Args:
            symbol: 交易对符号
            trade_type: 交易类型（"开仓做多"、"开仓做空"、"做多平仓"、"做空平仓"）
            trade_amount: 成交额（USDT）
            leverage: 杠杆倍数（现货为1）
            pnl: 平仓盈亏（USDT），开仓时为None
        """
        if not self.trading_record_file:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pnl_str = f"{pnl:.4f}" if pnl is not None else ""
            
            with self.trading_record_lock:
                with open(self.trading_record_file, 'a', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, symbol, trade_type, f"{trade_amount:.4f}", 
                                    leverage, pnl_str])
        
        except Exception as e:
            logger.error(f"记录交易失败: {e}")
    
    def _on_breakout(self, symbol: str, signal: int, details: Dict):
        """
        突破信号回调函数
        
        Args:
            symbol: 交易对符号
            signal: 信号（1=做多, -1=做空）
            details: 突破详情
        """
        try:
            price = details.get('current_price', 0)
            if price <= 0:
                logger.warning(f"{symbol} 突破信号价格无效: {price}")
                return
            
            logger.info(f"📊 处理突破信号: {symbol} {'做多' if signal == 1 else '做空'} @ {price:.4f}")
            
            # 执行交易
            self._execute_trade(symbol, signal, price, details)
            
        except Exception as e:
            logger.error(f"处理突破信号时出错 {symbol}: {e}")
    
    def _execute_trade(self, symbol: str, signal: int, price: float, details: Dict):
        """
        执行交易
        
        Args:
            symbol: 交易对符号
            signal: 信号（1=做多, -1=做空）
            price: 价格
            details: 交易详情
        """
        try:
            action = "LONG_OPEN" if signal == 1 else "SHORT_OPEN"
            trade_type = "开仓做多" if signal == 1 else "开仓做空"
            
            # 确定杠杆倍数（现货为1，杠杆模式使用配置的杠杆）
            actual_leverage = 1 if self.trade_mode == 1 else self.leverage
            
            # 记录模拟交易
            logger.info(f"[模拟交易] {symbol} {action}: 价格={price:.4f}")
            
            # 获取压缩事件（用于计算止损止盈）
            compression_event = details.get('compression_event')
            
            # 计算止损和止盈价格
            stop_loss = self.position_manager.calculate_stop_loss(
                symbol=symbol,
                entry_price=price,
                position=signal,
                compression_event=compression_event
            )
            
            take_profit = self.position_manager.calculate_take_profit(
                symbol=symbol,
                entry_price=price,
                stop_loss=stop_loss,
                position=signal,
                compression_event=compression_event
            )
            
            # 获取入场时的ATR（用于失败退出检查）
            entry_atr = None
            try:
                limit = self.atr_mid_period + 5
                df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
                if df is not None and len(df) >= 10:
                    from tools.technical_indicators import atr
                    atr_short = atr(df, 10)
                    if len(atr_short) > 0:
                        entry_atr = float(atr_short.iloc[-1])
            except:
                pass
            
            # 更新持仓
            if symbol not in self.positions:
                # 新开仓
                self.positions[symbol] = {
                    'position': signal,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'highest_price': price if signal == 1 else price,
                    'lowest_price': price if signal == -1 else price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_atr': entry_atr,
                    'compression_event': compression_event
                }
                
                logger.info(f"📊 {symbol} 开仓: 入场={price:.4f}, 止损={stop_loss:.4f}, 止盈={take_profit:.4f}")
                
                # 记录开仓交易
                self._record_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    trade_amount=self.trade_amount,
                    leverage=actual_leverage,
                    pnl=None  # 开仓时无盈亏
                )
            else:
                # 如果已有持仓，先平仓再开新仓
                old_position = self.positions[symbol]['position']
                old_entry_price = self.positions[symbol]['entry_price']
                
                if old_position != 0:
                    # 计算平仓收益
                    if old_position == 1:
                        return_rate = (price - old_entry_price) / old_entry_price
                        pnl = self.trade_amount * return_rate * actual_leverage
                    else:
                        return_rate = (old_entry_price - price) / old_entry_price
                        pnl = self.trade_amount * return_rate * actual_leverage
                    
                    logger.info(f"[模拟交易] {symbol} 平仓: 收益率={return_rate*100:.2f}%, 盈亏={pnl:.4f} USDT")
                    
                    # 记录平仓交易
                    close_trade_type = "做多平仓" if old_position == 1 else "做空平仓"
                    self._record_trade(
                        symbol=symbol,
                        trade_type=close_trade_type,
                        trade_amount=self.trade_amount,
                        leverage=actual_leverage,
                        pnl=pnl
                    )
                
                # 计算新持仓的止损止盈
                new_stop_loss = self.position_manager.calculate_stop_loss(
                    symbol=symbol,
                    entry_price=price,
                    position=signal,
                    compression_event=compression_event
                )
                
                new_take_profit = self.position_manager.calculate_take_profit(
                    symbol=symbol,
                    entry_price=price,
                    stop_loss=new_stop_loss,
                    position=signal,
                    compression_event=compression_event
                )
                
                # 获取入场时的ATR
                entry_atr = None
                try:
                    limit = self.atr_mid_period + 5
                    df = self.market_data_retriever.get_kline(symbol, self.bar, limit)
                    if df is not None and len(df) >= 10:
                        from tools.technical_indicators import atr
                        atr_short = atr(df, 10)
                        if len(atr_short) > 0:
                            entry_atr = float(atr_short.iloc[-1])
                except:
                    pass
                
                # 更新为新持仓
                self.positions[symbol] = {
                    'position': signal,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'highest_price': price if signal == 1 else price,
                    'lowest_price': price if signal == -1 else price,
                    'stop_loss': new_stop_loss,
                    'take_profit': new_take_profit,
                    'entry_atr': entry_atr,
                    'compression_event': compression_event
                }
                
                logger.info(f"📊 {symbol} 换仓: 入场={price:.4f}, 止损={new_stop_loss:.4f}, 止盈={new_take_profit:.4f}")
                
                # 记录新开仓交易
                self._record_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    trade_amount=self.trade_amount,
                    leverage=actual_leverage,
                    pnl=None  # 开仓时无盈亏
                )
            
            # 真实交易
            if self.trade and self.trader:
                try:
                    trade_result = self.trader.execute_trade(action, symbol, price)
                    if trade_result:
                        logger.info(f"✅ [真实交易] {symbol} {action} 成功")
                    else:
                        logger.error(f"❌ [真实交易] {symbol} {action} 失败")
                except Exception as e:
                    logger.error(f"❌ [真实交易] {symbol} {action} 异常: {e}")
            
        except Exception as e:
            logger.error(f"执行交易时出错 {symbol}: {e}")
    
    def _check_positions(self):
        """
        检查所有持仓的平仓条件
        
        包括：
        1. 硬止损检查
        2. 主动止盈检查
        3. 失败退出检查
        4. Break-even检查
        """
        if not self.positions:
            return
        
        positions_to_close = []
        
        for symbol, position_info in list(self.positions.items()):
            try:
                position = position_info.get('position', 0)
                if position == 0:
                    continue
                
                entry_price = position_info.get('entry_price', 0)
                entry_time = position_info.get('entry_time')
                stop_loss = position_info.get('stop_loss', 0)
                take_profit = position_info.get('take_profit', 0)
                entry_atr = position_info.get('entry_atr')
                compression_event = position_info.get('compression_event')
                
                if entry_price <= 0:
                    continue
                
                # 获取当前价格
                ticker = self.market_data_retriever.get_ticker_by_symbol(symbol)
                if not ticker or not ticker.last:
                    continue
                
                current_price = float(ticker.last)
                
                # 更新最高/最低价
                if position == 1:
                    position_info['highest_price'] = max(position_info.get('highest_price', current_price), current_price)
                else:
                    position_info['lowest_price'] = min(position_info.get('lowest_price', current_price), current_price)
                
                # 1. 检查硬止损
                should_close, reason = self.position_manager.check_hard_stop_loss(
                    symbol=symbol,
                    current_price=current_price,
                    position=position,
                    stop_loss=stop_loss
                )
                
                if should_close:
                    positions_to_close.append((symbol, reason, current_price))
                    continue
                
                # 2. 检查Break-even（更新止损）
                should_update_sl, new_stop_loss = self.position_manager.check_break_even(
                    symbol=symbol,
                    current_price=current_price,
                    position=position,
                    entry_price=entry_price,
                    stop_loss=stop_loss
                )
                
                if should_update_sl and new_stop_loss != stop_loss:
                    position_info['stop_loss'] = new_stop_loss
                    logger.info(f"🔄 {symbol} Break-even触发: 止损更新为 {new_stop_loss:.4f}")
                
                # 3. 检查主动止盈
                should_close, reason, new_take_profit = self.position_manager.check_take_profit(
                    symbol=symbol,
                    current_price=current_price,
                    position=position,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    compression_event=compression_event
                )
                
                if should_close:
                    positions_to_close.append((symbol, reason, current_price))
                    continue
                
                # 如果止盈价格更新（ATR跟踪）
                if new_take_profit and new_take_profit != take_profit:
                    position_info['take_profit'] = new_take_profit
                
                # 4. 检查失败退出
                if entry_time and entry_atr:
                    should_close, reason = self.position_manager.check_failure_exit(
                        symbol=symbol,
                        entry_time=entry_time,
                        entry_atr=entry_atr
                    )
                    
                    if should_close:
                        positions_to_close.append((symbol, reason, current_price))
                        continue
                
            except Exception as e:
                logger.error(f"检查持仓 {symbol} 时出错: {e}")
                continue
        
        # 执行平仓
        for symbol, reason, close_price in positions_to_close:
            self._close_position(symbol, reason, close_price)
    
    def _close_position(self, symbol: str, reason: str, close_price: float):
        """
        平仓
        
        Args:
            symbol: 交易对符号
            reason: 平仓原因
            close_price: 平仓价格
        """
        if symbol not in self.positions:
            return
        
        try:
            position_info = self.positions[symbol]
            position = position_info.get('position', 0)
            entry_price = position_info.get('entry_price', 0)
            
            if position == 0 or entry_price <= 0:
                return
            
            # 确定杠杆倍数
            actual_leverage = 1 if self.trade_mode == 1 else self.leverage
            
            # 计算盈亏
            if position == 1:
                return_rate = (close_price - entry_price) / entry_price
                pnl = self.trade_amount * return_rate * actual_leverage
                close_trade_type = "做多平仓"
            else:
                return_rate = (entry_price - close_price) / entry_price
                pnl = self.trade_amount * return_rate * actual_leverage
                close_trade_type = "做空平仓"
            
            logger.info(f"🔴 {symbol} 平仓 [{reason}]: 入场={entry_price:.4f}, 平仓={close_price:.4f}, "
                       f"收益率={return_rate*100:.2f}%, 盈亏={pnl:.4f} USDT")
            
            # 记录平仓交易
            self._record_trade(
                symbol=symbol,
                trade_type=close_trade_type,
                trade_amount=self.trade_amount,
                leverage=actual_leverage,
                pnl=pnl
            )
            
            # 真实交易平仓
            if self.trade and self.trader:
                try:
                    action = "LONG_CLOSE" if position == 1 else "SHORT_CLOSE"
                    trade_result = self.trader.execute_trade(action, symbol, close_price)
                    if trade_result:
                        logger.info(f"✅ [真实交易] {symbol} {action} 成功")
                    else:
                        logger.error(f"❌ [真实交易] {symbol} {action} 失败")
                except Exception as e:
                    logger.error(f"❌ [真实交易] {symbol} 平仓异常: {e}")
            
            # 移除持仓
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"平仓 {symbol} 时出错: {e}")
    
    def _scan_loop(self):
        """扫描循环（生产者线程）"""
        logger.info("启动市场扫描线程（生产者）...")
        
        while self.running:
            try:
                # 执行市场扫描
                new_compressions = self.scanner.scan_market(
                    bar=self.bar,
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period,
                    atr_ratio_threshold=self.atr_ratio_threshold,
                    bb_period=self.bb_period,
                    bb_std=self.bb_std,
                    bb_width_ratio=self.bb_width_ratio,
                    ttl_bars=self.ttl_bars
                )
                
                # 清理过期压缩事件
                self.strategy.cleanup_compression_pool(
                    bar=self.bar,
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period
                )
                
                # 等待下次扫描
                time.sleep(self.scan_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"扫描循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    def _watch_loop(self):
        """监控循环（消费者线程）"""
        logger.info("启动突破监控线程（消费者）...")
        
        # 计算等待时间（根据K线周期）
        if self.bar == '1m':
            wait_seconds = 60
        elif self.bar == '5m':
            wait_seconds = 300
        else:
            wait_seconds = 60  # 默认1分钟
        
        while self.running:
            try:
                # 监控压缩池中的币种
                breakouts = self.watcher.watch_compression_pool(
                    bar=self.bar,
                    volume_period=self.volume_period,
                    volume_multiplier=self.volume_multiplier
                )
                
                # 检查所有持仓的平仓条件
                self._check_positions()
                
                # 清理过期压缩事件
                self.strategy.cleanup_compression_pool(
                    bar=self.bar,
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period
                )
                
                # 等待下次检查（每根K线检查一次）
                time.sleep(wait_seconds)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续
    
    def start(self):
        """启动监控系统"""
        if self.running:
            logger.warning("监控系统已在运行")
            return
        
        self.running = True
        
        logger.info("=" * 60)
        logger.info("VCB市场监控系统启动")
        logger.info("=" * 60)
        logger.info(f"扫描参数:")
        logger.info(f"  - 最小交易量: {self.min_vol_ccy:,.0f} {self.currency}")
        logger.info(f"  - 扫描间隔: {self.scan_interval_minutes} 分钟")
        logger.info(f"  - 并行线程数: {self.max_workers}")
        logger.info(f"压缩检测参数:")
        logger.info(f"  - K线周期: {self.bar}")
        logger.info(f"  - ATR: {self.atr_short_period}/{self.atr_mid_period}, 阈值={self.atr_ratio_threshold}")
        logger.info(f"  - 布林带: {self.bb_period}, {self.bb_std}, 宽度比率={self.bb_width_ratio}")
        logger.info(f"  - TTL: {self.ttl_bars} 根K线")
        logger.info(f"突破检测参数:")
        logger.info(f"  - 成交量周期: {self.volume_period}, 倍数: {self.volume_multiplier}")
        logger.info(f"交易模式: {'真实交易' if self.trade else '模拟交易'}")
        if self.trade:
            trade_mode_names = {1: "现货", 2: "全仓杠杆", 3: "逐仓杠杆"}
            logger.info(f"  - 交易模式: {trade_mode_names.get(self.trade_mode, '未知')}")
            logger.info(f"  - 每次交易金额: {self.trade_amount} USDT")
            if self.trade_mode in [2, 3]:
                logger.info(f"  - 杠杆倍数: {self.leverage}x")
        if self.trading_record_file:
            logger.info(f"交易记录文件: {self.trading_record_file}")
        logger.info("=" * 60)
        
        # 启动扫描线程（生产者）
        self.scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.scan_thread.start()
        
        # 启动监控线程（消费者）
        self.watch_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.watch_thread.start()
        
        logger.info("✅ 监控系统已启动（按 Ctrl+C 停止）")
    
    def stop(self):
        """停止监控系统"""
        if not self.running:
            return
        
        logger.info("正在停止监控系统...")
        self.running = False
        
        # 等待线程结束
        if self.scan_thread:
            self.scan_thread.join(timeout=5)
        if self.watch_thread:
            self.watch_thread.join(timeout=5)
        
        # 打印统计信息
        scan_stats = self.scanner.get_scan_stats()
        watch_stats = self.watcher.get_watch_stats()
        
        logger.info("=" * 60)
        logger.info("监控系统已停止")
        logger.info("=" * 60)
        logger.info(f"扫描统计:")
        logger.info(f"  - 扫描次数: {scan_stats['scan_count']}")
        logger.info(f"  - 最后扫描时间: {scan_stats['last_scan_time']}")
        logger.info(f"  - 最后扫描币种数: {scan_stats['last_scan_symbols_count']}")
        logger.info(f"突破统计:")
        logger.info(f"  - 突破次数: {watch_stats['breakout_count']}")
        logger.info(f"  - 最后突破时间: {watch_stats['last_breakout_time']}")
        logger.info(f"当前状态:")
        logger.info(f"  - 压缩池大小: {self.strategy.get_compression_pool_size()}")
        logger.info(f"  - 持仓数量: {len(self.positions)}")
        logger.info("=" * 60)
    
    def run(self):
        """运行监控系统（阻塞）"""
        try:
            self.start()
            
            # 主循环：定期打印状态
            while self.running:
                time.sleep(60)  # 每分钟打印一次状态
                
                # 打印当前状态
                pool_size = self.strategy.get_compression_pool_size()
                pool_symbols = self.strategy.get_compression_pool_symbols()
                position_count = len([p for p in self.positions.values() if p['position'] != 0])
                
                logger.info(f"[状态] 压缩池: {pool_size} 个币种, 持仓: {position_count} 个")
                if pool_symbols:
                    logger.info(f"[状态] 压缩池币种: {', '.join(pool_symbols[:10])}" + 
                              (f" ... (共{len(pool_symbols)}个)" if len(pool_symbols) > 10 else ""))
                
        except KeyboardInterrupt:
            logger.info("\n收到停止信号...")
        finally:
            self.stop()


def load_config_from_file(config_path: str) -> dict:
    """从配置文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"从配置文件 {config_path} 加载配置成功")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return {}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VCB策略市场扫描监控系统')
    parser.add_argument('--config', type=str, help='配置文件路径', default=None)
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        default_config = load_config_from_file(args.config)
    else:
        config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.json')
        if os.path.exists(config_path):
            default_config = load_config_from_file(config_path)
        else:
            default_config = {}
    
    # 使用配置或默认值
    monitor = VCBMarketMonitor(
        min_vol_ccy=default_config.get('min_vol_ccy', 10000000),
        currency=default_config.get('currency', 'USDT'),
        inst_type=default_config.get('inst_type', 'SWAP'),
        scan_interval_minutes=default_config.get('scan_interval_minutes', 5),
        max_workers=default_config.get('max_workers', 10),
        bar=default_config.get('bar', '1m'),
        atr_short_period=default_config.get('atr_short_period', 10),
        atr_mid_period=default_config.get('atr_mid_period', 60),
        atr_ratio_threshold=default_config.get('atr_ratio_threshold', 0.5),
        bb_period=default_config.get('bb_period', 20),
        bb_std=default_config.get('bb_std', 2),
        bb_width_ratio=default_config.get('bb_width_ratio', 0.7),
        ttl_bars=default_config.get('ttl_bars', 30),
        volume_period=default_config.get('volume_period', 20),
        volume_multiplier=default_config.get('volume_multiplier', 1.0),
        trade=default_config.get('trade', False),
        trade_amount=default_config.get('trade_amount', 10.0),
        trade_mode=default_config.get('trade_mode', 3),
        leverage=default_config.get('leverage', 3),
        trailing_stop_pct=default_config.get('trailing_stop_pct', 1.0),
        stop_loss_atr_multiplier=default_config.get('stop_loss_atr_multiplier', 0.8),
        take_profit_r=default_config.get('take_profit_r', 2.0)
    )
    
    monitor.run()


if __name__ == "__main__":
    main()

