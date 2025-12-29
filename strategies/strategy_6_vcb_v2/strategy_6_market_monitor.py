#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : strategy_6_market_monitor.py
@Description: VCB策略市场扫描监控系统 - 扫描整个市场寻找压缩和突破
"""

import argparse
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 加载环境变量
from dotenv import load_dotenv

load_dotenv()

from strategies.strategy_6_vcb_v2.compression_scanner import CompressionScanner
from strategies.strategy_6_vcb_v2.breakout_watcher import BreakoutWatcher
from strategies.strategy_6_vcb_v2.methods.trader import Strategy6Trader
from strategies.strategy_6_vcb_v2.methods.position_manager import PositionManager
from strategies.strategy_6_vcb_v2.strategy_6 import VCBStrategy
from apis.okx_api.client import get_okx_client
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
                 only_major_coins: bool = False,

                 # v2.1新增参数：压缩评分相关
                 compression_score_threshold: float = 70.0,
                 compression_score_min: float = 60.0,
                 atr_ratio_invalidation_threshold: float = 0.7,

                 # v2.1新增参数：突破检测相关
                 breakout_threshold: float = 0.002,
                 breakout_invalidation_threshold: float = 0.03,
                 pre_breakout_protection_zone: float = 0.005,
                 breakout_body_atr_multiplier: float = 0.4,
                 breakout_shadow_ratio: float = 0.5,
                 breakout_volume_min_multiplier: float = 1.5,
                 breakout_new_high_low_lookback: int = 10,

                 # v2.1新增参数：15分钟验证相关
                 validation_price_deviation_threshold: float = 2.0,
                 validation_atr_relative_threshold: float = 1.5,
                 validation_amplitude_ratio_threshold: float = 0.4,

                 # v2.1新增参数：压缩评分权重
                 score_weight_atr: float = 0.3,
                 score_weight_duration: float = 0.25,
                 score_weight_volume: float = 0.2,
                 score_weight_range: float = 0.15,
                 score_weight_ma: float = 0.1):
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

        # v2.1新增参数
        self.compression_score_threshold = compression_score_threshold
        self.compression_score_min = compression_score_min
        self.atr_ratio_invalidation_threshold = atr_ratio_invalidation_threshold
        self.breakout_threshold = breakout_threshold
        self.breakout_invalidation_threshold = breakout_invalidation_threshold
        self.pre_breakout_protection_zone = pre_breakout_protection_zone
        self.breakout_body_atr_multiplier = breakout_body_atr_multiplier
        self.breakout_shadow_ratio = breakout_shadow_ratio
        self.breakout_volume_min_multiplier = breakout_volume_min_multiplier
        self.breakout_new_high_low_lookback = breakout_new_high_low_lookback
        self.validation_price_deviation_threshold = validation_price_deviation_threshold
        self.validation_atr_relative_threshold = validation_atr_relative_threshold
        self.validation_amplitude_ratio_threshold = validation_amplitude_ratio_threshold
        self.score_weight_atr = score_weight_atr
        self.score_weight_duration = score_weight_duration
        self.score_weight_volume = score_weight_volume
        self.score_weight_range = score_weight_range
        self.score_weight_ma = score_weight_ma

        self.client = get_okx_client()
        self.strategy = VCBStrategy(self.client)
        self.market_data_retriever = MarketDataRetriever(self.client)

        # 初始化仓位管理器
        self.position_manager = PositionManager(
            market_data_retriever=self.market_data_retriever,
            bar='5m',
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
            trading_records_dir = os.path.join(project_root, "trading_records", "strategy_6_vcb_v2")

            if not os.path.exists(trading_records_dir):
                os.makedirs(trading_records_dir)
                logger.info(f"创建交易记录目录: {trading_records_dir}")

            # 生成文件名（启动日期时间）
            start_time = datetime.now()
            filename = start_time.strftime("%Y%m%d_%H%M%S.csv")
            filepath = os.path.join(trading_records_dir, filename)

            self.trading_record_file = filepath

            # 创建CSV文件并写入表头（v2添加压缩评分）
            headers = ['时间', '币种', '交易类型', '成交价格', '成交额(USDT)', '手续费(USDT)', '杠杆倍数',
                       '压缩评分', '平仓盈亏(USDT)']
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

            logger.info(f"交易记录文件已创建: {filepath}")

        except Exception as e:
            logger.error(f"初始化交易记录文件失败: {e}")
            self.trading_record_file = None

    def _record_trade(self, symbol: str, trade_type: str, price: float, trade_amount: float,
                      fee: float, leverage: int, compression_score: Optional[float] = None,
                      pnl: Optional[float] = None):
        """
        记录交易到CSV文件（v2添加压缩评分）

        Args:
            symbol: 交易对符号
            trade_type: 交易类型（"开仓做多"、"开仓做空"、"做多平仓"、"做空平仓"）
            price: 成交价格
            trade_amount: 成交额（USDT）
            fee: 手续费（USDT）
            leverage: 杠杆倍数（现货为1）
            compression_score: 压缩评分（v2新增）
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
                    compression_score_str = f"{compression_score:.2f}" if compression_score is not None else ""
                    writer.writerow([timestamp, symbol, trade_type, f"{price:.8f}",
                                     f"{trade_amount:.4f}", f"{fee:.4f}", leverage,
                                     compression_score_str, pnl_str])

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

            logger.info(f"📊 处理突破信号: {symbol} {'做多' if signal == 1 else '做空'} @ {price:.4f}, 压缩评分={details.get('compression_score', 0):.1f}, 突破质量得分={details.get('breakout_quality', {}).get('conditions_met', 0)}/4")

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

            # 获取压缩评分（v2新增）
            compression_score = compression_event.compression_score if compression_event else None

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

            # 获取入场时的ATR（用于失败退出检查和结构验证）
            entry_atr = None
            entry_atr_short = None
            try:
                limit = self.atr_mid_period + 5
                df = self.market_data_retriever.get_kline(symbol, '5m', limit)  # v2.1：使用5分钟K线获取ATR
                if df is not None and len(df) >= 10:
                    from tools.technical_indicators import atr
                    atr_short = atr(df, 10)
                    if len(atr_short) > 0:
                        entry_atr_short = float(atr_short.iloc[-1])
                    # 用于失败退出检查的ATR（中期）
                    atr_mid = atr(df, self.atr_mid_period)
                    if len(atr_mid) > 0:
                        entry_atr = float(atr_mid.iloc[-1])
            except:
                pass

            # 计算手续费（实际投入金额的0.05%）
            fee = self.trade_amount * 0.0005  # 0.05%

                # 更新持仓
            if symbol not in self.positions:
                # 新开仓
                # v2.1新增：记录突破边界用于延迟确认
                if compression_event and compression_event.breakout_levels:
                    breakout_up = compression_event.breakout_levels.get('up')
                    breakout_down = compression_event.breakout_levels.get('down')
                    # 如果breakout_levels存在但值为None，使用配置的threshold计算
                    if breakout_up is None and compression_event.compression_high:
                        breakout_up = compression_event.compression_high * (1 + self.breakout_threshold)
                    if breakout_down is None and compression_event.compression_low:
                        breakout_down = compression_event.compression_low * (1 - self.breakout_threshold)
                else:
                    breakout_up = None
                    breakout_down = None
                entry_volume = details.get('current_volume', 0)  # v2.1新增：记录入场时的成交量
                
                self.positions[symbol] = {
                    'position': signal,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'highest_price': price if signal == 1 else price,
                    'lowest_price': price if signal == -1 else price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'entry_atr': entry_atr,  # ATR(60) 用于失败退出
                    'entry_atr_short': entry_atr_short,  # ATR(10) 用于结构验证
                    'compression_event': compression_event,
                    'entry_fee': fee,  # 开仓手续费
                    'breakout_up': breakout_up,  # v2.1新增：突破上边界（用于延迟确认）
                    'breakout_down': breakout_down,  # v2.1新增：突破下边界（用于延迟确认）
                    'entry_volume': entry_volume  # v2.1新增：入场时的成交量（用于延迟确认）
                }

                logger.info(f"📊 {symbol} 开仓: 入场={price:.4f}, 止损={stop_loss:.4f}, 止盈={take_profit:.4f}")

                # 记录开仓交易（v2添加压缩评分）
                self._record_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    price=price,
                    trade_amount=self.trade_amount,
                    fee=fee,
                    leverage=actual_leverage,
                    compression_score=compression_score,  # v2新增
                    pnl=None  # 开仓时无盈亏
                )
            else:
                # 如果已有持仓，先平仓再开新仓
                old_position = self.positions[symbol]['position']
                old_entry_price = self.positions[symbol]['entry_price']
                # 获取旧仓位的开仓手续费（如果存在）
                old_entry_fee = self.positions[symbol].get('entry_fee', self.trade_amount * 0.0005)
                # 获取旧仓位的压缩事件和评分（v2新增）
                old_compression_event = self.positions[symbol].get('compression_event')
                old_compression_score = old_compression_event.compression_score if old_compression_event else None

                if old_position != 0:
                    # 计算平仓收益
                    if old_position == 1:
                        return_rate = (price - old_entry_price) / old_entry_price
                        gross_pnl = self.trade_amount * return_rate * actual_leverage
                    else:
                        return_rate = (old_entry_price - price) / old_entry_price
                        gross_pnl = self.trade_amount * return_rate * actual_leverage

                    # 计算平仓手续费（实际投入金额的0.05%）
                    close_fee = self.trade_amount * 0.0005  # 0.05%

                    # 净盈亏 = 毛利 - 开仓手续费 - 平仓手续费
                    net_pnl = gross_pnl - old_entry_fee - close_fee

                    logger.info(f"[模拟交易] {symbol} 平仓: 收益率={return_rate * 100:.2f}%, "
                                f"毛利={gross_pnl:.4f} USDT, 手续费={old_entry_fee + close_fee:.4f} USDT, "
                                f"净盈亏={net_pnl:.4f} USDT")

                    # 记录平仓交易（v2添加压缩评分）
                    close_trade_type = "做多平仓" if old_position == 1 else "做空平仓"
                    self._record_trade(
                        symbol=symbol,
                        trade_type=close_trade_type,
                        price=price,
                        trade_amount=self.trade_amount,
                        fee=close_fee,
                        leverage=actual_leverage,
                        compression_score=old_compression_score,  # v2新增
                        pnl=net_pnl  # 记录净盈亏
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
                entry_atr_short = None
                try:
                    limit = self.atr_mid_period + 5
                    df = self.market_data_retriever.get_kline(symbol, '5m', limit)  # v2.1：使用5分钟K线获取ATR
                    if df is not None and len(df) >= 10:
                        from tools.technical_indicators import atr
                        atr_short = atr(df, 10)
                        if len(atr_short) > 0:
                            entry_atr_short = float(atr_short.iloc[-1])
                        # 用于失败退出检查的ATR（中期）
                        atr_mid = atr(df, self.atr_mid_period)
                        if len(atr_mid) > 0:
                            entry_atr = float(atr_mid.iloc[-1])
                except:
                    pass

                # 计算手续费（实际投入金额的0.05%）
                fee = self.trade_amount * 0.0005  # 0.05%

                # 更新为新持仓
                # v2.1新增：记录突破边界用于延迟确认
                if compression_event and compression_event.breakout_levels:
                    breakout_up = compression_event.breakout_levels.get('up')
                    breakout_down = compression_event.breakout_levels.get('down')
                    # 如果breakout_levels存在但值为None，使用配置的threshold计算
                    if breakout_up is None and compression_event.compression_high:
                        breakout_up = compression_event.compression_high * (1 + self.breakout_threshold)
                    if breakout_down is None and compression_event.compression_low:
                        breakout_down = compression_event.compression_low * (1 - self.breakout_threshold)
                else:
                    breakout_up = None
                    breakout_down = None
                entry_volume = details.get('current_volume', 0)  # v2.1新增：记录入场时的成交量
                
                self.positions[symbol] = {
                    'position': signal,
                    'entry_price': price,
                    'entry_time': datetime.now(),
                    'highest_price': price if signal == 1 else price,
                    'lowest_price': price if signal == -1 else price,
                    'stop_loss': new_stop_loss,
                    'take_profit': new_take_profit,
                    'entry_atr': entry_atr,  # ATR(60) 用于失败退出
                    'entry_atr_short': entry_atr_short,  # ATR(10) 用于结构验证
                    'compression_event': compression_event,
                    'entry_fee': fee,  # 开仓手续费
                    'breakout_up': breakout_up,  # v2.1新增：突破上边界（用于延迟确认）
                    'breakout_down': breakout_down,  # v2.1新增：突破下边界（用于延迟确认）
                    'entry_volume': entry_volume  # v2.1新增：入场时的成交量（用于延迟确认）
                }

                logger.info(f"📊 {symbol} 换仓: 入场={price:.4f}, 止损={new_stop_loss:.4f}, 止盈={new_take_profit:.4f}")

                # 记录新开仓交易（v2添加压缩评分）
                self._record_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    price=price,
                    trade_amount=self.trade_amount,
                    fee=fee,
                    leverage=actual_leverage,
                    compression_score=compression_score,  # v2新增
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
                entry_atr_short = position_info.get('entry_atr_short')
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
                    position_info['highest_price'] = max(position_info.get('highest_price', current_price),
                                                         current_price)
                else:
                    position_info['lowest_price'] = min(position_info.get('lowest_price', current_price), current_price)

                # 计算从入场到现在经过了多少根K线（用于判断是否在验证期内）
                # v2.1：延迟确认和结构验证都使用1分钟K线
                if entry_time:
                    time_diff = datetime.now() - entry_time
                    bars_elapsed = int(time_diff.total_seconds() / 60)  # 使用1分钟K线计算
                else:
                    bars_elapsed = 999  # 如果没有入场时间，假设不在验证期内

                # 0. 检查结构验证（验证期内优先检查，避免过早止损）
                should_close, reason = self.position_manager.check_structure_validation(
                    symbol=symbol,
                    current_price=current_price,
                    position=position,
                    entry_time=entry_time,
                    entry_bar='1m',  # 延迟确认使用1分钟K线
                    compression_event=compression_event,
                    entry_atr_short=entry_atr_short
                )

                if should_close:
                    positions_to_close.append((symbol, reason, current_price))
                    continue

                # v2.1新增：延迟确认机制（第三层过滤，反噪声）
                # 入场后观察1-2根K线，不允许价格重新回到突破边界内，成交量不能快速塌缩
                if bars_elapsed <= 2:  # 前2根K线内
                    breakout_up = position_info.get('breakout_up')
                    breakout_down = position_info.get('breakout_down')
                    entry_volume = position_info.get('entry_volume', 0)
                    
                    if breakout_up is not None and breakout_down is not None:
                        # 检查价格是否回到突破边界内
                        price_back_inside = False
                        if position == 1:  # 做多
                            if current_price < breakout_up:
                                price_back_inside = True
                        else:  # 做空
                            if current_price > breakout_down:
                                price_back_inside = True
                        
                        if price_back_inside:
                            # 价格回到突破边界内，延迟确认失败
                            logger.warning(f"{symbol} 延迟确认失败：价格回到突破边界内（入场后{bars_elapsed}根K线）")
                            positions_to_close.append((symbol, "DELAYED_CONFIRMATION_FAIL", current_price))
                            continue
                        
                        # 检查成交量是否快速塌缩（当前成交量 < 0.5 × 入场成交量）
                        try:
                            # 获取当前成交量
                            limit = 5
                            df = self.market_data_retriever.get_kline(symbol, '1m', limit)
                            if df is not None and len(df) >= 1:
                                current_volume = float(df['vol'].iloc[-1] if 'vol' in df.columns else df['volume'].iloc[-1])
                                if entry_volume > 0 and current_volume < 0.5 * entry_volume:
                                    logger.warning(f"{symbol} 延迟确认失败：成交量快速塌缩（入场后{bars_elapsed}根K线）")
                                    positions_to_close.append((symbol, "DELAYED_CONFIRMATION_FAIL", current_price))
                                    continue
                        except:
                            pass  # 如果无法获取成交量，跳过此检查

                # 1. 检查硬止损（验证期外才检查，避免过早止损）
                # 验证期内（前2根K线）不触发硬止损，只检查结构验证和延迟确认
                if bars_elapsed > 2:
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

            # 获取开仓手续费（如果存在）
            entry_fee = position_info.get('entry_fee', self.trade_amount * 0.0005)
            # 获取压缩事件和评分（v2新增）
            compression_event = position_info.get('compression_event')
            compression_score = compression_event.compression_score if compression_event else None

            # 计算盈亏
            if position == 1:
                return_rate = (close_price - entry_price) / entry_price
                gross_pnl = self.trade_amount * return_rate * actual_leverage
                close_trade_type = "做多平仓"
            else:
                return_rate = (entry_price - close_price) / entry_price
                gross_pnl = self.trade_amount * return_rate * actual_leverage
                close_trade_type = "做空平仓"

            # 计算平仓手续费（实际投入金额的0.05%）
            close_fee = self.trade_amount * 0.0005  # 0.05%

            # 净盈亏 = 毛利 - 开仓手续费 - 平仓手续费
            net_pnl = gross_pnl - entry_fee - close_fee

            logger.info(f"🔴 {symbol} 平仓 [{reason}]: 入场={entry_price:.4f}, 平仓={close_price:.4f}, "
                        f"收益率={return_rate * 100:.2f}%, 毛利={gross_pnl:.4f} USDT, "
                        f"手续费={entry_fee + close_fee:.4f} USDT, 净盈亏={net_pnl:.4f} USDT")

            # 记录平仓交易（v2添加压缩评分）
            self._record_trade(
                symbol=symbol,
                trade_type=close_trade_type,
                price=close_price,
                trade_amount=self.trade_amount,
                fee=close_fee,
                leverage=actual_leverage,
                compression_score=compression_score,  # v2新增
                pnl=net_pnl  # 记录净盈亏
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
        
        # v2.1修改：第一次启动时，如果还在5分钟间隔的第一分钟内（分钟数%5==0且秒数<60），可以直接开始
        # 否则等待到下一个5分钟整数倍时间点
        from datetime import timedelta
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        
        # 检查是否在5分钟间隔的第一分钟内
        is_first_minute = (current_minute % 5 == 0) and (current_second < 60)
        
        if is_first_minute:
            # 在第一分钟内，可以直接开始第一次扫描
            logger.info(f"✅ 当前时间在5分钟间隔的第一分钟内 ({now.strftime('%H:%M:%S')})，立即开始第一次扫描")
        else:
            # 不在第一分钟内，等待到下一个5分钟整数倍时间点
            remainder = current_minute % 5
            if remainder == 0:
                # 如果正好是5的倍数，等待下一个5分钟周期（5分钟）
                minutes_to_wait = 5
            else:
                # 否则等待到下一个5分钟整数倍
                minutes_to_wait = 5 - remainder
            
            # 计算下一个5分钟整数倍的时间点（秒数和微秒归零）
            next_scan_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_wait)
            
            wait_seconds = (next_scan_time - now).total_seconds()
            if wait_seconds > 0:
                logger.info(f"⏰ 等待到下一个5分钟整数倍时间点 ({next_scan_time.strftime('%H:%M')}) 再开始第一次扫描（还需等待 {int(wait_seconds)} 秒）...")
                time.sleep(wait_seconds)
                logger.info(f"✅ 到达扫描时间点 ({datetime.now().strftime('%H:%M:%S')})，开始第一次市场扫描")
        
        # 标记第一次扫描已完成
        first_scan_done = False

        while self.running:
            try:
                # 执行市场扫描
                new_compressions = self.scanner.scan_market(
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period,
                    atr_ratio_threshold=self.atr_ratio_threshold,
                    bb_period=self.bb_period,
                    bb_std=self.bb_std,
                    bb_width_ratio=self.bb_width_ratio,
                    ttl_bars=self.ttl_bars,
                    compression_score_threshold=self.compression_score_threshold,
                    validation_price_deviation_threshold=self.validation_price_deviation_threshold,
                    validation_atr_relative_threshold=self.validation_atr_relative_threshold,
                    validation_amplitude_ratio_threshold=self.validation_amplitude_ratio_threshold,
                    breakout_threshold=self.breakout_threshold,
                    breakout_invalidation_threshold=self.breakout_invalidation_threshold,
                    score_weight_atr=self.score_weight_atr,
                    score_weight_duration=self.score_weight_duration,
                    score_weight_volume=self.score_weight_volume,
                    score_weight_range=self.score_weight_range,
                    score_weight_ma=self.score_weight_ma
                )

                # 清理过期压缩事件
                self.strategy.cleanup_compression_pool(
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period,
                    compression_score_min=self.compression_score_min,
                    atr_ratio_invalidation_threshold=self.atr_ratio_invalidation_threshold,
                    pre_breakout_protection_zone=self.pre_breakout_protection_zone
                )

                # 标记第一次扫描已完成
                first_scan_done = True
                
                # 等待到下一个5分钟整数倍时间点（保持完美的5分钟间隔）
                now = datetime.now()
                current_minute = now.minute
                remainder = current_minute % 5
                
                if remainder == 0:
                    # 如果正好是5的倍数，等待下一个5分钟周期（5分钟）
                    minutes_to_wait = 5
                else:
                    # 否则等待到下一个5分钟整数倍
                    minutes_to_wait = 5 - remainder
                
                # 计算下一个5分钟整数倍的时间点（秒数和微秒归零）
                next_scan_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_wait)
                wait_seconds = (next_scan_time - now).total_seconds()
                
                if wait_seconds > 0:
                    logger.debug(f"等待到下一个5分钟整数倍时间点 ({next_scan_time.strftime('%H:%M:%S')}) 再开始下次扫描（还需等待 {int(wait_seconds)} 秒）")
                    time.sleep(wait_seconds)
                else:
                    # 如果已经过了，等待5分钟
                    time.sleep(self.scan_interval_minutes * 60)

            except Exception as e:
                logger.error(f"扫描循环出错: {e}")
                time.sleep(60)  # 出错后等待1分钟再继续

    def _watch_loop(self):
        """监控循环（消费者线程）"""
        logger.info("启动突破监控线程（消费者）...")

        # 等待到下一分钟开始时启动第一次扫描
        from datetime import timedelta
        now = datetime.now()
        # 计算到下一分钟开始的时间（秒数和微秒归零，分钟+1）
        next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        wait_seconds = (next_minute - now).total_seconds()
        if wait_seconds > 0:
            logger.info(f"⏰ 突破监控等待到下一分钟开始 ({next_minute.strftime('%H:%M:%S')}) 再启动（还需等待 {int(wait_seconds)} 秒）...")
            time.sleep(wait_seconds)
            logger.info(f"✅ 到达突破监控时间点 ({datetime.now().strftime('%H:%M:%S')})，开始第一次突破扫描")

        while self.running:
            try:
                # 记录扫描开始时间
                scan_start_time = datetime.now()
                
                # 监控压缩池中的币种
                breakouts = self.watcher.watch_compression_pool(
                    volume_period=self.volume_period,
                    volume_multiplier=self.volume_multiplier,
                    breakout_threshold=self.breakout_threshold,  # v2.1新增
                    breakout_body_atr_multiplier=self.breakout_body_atr_multiplier,
                    breakout_shadow_ratio=self.breakout_shadow_ratio,
                    breakout_volume_min_multiplier=self.breakout_volume_min_multiplier,
                    breakout_new_high_low_lookback=self.breakout_new_high_low_lookback
                )

                # 检查所有持仓的平仓条件
                self._check_positions()

                # 清理过期压缩事件
                self.strategy.cleanup_compression_pool(
                    atr_short_period=self.atr_short_period,
                    atr_mid_period=self.atr_mid_period,
                    compression_score_min=self.compression_score_min,
                    atr_ratio_invalidation_threshold=self.atr_ratio_invalidation_threshold,
                    pre_breakout_protection_zone=self.pre_breakout_protection_zone
                )

                # 计算扫描耗时
                scan_end_time = datetime.now()
                scan_duration = (scan_end_time - scan_start_time).total_seconds()
                
                # 计算到下一分钟开始还需要等待的时间
                current_time = datetime.now()
                next_minute_start = current_time.replace(second=0, microsecond=0) + timedelta(minutes=1)
                wait_seconds = (next_minute_start - current_time).total_seconds()
                
                # 如果扫描耗时超过1分钟，立即开始下一次（不等待）
                if wait_seconds <= 0:
                    logger.warning(f"⚠️ 突破扫描耗时 {scan_duration:.2f} 秒，超过1分钟，立即开始下一次扫描")
                    continue
                
                # 等待到下一分钟开始
                logger.debug(f"突破扫描完成，耗时 {scan_duration:.2f} 秒，等待 {wait_seconds:.2f} 秒到下一分钟开始")
                time.sleep(wait_seconds)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                # 出错后也等待到下一分钟开始
                now = datetime.now()
                next_minute = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                wait_seconds = (next_minute - now).total_seconds()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                else:
                    time.sleep(1)  # 如果已经过了，只等1秒

    def start(self):
        """启动监控系统"""
        if self.running:
            logger.warning("监控系统已在运行")
            return

        self.running = True

        logger.info("=" * 60)
        logger.info("VCB市场监控系统启动 (V2.1)")
        logger.info("=" * 60)
        logger.info(f"扫描参数:")
        logger.info(f"  - 最小交易量: {self.min_vol_ccy:,.0f} {self.currency}")
        logger.info(f"  - 扫描间隔: {self.scan_interval_minutes} 分钟")
        logger.info(f"压缩检测参数:")
        logger.info(f"  - ATR比率阈值: {self.atr_ratio_threshold} (短期/中期)")
        logger.info(f"  - 压缩评分阈值: ≥{self.compression_score_threshold} (最低保留: {self.compression_score_min})")
        logger.info(f"  - 临界保护区: ±{self.pre_breakout_protection_zone*100:.1f}% (v2.1新增)")
        logger.info(f"突破检测参数:")
        logger.info(f"  - 突破幅度: {self.breakout_threshold*100:.2f}% (v2.1从1%降低)")
        logger.info(f"  - 成交量倍数: {self.volume_multiplier}× (v2.1从1.5降低)")
        logger.info(f"  - 影线控制: <{self.breakout_shadow_ratio*100:.0f}%实体 (v2.1从30%放宽)")
        logger.info(f"风险管理参数:")
        take_profit_mode_names = {
            'r_multiple': 'R倍止盈',
            'bb_middle': '布林中轨止盈',
            'bb_opposite': '对侧轨道止盈',
            'atr_trailing': 'ATR跟踪止盈'
        }
        logger.info(f"  - 止盈模式: {take_profit_mode_names.get(self.take_profit_mode, self.take_profit_mode)}")
        logger.info(f"  - 止盈R倍数: 主流币={self.take_profit_r_major}R, 山寨币={self.take_profit_r_alt}R")
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
        take_profit_r=default_config.get('take_profit_r', 2.0),
        take_profit_mode=default_config.get('take_profit_mode', 'r_multiple'),
        take_profit_r_major=default_config.get('take_profit_r_major', 1.5),
        take_profit_r_alt=default_config.get('take_profit_r_alt', 2.5),
        failure_exit_bars=default_config.get('failure_exit_bars', 10),
        failure_exit_atr_threshold=default_config.get('failure_exit_atr_threshold', 1.2),
        break_even_r=default_config.get('break_even_r', 1.0),
        only_major_coins=default_config.get('only_major_coins', False),
        # v2.1新增参数
        compression_score_threshold=default_config.get('compression_score_threshold', 70.0),
        compression_score_min=default_config.get('compression_score_min', 60.0),
        atr_ratio_invalidation_threshold=default_config.get('atr_ratio_invalidation_threshold', 0.7),
        breakout_threshold=default_config.get('breakout_threshold', 0.002),
        breakout_invalidation_threshold=default_config.get('breakout_invalidation_threshold', 0.03),
        pre_breakout_protection_zone=default_config.get('pre_breakout_protection_zone', 0.005),
        breakout_body_atr_multiplier=default_config.get('breakout_body_atr_multiplier', 0.4),
        breakout_shadow_ratio=default_config.get('breakout_shadow_ratio', 0.5),
        breakout_volume_min_multiplier=default_config.get('breakout_volume_min_multiplier', 1.5),
        breakout_new_high_low_lookback=default_config.get('breakout_new_high_low_lookback', 10),
        validation_price_deviation_threshold=default_config.get('validation_price_deviation_threshold', 2.0),
        validation_atr_relative_threshold=default_config.get('validation_atr_relative_threshold', 1.5),
        validation_amplitude_ratio_threshold=default_config.get('validation_amplitude_ratio_threshold', 0.4),
        score_weight_atr=default_config.get('score_weight_atr', 0.3),
        score_weight_duration=default_config.get('score_weight_duration', 0.25),
        score_weight_volume=default_config.get('score_weight_volume', 0.2),
        score_weight_range=default_config.get('score_weight_range', 0.15),
        score_weight_ma=default_config.get('score_weight_ma', 0.1)
    )

    monitor.run()


if __name__ == "__main__":
    main()
