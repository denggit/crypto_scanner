#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 2025
@File       : compression_scanner.py
@Description: Compression Scanner（生产者）- 扫描整个市场寻找压缩事件
"""

import os
import sys
import time
import concurrent.futures
from datetime import datetime
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from apis.okx_api.client import OKXClient
from apis.okx_api.market_data import MarketDataRetriever
from strategies.strategy_6_vcb.strategy_6 import VCBStrategy, CompressionEvent
from tools.market_scanner import CryptoScanner
from utils.logger import logger


class CompressionScanner:
    """
    压缩扫描器（生产者）
    
    职责：
    1. 扫描 300+ 币种
    2. 检测压缩事件
    3. 将新压缩事件放入压缩池
    """
    
    def __init__(self, client: OKXClient, strategy: VCBStrategy,
                 min_vol_ccy: float = 10000000,  # 最小24h交易量（USDT）
                 currency: str = 'USDT',
                 inst_type: str = 'SWAP',
                 max_workers: int = 10,
                 only_major_coins: bool = False):
        """
        初始化压缩扫描器
        
        Args:
            client: OKX客户端
            strategy: VCB策略实例（共享压缩池）
            min_vol_ccy: 最小24小时交易量（USDT）
            currency: 交易对货币（默认USDT）
            inst_type: 合约类型（默认SWAP）
            max_workers: 并行扫描的最大线程数
            only_major_coins: 是否只扫描主流币
        """
        self.client = client
        self.strategy = strategy
        self.min_vol_ccy = min_vol_ccy
        self.currency = currency
        self.inst_type = inst_type
        self.max_workers = max_workers
        self.only_major_coins = only_major_coins
        
        self.market_data_retriever = MarketDataRetriever(client)
        self.scanner = CryptoScanner(client)
        
        # 扫描统计
        self.scan_count = 0
        self.last_scan_time = None
        self.last_scan_symbols_count = 0
    
    def scan_market(self, bar: str = '1m',
                    atr_short_period: int = 10,
                    atr_mid_period: int = 60,
                    atr_ratio_threshold: float = 0.5,
                    bb_period: int = 20,
                    bb_std: int = 2,
                    bb_width_ratio: float = 0.7,
                    ttl_bars: int = 30) -> Dict[str, CompressionEvent]:
        """
        扫描整个市场，寻找压缩事件
        
        Args:
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            atr_ratio_threshold: ATR比率阈值
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            bb_width_ratio: 布林带宽度收缩比率
            ttl_bars: 压缩事件TTL（K线数量）
            
        Returns:
            dict: 本次扫描发现的新压缩事件 {symbol: CompressionEvent}
        """
        try:
            logger.info(f"开始扫描市场寻找压缩事件（最小交易量: {self.min_vol_ccy:,.0f} {self.currency}）...")
            
            # 获取符合条件的币种列表
            symbols = self._get_symbols_to_scan()
            
            if not symbols:
                logger.warning(f"未找到符合条件的币种（24h交易量 >= {self.min_vol_ccy:,.0f} {self.currency}）")
                return {}
            
            logger.info(f"找到 {len(symbols)} 个符合条件的币种，开始并行扫描...")
            
            # 并行扫描所有币种
            new_compressions = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有扫描任务
                future_to_symbol = {
                    executor.submit(
                        self._scan_single_symbol,
                        symbol, bar, atr_short_period, atr_mid_period,
                        atr_ratio_threshold, bb_period, bb_std, bb_width_ratio, ttl_bars
                    ): symbol
                    for symbol in symbols
                }
                
                # 收集结果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        compression = future.result()
                        if compression:
                            new_compressions[symbol] = compression
                            logger.info(f"✅ 发现压缩: {symbol} (ATR比率={compression.atr_ratio:.4f})")
                        completed += 1

                        # 每完成20%显示进度
                        if completed % max(1, len(symbols) // 5) == 0:
                            logger.info(f"扫描进度: {completed}/{len(symbols)} ({completed*100//len(symbols)}%)")
                    except Exception as e:
                        logger.warning(f"扫描 {symbol} 时出错: {e}")
                        completed += 1
            
            # 更新统计
            self.scan_count += 1
            self.last_scan_time = datetime.now()
            self.last_scan_symbols_count = len(symbols)
            
            logger.info(f"扫描完成: 共扫描 {len(symbols)} 个币种，发现 {len(new_compressions)} 个新压缩事件")
            logger.info(f"当前压缩池大小: {self.strategy.get_compression_pool_size()}")
            
            return new_compressions
            
        except Exception as e:
            logger.error(f"扫描市场时出错: {e}")
            return {}
    
    def _scan_single_symbol(self, symbol: str, bar: str,
                           atr_short_period: int, atr_mid_period: int,
                           atr_ratio_threshold: float,
                           bb_period: int, bb_std: int,
                           bb_width_ratio: float, ttl_bars: int) -> CompressionEvent:
        """
        扫描单个币种
        
        Args:
            symbol: 交易对符号
            bar: K线周期
            atr_short_period: 短期ATR周期
            atr_mid_period: 中期ATR周期
            atr_ratio_threshold: ATR比率阈值
            bb_period: 布林带周期
            bb_std: 布林带标准差倍数
            bb_width_ratio: 布林带宽度收缩比率
            ttl_bars: 压缩事件TTL
            
        Returns:
            CompressionEvent: 如果发现压缩，返回压缩事件；否则返回None
        """
        try:
            # 使用策略的检测方法（会自动管理压缩池）
            compression = self.strategy.detect_compression(
                symbol=symbol,
                bar=bar,
                atr_short_period=atr_short_period,
                atr_mid_period=atr_mid_period,
                atr_ratio_threshold=atr_ratio_threshold,
                bb_period=bb_period,
                bb_std=bb_std,
                bb_width_ratio=bb_width_ratio,
                ttl_bars=ttl_bars
            )
            
            return compression
            
        except Exception as e:
            # 静默失败，不记录每个币种的错误（避免日志过多）
            return None
    
    def _get_symbols_to_scan(self) -> List[str]:
        """
        获取需要扫描的币种列表
        
        Returns:
            list: 币种符号列表
        """
        try:
            # 使用 CryptoScanner 获取符合条件的币种，扫描现货交易量比扫描永续合约交易量更加安全
            symbols = self.scanner._get_volume_filtered_symbols(
                currency=self.currency,
                min_vol_ccy=self.min_vol_ccy,
                use_cache=True,
                inst_type=self.inst_type
            )

            # 如果只做主流币，进行过滤
            if self.only_major_coins:
                from strategies.strategy_6_vcb.methods.position_manager import MAJOR_COINS
                filtered_symbols = []
                for symbol in symbols:
                    base_coin = symbol.split('-')[0]
                    if base_coin in MAJOR_COINS:
                        filtered_symbols.append(symbol)
                symbols = filtered_symbols
                logger.info(f"主流币过滤后剩余 {len(symbols)} 个币种")
            
            return symbols if symbols else []
            
        except Exception as e:
            logger.error(f"获取币种列表时出错: {e}")
            return []
    
    def get_scan_stats(self) -> Dict:
        """获取扫描统计信息"""
        return {
            'scan_count': self.scan_count,
            'last_scan_time': self.last_scan_time.strftime('%Y-%m-%d %H:%M:%S') if self.last_scan_time else None,
            'last_scan_symbols_count': self.last_scan_symbols_count,
            'current_pool_size': self.strategy.get_compression_pool_size()
        }

