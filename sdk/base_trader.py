#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : base_trader.py
@Description: Base Trader class for real trading
"""

import os
import sys
from abc import ABC
from typing import Optional

from apis.okx_api import Trader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from utils.logger import logger


class BaseTrader(ABC):
    """Base class for real trading execution"""

    TRADE_MODE_SPOT = 1
    TRADE_MODE_CROSS = 2
    TRADE_MODE_ISOLATED = 3

    def __init__(self, client, trade_amount: float = 10.0, trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Base Trader
        
        Args:
            client: Trading client instance
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        self.client = client
        self.trader = Trader(client)
        self.trade_amount = trade_amount
        self.trade_mode = trade_mode
        self.leverage = leverage
        self.leverage_setup_done = {}

        # Cache for instrument information to avoid repeated API calls
        self._instrument_cache = {}

        self.td_mode_map = {
            self.TRADE_MODE_SPOT: 'cash',
            self.TRADE_MODE_CROSS: 'cross',
            self.TRADE_MODE_ISOLATED: 'isolated'
        }

    def get_inst_id(self, symbol: str) -> str:
        """
        Get instrument ID based on trade mode

        Args:
            symbol: Trading pair symbol (e.g., BTC-USDT)

        Returns:
            Instrument ID for OKX API
        """
        if self.is_leverage_mode() and not symbol.endswith("-SWAP"):
            return f"{symbol}-SWAP"
        return symbol

    def get_td_mode(self) -> str:
        """Get OKX API tdMode parameter based on trade_mode"""
        return self.td_mode_map.get(self.trade_mode, 'isolated')

    def get_margin_mode(self) -> str:
        """Get OKX API margin mode parameter"""
        if self.trade_mode == self.TRADE_MODE_CROSS:
            return 'cross'
        elif self.trade_mode == self.TRADE_MODE_ISOLATED:
            return 'isolated'
        return None

    def is_leverage_mode(self) -> bool:
        """Check if using leverage mode"""
        return self.trade_mode in [self.TRADE_MODE_CROSS, self.TRADE_MODE_ISOLATED]

    def setup_leverage(self, symbol: str) -> bool:
        """
        Setup leverage for the instrument

        Args:
            symbol: Trading pair symbol

        Returns:
            True if successful, False otherwise
        """
        if not self.is_leverage_mode():
            return True

        inst_id = self.get_inst_id(symbol)

        if inst_id in self.leverage_setup_done:
            return True

        try:
            mgn_mode = self.get_margin_mode()
            result_long = self.client.set_leverage(
                instId=inst_id,
                lever=str(self.leverage),
                mgnMode=mgn_mode,
                posSide='long'
            )
            result_short = self.client.set_leverage(
                instId=inst_id,
                lever=str(self.leverage),
                mgnMode=mgn_mode,
                posSide='short'
            )

            if result_long.get('code') == '0' and result_short.get('code') == '0':
                logger.info(f"✅ 杠杆设置成功: {inst_id} {self.leverage}x {mgn_mode}")
                self.leverage_setup_done[inst_id] = True
                return True
            elif result_long.get('code') != '0':
                logger.warning(f"⚠️  开多杠杆设置失败: {result_long.get('msg', 'Unknown error')}")
                return False
            elif result_short.get('code') != '0':
                logger.warning(f"⚠️  开空杠杆设置失败: {result_short.get('msg', 'Unknown error')}")
                return False
        except Exception as e:
            logger.error(f"⚠️  杠杆设置异常: {e}")
            return False

    def execute_open_long(self, symbol: str, price: float = None) -> Optional[any]:
        """
        Execute open long position

        Args:
            symbol: Trading pair symbol
            price: Current price (目前先只做市价）

        Returns:
            Order object if successful, None otherwise
        """
        # 强制设置杠杆（只在第一次需要）
        if not self.setup_leverage(symbol):
            logger.error(f"❌ {symbol} 杠杆设置失败，取消开多交易")
            return None

        inst_id = self.get_inst_id(symbol)
        td_mode = self.get_td_mode()

        # 计算正确的下单数量
        if price is None:
            # 如果没有提供价格，获取当前价格
            from apis.okx_api.market_data import MarketDataRetriever
            market_retriever = MarketDataRetriever(self.client)
            ticker = market_retriever.get_ticker_by_symbol(inst_id)
            if ticker:
                price = ticker.last
            else:
                logger.error(f"❌ 无法获取 {inst_id} 的价格")
                return None

        order_size = self.calculate_order_size(inst_id, price)

        order = self.trader.place_market_order(
            instId=inst_id,
            side='buy',
            sz=order_size,
            tdMode=td_mode,
            posSide='long'
        )
        return order

    def execute_open_short(self, symbol: str, price: float = None) -> Optional[any]:
        """
        Execute open short position

        Args:
            symbol: Trading pair symbol
            price: Current price (目前先只做市价）

        Returns:
            Order object if successful, None otherwise
        """
        # 强制设置杠杆（只在第一次需要）
        if not self.setup_leverage(symbol):
            logger.error(f"❌ {symbol} 杠杆设置失败，取消开空交易")
            return None

        inst_id = self.get_inst_id(symbol)
        td_mode = self.get_td_mode()

        # 计算正确的下单数量
        if price is None:
            # 如果没有提供价格，获取当前价格
            from apis.okx_api.market_data import MarketDataRetriever
            market_retriever = MarketDataRetriever(self.client)
            ticker = market_retriever.get_ticker_by_symbol(inst_id)
            if ticker:
                price = ticker.last
            else:
                logger.error(f"❌ 无法获取 {inst_id} 的价格")
                return None

        order_size = self.calculate_order_size(inst_id, price)

        order = self.trader.place_market_order(
            instId=inst_id,
            side='sell',
            sz=order_size,
            tdMode=td_mode,
            posSide="short"
        )
        return order

    def execute_close_long(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute close long position (sell all)

        Args:
            symbol: Trading pair symbol
            price: Current price

        Returns:
            Order object if successful, None otherwise
        """
        inst_id = self.get_inst_id(symbol)
        td_mode = self.get_td_mode()

        if self.is_leverage_mode():
            positions = self.client.get_positions(instId=inst_id)
            logger.info(f"检查仓位: {inst_id}, positions={positions}")
            if positions and 'data' in positions and len(positions['data']) > 0:
                found_position = False
                for pos in positions['data']:
                    logger.info(
                        f"仓位详情: instId={pos.get('instId')}, pos={pos.get('pos', 0)}, posSide={pos.get('posSide')}")
                    if pos.get('instId') == inst_id and pos.get("posSide") == 'long':
                        available_sz = pos.get('pos', '0')
                        logger.info(f"找到多仓仓位: {inst_id}, 数量={available_sz}")
                        order = self.trader.place_market_order(
                            instId=inst_id,
                            side='sell',
                            sz=available_sz,
                            tdMode=td_mode,
                            reduceOnly=True,
                            posSide='long'
                        )
                        return order
                    elif pos.get('instId') == inst_id:
                        found_position = True
                        logger.info(
                            f"找到仓位但不是多仓: {inst_id}, 数量={pos.get('pos', 0)}, posSide={pos.get('posSide')}")

                if not found_position:
                    logger.warning(f"未找到 {inst_id} 的仓位信息")
            else:
                logger.warning(f"未获取到仓位数据或数据为空: {positions}")
        else:
            balance = self.trader.get_account_balance()
            if balance and 'data' in balance and len(balance['data']) > 0:
                for detail in balance['data'][0].get('details', []):
                    if detail['ccy'] == symbol.split('-')[0]:
                        available_sz = detail.get('availBal', '0')
                        if float(available_sz) > 0:
                            order = self.trader.place_market_order(
                                instId=inst_id,
                                side='sell',
                                sz=available_sz,
                                tdMode=td_mode,
                                posSide='long'
                            )
                            return order
                        break
        return None

    def execute_close_short(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute close short position (buy all)

        Args:
            symbol: Trading pair symbol
            price: Current price

        Returns:
            Order object if successful, None otherwise
        """
        inst_id = self.get_inst_id(symbol)
        td_mode = self.get_td_mode()

        if self.is_leverage_mode():
            positions = self.client.get_positions(instId=inst_id)
            logger.info(f"检查仓位: {inst_id}, positions={positions}")
            if positions and 'data' in positions and len(positions['data']) > 0:
                found_position = False
                for pos in positions['data']:
                    logger.info(
                        f"仓位详情: instId={pos.get('instId')}, pos={pos.get('pos', 0)}, posSide={pos.get('posSide')}")
                    if pos.get('instId') == inst_id and pos.get("posSide") == 'short':
                        available_sz = str(abs(float(pos.get('pos', '0'))))
                        logger.info(f"找到空仓仓位: {inst_id}, 数量={available_sz}")
                        order = self.trader.place_market_order(
                            instId=inst_id,
                            side='buy',
                            sz=available_sz,
                            tdMode=td_mode,
                            reduceOnly=True,
                            posSide='short'
                        )
                        return order
                    elif pos.get('instId') == inst_id:
                        found_position = True
                        logger.info(
                            f"找到仓位但不是空仓: {inst_id}, 数量={pos.get('pos', 0)}, posSide={pos.get('posSide')}")

                if not found_position:
                    logger.warning(f"未找到 {inst_id} 的仓位信息")
            else:
                logger.warning(f"未获取到仓位数据或数据为空: {positions}")
        else:
            balance = self.trader.get_account_balance()
            if balance and 'data' in balance and len(balance['data']) > 0:
                for detail in balance['data'][0].get('details', []):
                    if detail['ccy'] == symbol.split('-')[0]:
                        available_sz = detail.get('availBal', '0')
                        if float(available_sz) > 0:
                            order = self.trader.place_market_order(
                                instId=inst_id,
                                side='buy',
                                sz=available_sz,
                                tdMode=td_mode,
                                posSide='short'
                            )
                            return order
                        break
        return None

    def execute_trade(self, action: str, symbol: str, price: float) -> bool:
        """
        Execute trade based on action
        
        Args:
            action: Trade action (LONG_OPEN, SHORT_OPEN, etc.)
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if action == "LONG_OPEN":
                order = self.execute_open_long(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 做多成功: 订单ID={order.ordId}, 价格={price:.4f}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 做多失败")
                    return False

            elif action == "SHORT_CLOSE_LONG_OPEN":
                # 先平空仓，再开多仓
                close_order = self.execute_close_short(symbol, price)
                if close_order:
                    logger.info(f"✅ [真实交易] {symbol} 平空成功: 订单ID={close_order.ordId}")
                    # 平空成功后开多仓
                    open_order = self.execute_open_long(symbol, price)
                    if open_order:
                        logger.info(f"✅ [真实交易] {symbol} 做多成功: 订单ID={open_order.ordId}, 价格={price:.4f}")
                        return True
                    else:
                        logger.error(f"❌ [真实交易] {symbol} 平空成功但做多失败")
                        return False
                else:
                    logger.error(f"❌ [真实交易] {symbol} 平空失败")
                    return False

            elif action == "SHORT_OPEN":
                order = self.execute_open_short(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 做空成功: 订单ID={order.ordId}, 价格={price:.4f}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 做空失败")
                    return False

            elif action == "LONG_CLOSE_SHORT_OPEN":
                # 先平多仓，再开空仓
                close_order = self.execute_close_long(symbol, price)
                if close_order:
                    logger.info(f"✅ [真实交易] {symbol} 平多成功: 订单ID={close_order.ordId}")
                    # 平多成功后开空仓
                    open_order = self.execute_open_short(symbol, price)
                    if open_order:
                        logger.info(f"✅ [真实交易] {symbol} 做空成功: 订单ID={open_order.ordId}, 价格={price:.4f}")
                        return True
                    else:
                        logger.error(f"❌ [真实交易] {symbol} 平多成功但做空失败")
                        return False
                else:
                    logger.error(f"❌ [真实交易] {symbol} 平多失败")
                    return False

            elif action in ["LONG_CLOSE_TRAILING_STOP"]:
                order = self.execute_close_long(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 平多成功: 订单ID={order.ordId}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 平多失败")
                    return False

            elif action in ["SHORT_CLOSE_TRAILING_STOP"]:
                order = self.execute_close_short(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 平空成功: 订单ID={order.ordId}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 平空失败")
                    return False

            elif action in ["LONG_CLOSE_VOLATILITY"]:
                order = self.execute_close_long(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 波动率平多成功: 订单ID={order.ordId}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 波动率平多失败")
                    return False

            elif action in ["SHORT_CLOSE_VOLATILITY"]:
                order = self.execute_close_short(symbol, price)
                if order:
                    logger.info(f"✅ [真实交易] {symbol} 波动率平空成功: 订单ID={order.ordId}")
                    return True
                else:
                    logger.error(f"❌ [真实交易] {symbol} 波动率平空失败")
                    return False

        except Exception as e:
            logger.exception(f"❌ [真实交易] {symbol} 执行交易时出错: {e}")
            return False

        return False

    def calculate_order_size(self, symbol: str, price: float) -> str:
        """
        Calculate the correct order size based on instrument parameters and trade amount
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC-USDT')
            price: Current price of the instrument
            
        Returns:
            str: Order size as string formatted for API
        """
        try:
            # Check if instrument info is already cached
            if symbol in self._instrument_cache:
                instrument = self._instrument_cache[symbol]
                logger.debug(f"使用缓存的合约信息: {symbol}")
            else:
                # Get instrument information from API
                from apis.okx_api import MarketDataRetriever
                market_retriever = MarketDataRetriever(self.client)
                instrument = market_retriever.get_instrument_info(symbol)

                if instrument:
                    # Cache the instrument info for future use
                    self._instrument_cache[symbol] = instrument
                    logger.debug(f"缓存合约信息: {symbol}")

                    # Save to JSON cache file
                    from apis.okx_api.instrument_cache import InstrumentCache
                    cache = InstrumentCache()
                    cache.save_instrument(symbol, instrument)
                else:
                    # Try to load from JSON cache as fallback
                    from apis.okx_api.instrument_cache import InstrumentCache
                    cache = InstrumentCache()
                    cached_instrument = cache.get_instrument(symbol)

                    if cached_instrument:
                        # Cached data is now a dictionary, we need to convert it back to Instrument object
                        # For now, we'll use it as a dictionary since the code accesses attributes like .minSz, .ctVal
                        instrument = cached_instrument
                        self._instrument_cache[symbol] = instrument
                        logger.info(f"📖 从JSON缓存读取instrument信息: {symbol}")
                    else:
                        logger.warning(f"无法获取 {symbol} 的合约信息，使用默认下单数量")
                        return str(self.trade_amount)

            # Get contract parameters
            # Handle both Instrument object and dictionary
            if hasattr(instrument, 'minSz'):
                # It's an Instrument object
                min_sz = float(instrument.minSz)
                ct_val = float(instrument.ctVal)
            else:
                # It's a dictionary from cache
                min_sz = float(instrument.get('minSz', '0'))
                ct_val = float(instrument.get('ctVal', '0'))

            # Calculate single currency value
            single_currency_value = price

            # Calculate required sz based on trade amount
            if ct_val > 0:
                # For derivatives: sz = trade_amount / (single_currency_value * ct_val)
                calculated_sz = self.trade_amount / (single_currency_value * ct_val)
            else:
                # For spot: sz = trade_amount / single_currency_value
                calculated_sz = self.trade_amount / single_currency_value

            # Ensure minimum order size and respect lot size
            # First ensure minimum size
            final_sz = max(min_sz, calculated_sz)
            # Then round to nearest multiple of lot_sz (if available)
            if hasattr(instrument, 'lotSz'):
                # It's an Instrument object
                lot_sz = float(instrument.lotSz)
            else:
                # It's a dictionary from cache
                lot_sz = float(instrument.get('lotSz', '1'))

            if lot_sz > 0:
                # Ensure minimum order size
                final_sz = math.ceil(max(min_sz, calculated_sz) * (1 / lot_sz)) / (1 / lot_sz)

            # Ensure we don't exceed maximum order size
            if hasattr(instrument, 'maxMktSz'):
                max_mkt_sz = float(instrument.maxMktSz)
                final_sz = min(final_sz, max_mkt_sz)
            elif hasattr(instrument, 'get'):
                max_mkt_sz = float(instrument.get('maxMktSz', '1000000'))
                final_sz = min(final_sz, max_mkt_sz)

            logger.info(f"下单数量计算: symbol={symbol}, trade_amount={self.trade_amount}, price={price:.8f}, "
                        f"min_sz={min_sz}, lot_sz={lot_sz}, ct_val={ct_val}, calculated_sz={calculated_sz:.8f}, final_sz={final_sz:.8f}")

            return str(final_sz)

        except Exception as e:
            logger.error(f"计算下单数量失败: {e}")
            return str(self.trade_amount)
