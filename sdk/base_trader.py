#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : base_trader.py
@Description: Base Trader class for real trading
"""

from abc import ABC, abstractmethod
from typing import Optional
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        self.trade_amount = trade_amount
        self.trade_mode = trade_mode
        self.leverage = leverage
        
        self.td_mode_map = {
            self.TRADE_MODE_SPOT: 'cash',
            self.TRADE_MODE_CROSS: 'cross',
            self.TRADE_MODE_ISOLATED: 'isolated'
        }
    
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
    
    @abstractmethod
    def setup_leverage(self, symbol: str) -> bool:
        """
        Setup leverage for the instrument
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_open_long(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute open long position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def execute_open_short(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute open short position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def execute_close_long(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute close long position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
    @abstractmethod
    def execute_close_short(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute close short position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        pass
    
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
                    
        except Exception as e:
            logger.error(f"❌ [真实交易] {symbol} 执行交易时出错: {e}")
            return False
        
        return False
