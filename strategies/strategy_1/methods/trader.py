#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : trader.py
@Description: Strategy 1 specific trader implementation
"""

import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_trader import BaseTrader
from okx_api.trader import Trader


class Strategy1Trader(BaseTrader):
    """EMA Crossover Strategy Trader"""
    
    def __init__(self, client, trade_amount: float = 10.0, trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Strategy 1 Trader
        
        Args:
            client: OKX client instance
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        super().__init__(client, trade_amount, trade_mode, leverage)
        self.trader = Trader(client)
        self.leverage_setup_done = {}
    
    def _get_inst_id(self, symbol: str) -> str:
        """
        Get instrument ID based on trade mode
        
        Args:
            symbol: Trading pair symbol (e.g., BTC-USDT)
            
        Returns:
            Instrument ID for OKX API
        """
        if self.is_leverage_mode():
            return f"{symbol}-SWAP"
        return symbol
    
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
        
        inst_id = self._get_inst_id(symbol)
        
        if inst_id in self.leverage_setup_done:
            return True
        
        try:
            mgn_mode = self.get_margin_mode()
            result = self.client.set_leverage(
                instId=inst_id,
                lever=str(self.leverage),
                mgnMode=mgn_mode,
                posSide='net'
            )
            
            if result.get('code') == '0':
                print(f"✅ 杠杆设置成功: {inst_id} {self.leverage}x {mgn_mode}")
                self.leverage_setup_done[inst_id] = True
                return True
            else:
                print(f"⚠️  杠杆设置失败: {result.get('msg', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"⚠️  杠杆设置异常: {e}")
            return False
    
    def execute_open_long(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute open long position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        self.setup_leverage(symbol)
        
        inst_id = self._get_inst_id(symbol)
        td_mode = self.get_td_mode()
        
        order = self.trader.place_market_order(
            instId=inst_id,
            side='buy',
            sz=str(self.trade_amount),
            tdMode=td_mode,
            tgtCcy='quote_ccy'
        )
        return order
    
    def execute_open_short(self, symbol: str, price: float) -> Optional[any]:
        """
        Execute open short position
        
        Args:
            symbol: Trading pair symbol
            price: Current price
            
        Returns:
            Order object if successful, None otherwise
        """
        self.setup_leverage(symbol)
        
        inst_id = self._get_inst_id(symbol)
        td_mode = self.get_td_mode()
        
        order = self.trader.place_market_order(
            instId=inst_id,
            side='sell',
            sz=str(self.trade_amount),
            tdMode=td_mode,
            tgtCcy='quote_ccy'
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
        inst_id = self._get_inst_id(symbol)
        td_mode = self.get_td_mode()
        
        if self.is_leverage_mode():
            positions = self.client.get_positions(instId=inst_id)
            if positions and 'data' in positions and len(positions['data']) > 0:
                for pos in positions['data']:
                    if pos.get('instId') == inst_id and float(pos.get('pos', 0)) > 0:
                        available_sz = pos.get('pos', '0')
                        order = self.trader.place_market_order(
                            instId=inst_id,
                            side='sell',
                            sz=available_sz,
                            tdMode=td_mode,
                            reduceOnly=True
                        )
                        return order
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
                                tdMode=td_mode
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
        inst_id = self._get_inst_id(symbol)
        td_mode = self.get_td_mode()
        
        if self.is_leverage_mode():
            positions = self.client.get_positions(instId=inst_id)
            if positions and 'data' in positions and len(positions['data']) > 0:
                for pos in positions['data']:
                    if pos.get('instId') == inst_id and float(pos.get('pos', 0)) < 0:
                        available_sz = str(abs(float(pos.get('pos', '0'))))
                        order = self.trader.place_market_order(
                            instId=inst_id,
                            side='buy',
                            sz=available_sz,
                            tdMode=td_mode,
                            reduceOnly=True
                        )
                        return order
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
                                tdMode=td_mode
                            )
                            return order
                        break
        return None
