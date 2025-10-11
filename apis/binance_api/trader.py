#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : trader.py
@Description: Binance Trading functionality
"""

from typing import Optional
from .client import BinanceClient
from .models import Order


class Trader:
    """
    Trading functionality for Binance
    """

    def __init__(self, client: BinanceClient):
        """
        Initialize Trader

        Args:
            client: BinanceClient instance
        """
        self.client = client

    def place_market_order(self, symbol: str, side: str, quantity: float = None, 
                          quote_order_qty: float = None) -> Optional[Order]:
        """
        Place a market order

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            side: Order side ('BUY' or 'SELL')
            quantity: Quantity in base asset
            quote_order_qty: Quantity in quote asset (USDT)

        Returns:
            Order object if successful, None otherwise
        """
        try:
            response = self.client.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                quote_order_qty=quote_order_qty
            )

            return Order(
                symbol=response.get('symbol', ''),
                orderId=response.get('orderId', ''),
                clientOrderId=response.get('clientOrderId', ''),
                price=float(response.get('price', 0)),
                origQty=float(response.get('origQty', 0)),
                executedQty=float(response.get('executedQty', 0)),
                cummulativeQuoteQty=float(response.get('cummulativeQuoteQty', 0)),
                status=response.get('status', ''),
                timeInForce=response.get('timeInForce', ''),
                type=response.get('type', ''),
                side=response.get('side', ''),
                stopPrice=float(response.get('stopPrice', 0)),
                icebergQty=float(response.get('icebergQty', 0)),
                time=int(response.get('time', 0)),
                updateTime=int(response.get('updateTime', 0)),
                isWorking=response.get('isWorking', False),
                origQuoteOrderQty=float(response.get('origQuoteOrderQty', 0))
            )
        except Exception:
            return None

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float,
                         time_in_force: str = 'GTC') -> Optional[Order]:
        """
        Place a limit order

        Args:
            symbol: Trading pair symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Quantity
            price: Price
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Order object if successful, None otherwise
        """
        try:
            response = self.client.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                time_in_force=time_in_force
            )

            return Order(
                symbol=response.get('symbol', ''),
                orderId=response.get('orderId', ''),
                clientOrderId=response.get('clientOrderId', ''),
                price=float(response.get('price', 0)),
                origQty=float(response.get('origQty', 0)),
                executedQty=float(response.get('executedQty', 0)),
                cummulativeQuoteQty=float(response.get('cummulativeQuoteQty', 0)),
                status=response.get('status', ''),
                timeInForce=response.get('timeInForce', ''),
                type=response.get('type', ''),
                side=response.get('side', ''),
                stopPrice=float(response.get('stopPrice', 0)),
                icebergQty=float(response.get('icebergQty', 0)),
                time=int(response.get('time', 0)),
                updateTime=int(response.get('updateTime', 0)),
                isWorking=response.get('isWorking', False),
                origQuoteOrderQty=float(response.get('origQuoteOrderQty', 0))
            )
        except Exception:
            return None

    def get_pending_orders(self, symbol: str = None) -> list:
        """
        Get pending orders

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            List of pending orders
        """
        try:
            response = self.client.get_open_orders(symbol)
            orders = []
            for order_data in response:
                order = Order(
                    symbol=order_data.get('symbol', ''),
                    orderId=order_data.get('orderId', ''),
                    clientOrderId=order_data.get('clientOrderId', ''),
                    price=float(order_data.get('price', 0)),
                    origQty=float(order_data.get('origQty', 0)),
                    executedQty=float(order_data.get('executedQty', 0)),
                    cummulativeQuoteQty=float(order_data.get('cummulativeQuoteQty', 0)),
                    status=order_data.get('status', ''),
                    timeInForce=order_data.get('timeInForce', ''),
                    type=order_data.get('type', ''),
                    side=order_data.get('side', ''),
                    stopPrice=float(order_data.get('stopPrice', 0)),
                    icebergQty=float(order_data.get('icebergQty', 0)),
                    time=int(order_data.get('time', 0)),
                    updateTime=int(order_data.get('updateTime', 0)),
                    isWorking=order_data.get('isWorking', False),
                    origQuoteOrderQty=float(order_data.get('origQuoteOrderQty', 0))
                )
                orders.append(order)
            return orders
        except Exception:
            return []

    def cancel_order(self, symbol: str, order_id: str = None, orig_client_order_id: str = None) -> bool:
        """
        Cancel an order

        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            orig_client_order_id: Original client order ID

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.cancel_order(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=orig_client_order_id
            )
            return 'orderId' in response or 'origClientOrderId' in response
        except Exception:
            return False

    def get_account_balance(self):
        """
        Get account balance

        Returns:
            Account balance data
        """
        try:
            return self.client.get_account_balance()
        except Exception:
            return {}

    def get_order_status(self, symbol: str, order_id: str = None, orig_client_order_id: str = None) -> Optional[Order]:
        """
        Get order status

        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            orig_client_order_id: Original client order ID

        Returns:
            Order object if successful, None otherwise
        """
        try:
            response = self.client.get_order(
                symbol=symbol,
                order_id=order_id,
                orig_client_order_id=orig_client_order_id
            )

            return Order(
                symbol=response.get('symbol', ''),
                orderId=response.get('orderId', ''),
                clientOrderId=response.get('clientOrderId', ''),
                price=float(response.get('price', 0)),
                origQty=float(response.get('origQty', 0)),
                executedQty=float(response.get('executedQty', 0)),
                cummulativeQuoteQty=float(response.get('cummulativeQuoteQty', 0)),
                status=response.get('status', ''),
                timeInForce=response.get('timeInForce', ''),
                type=response.get('type', ''),
                side=response.get('side', ''),
                stopPrice=float(response.get('stopPrice', 0)),
                icebergQty=float(response.get('icebergQty', 0)),
                time=int(response.get('time', 0)),
                updateTime=int(response.get('updateTime', 0)),
                isWorking=response.get('isWorking', False),
                origQuoteOrderQty=float(response.get('origQuoteOrderQty', 0))
            )
        except Exception:
            return None

    # Futures Trading Methods

    def futures_place_market_order(self, symbol: str, side: str, quantity: float, 
                                  position_side: str = 'BOTH', reduce_only: bool = False) -> Optional[Order]:
        """
        Place a futures market order

        Args:
            symbol: Trading pair symbol
            side: Order side ('BUY' or 'SELL')
            quantity: Quantity
            position_side: Position side ('LONG', 'SHORT', 'BOTH')
            reduce_only: Reduce only flag

        Returns:
            Order object if successful, None otherwise
        """
        try:
            response = self.client.futures_place_order(
                symbol=symbol,
                side=side,
                order_type='MARKET',
                quantity=quantity,
                position_side=position_side,
                reduce_only=reduce_only
            )

            return Order(
                symbol=response.get('symbol', ''),
                orderId=response.get('orderId', ''),
                clientOrderId=response.get('clientOrderId', ''),
                price=float(response.get('price', 0)),
                origQty=float(response.get('origQty', 0)),
                executedQty=float(response.get('executedQty', 0)),
                cummulativeQuoteQty=float(response.get('cumQuote', 0)),
                status=response.get('status', ''),
                timeInForce=response.get('timeInForce', ''),
                type=response.get('type', ''),
                side=response.get('side', ''),
                stopPrice=float(response.get('stopPrice', 0)),
                icebergQty=float(response.get('icebergQty', 0)),
                time=int(response.get('time', 0)),
                updateTime=int(response.get('updateTime', 0)),
                isWorking=response.get('isWorking', False),
                origQuoteOrderQty=float(response.get('origQuoteOrderQty', 0))
            )
        except Exception:
            return None

    def futures_get_positions(self, symbol: str = None):
        """
        Get futures positions

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            Positions data
        """
        try:
            return self.client.futures_get_positions(symbol)
        except Exception:
            return {}

    def futures_change_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Change leverage for futures trading

        Args:
            symbol: Trading pair symbol
            leverage: Leverage value

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.futures_change_leverage(symbol, leverage)
            return 'leverage' in response
        except Exception:
            return False

    def futures_change_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Change margin type for futures trading

        Args:
            symbol: Trading pair symbol
            margin_type: Margin type ('ISOLATED', 'CROSSED')

        Returns:
            True if successful, False otherwise
        """
        try:
            response = self.client.futures_change_margin_type(symbol, margin_type)
            return 'code' in response and response['code'] == 200
        except Exception:
            return False

    def futures_get_account_balance(self):
        """
        Get futures account balance

        Returns:
            Futures account balance data
        """
        try:
            return self.client.futures_get_account_balance()
        except Exception:
            return {}

    def futures_get_account_info(self):
        """
        Get futures account information

        Returns:
            Futures account information
        """
        try:
            return self.client.futures_get_account_info()
        except Exception:
            return {}