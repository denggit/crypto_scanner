#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : client.py
@Description: Binance API Client for market data and trading
"""

import hashlib
import hmac
import json
import time
import requests
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode
import threading
from time import sleep


class BinanceAPIException(Exception):
    """Custom exception for Binance API errors"""
    pass


class BinanceClient:
    """
    Binance API Client for market data and trading
    """

    def __init__(self, api_key: str = None, api_secret: str = None, testnet: bool = False):
        """
        Initialize Binance Client

        Args:
            api_key: Binance API Key
            api_secret: Binance API Secret
            testnet: Use testnet endpoint
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        if testnet:
            self.base_url = "https://testnet.binance.vision"
        else:
            self.base_url = "https://api.binance.com"

        self.session = requests.Session()

        # Rate limiting
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms minimum between requests

    def _sign_request(self, params: Dict) -> str:
        """Sign request with HMAC SHA256"""
        query_string = urlencode(params)
        return hmac.new(
            bytes(self.api_secret, encoding='utf-8'),
            bytes(query_string, encoding='utf-8'),
            digestmod='sha256'
        ).hexdigest()

    def _wait_for_rate_limit(self):
        """Enforce rate limiting between requests"""
        with self.rate_limit_lock:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep(self.min_request_interval - time_since_last_request)
            self.last_request_time = time.time()

    def _make_request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        """
        Make HTTP request to Binance API

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature

        Returns:
            API response as dictionary

        Raises:
            BinanceAPIException: If API returns an error
        """
        # Enforce rate limiting
        self._wait_for_rate_limit()

        url = self.base_url + endpoint
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        if signed and self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key

        # Add timestamp for signed requests
        if signed:
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign_request(params)

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method.upper() == 'POST':
                response = self.session.post(url, headers=headers, data=params, timeout=10)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url, headers=headers, data=params, timeout=10)
            else:
                raise BinanceAPIException(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            result = response.json()

            # Check for API errors
            if 'code' in result and result['code'] != 200:
                error_code = result.get('code', 'unknown')
                error_msg = result.get('msg', 'Unknown error')
                raise BinanceAPIException(f"Binance API Error {error_code}: {error_msg}")

            return result

        except requests.exceptions.RequestException as e:
            raise BinanceAPIException(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise BinanceAPIException(f"JSON decode error: {str(e)}")
        except Exception as e:
            raise BinanceAPIException(f"Unexpected error: {str(e)}")

    # Market Data Endpoints

    def get_exchange_info(self, symbol: str = None) -> Dict:
        """
        Get exchange information

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            Exchange information
        """
        endpoint = "/api/v3/exchangeInfo"
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', endpoint, params)

    def get_tickers(self) -> Dict:
        """
        Get tickers for all symbols

        Returns:
            Tickers data
        """
        endpoint = "/api/v3/ticker/24hr"
        return self._make_request('GET', endpoint)

    def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker for specific symbol

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)

        Returns:
            Ticker data
        """
        endpoint = "/api/v3/ticker/24hr"
        params = {'symbol': symbol}
        return self._make_request('GET', endpoint, params)

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book for symbol

        Args:
            symbol: Trading pair symbol
            limit: Size of order book (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book data
        """
        endpoint = "/api/v3/depth"
        params = {'symbol': symbol, 'limit': limit}
        return self._make_request('GET', endpoint, params)

    def get_kline(self, symbol: str, interval: str = '1m', limit: int = 500) -> Dict:
        """
        Get kline/candlestick data

        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
            limit: Number of bars (max 1000)

        Returns:
            Kline data
        """
        endpoint = "/api/v3/klines"
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        return self._make_request('GET', endpoint, params)

    # Account Endpoints

    def get_account_balance(self) -> Dict:
        """
        Get account balance (requires authentication)

        Returns:
            Account balance data
        """
        endpoint = "/api/v3/account"
        return self._make_request('GET', endpoint, signed=True)

    def get_account_info(self) -> Dict:
        """
        Get account information (requires authentication)

        Returns:
            Account information
        """
        endpoint = "/api/v3/account"
        return self._make_request('GET', endpoint, signed=True)

    # Trading Endpoints

    def place_order(self, symbol: str, side: str, order_type: str,
                   quantity: float = None, quote_order_qty: float = None,
                   price: float = None, stop_price: float = None,
                   time_in_force: str = 'GTC', new_client_order_id: str = None,
                   iceberg_qty: float = None, new_order_resp_type: str = 'ACK') -> Dict:
        """
        Place order (requires authentication)

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY, SELL)
            order_type: Order type (MARKET, LIMIT, STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT, LIMIT_MAKER)
            quantity: Quantity
            quote_order_qty: Quote order quantity (for MARKET orders)
            price: Price (required for LIMIT orders)
            stop_price: Stop price (required for STOP_LOSS, STOP_LOSS_LIMIT, TAKE_PROFIT, TAKE_PROFIT_LIMIT)
            time_in_force: Time in force (GTC, IOC, FOK)
            new_client_order_id: A unique id for the order
            iceberg_qty: Iceberg quantity for LIMIT orders
            new_order_resp_type: Response type (ACK, RESULT, FULL)

        Returns:
            Order result
        """
        endpoint = "/api/v3/order"
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'newOrderRespType': new_order_resp_type
        }

        if quantity:
            params['quantity'] = quantity
        if quote_order_qty:
            params['quoteOrderQty'] = quote_order_qty
        if price:
            params['price'] = price
        if stop_price:
            params['stopPrice'] = stop_price
        if time_in_force:
            params['timeInForce'] = time_in_force
        if new_client_order_id:
            params['newClientOrderId'] = new_client_order_id
        if iceberg_qty:
            params['icebergQty'] = iceberg_qty

        return self._make_request('POST', endpoint, params, signed=True)

    def place_market_order(self, symbol: str, side: str, quantity: float = None, quote_order_qty: float = None) -> Dict:
        """
        Place market order

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY, SELL)
            quantity: Quantity
            quote_order_qty: Quote order quantity

        Returns:
            Order result
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='MARKET',
            quantity=quantity,
            quote_order_qty=quote_order_qty
        )

    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float,
                         time_in_force: str = 'GTC') -> Dict:
        """
        Place limit order

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY, SELL)
            quantity: Quantity
            price: Price
            time_in_force: Time in force (GTC, IOC, FOK)

        Returns:
            Order result
        """
        return self.place_order(
            symbol=symbol,
            side=side,
            order_type='LIMIT',
            quantity=quantity,
            price=price,
            time_in_force=time_in_force
        )

    def get_order(self, symbol: str, order_id: str = None, orig_client_order_id: str = None) -> Dict:
        """
        Get order status

        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            orig_client_order_id: Original client order ID

        Returns:
            Order data
        """
        endpoint = "/api/v3/order"
        params = {'symbol': symbol}

        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise BinanceAPIException("Either order_id or orig_client_order_id must be provided")

        return self._make_request('GET', endpoint, params, signed=True)

    def cancel_order(self, symbol: str, order_id: str = None, orig_client_order_id: str = None) -> Dict:
        """
        Cancel order

        Args:
            symbol: Trading pair symbol
            order_id: Order ID
            orig_client_order_id: Original client order ID

        Returns:
            Cancel result
        """
        endpoint = "/api/v3/order"
        params = {'symbol': symbol}

        if order_id:
            params['orderId'] = order_id
        elif orig_client_order_id:
            params['origClientOrderId'] = orig_client_order_id
        else:
            raise BinanceAPIException("Either order_id or orig_client_order_id must be provided")

        return self._make_request('DELETE', endpoint, params, signed=True)

    def get_open_orders(self, symbol: str = None) -> Dict:
        """
        Get open orders

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            Open orders data
        """
        endpoint = "/api/v3/openOrders"
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', endpoint, params, signed=True)

    def get_all_orders(self, symbol: str, order_id: str = None, limit: int = 500) -> Dict:
        """
        Get all orders

        Args:
            symbol: Trading pair symbol
            order_id: Order ID (optional)
            limit: Number of orders (max 1000)

        Returns:
            All orders data
        """
        endpoint = "/api/v3/allOrders"
        params = {'symbol': symbol, 'limit': limit}
        if order_id:
            params['orderId'] = order_id
        return self._make_request('GET', endpoint, params, signed=True)

    # Futures Endpoints

    def futures_get_account_balance(self) -> Dict:
        """
        Get futures account balance (requires authentication)

        Returns:
            Futures account balance data
        """
        endpoint = "/fapi/v2/balance"
        return self._make_request('GET', endpoint, signed=True)

    def futures_get_account_info(self) -> Dict:
        """
        Get futures account information (requires authentication)

        Returns:
            Futures account information
        """
        endpoint = "/fapi/v2/account"
        return self._make_request('GET', endpoint, signed=True)

    def futures_get_positions(self, symbol: str = None) -> Dict:
        """
        Get futures positions (requires authentication)

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            Positions data
        """
        endpoint = "/fapi/v2/positionRisk"
        params = {}
        if symbol:
            params['symbol'] = symbol
        return self._make_request('GET', endpoint, params, signed=True)

    def futures_place_order(self, symbol: str, side: str, order_type: str,
                           quantity: float, position_side: str = 'BOTH',
                           price: float = None, stop_price: float = None,
                           time_in_force: str = 'GTC', reduce_only: bool = False,
                           close_position: bool = False, activation_price: float = None,
                           callback_rate: float = None, working_type: str = 'CONTRACT_PRICE') -> Dict:
        """
        Place futures order (requires authentication)

        Args:
            symbol: Trading pair symbol
            side: Order side (BUY, SELL)
            order_type: Order type (MARKET, LIMIT, STOP, STOP_MARKET, TAKE_PROFIT, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET)
            quantity: Quantity
            position_side: Position side (LONG, SHORT, BOTH)
            price: Price (required for LIMIT orders)
            stop_price: Stop price
            time_in_force: Time in force (GTC, IOC, FOK, GTX)
            reduce_only: Reduce only flag
            close_position: Close position flag
            activation_price: Activation price for TRAILING_STOP_MARKET
            callback_rate: Callback rate for TRAILING_STOP_MARKET
            working_type: Working type (MARK_PRICE, CONTRACT_PRICE)

        Returns:
            Order result
        """
        endpoint = "/fapi/v1/order"
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'quantity': quantity,
            'positionSide': position_side
        }

        if price:
            params['price'] = price
        if stop_price:
            params['stopPrice'] = stop_price
        if time_in_force:
            params['timeInForce'] = time_in_force
        if reduce_only:
            params['reduceOnly'] = reduce_only
        if close_position:
            params['closePosition'] = close_position
        if activation_price:
            params['activationPrice'] = activation_price
        if callback_rate:
            params['callbackRate'] = callback_rate
        if working_type:
            params['workingType'] = working_type

        return self._make_request('POST', endpoint, params, signed=True)

    def futures_change_leverage(self, symbol: str, leverage: int) -> Dict:
        """
        Change leverage for futures trading

        Args:
            symbol: Trading pair symbol
            leverage: Leverage value

        Returns:
            Leverage change result
        """
        endpoint = "/fapi/v1/leverage"
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        return self._make_request('POST', endpoint, params, signed=True)

    def futures_change_margin_type(self, symbol: str, margin_type: str) -> Dict:
        """
        Change margin type for futures trading

        Args:
            symbol: Trading pair symbol
            margin_type: Margin type (ISOLATED, CROSSED)

        Returns:
            Margin type change result
        """
        endpoint = "/fapi/v1/marginType"
        params = {
            'symbol': symbol,
            'marginType': margin_type
        }
        return self._make_request('POST', endpoint, params, signed=True)