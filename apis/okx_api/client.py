import hmac
import base64
import hmac
import json
import threading
import time
from time import sleep
from typing import Dict
from urllib.parse import urlencode

import requests


class OKXAPIException(Exception):
    """Custom exception for OKX API errors"""
    pass


class OKXClient:
    """
    OKX API Client for market data and trading
    """

    def __init__(self, api_key: str = None, api_secret: str = None, passphrase: str = None, demo: bool = False):
        """
        Initialize OKX Client

        Args:
            api_key: OKX API Key
            api_secret: OKX API Secret
            passphrase: OKX API Passphrase
            demo: Use demo trading endpoint
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.demo = demo

        if demo:
            self.base_url = "https://www.okx.com"
        else:
            self.base_url = "https://www.okx.com"

        self.session = requests.Session()

        # Rate limiting
        self.rate_limit_lock = threading.Lock()
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms minimum between requests

        # Error codes
        self.error_codes = {
            '1': 'All operations failed',
            '50001': 'Invalid API key',
            '50002': 'Invalid API secret',
            '50003': 'Invalid timestamp',
            '50004': 'Invalid signature',
            '50005': 'Invalid passphrase',
            '50006': 'Invalid authorization',
            '50007': 'Too many requests',
            '50008': 'User IP not authorized',
            '50009': 'Account suspended',
            '50010': 'User not logged in',
            '50011': 'Invalid request',
            '50012': 'Invalid argument',
            '50013': 'Service unavailable',
            '50014': 'Order not found',
            '50015': 'Insufficient funds',
            '50016': 'Invalid amount',
            '50017': 'Invalid price',
            '50018': 'Order already cancelled',
            '50019': 'Order already completed',
            '50020': 'Order price exceeds limit',
            '50021': 'Order quantity exceeds limit',
            '50022': 'Unsupported transfer currency',
            '50023': 'Invalid transfer amount',
            '50024': 'Account not authorized',
            '50025': 'Duplicate order ID',
            '50026': 'Order price deviation exceeds limit',
            '50027': 'Invalid leverage',
            '50028': 'Position not found',
            '50029': 'Position already closed',
            '50030': 'Futures contract expired',
            '50031': 'Invalid position side',
            '50032': 'Exceed maximum order quantity',
            '50033': 'Exceed maximum order price',
            '50034': 'Exceed maximum order value',
            '50035': 'Instrument not supported',
            '50036': 'Order quantity too small',
            '50037': 'Invalid order type',
            '50038': 'Invalid position type',
            '50039': 'Invalid time in force',
            '50040': 'Invalid trigger price',
            '50041': 'Invalid algorithm order type',
            '50042': 'Invalid reduce only',
            '50043': 'Invalid close position',
            '50044': 'Invalid currency',
            '50045': 'Invalid account',
            '50046': 'Invalid position mode',
            '50047': 'Invalid margin mode',
            '50048': 'Invalid loan amount',
            '50049': 'Invalid repayment amount',
            '50050': 'Invalid loan currency'
        }

    def _get_server_time(self) -> str:
        """Get server timestamp in ISO format"""
        endpoint = "/api/v5/public/time"
        response = self.session.get(self.base_url + endpoint)
        data = response.json()
        # Convert milliseconds to ISO format
        ts_ms = int(data['data'][0]['ts'])
        ts_sec = ts_ms / 1000.0
        from datetime import datetime
        return datetime.utcfromtimestamp(ts_sec).isoformat()[:-3] + 'Z'

    def _sign_request(self, timestamp: str, method: str, endpoint: str, body: str = '') -> str:
        """Sign request with HMAC SHA256"""
        message = timestamp + method.upper() + endpoint + body
        mac = hmac.new(
            bytes(self.api_secret, encoding='utf-8'),
            bytes(message, encoding='utf-8'),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()

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
        Make HTTP request to OKX API

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether request needs signature

        Returns:
            API response as dictionary

        Raises:
            OKXAPIException: If API returns an error
        """
        # Enforce rate limiting
        self._wait_for_rate_limit()

        url = self.base_url + endpoint
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        if signed and self.api_key and self.api_secret and self.passphrase:
            timestamp = self._get_server_time()
            headers['OK-ACCESS-KEY'] = self.api_key
            headers['OK-ACCESS-SIGN'] = self._sign_request(timestamp, method, endpoint,
                                                           json.dumps(params) if params else '')
            headers['OK-ACCESS-TIMESTAMP'] = timestamp
            headers['OK-ACCESS-PASSPHRASE'] = self.passphrase

        try:
            if method.upper() == 'GET':
                if params:
                    url += '?' + urlencode(params)
                response = self.session.get(url, headers=headers, timeout=10)
            else:
                response = self.session.post(url, headers=headers, json=params or {}, timeout=10)

            response.raise_for_status()
            result = response.json()

            # Check for API errors
            if result.get('code') != '0':
                error_code = result.get('code', 'unknown')
                error_msg = result.get('msg', 'Unknown error')
                full_error = self.error_codes.get(error_code, f"Unknown error code: {error_code}")
                raise OKXAPIException(f"OKX API Error {error_code}: {full_error} - {error_msg}")

            return result

        except requests.exceptions.RequestException as e:
            raise OKXAPIException(f"Network error: {str(e)}")
        except json.JSONDecodeError as e:
            raise OKXAPIException(f"JSON decode error: {str(e)}")
        except Exception as e:
            raise OKXAPIException(f"Unexpected error: {str(e)}")

    def get_tickers(self, instType: str = 'SPOT') -> Dict:
        """
        Get tickers for all instruments

        Args:
            instType: Instrument type (SPOT, FUTURES, SWAP, OPTION)

        Returns:
            Tickers data
        """
        endpoint = "/api/v5/market/tickers"
        params = {'instType': instType}
        return self._make_request('GET', endpoint, params)

    def get_ticker(self, instId: str) -> Dict:
        """
        Get ticker for specific instrument

        Args:
            instId: Instrument ID (e.g., BTC-USDT)

        Returns:
            Ticker data
        """
        endpoint = "/api/v5/market/ticker"
        params = {'instId': instId}
        return self._make_request('GET', endpoint, params)

    def get_order_book(self, instId: str, sz: int = 5) -> Dict:
        """
        Get order book for instrument

        Args:
            instId: Instrument ID
            sz: Size of order book (1-400)

        Returns:
            Order book data

        返回参数
            参数名	类型	描述
            asks	Array of Arrays	卖方深度
            bids	Array of Arrays	买方深度
            ts	String	深度产生的时间
        """
        endpoint = "/api/v5/market/books"
        params = {'instId': instId, 'sz': sz}
        return self._make_request('GET', endpoint, params)

    def get_kline(self, instId: str, bar: str = '1m', limit: int = 100) -> Dict:
        """
        Get kline/candlestick data

        Args:
            instId: Instrument ID
            bar: Bar size (1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 6H, 12H, 1D, 1W, 1M)
            limit: Number of bars (max 100)

        Returns:
            Kline data
        """
        endpoint = "/api/v5/market/candles"
        params = {'instId': instId, 'bar': bar, 'limit': limit}
        return self._make_request('GET', endpoint, params)

    def get_account_balance(self) -> Dict:
        """
        Get account balance (requires authentication)

        Returns:
            Account balance data
        """
        endpoint = "/api/v5/account/balance"
        return self._make_request('GET', endpoint, signed=True)

    def place_order(self, instId: str, tdMode: str, side: str, ordType: str,
                    sz: str = None, px: str = None, reduceOnly: bool = False,
                    tgtCcy: str = 'quote_ccy', posSide: str = None) -> Dict:
        """
        Place order (requires authentication)

        Args:
            instId: Instrument ID
            tdMode: Trade mode (cash, cross, isolated)
            side: Order side (buy, sell)
            ordType: Order type (market, limit, post_only, fok, ioc)
            sz: Quantity (can be in base_ccy or quote_ccy depending on tgtCcy)
            px: Price (required for limit orders)
            reduceOnly: Reduce only flag
            tgtCcy: Order quantity unit (base_ccy, quote_ccy) - if quote_ccy, sz is in USDT
            posSide: long_short_mode or net_mode, check by client._make_request("GET", "/api/v5/account/config", {}, signed=True)

        Returns:
            Order result
        """
        endpoint = "/api/v5/trade/order"
        params = {
            'instId': instId,
            'tdMode': tdMode,
            'side': side,
            'ordType': ordType
        }

        if sz:
            params['sz'] = sz
        if px:
            params['px'] = px
        if reduceOnly:
            params['reduceOnly'] = reduceOnly
        if tgtCcy:
            params['tgtCcy'] = tgtCcy
        if posSide:
            params['posSide'] = posSide  # 👈 关键加上这一行

        return self._make_request('POST', endpoint, params, signed=True)

    def get_orders(self, instId: str = None, state: str = None) -> Dict:
        """
        Get order list (requires authentication)

        Args:
            instId: Instrument ID
            state: Order state (live, filled, canceled)

        Returns:
            Orders data
        """
        endpoint = "/api/v5/trade/orders-pending"
        params = {}
        if instId:
            params['instId'] = instId
        if state:
            params['state'] = state
        return self._make_request('GET', endpoint, params, signed=True)

    def set_leverage(self, instId: str, lever: str, mgnMode: str = 'isolated', posSide: str = 'long') -> Dict:
        """
        Set leverage for instrument (requires authentication)
        
        Args:
            instId: Instrument ID (e.g., BTC-USDT-SWAP)
            lever: Leverage (e.g., "3" for 3x)
            mgnMode: Margin mode (cross, isolated)
            posSide: Position side (long, short, net) - required for long/short mode
            
        Returns:
            Set leverage result
        """
        endpoint = "/api/v5/account/set-leverage"
        params = {
            'instId': instId,
            'lever': lever,
            'mgnMode': mgnMode,
            'posSide': posSide
        }
        return self._make_request('POST', endpoint, params, signed=True)

    def get_positions(self, instId: str = None) -> Dict:
        """
        Get positions (requires authentication)
        
        Args:
            instId: Instrument ID (optional)
            
        Returns:
            Positions data
        """
        endpoint = "/api/v5/account/positions"
        params = {}
        # 有bug，当前不支持指定instId
        # if instId:
        #     params['instId'] = instId
        return self._make_request('GET', endpoint, params, signed=True)

    def get_instruments(self, instType: str, instId: str = None) -> Dict:
        """
        Get instrument information
        
        Args:
            instType: Instrument type (SPOT, SWAP, FUTURES, OPTION)
            instId: Instrument ID (optional)
            
        Returns:
            Instrument data including contract size, tick size, etc.
        """
        endpoint = "/api/v5/public/instruments"
        params = {'instType': instType}
        if instId:
            params['instId'] = instId
        return self._make_request('GET', endpoint, params)


def get_okx_client():
    import os
    # 加载环境变量
    from dotenv import load_dotenv

    load_dotenv()
    # 从环境变量获取API凭证（如果存在）
    api_key = os.getenv('OK-ACCESS-KEY')
    api_secret = os.getenv('OK-ACCESS-SECRET')
    passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

    client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)

    return client
