#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : market_data.py
@Description: Binance Market Data Retriever
"""

import pandas as pd
from typing import Dict, List, Optional
from .client import BinanceClient
from .models import Ticker, Kline


class MarketDataRetriever:
    """
    Market data retriever for Binance
    """

    def __init__(self, client: BinanceClient):
        """
        Initialize Market Data Retriever

        Args:
            client: BinanceClient instance
        """
        self.client = client

    def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """
        Get ticker for specific symbol

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)

        Returns:
            Ticker object if successful, None otherwise
        """
        try:
            response = self.client.get_ticker(symbol)
            return Ticker(
                symbol=response.get('symbol', ''),
                priceChange=float(response.get('priceChange', 0)),
                priceChangePercent=float(response.get('priceChangePercent', 0)),
                weightedAvgPrice=float(response.get('weightedAvgPrice', 0)),
                prevClosePrice=float(response.get('prevClosePrice', 0)),
                lastPrice=float(response.get('lastPrice', 0)),
                lastQty=float(response.get('lastQty', 0)),
                bidPrice=float(response.get('bidPrice', 0)),
                bidQty=float(response.get('bidQty', 0)),
                askPrice=float(response.get('askPrice', 0)),
                askQty=float(response.get('askQty', 0)),
                openPrice=float(response.get('openPrice', 0)),
                highPrice=float(response.get('highPrice', 0)),
                lowPrice=float(response.get('lowPrice', 0)),
                volume=float(response.get('volume', 0)),
                quoteVolume=float(response.get('quoteVolume', 0)),
                openTime=int(response.get('openTime', 0)),
                closeTime=int(response.get('closeTime', 0)),
                firstId=int(response.get('firstId', 0)),
                lastId=int(response.get('lastId', 0)),
                count=int(response.get('count', 0))
            )
        except Exception:
            return None

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price, 0 if failed
        """
        try:
            response = self.client.get_ticker(symbol)
            return float(response.get('lastPrice', 0))
        except Exception:
            return 0.0

    def get_kline_data(self, symbol: str, interval: str = '1m', limit: int = 500) -> List[Kline]:
        """
        Get kline data

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of bars

        Returns:
            List of Kline objects
        """
        try:
            response = self.client.get_kline(symbol, interval, limit)
            klines = []
            for kline in response:
                klines.append(Kline(
                    open_time=int(kline[0]),
                    open=float(kline[1]),
                    high=float(kline[2]),
                    low=float(kline[3]),
                    close=float(kline[4]),
                    volume=float(kline[5]),
                    close_time=int(kline[6]),
                    quote_asset_volume=float(kline[7]),
                    number_of_trades=int(kline[8]),
                    taker_buy_base_asset_volume=float(kline[9]),
                    taker_buy_quote_asset_volume=float(kline[10]),
                    ignore=int(kline[11])
                ))
            return klines
        except Exception:
            return []

    def get_kline_dataframe(self, symbol: str, interval: str = '1m', limit: int = 500) -> pd.DataFrame:
        """
        Get kline data as pandas DataFrame

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of bars

        Returns:
            DataFrame with kline data
        """
        klines = self.get_kline_data(symbol, interval, limit)
        if not klines:
            return pd.DataFrame()

        data = []
        for kline in klines:
            data.append({
                'timestamp': kline.open_time,
                'open': kline.open,
                'high': kline.high,
                'low': kline.low,
                'close': kline.close,
                'volume': kline.volume,
                'close_time': kline.close_time,
                'quote_volume': kline.quote_asset_volume,
                'trades': kline.number_of_trades,
                'taker_buy_base_volume': kline.taker_buy_base_asset_volume,
                'taker_buy_quote_volume': kline.taker_buy_quote_asset_volume
            })

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """
        Get order book

        Args:
            symbol: Trading pair symbol
            limit: Size of order book

        Returns:
            Order book data
        """
        try:
            return self.client.get_order_book(symbol, limit)
        except Exception:
            return {'bids': [], 'asks': []}

    def get_24hr_ticker(self, symbol: str) -> Dict:
        """
        Get 24hr ticker statistics

        Args:
            symbol: Trading pair symbol

        Returns:
            24hr ticker data
        """
        try:
            return self.client.get_ticker(symbol)
        except Exception:
            return {}

    def get_all_tickers(self) -> List[Dict]:
        """
        Get all tickers

        Returns:
            List of all tickers
        """
        try:
            response = self.client.get_tickers()
            return response if isinstance(response, list) else []
        except Exception:
            return []

    def get_exchange_info(self, symbol: str = None) -> Dict:
        """
        Get exchange information

        Args:
            symbol: Trading pair symbol (optional)

        Returns:
            Exchange information
        """
        try:
            return self.client.get_exchange_info(symbol)
        except Exception:
            return {}

    def get_symbol_info(self, symbol: str) -> Dict:
        """
        Get symbol information

        Args:
            symbol: Trading pair symbol

        Returns:
            Symbol information
        """
        try:
            response = self.client.get_exchange_info(symbol)
            if 'symbols' in response and response['symbols']:
                return response['symbols'][0]
            return {}
        except Exception:
            return {}

    def get_price_change_stats(self, symbol: str) -> Dict:
        """
        Get price change statistics

        Args:
            symbol: Trading pair symbol

        Returns:
            Price change statistics
        """
        try:
            ticker = self.get_ticker(symbol)
            if ticker:
                return {
                    'symbol': ticker.symbol,
                    'price_change': ticker.priceChange,
                    'price_change_percent': ticker.priceChangePercent,
                    'high_price': ticker.highPrice,
                    'low_price': ticker.lowPrice,
                    'volume': ticker.volume,
                    'quote_volume': ticker.quoteVolume
                }
            return {}
        except Exception:
            return {}

    def get_volume_data(self, symbol: str, interval: str = '1m', limit: int = 100) -> Dict:
        """
        Get volume data

        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            limit: Number of bars

        Returns:
            Volume statistics
        """
        try:
            df = self.get_kline_dataframe(symbol, interval, limit)
            if df.empty:
                return {}

            return {
                'current_volume': df['volume'].iloc[-1],
                'avg_volume': df['volume'].mean(),
                'max_volume': df['volume'].max(),
                'min_volume': df['volume'].min(),
                'volume_ratio': df['volume'].iloc[-1] / df['volume'].mean() if df['volume'].mean() > 0 else 0
            }
        except Exception:
            return {}