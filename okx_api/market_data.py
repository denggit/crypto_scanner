from typing import List, Dict, Any
from .client import OKXClient
from .models import Ticker, OrderBook, Candle, MarketData
import time
import pandas as pd
import numpy as np

class MarketDataRetriever:
    """
    Retrieve and process market data from OKX
    """

    def __init__(self, client: OKXClient):
        """
        Initialize MarketDataRetriever

        Args:
            client (OKXClient): OKXClient instance for making API requests
        """
        self.client = client

    def get_all_tickers(self, instType: str = 'SPOT') -> List[Ticker]:
        """
        Get all tickers for specified instrument type

        Args:
            instType (str): Instrument type to retrieve tickers for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts

        Returns:
            List[Ticker]: List of Ticker objects containing market data for all instruments of the specified type
        """
        response = self.client.get_tickers(instType)
        tickers = []

        if response.get('code') == '0' and 'data' in response:
            for ticker_data in response['data']:
                ticker = Ticker(
                    instId=ticker_data.get('instId', ''),
                    last=float(ticker_data.get('last', 0)),
                    open24h=float(ticker_data.get('open24h', 0)),
                    high24h=float(ticker_data.get('high24h', 0)),
                    low24h=float(ticker_data.get('low24h', 0)),
                    vol24h=float(ticker_data.get('vol24h', 0)),
                    volCcy24h=float(ticker_data.get('volCcy24h', 0)),
                    ts=int(ticker_data.get('ts', 0)),
                    bidPx=float(ticker_data.get('bidPx', 0)),
                    askPx=float(ticker_data.get('askPx', 0)),
                    bidSz=float(ticker_data.get('bidSz', 0)),
                    askSz=float(ticker_data.get('askSz', 0))
                )
                tickers.append(ticker)

        return tickers

    def get_ticker_by_symbol(self, instId: str) -> Ticker:
        """
        Get ticker for specific symbol

        Args:
            instId (str): Instrument ID to retrieve ticker data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')

        Returns:
            Ticker: Ticker object containing market data for the specified instrument, or None if not found
        """
        response = self.client.get_ticker(instId)

        if response.get('code') == '0' and 'data' in response:
            ticker_data = response['data'][0]
            return Ticker(
                instId=ticker_data.get('instId', ''),
                last=float(ticker_data.get('last', 0)),
                open24h=float(ticker_data.get('open24h', 0)),
                high24h=float(ticker_data.get('high24h', 0)),
                low24h=float(ticker_data.get('low24h', 0)),
                vol24h=float(ticker_data.get('vol24h', 0)),
                volCcy24h=float(ticker_data.get('volCcy24h', 0)),
                ts=int(ticker_data.get('ts', 0)),
                bidPx=float(ticker_data.get('bidPx', 0)),
                askPx=float(ticker_data.get('askPx', 0)),
                bidSz=float(ticker_data.get('bidSz', 0)),
                askSz=float(ticker_data.get('askSz', 0))
            )
        return None

    def get_order_book(self, instId: str, sz: int = 5) -> OrderBook:
        """
        Get order book for instrument

        Args:
            instId (str): Instrument ID to retrieve order book data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')
            sz (int): Number of bid/ask levels to retrieve. Defaults to 5. Maximum is 400.

        Returns:
            OrderBook: OrderBook object containing bid/ask data for the specified instrument, or None if not found
        """
        response = self.client.get_order_book(instId, sz)

        if response.get('code') == '0' and 'data' in response:
            book_data = response['data'][0]
            return OrderBook(
                instId=book_data.get('instId', ''),
                bids=book_data.get('bids', []),
                asks=book_data.get('asks', []),
                ts=int(book_data.get('ts', 0))
            )
        return None

    def get_kline(self, instId: str, bar: str = '1m', limit: int = 100) -> List[Candle]:
        """
        Get kline/candlestick data

        Args:
            instId (str): Instrument ID to retrieve kline data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')
            bar (str): Bar size/candle interval. Defaults to '1m'.
                      Options: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
            limit (int): Number of bars/candles to retrieve. Defaults to 100. Maximum is 100.

        Returns:
            List[Candle]: List of Candle objects containing OHLCV data for the specified instrument
        """
        response = self.client.get_kline(instId, bar, limit)
        candles = []

        if response.get('code') == '0' and 'data' in response:
            for candle_data in response['data']:
                candle = Candle(
                    ts=int(candle_data[0]),
                    o=float(candle_data[1]),
                    h=float(candle_data[2]),
                    l=float(candle_data[3]),
                    c=float(candle_data[4]),
                    vol=float(candle_data[5]),
                    volCcy=float(candle_data[6])
                )
                candles.append(candle)

        return candles

    def get_market_data(self, instId: str, bar: str = '1m', kline_limit: int = 100) -> MarketData:
        """
        Get complete market data for an instrument

        Args:
            instId (str): Instrument ID to retrieve complete market data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')
            bar (str): Bar size/candle interval for kline data. Defaults to '1m'.
                      Options: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
            kline_limit (int): Number of kline bars/candles to retrieve. Defaults to 100. Maximum is 100.

        Returns:
            MarketData: MarketData object containing ticker, order book, and kline data for the specified instrument
        """
        ticker = self.get_ticker_by_symbol(instId)
        order_book = self.get_order_book(instId)
        candles = self.get_kline(instId, bar, kline_limit)

        return MarketData(
            ticker=ticker,
            order_book=order_book,
            candles=candles,
            timestamp=int(time.time() * 1000)
        )

    def get_all_market_data(self, instType: str = 'SPOT', bar: str = '1m',
                           kline_limit: int = 100) -> Dict[str, MarketData]:
        """
        Get complete market data for all instruments

        Args:
            instType (str): Instrument type to retrieve market data for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts
            bar (str): Bar size/candle interval for kline data. Defaults to '1m'.
                      Options: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
            kline_limit (int): Number of kline bars/candles to retrieve per instrument. Defaults to 100. Maximum is 100.

        Returns:
            Dict[str, MarketData]: Dictionary mapping instrument IDs to MarketData objects containing complete market data
        """
        all_data = {}
        tickers = self.get_all_tickers(instType)

        for ticker in tickers:
            instId = ticker.instId
            try:
                order_book = self.get_order_book(instId)
                candles = self.get_kline(instId, bar, kline_limit)

                market_data = MarketData(
                    ticker=ticker,
                    order_book=order_book,
                    candles=candles,
                    timestamp=int(time.time() * 1000)
                )

                all_data[instId] = market_data

                # Add delay to respect rate limits
                time.sleep(0.1)
            except Exception as e:
                print(f"Error fetching data for {instId}: {e}")
                continue

        return all_data

    def get_kline_with_ma(self, instId: str, bar: str = '1m', limit: int = 100, ma_periods: List[int] = None) -> pd.DataFrame:
        """
        Get kline data with moving averages

        Args:
            instId (str): Instrument ID to retrieve kline data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')
            bar (str): Bar size/candle interval. Defaults to '1m'.
                      Options: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
            limit (int): Number of bars/candles to retrieve. Defaults to 100. Maximum is 100.
            ma_periods (List[int]): List of moving average periods to calculate (e.g., [5, 10, 20, 50])

        Returns:
            pd.DataFrame: DataFrame containing kline data with calculated moving averages
        """
        if ma_periods is None:
            ma_periods = [5, 10, 20]

        # Get kline data
        kline_data = self.get_kline(instId, bar, limit)

        if not kline_data:
            return pd.DataFrame()

        # Convert to DataFrame more efficiently using numpy
        timestamps = np.array([candle.ts for candle in kline_data])
        opens = np.array([candle.o for candle in kline_data])
        highs = np.array([candle.h for candle in kline_data])
        lows = np.array([candle.l for candle in kline_data])
        closes = np.array([candle.c for candle in kline_data])
        volumes = np.array([candle.vol for candle in kline_data])
        volume_ccys = np.array([candle.volCcy for candle in kline_data])

        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'volume_currency': volume_ccys
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Calculate moving averages
        for period in ma_periods:
            if period <= len(df):
                df[f'MA_{period}'] = df['close'].rolling(window=period).mean()

        return df