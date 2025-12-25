import time
from typing import List, Dict, Union

import numpy as np
import pandas as pd

from .client import OKXClient
from .models import Ticker, OrderBook, Candle, MarketData, Instrument
from utils.logger import logger


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

    def get_all_symbol(self, instType: str = 'SPOT', currency: str = 'USDT') -> List[str]:
        """
        Get all instrument IDs for specified instrument type and currency

        Args:
            instType (str): Instrument type to retrieve symbols for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts
            currency (str): Currency to filter by (e.g., 'USDT', 'USD'). Defaults to 'USDT'.

        Returns:
            List[str]: List of instrument IDs matching the specified type and currency
        """
        response = self.client.get_tickers(instType)
        symbols = []

        if response.get('code') == '0' and 'data' in response:
            for ticker_data in response['data']:
                instId = ticker_data.get('instId', '')
                # If currency filter is specified, only add symbols that match
                if currency is None or f'-{currency}' in instId:
                    symbols.append(instId)

        return symbols

    def get_all_tickers(self, instType: str = 'SPOT', currency: str = 'USDT') -> List[Ticker]:
        """
        Get all tickers for specified instrument type

        Args:
            instType (str): Instrument type to retrieve tickers for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts
            currency (str, optional): Currency to filter by (e.g., 'USDT', 'USD').
                                    If provided, only tickers ending with this currency will be returned.

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

                # If currency filter is specified, only add tickers that match
                if currency is None or f'-{currency}' in ticker.instId:
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

    def get_kline(self, instId: str, bar: str = '1m', limit: int = 100, return_dataframe: bool = True) -> Union[List[Candle], pd.DataFrame]:
        """
        Get kline/candlestick data

        Args:
            instId (str): Instrument ID to retrieve kline data for (e.g., 'BTC-USDT', 'ETH-USD-SWAP')
            bar (str): Bar size/candle interval. Defaults to '1m'.
                      Options: '1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M'
            limit (int): Number of bars/candles to retrieve. Defaults to 100. Maximum is 300.
            return_dataframe (bool): If True, returns pandas DataFrame. If False, returns List[Candle]. Defaults to True.

        Returns:
            Union[List[Candle], pd.DataFrame]: Kline data in specified format
        """
        response = self.client.get_kline(instId, bar, limit)

        if response.get('code') != '0' or 'data' not in response:
            return pd.DataFrame() if return_dataframe else []

        # If not returning DataFrame, use original logic
        if not return_dataframe:
            candles = []
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

        # Convert to DataFrame efficiently using list comprehension (proven to be faster)
        df = pd.DataFrame({
            'timestamp': [int(candle[0]) for candle in response['data']],
            'open': [float(candle[1]) for candle in response['data']],
            'high': [float(candle[2]) for candle in response['data']],
            'low': [float(candle[3]) for candle in response['data']],
            'close': [float(candle[4]) for candle in response['data']],
            'volume': [float(candle[5]) for candle in response['data']],
            'volume_currency': [float(candle[6]) for candle in response['data']]
        })

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def get_volume_filtered_symbols(self, instType: str = 'SPOT', currency: str = 'USDT',
                                  min_vol_ccy: float = 100000) -> List[str]:
        """
        Get symbols filtered by minimum 24h volume

        Args:
            instType (str): Instrument type to retrieve symbols for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts
            currency (str): Currency to filter by (e.g., 'USDT', 'USD'). Defaults to 'USDT'.
            min_vol_ccy (float): Minimum 24h volume in currency to include. Defaults to 100,000.

        Returns:
            List[str]: List of instrument IDs with 24h volume >= min_vol_ccy, sorted by volume (descending)
        """
        tickers = self.get_all_tickers(instType, currency)

        # Filter by minimum volume (确保 volCcy24h 是有效的数值)
        filtered_symbols = []
        volume_dict = {}
        
        for ticker in tickers:
            # 确保 volCcy24h 是有效的数值类型
            vol_ccy = ticker.volCcy24h
            if vol_ccy is None:
                continue
            
            # 转换为 float（处理可能的字符串类型）
            try:
                vol_ccy_float = float(vol_ccy)
            except (ValueError, TypeError):
                continue
            
            # 检查是否满足最小交易量要求
            if vol_ccy_float >= min_vol_ccy:
                filtered_symbols.append(ticker.instId)
                volume_dict[ticker.instId] = vol_ccy_float

        # Sort by volume (descending)
        filtered_symbols.sort(key=lambda x: volume_dict.get(x, 0), reverse=True)

        return filtered_symbols

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
        # Get kline data as Candle objects for MarketData
        candles = self.get_kline(instId, bar, kline_limit, return_dataframe=False)

        return MarketData(
            ticker=ticker,
            order_book=order_book,
            candles=candles,
            timestamp=int(time.time() * 1000)
        )

    def get_all_market_data(self, instType: str = 'SPOT', bar: str = '1m',
                            kline_limit: int = 100, currency: str = None) -> Dict[str, MarketData]:
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
            currency (str, optional): Currency to filter by (e.g., 'USDT', 'USD').
                                    If provided, only instruments ending with this currency will be returned.

        Returns:
            Dict[str, MarketData]: Dictionary mapping instrument IDs to MarketData objects containing complete market data
        """
        all_data = {}
        tickers = self.get_all_tickers(instType, currency)

        for ticker in tickers:
            instId = ticker.instId
            try:
                order_book = self.get_order_book(instId)
                # Get kline data as Candle objects for MarketData
                candles = self.get_kline(instId, bar, kline_limit, return_dataframe=False)

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
                logger.error(f"Error fetching data for {instId}: {e}")
                continue

        return all_data

    def get_kline_with_ma(self, instId: str, bar: str = '1m', limit: int = 100,
                          ma_periods: List[int] = None) -> pd.DataFrame:
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

        # Get kline data (now returns DataFrame directly)
        df = self.get_kline(instId, bar, limit)

        if df.empty:
            return pd.DataFrame()

        # Calculate moving averages
        for period in ma_periods:
            if period <= len(df):
                df[f'MA_{period}'] = df['close'].rolling(window=period).mean()

        return df

    def get_instruments(self, instType: str, instId: str = None, uly: str = None, state: str = 'live') -> List[Instrument]:
        """
        Get instrument information

        Args:
            instType (str): Instrument type to retrieve instruments for.
                           Options are:
                           - 'SPOT': Spot trading pairs (e.g., BTC-USDT)
                           - 'FUTURES': Futures contracts with expiration dates
                           - 'SWAP': Perpetual swap contracts without expiration
                           - 'OPTION': Options contracts
            instId (str, optional): Specific instrument ID to retrieve information for.
            uly (str, optional): Underlying asset to filter by.
            state (str, optional): Instrument state to filter by. Defaults to 'live'.
                                  Options: 'live', 'suspend', 'preopen', 'settlement'

        Returns:
            List[Instrument]: List of Instrument objects containing detailed instrument information
        """
        response = self.client.get_instruments(instType, instId)
        instruments = []

        if response.get('code') == '0' and 'data' in response:
            for inst_data in response['data']:
                instrument = Instrument(
                    instId=inst_data.get('instId', ''),
                    uly=inst_data.get('uly', ''),
                    category=inst_data.get('category', ''),
                    baseCcy=inst_data.get('baseCcy', ''),
                    quoteCcy=inst_data.get('quoteCcy', ''),
                    settleCcy=inst_data.get('settleCcy', ''),
                    ctVal=inst_data.get('ctVal', 0),
                    ctMult=inst_data.get('ctMult', 0),
                    ctValCcy=inst_data.get('ctValCcy', ''),
                    optType=inst_data.get('optType', ''),
                    stk=inst_data.get('stk', 0),
                    listTime=inst_data.get('listTime', 0),
                    expTime=inst_data.get('expTime', 0),
                    lever=inst_data.get('lever', 0),
                    tickSz=inst_data.get('tickSz', 0),
                    lotSz=inst_data.get('lotSz', 0),
                    minSz=inst_data.get('minSz', 0),
                    ctType=inst_data.get('ctType', ''),
                    alias=inst_data.get('alias', ''),
                    state=inst_data.get('state', ''),
                    maxLmtSz=inst_data.get('maxLmtSz', 0),
                    maxMktSz=inst_data.get('maxMktSz', 0),
                    maxTwapSz=inst_data.get('maxTwapSz', 0),
                    maxIcebergSz=inst_data.get('maxIcebergSz', 0),
                    maxTriggerSz=inst_data.get('maxTriggerSz', 0),
                    maxStopSz=inst_data.get('maxStopSz', 0)
                )

                # Apply filters
                if uly and instrument.uly != uly:
                    continue
                if state and instrument.state != state:
                    continue

                instruments.append(instrument)

        return instruments

    def get_instrument_info(self, instId: str) -> Instrument:
        """
        Get detailed information for a specific instrument

        Args:
            instId (str): Instrument ID to retrieve information for (e.g., 'BTC-USDT', 'BTC-USDT-SWAP')

        Returns:
            Instrument: Instrument object containing detailed information, or None if not found
        """
        # Determine instrument type from instId
        if instId.endswith('-SWAP'):
            instType = 'SWAP'
        elif instId.endswith('-FUTURES'):
            instType = 'FUTURES'
        elif instId.endswith('-OPTION'):
            instType = 'OPTION'
        else:
            instType = 'SPOT'

        instruments = self.get_instruments(instType, instId)
        return instruments[0] if instruments else None

    def get_instruments_by_currency(self, instType: str, currency: str, state: str = 'live') -> List[Instrument]:
        """
        Get instruments filtered by currency

        Args:
            instType (str): Instrument type to retrieve instruments for.
            currency (str): Currency to filter by (e.g., 'USDT', 'USD').
            state (str, optional): Instrument state to filter by. Defaults to 'live'.

        Returns:
            List[Instrument]: List of Instrument objects for the specified currency
        """
        instruments = self.get_instruments(instType, state=state)
        
        if instType == 'SPOT':
            # For SPOT, filter by quote currency
            return [inst for inst in instruments if inst.quoteCcy == currency]
        else:
            # For derivatives, filter by settle currency
            return [inst for inst in instruments if inst.settleCcy == currency]

    def get_contract_details(self, instId: str) -> Dict[str, float]:
        """
        Get contract details for an instrument (contract value, multiplier, etc.)

        Args:
            instId (str): Instrument ID to retrieve contract details for

        Returns:
            Dict[str, float]: Dictionary containing contract details
        """
        instrument = self.get_instrument_info(instId)
        if not instrument:
            return {}

        return {
            'contract_value': instrument.ctVal,
            'contract_multiplier': instrument.ctMult,
            'tick_size': instrument.tickSz,
            'lot_size': instrument.lotSz,
            'min_order_size': instrument.minSz,
            'max_limit_size': instrument.maxLmtSz,
            'max_market_size': instrument.maxMktSz
        }

    def get_trading_parameters(self, instId: str) -> Dict[str, any]:
        """
        Get trading parameters for an instrument

        Args:
            instId (str): Instrument ID to retrieve trading parameters for

        Returns:
            Dict[str, any]: Dictionary containing trading parameters
        """
        instrument = self.get_instrument_info(instId)
        if not instrument:
            return {}

        return {
            'instrument_id': instrument.instId,
            'base_currency': instrument.baseCcy,
            'quote_currency': instrument.quoteCcy,
            'settle_currency': instrument.settleCcy,
            'tick_size': instrument.tickSz,
            'lot_size': instrument.lotSz,
            'min_order_size': instrument.minSz,
            'max_leverage': instrument.lever,
            'instrument_state': instrument.state,
            'contract_type': instrument.ctType,
            'alias': instrument.alias
        }
