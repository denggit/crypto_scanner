"""
Example usage of OKX API modules for market data and trading
"""

import os
import sys
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from okx_api.trader import Trader

def main():
    # Initialize OKX client
    # Note: You need to set your API credentials in environment variables
    api_key = os.getenv('OKX_API_KEY')
    api_secret = os.getenv('OKX_API_SECRET')
    passphrase = os.getenv('OKX_PASSPHRASE')

    # Initialize client
    client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)

    # Initialize market data retriever
    market_data_retriever = MarketDataRetriever(client)

    # Initialize trader
    trader = Trader(client)

    print("OKX API Example")
    print("=" * 50)

    # Example 1: Get all spot tickers
    print("\n1. Getting all spot tickers...")
    try:
        tickers = market_data_retriever.get_all_tickers('SPOT')
        print(f"Retrieved {len(tickers)} tickers")
        if tickers:
            # Show top 5 tickers by volume
            sorted_tickers = sorted(tickers, key=lambda x: x.volCcy24h, reverse=True)
            print("\nTop 5 cryptocurrencies by 24h volume:")
            for i, ticker in enumerate(sorted_tickers[:5]):
                print(f"  {i+1}. {ticker.instId}: {ticker.last} (24h vol: {ticker.volCcy24h})")
    except Exception as e:
        print(f"Error getting tickers: {e}")

    # Example 2: Get specific ticker data
    print("\n2. Getting BTC-USDT ticker...")
    try:
        btc_ticker = market_data_retriever.get_ticker_by_symbol('BTC-USDT')
        if btc_ticker:
            print(f"BTC-USDT Price: {btc_ticker.last}")
            print(f"24h High: {btc_ticker.high24h}")
            print(f"24h Low: {btc_ticker.low24h}")
            print(f"24h Volume: {btc_ticker.volCcy24h}")
    except Exception as e:
        print(f"Error getting BTC-USDT ticker: {e}")

    # Example 3: Get order book
    print("\n3. Getting BTC-USDT order book...")
    try:
        order_book = market_data_retriever.get_order_book('BTC-USDT', sz=5)
        if order_book:
            print("Top 5 bids:")
            for bid in order_book.bids[:5]:
                print(f"  Price: {bid[0]}, Size: {bid[1]}")
            print("Top 5 asks:")
            for ask in order_book.asks[:5]:
                print(f"  Price: {ask[0]}, Size: {ask[1]}")
    except Exception as e:
        print(f"Error getting order book: {e}")

    # Example 4: Get kline data
    print("\n4. Getting BTC-USDT 1-hour kline data...")
    try:
        klines = market_data_retriever.get_kline('BTC-USDT', bar='1H', limit=10)
        if klines:
            print("Last 10 hourly candles:")
            for candle in klines:
                print(f"  Time: {candle.ts}, Open: {candle.o}, High: {candle.h}, Low: {candle.l}, Close: {candle.c}, Volume: {candle.volCcy}")
    except Exception as e:
        print(f"Error getting kline data: {e}")

    # Example 5: Get account balance (requires authentication)
    if api_key and api_secret and passphrase:
        print("\n5. Getting account balance...")
        try:
            balance = trader.get_account_balance()
            if balance.get('code') == '0':
                print("Account balances:")
                for bal in balance['data'][0]['details']:
                    if float(bal['availBal']) > 0:
                        print(f"  {bal['ccy']}: Available {bal['availBal']}, Total {bal['cashBal']}")
            else:
                print(f"Error getting balance: {balance.get('msg')}")
        except Exception as e:
            print(f"Error getting account balance: {e}")
    else:
        print("\n5. Account balance - Skipping (no API credentials provided)")

    print("\n" + "=" * 50)
    print("Example completed. For trading operations, ensure you have valid API credentials.")

if __name__ == "__main__":
    main()