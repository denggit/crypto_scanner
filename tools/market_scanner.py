"""
Cryptocurrency Market Scanner using OKX API
"""

import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever

class CryptoScanner:
    """Cryptocurrency market scanner for quant research"""

    def __init__(self, client: OKXClient):
        self.client = client
        self.market_data_retriever = MarketDataRetriever(client)

    def scan_top_coins(self, limit: int = 20) -> list:
        """
        Scan top cryptocurrencies by 24h volume

        Args:
            limit: Number of top coins to return

        Returns:
            List of top coins with their market data
        """
        try:
            tickers = self.market_data_retriever.get_all_tickers('SPOT')

            # Sort by 24h volume (in currency)
            sorted_tickers = sorted(tickers, key=lambda x: x.volCcy24h, reverse=True)

            top_coins = []
            for i, ticker in enumerate(sorted_tickers[:limit]):
                coin_data = {
                    'rank': i + 1,
                    'symbol': ticker.instId,
                    'price': ticker.last,
                    'price_change_24h': ((ticker.last - ticker.open24h) / ticker.open24h * 100) if ticker.open24h > 0 else 0,
                    'high_24h': ticker.high24h,
                    'low_24h': ticker.low24h,
                    'volume_24h': ticker.volCcy24h,
                    'volume_change_24h': 0  # Would need additional data to calculate
                }
                top_coins.append(coin_data)

            return top_coins
        except Exception as e:
            print(f"Error scanning top coins: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return []

    def scan_volatility(self, symbol: str, bar: str = '15m', limit: int = 96) -> dict:
        """
        Scan volatility for a specific symbol (96 periods of 15m = 24 hours)

        Args:
            symbol: Trading pair (e.g., BTC-USDT)
            bar: Time interval
            limit: Number of periods

        Returns:
            Volatility data
        """
        try:
            candles = self.market_data_retriever.get_kline(symbol, bar, limit)

            if not candles:
                return {}

            prices = [candle.c for candle in candles]

            if len(prices) < 2:
                return {}

            # Calculate price changes
            price_changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1] * 100
                price_changes.append(change)

            # Calculate volatility (standard deviation of price changes)
            if price_changes:
                avg_change = sum(price_changes) / len(price_changes)
                variance = sum((x - avg_change) ** 2 for x in price_changes) / len(price_changes)
                volatility = variance ** 0.5
            else:
                volatility = 0

            return {
                'symbol': symbol,
                'volatility': volatility,
                'avg_price_change': avg_change if price_changes else 0,
                'max_price_change': max(price_changes) if price_changes else 0,
                'min_price_change': min(price_changes) if price_changes else 0,
                'periods': len(candles)
            }
        except Exception as e:
            print(f"Error scanning volatility for {symbol}: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return {}

    def scan_liquidity(self, symbol: str, depth: int = 10) -> dict:
        """
        Scan liquidity for a specific symbol

        Args:
            symbol: Trading pair (e.g., BTC-USDT)
            depth: Order book depth to analyze

        Returns:
            Liquidity data
        """
        try:
            order_book = self.market_data_retriever.get_order_book(symbol, depth)

            if not order_book:
                return {}

            # Calculate total bid and ask volume
            total_bids = sum(float(bid[1]) for bid in order_book.bids)
            total_asks = sum(float(ask[1]) for ask in order_book.asks)

            # Calculate spread
            best_bid = float(order_book.bids[0][0]) if order_book.bids else 0
            best_ask = float(order_book.asks[0][0]) if order_book.asks else 0
            spread = (best_ask - best_bid) / best_bid * 100 if best_bid > 0 else 0

            return {
                'symbol': symbol,
                'bid_volume': total_bids,
                'ask_volume': total_asks,
                'spread_percent': spread,
                'best_bid': best_bid,
                'best_ask': best_ask
            }
        except Exception as e:
            print(f"Error scanning liquidity for {symbol}: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return {}

    def generate_market_report(self) -> dict:
        """
        Generate a comprehensive market report

        Returns:
            Market report data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'top_coins': self.scan_top_coins(10),
            'volatility_data': [],
            'liquidity_data': []
        }

        # Analyze top 5 coins for volatility and liquidity
        top_symbols = [coin['symbol'] for coin in report['top_coins'][:5]]

        for symbol in top_symbols:
            volatility = self.scan_volatility(symbol)
            liquidity = self.scan_liquidity(symbol)

            if volatility:
                report['volatility_data'].append(volatility)
            if liquidity:
                report['liquidity_data'].append(liquidity)

            # Rate limiting
            time.sleep(0.1)

        return report

def main():
    print("Cryptocurrency Market Scanner")
    print("=" * 50)
    print("Note: This tool requires internet connectivity to access OKX API")
    print("If you encounter connection errors, please check your network settings\n")

    # 从环境变量获取API凭证（如果存在）
    import os
    api_key = os.getenv('OK-ACCESS-KEY')
    api_secret = os.getenv('OK-ACCESS-SECRET')
    passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

    # Initialize client (with auth if credentials are available)
    if api_key and api_secret and passphrase:
        print("Using authenticated client with API credentials")
        client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)
    else:
        print("Using public client (no authentication)")
        client = OKXClient()

    scanner = CryptoScanner(client)

    # Generate market report
    print("\nGenerating market report...")
    report = scanner.generate_market_report()

    print(f"\nReport generated at: {report['timestamp']}")

    if not report['top_coins']:
        print("Warning: Unable to retrieve market data. This may be due to:")
        print("  1. Network connectivity issues")
        print("  2. Firewall or proxy restrictions")
        print("  3. Temporary OKX API unavailability")
        print("\nPlease check your internet connection and try again.")
        return

    print("\nTop 10 Cryptocurrencies by Volume:")
    print("-" * 50)
    for coin in report['top_coins']:
        print(f"{coin['rank']:2d}. {coin['symbol']:10s} | "
              f"Price: ${coin['price']:>10.2f} | "
              f"24h Change: {coin['price_change_24h']:>6.2f}% | "
              f"Volume: ${coin['volume_24h']:>12,.0f}")

    if report['volatility_data']:
        print("\nVolatility Analysis (Top 5 coins):")
        print("-" * 50)
        for vol in report['volatility_data']:
            print(f"{vol['symbol']:10s} | "
                  f"Volatility: {vol['volatility']:>6.2f} | "
                  f"Avg Change: {vol['avg_price_change']:>6.2f}%")

    if report['liquidity_data']:
        print("\nLiquidity Analysis (Top 5 coins):")
        print("-" * 50)
        for liq in report['liquidity_data']:
            print(f"{liq['symbol']:10s} | "
                  f"Spread: {liq['spread_percent']:>5.2f}% | "
                  f"Bid Volume: {liq['bid_volume']:>10.2f} | "
                  f"Ask Volume: {liq['ask_volume']:>10.2f}")

    print("\n[INFO] For authenticated trading features, set your OKX API credentials in the .env file")

if __name__ == "__main__":
    main()