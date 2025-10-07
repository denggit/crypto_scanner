"""
Cryptocurrency Market Scanner using OKX API
"""

import os
import sys
import time
from datetime import datetime
import asyncio
import concurrent.futures
from functools import lru_cache

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
        self._symbol_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._last_cache_update = 0

    def scan_top_coins(self, limit: int = 20, currency='USDT') -> list:
        """
        Scan top cryptocurrencies by 24h volume

        Args:
            limit: Number of top coins to return
            currency: Currency to filter by (default: USDT)

        Returns:
            List of top coins with their market data
        """
        try:
            tickers = self.market_data_retriever.get_all_tickers('SPOT', currency)

            # Sort by 24h volume (in currency)
            sorted_tickers = sorted(tickers, key=lambda x: x.volCcy24h, reverse=True)

            top_coins = []
            for i, ticker in enumerate(sorted_tickers[:limit]):
                coin_data = {
                    'rank': i + 1,
                    'symbol': ticker.instId,
                    'price': ticker.last,
                    'price_change_24h': (
                                (ticker.last - ticker.open24h) / ticker.open24h * 100) if ticker.open24h > 0 else 0,
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

    def scan_bullish_coins(self, currency: str = 'USDT', ma_periods: list = None,
                           bar: str = '5m', min_vol_ccy: float = 1000000, use_parallel: bool = True,
                           use_cache: bool = True) -> list:
        """
        Scan for cryptocurrencies with bullish moving average alignment (golden cross/multi MA alignment)
        Optimized for performance with parallel processing and caching

        Args:
            currency: Currency to filter by (default: USDT)
            ma_periods: List of MA periods to check alignment (default: [5, 20, 60, 120, 200])
            bar: Time interval for kline data (default: 5m)
            min_vol_ccy: Minimum 24h volume in currency to include (default: 100,000)
            use_parallel: Use parallel processing for faster scanning (default: True)
            use_cache: Use caching for symbol data (default: True)

        Returns:
            List of coins with bullish MA alignment
        """
        if ma_periods is None:
            ma_periods = [5, 20, 60, 120, 200]

        # 保持有限的K线数量以加快网络请求；仅用于计算末端均线
        limit = max(ma_periods) + 1

        try:
            # Get symbols filtered by volume
            symbols = self._get_volume_filtered_symbols(currency, min_vol_ccy, use_cache)

            if not symbols:
                print(f"No symbols found with 24h volume >= {min_vol_ccy:,.0f} {currency}")
                return []

            print(f"Scanning {len(symbols)} symbols with 24h volume >= {min_vol_ccy:,.0f} {currency}")

            if use_parallel and len(symbols) > 1:
                return self._scan_bullish_coins_parallel(symbols, ma_periods, bar, limit)
            else:
                return self._scan_bullish_coins_sequential(symbols, ma_periods, bar, limit)

        except Exception as e:
            print(f"Error scanning bullish coins: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return []

    def _scan_bullish_coins_sequential(self, symbols: list, ma_periods: list, bar: str, limit: int) -> list:
        """Sequential scanning implementation"""
        bullish_coins = []

        for symbol in symbols:
            try:
                result = self._analyze_symbol_bullish(symbol, ma_periods, bar, limit)
                if result:
                    bullish_coins.append(result)
            except Exception:
                continue

        # Sort by trend strength
        bullish_coins.sort(key=lambda x: x['trend_strength'], reverse=True)
        return bullish_coins

    def _scan_bullish_coins_parallel(self, symbols: list, ma_periods: list, bar: str, limit: int) -> list:
        """Parallel scanning implementation using ThreadPoolExecutor"""
        bullish_coins = []

        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._analyze_symbol_bullish, symbol, ma_periods, bar, limit): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        bullish_coins.append(result)
                except Exception:
                    continue

        # Sort by trend strength
        bullish_coins.sort(key=lambda x: x['trend_strength'], reverse=True)
        return bullish_coins

    def _analyze_symbol_bullish(self, symbol: str, ma_periods: list, bar: str, limit: int) -> dict:
        """Analyze a single symbol for bullish MA alignment"""
        try:
            # Get kline data
            df = self.market_data_retriever.get_kline(symbol, bar, limit)

            if df is None or len(df) == 0:
                return None

            # 仅计算最后一个时间点的均线；支持数据不足时的部分均线
            closes = df['c'] if 'c' in df.columns else df['close']
            if len(closes) == 0:
                return None

            # 优化：预先计算所有可能的MA值，避免重复切片
            available_periods = [p for p in sorted(ma_periods) if len(closes) >= p]
            if not available_periods:
                return None

            # 优化：一次性计算所有MA值
            latest_ma_values = {}
            current_price = float(closes.iloc[-1])

            # 使用更高效的MA计算方法
            for p in available_periods:
                # 使用numpy数组进行更快的计算
                close_values = closes.iloc[-p:].values
                latest_ma_values[p] = float(close_values.mean())

            # 多头排列：close > ma_min > ... > ma_max（仅基于可计算的周期）
            ordered_periods = available_periods  # 已按升序
            is_bullish_chain = True
            prev_value = current_price

            # 优化：提前检查是否可能形成多头排列
            if current_price <= latest_ma_values[ordered_periods[0]]:
                return None

            for p in ordered_periods:
                if prev_value <= latest_ma_values[p]:
                    is_bullish_chain = False
                    break
                prev_value = latest_ma_values[p]

            if is_bullish_chain:
                longest_p = ordered_periods[-1]
                longest_ma = latest_ma_values[longest_p]
                trend_strength = ((current_price - longest_ma) / longest_ma * 100) if longest_ma > 0 else 0

                return {
                    'symbol': symbol,
                    'price': current_price,
                    'trend_strength': trend_strength,
                    'ma_values': latest_ma_values
                }

        except Exception as e:
            return None

        return None

    def _get_cached_symbols(self, currency: str, use_cache: bool = True) -> list:
        """Get symbols with caching support"""
        cache_key = f"symbols_{currency}"
        current_time = time.time()

        if use_cache and cache_key in self._symbol_cache:
            cached_time, symbols = self._symbol_cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                return symbols

        # Fetch fresh symbols
        symbols = self.market_data_retriever.get_all_symbol('SPOT', currency)

        if use_cache:
            self._symbol_cache[cache_key] = (current_time, symbols)
            self._last_cache_update = current_time

        return symbols

    def _get_volume_filtered_symbols(self, currency: str, min_vol_ccy: float, use_cache: bool = True) -> list:
        """Get symbols filtered by 24h volume using market_data_retriever method"""
        cache_key = f"tickers_{currency}_{min_vol_ccy}"
        current_time = time.time()

        if use_cache and cache_key in self._symbol_cache:
            cached_time, symbols = self._symbol_cache[cache_key]
            if current_time - cached_time < self._cache_ttl:
                return symbols

        try:
            # Use the new method from market_data_retriever
            filtered_symbols = self.market_data_retriever.get_volume_filtered_symbols(
                'SPOT', currency, min_vol_ccy
            )

            if use_cache:
                self._symbol_cache[cache_key] = (current_time, filtered_symbols)
                self._last_cache_update = current_time

            return filtered_symbols

        except Exception as e:
            print(f"Error getting volume filtered symbols: {e}")
            # Fallback to basic symbol list without volume filtering
            return self._get_cached_symbols(currency, use_cache)

    def clear_cache(self):
        """Clear the symbol cache"""
        self._symbol_cache.clear()
        self._last_cache_update = 0

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
            candles = self.market_data_retriever.get_kline(symbol, bar, limit, return_dataframe=False)

            if not candles:
                return {}

            prices = [candle.c for candle in candles]

            if len(prices) < 2:
                return {}

            # Calculate price changes
            price_changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i - 1]) / prices[i - 1] * 100
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
            'bullish_coins': self.scan_bullish_coins(min_vol_ccy=100000),  # Add bullish coins with volume filter
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

    # Display bullish coins
    if report['bullish_coins']:
        print("\nBullish Coins (Multi MA Alignment):")
        print("-" * 50)
        for coin in report['bullish_coins'][:10]:  # Show top 10 bullish coins
            print(f"{coin['symbol']:12s} | "
                  f"Price: ${coin['price']:>10.2f} | "
                  f"Trend Strength: {coin['trend_strength']:>6.2f}%")
    else:
        print("\nNo bullish coins found with current criteria.")

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
