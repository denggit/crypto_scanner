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
from tools.technical_indicators import sma, ema


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

    def scan_ma_alignment(self, currency: str = 'USDT', ma_periods: list = None,
                         bar: str = '5m', min_vol_ccy: float = 1000000, use_parallel: bool = True,
                         use_cache: bool = True, symbols: list = None) -> list:
        """
        Scan for cryptocurrencies with moving average alignment (golden cross/multi MA alignment)
        Optimized for performance with parallel processing and caching

        Args:
            currency: Currency to filter by (default: USDT)
            ma_periods: List of MA periods to check alignment (default: [5, 20, 60, 120, 200])
            bar: Time interval for kline data (default: 5m)
            min_vol_ccy: Minimum 24h volume in currency to include (default: 1,000,000)
            use_parallel: Use parallel processing for faster scanning (default: True)
            use_cache: Use caching for symbol data (default: True)
            symbols: Optional list of symbols to scan. If provided, skips volume filtering.

        Returns:
            List of coins with MA alignment
        """
        if ma_periods is None:
            ma_periods = [5, 20, 60, 120, 200]

        # 保持有限的K线数量以加快网络请求；仅用于计算末端均线
        limit = max(ma_periods) + 1

        try:
            # Use provided symbols if available, otherwise get volume filtered symbols
            if symbols is not None and len(symbols) > 0:
                print(f"Scanning {len(symbols)} provided symbols")
            else:
                # Get symbols filtered by volume
                symbols = self._get_volume_filtered_symbols(currency, min_vol_ccy, use_cache)

                if not symbols:
                    print(f"No symbols found with 24h volume >= {min_vol_ccy:,.0f} {currency}")
                    return []

                print(f"Scanning {len(symbols)} symbols with 24h volume >= {min_vol_ccy:,.0f} {currency}")

            if use_parallel and len(symbols) > 1:
                return self._scan_ma_alignment_parallel(symbols, ma_periods, bar, limit)
            else:
                return self._scan_ma_alignment_sequential(symbols, ma_periods, bar, limit)

        except Exception as e:
            print(f"Error scanning MA alignment: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return []

    def _scan_ma_alignment_sequential(self, symbols: list, ma_periods: list, bar: str, limit: int) -> list:
        """Sequential scanning implementation"""
        ma_alignment_coins = []

        for symbol in symbols:
            try:
                result = self._analyze_symbol_ma_alignment(symbol, ma_periods, bar, limit)
                if result:
                    ma_alignment_coins.append(result)
            except Exception:
                continue

        # Sort by trend strength
        ma_alignment_coins.sort(key=lambda x: x['trend_strength'], reverse=True)
        return ma_alignment_coins

    def _scan_ma_alignment_parallel(self, symbols: list, ma_periods: list, bar: str, limit: int) -> list:
        """Parallel scanning implementation using ThreadPoolExecutor"""
        ma_alignment_coins = []

        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._analyze_symbol_ma_alignment, symbol, ma_periods, bar, limit): symbol
                for symbol in symbols
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        ma_alignment_coins.append(result)
                except Exception:
                    continue

        # Sort by trend strength
        ma_alignment_coins.sort(key=lambda x: x['trend_strength'], reverse=True)
        return ma_alignment_coins

    def _analyze_symbol_ma_alignment(self, symbol: str, ma_periods: list, bar: str, limit: int) -> dict:
        """Analyze a single symbol for MA alignment"""
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

            # 使用 technical_indicators 模块计算均线值
            latest_ma_values = {}
            current_price = float(closes.iloc[-1])

            for p in available_periods:
                # 使用 SMA 计算均线
                ma_series = sma(closes, p)
                latest_ma_values[p] = float(ma_series.iloc[-1])

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

    def scan_ma_convergence_breakout(self, currency: str = 'USDT', ma_periods: list = None,
                                    bar: str = '15m', min_vol_ccy: float = 1000000,
                                    convergence_threshold: float = 0.02,
                                    breakout_strength: float = 0.01,
                                    use_parallel: bool = True,
                                    use_cache: bool = True) -> list:
        """
        扫描均线粘合+发散形态（趋势启动点检测）

        检测均线之间的距离是否非常接近（粘合）并开始向上发散
        这种形态往往是趋势启动点

        Args:
            currency: 交易对货币 (default: USDT)
            ma_periods: 均线周期列表 (default: [5, 20, 60])
            bar: K线时间间隔 (default: 15m)
            min_vol_ccy: 最小24小时交易量 (default: 1,000,000)
            convergence_threshold: 均线粘合阈值 (default: 2%)
            breakout_strength: 突破强度阈值 (default: 1%)
            use_parallel: 是否使用并行处理 (default: True)
            use_cache: 是否使用缓存 (default: True)

        Returns:
            均线粘合+发散形态的币种列表
        """
        if ma_periods is None:
            ma_periods = [5, 20, 60]

        # 需要更多K线数据来计算均线斜率
        limit = max(ma_periods) + 20  # 额外数据用于计算斜率

        try:
            # 获取交易量过滤的币种
            symbols = self._get_volume_filtered_symbols(currency, min_vol_ccy, use_cache)

            if not symbols:
                print(f"No symbols found with 24h volume >= {min_vol_ccy:,.0f} {currency}")
                return []

            print(f"Scanning {len(symbols)} symbols for MA convergence + breakout patterns")

            if use_parallel and len(symbols) > 1:
                return self._scan_ma_convergence_breakout_parallel(
                    symbols, ma_periods, bar, limit, convergence_threshold, breakout_strength
                )
            else:
                return self._scan_ma_convergence_breakout_sequential(
                    symbols, ma_periods, bar, limit, convergence_threshold, breakout_strength
                )

        except Exception as e:
            print(f"Error scanning MA convergence breakout: {e}")
            print("This may be due to network connectivity issues or API restrictions.")
            print("Please check your internet connection and firewall settings.")
            return []

    def _scan_ma_convergence_breakout_sequential(self, symbols: list, ma_periods: list,
                                                bar: str, limit: int,
                                                convergence_threshold: float,
                                                breakout_strength: float) -> list:
        """顺序扫描均线粘合+发散形态"""
        convergence_breakout_coins = []

        for symbol in symbols:
            try:
                result = self._analyze_symbol_ma_convergence_breakout(
                    symbol, ma_periods, bar, limit, convergence_threshold, breakout_strength
                )
                if result:
                    convergence_breakout_coins.append(result)
            except Exception:
                continue

        # 按突破强度排序
        convergence_breakout_coins.sort(key=lambda x: x['breakout_strength'], reverse=True)
        return convergence_breakout_coins

    def _scan_ma_convergence_breakout_parallel(self, symbols: list, ma_periods: list,
                                              bar: str, limit: int,
                                              convergence_threshold: float,
                                              breakout_strength: float) -> list:
        """并行扫描均线粘合+发散形态"""
        convergence_breakout_coins = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
            future_to_symbol = {
                executor.submit(
                    self._analyze_symbol_ma_convergence_breakout,
                    symbol, ma_periods, bar, limit, convergence_threshold, breakout_strength
                ): symbol
                for symbol in symbols
            }

            for future in concurrent.futures.as_completed(future_to_symbol):
                try:
                    result = future.result()
                    if result:
                        convergence_breakout_coins.append(result)
                except Exception:
                    continue

        # 按突破强度排序
        convergence_breakout_coins.sort(key=lambda x: x['breakout_strength'], reverse=True)
        return convergence_breakout_coins

    def _analyze_symbol_ma_convergence_breakout(self, symbol: str, ma_periods: list,
                                               bar: str, limit: int,
                                               convergence_threshold: float,
                                               breakout_strength: float) -> dict:
        """
        分析单个币种的均线粘合+发散形态

        判断逻辑：
        1. 均线粘合：所有均线之间的最大距离 < convergence_threshold
        2. 向上发散：MA5 突破 MA20，且 MA20 斜率为正
        """
        try:
            # 获取K线数据
            df = self.market_data_retriever.get_kline(symbol, bar, limit)

            if df is None or len(df) < max(ma_periods) + 5:  # 需要足够数据计算斜率和粘合
                return None

            closes = df['c'] if 'c' in df.columns else df['close']
            if len(closes) < max(ma_periods) + 5:
                return None

            # 使用 technical_indicators 模块计算均线值
            current_ma_values = {}
            for p in ma_periods:
                if len(closes) >= p:
                    # 使用 SMA 计算均线
                    ma_series = sma(closes, p)
                    current_ma_values[p] = float(ma_series.iloc[-1])

            if len(current_ma_values) < len(ma_periods):
                return None

            # 计算前一个周期的均线值（用于计算斜率）
            prev_ma_values = {}
            for p in ma_periods:
                if len(closes) >= p + 5:  # 5个周期前的数据
                    ma_series = sma(closes, p)
                    prev_ma_values[p] = float(ma_series.iloc[-6])  # 5个周期前的值

            if len(prev_ma_values) < len(ma_periods):
                return None

            # 检查均线粘合：所有均线之间的最大距离是否小于阈值
            ma_values_list = list(current_ma_values.values())
            max_ma = max(ma_values_list)
            min_ma = min(ma_values_list)

            convergence_ratio = (max_ma - min_ma) / min_ma if min_ma > 0 else float('inf')

            # 检查向上发散条件
            # 1. MA5 > MA20 (突破)
            ma5_break_ma20 = current_ma_values[5] > current_ma_values[20]

            # 2. MA20 斜率为正
            ma20_slope = (current_ma_values[20] - prev_ma_values[20]) / prev_ma_values[20] if prev_ma_values[20] > 0 else 0

            # 3. MA5 突破强度
            ma5_breakout_strength = (current_ma_values[5] - current_ma_values[20]) / current_ma_values[20] if current_ma_values[20] > 0 else 0

            # 判断是否满足粘合+发散条件
            if (convergence_ratio <= convergence_threshold and
                ma5_break_ma20 and
                ma20_slope > 0 and
                ma5_breakout_strength >= breakout_strength):

                return {
                    'symbol': symbol,
                    'current_price': float(closes.iloc[-1]),
                    'convergence_ratio': convergence_ratio * 100,  # 转换为百分比
                    'ma5_breakout_strength': ma5_breakout_strength * 100,
                    'ma20_slope': ma20_slope * 100,
                    'breakout_strength': ma5_breakout_strength * 100,  # 综合突破强度
                    'ma_values': current_ma_values
                }

        except Exception as e:
            return None

        return None

    def generate_market_report(self) -> dict:
        """
        Generate a comprehensive market report

        Returns:
            Market report data
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'top_coins': self.scan_top_coins(10),
            'ma_alignment_coins': self.scan_ma_alignment(min_vol_ccy=100000),  # Add MA alignment coins with volume filter
            'ma_convergence_breakout_coins': self.scan_ma_convergence_breakout(min_vol_ccy=100000),  # Add MA convergence breakout coins
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

    # Display MA alignment coins
    if report['ma_alignment_coins']:
        print("\nBullish Coins (Multi MA Alignment):")
        print("-" * 50)
        for coin in report['ma_alignment_coins'][:10]:  # Show top 10 bullish coins
            print(f"{coin['symbol']:12s} | "
                  f"Price: ${coin['price']:>10.2f} | "
                  f"Trend Strength: {coin['trend_strength']:>6.2f}%")
    else:
        print("\nNo bullish coins found with current criteria.")

    # Display MA convergence breakout coins
    if report['ma_convergence_breakout_coins']:
        print("\nMA Convergence + Breakout Coins (Trend Start):")
        print("-" * 50)
        for coin in report['ma_convergence_breakout_coins'][:10]:  # Show top 10 convergence breakout coins
            print(f"{coin['symbol']:12s} | "
                  f"Price: ${coin['current_price']:>10.2f} | "
                  f"Convergence: {coin['convergence_ratio']:>5.2f}% | "
                  f"Breakout: {coin['breakout_strength']:>5.2f}%")
    else:
        print("\nNo MA convergence + breakout coins found with current criteria.")

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
