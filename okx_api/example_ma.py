"""
Example usage of the moving average functionality in OKX API
"""

import os
import sys
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever

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

    print("OKX API Moving Average Example")
    print("=" * 50)

    # Example: Get kline data with moving averages
    print("\n1. Getting BTC-USDT kline data with moving averages...")
    try:
        start_time = time.time()
        # Get kline data with default MA periods (5, 10, 20)
        df = market_data_retriever.get_kline_with_ma('BTC-USDT', bar='1H', limit=50)
        end_time = time.time()
        print(f"Data with default MA periods (5, 10, 20) - Retrieved in {end_time - start_time:.4f} seconds:")
        print(df.tail(10))  # Show last 10 rows

        # Get kline data with custom MA periods
        print("\n2. Getting BTC-USDT kline data with custom MA periods...")
        start_time = time.time()
        custom_ma_periods = [7, 25, 50, 100]
        df_custom = market_data_retriever.get_kline_with_ma(
            'BTC-USDT',
            bar='1H',
            limit=110,  # Need at least 110 data points for 100-period MA
            ma_periods=custom_ma_periods
        )
        end_time = time.time()
        print(f"Data with custom MA periods {custom_ma_periods} - Retrieved in {end_time - start_time:.4f} seconds:")
        print(df_custom.tail(10))  # Show last 10 rows

        # Show column names
        print(f"\nColumns in the DataFrame: {list(df_custom.columns)}")

        # Show DataFrame info
        print(f"\nDataFrame shape: {df_custom.shape}")

    except Exception as e:
        print(f"Error getting kline data with moving averages: {e}")

    print("\n" + "=" * 50)
    print("Moving average example completed.")

if __name__ == "__main__":
    main()