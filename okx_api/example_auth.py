"""
Example usage of OKX API with authentication
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from okx_api.trader import Trader

def main():
    # Initialize OKX client with environment variables
    # Note: OK-ACCESS-SIGN is generated dynamically, not stored
    api_key = os.getenv('OK-ACCESS-KEY')
    api_secret = os.getenv('OK-ACCESS-SECRET')  # This should be in your .env file
    passphrase = os.getenv('OK-ACCESS-PASSPHRASE')

    if not api_key or not api_secret or not passphrase:
        print("Missing API credentials. Please set the following environment variables:")
        print("- OK-ACCESS-KEY")
        print("- OK-ACCESS-SECRET")
        print("- OK-ACCESS-PASSPHRASE")
        return

    # Initialize client
    client = OKXClient(api_key=api_key, api_secret=api_secret, passphrase=passphrase)

    # Initialize market data retriever
    market_data_retriever = MarketDataRetriever(client)

    # Initialize trader
    trader = Trader(client)

    print("OKX API Authenticated Example")
    print("=" * 50)

    # Example 1: Get account balance (requires authentication)
    print("\n1. Getting account balance...")
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

    # Example 2: Get pending orders (requires authentication)
    print("\n2. Getting pending orders...")
    try:
        orders = trader.get_pending_orders()
        print(f"Found {len(orders)} pending orders")
        for order in orders[:3]:  # Show first 3 orders
            print(f"  {order.instId}: {order.side} {order.sz} @ {order.px}")
    except Exception as e:
        print(f"Error getting pending orders: {e}")

    print("\n" + "=" * 50)
    print("Authenticated example completed.")

if __name__ == "__main__":
    main()