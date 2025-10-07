# OKX API Modules

This package provides a comprehensive Python wrapper for the OKX cryptocurrency exchange API, enabling you to retrieve market data and execute trades programmatically.

## Features

- **Market Data Retrieval**: Get real-time ticker prices, order books, and candlestick data for all cryptocurrencies
- **Trading Functions**: Place market and limit orders, manage positions
- **Account Management**: Retrieve account balances and order history
- **Error Handling**: Comprehensive error handling with descriptive messages
- **Rate Limiting**: Built-in rate limiting to comply with OKX API requirements
- **Type Safety**: Data classes for all API responses for better type checking

## Installation

Ensure you have the required dependencies installed:

```bash
pip install requests
```

## Setup

1. Sign up for an account at [OKX](https://www.okx.com/)
2. Generate API credentials in your account settings
3. Set the following environment variables:
   - `OKX_API_KEY`
   - `OKX_API_SECRET`
   - `OKX_PASSPHRASE`

## Modules

### 1. Client (`client.py`)

The main API client that handles authentication, requests, and responses.

```python
from okx_api.client import OKXClient

# Initialize client
client = OKXClient(
    api_key="your_api_key",
    api_secret="your_api_secret",
    passphrase="your_passphrase"
)
```

### 2. Market Data (`market_data.py`)

Retrieve market data including tickers, order books, and candlestick data.

```python
from okx_api.market_data import MarketDataRetriever

# Initialize market data retriever
market_data = MarketDataRetriever(client)

# Get all spot tickers
tickers = market_data.get_all_tickers('SPOT')

# Get specific ticker
btc_ticker = market_data.get_ticker_by_symbol('BTC-USDT')

# Get order book
order_book = market_data.get_order_book('BTC-USDT', sz=10)

# Get kline data
klines = market_data.get_kline('BTC-USDT', bar='1H', limit=100)
```

### 3. Trading (`trader.py`)

Execute trades and manage orders.

```python
from okx_api.trader import Trader

# Initialize trader
trader = Trader(client)

# Place market order
order = trader.place_market_order(
    instId='BTC-USDT',
    side='buy',
    sz='0.001'
)

# Place limit order
order = trader.place_limit_order(
    instId='BTC-USDT',
    side='sell',
    sz='0.001',
    px='50000'
)

# Get pending orders
orders = trader.get_pending_orders()

# Get account balance
balance = trader.get_account_balance()
```

### 4. Data Models (`models.py`)

Data classes for all API responses:

- `Ticker`: Market ticker data
- `OrderBook`: Order book data
- `Candle`: K-line/candlestick data
- `Balance`: Account balance data
- `Order`: Order data
- `MarketData`: Complete market data for an instrument
- `AccountData`: Complete account data

## Example Usage

See `example.py` for a complete example of how to use all modules.

```python
import os
from okx_api.client import OKXClient
from okx_api.market_data import MarketDataRetriever
from okx_api.trader import Trader

# Initialize
client = OKXClient(
    api_key=os.getenv('OKX_API_KEY'),
    api_secret=os.getenv('OKX_API_SECRET'),
    passphrase=os.getenv('OKX_PASSPHRASE')
)

market_data = MarketDataRetriever(client)
trader = Trader(client)

# Get market data
tickers = market_data.get_all_tickers('SPOT')
btc_data = market_data.get_market_data('BTC-USDT')

# Place a trade
order = trader.place_market_order('BTC-USDT', 'buy', '0.001')
```

## Rate Limiting

The client implements automatic rate limiting to comply with OKX API requirements (100ms minimum between requests).

## Error Handling

All API errors are raised as `OKXAPIException` with descriptive error messages.

## License

MIT