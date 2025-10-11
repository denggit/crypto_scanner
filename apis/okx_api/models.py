from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Ticker:
    """Market ticker data"""
    instId: str          # Instrument ID
    last: float          # Last traded price
    open24h: float       # Open price in last 24 hours
    high24h: float       # Highest price in last 24 hours
    low24h: float        # Lowest price in last 24 hours
    vol24h: float        # Volume in last 24 hours (in contracts)
    volCcy24h: float     # Volume in last 24 hours (in currency)
    ts: int              # Timestamp
    bidPx: float         # Best bid price
    askPx: float         # Best ask price
    bidSz: float         # Best bid size
    askSz: float         # Best ask size

@dataclass
class OrderBook:
    """Order book data"""
    instId: str          # Instrument ID
    bids: List[List[str]] # Bid levels [price, size, number of orders]
    asks: List[List[str]] # Ask levels [price, size, number of orders]
    ts: int              # Timestamp

@dataclass
class Candle:
    """K-line/Candlestick data"""
    ts: int              # Timestamp
    o: float             # Open price
    h: float             # Highest price
    l: float             # Lowest price
    c: float             # Close price
    vol: float           # Volume (in contracts)
    volCcy: float        # Volume (in currency)

@dataclass
class Balance:
    """Account balance data"""
    ccy: str             # Currency
    bal: float           # Balance
    frozenBal: float     # Frozen balance
    availBal: float      # Available balance

@dataclass
class Order:
    """Order data"""
    instId: str          # Instrument ID
    ordId: str           # Order ID
    clOrdId: str         # Client order ID
    px: float            # Price
    sz: float            # Quantity
    ordType: str         # Order type
    side: str            # Order side
    state: str           # Order state
    accFillSz: float     # Accumulated fill quantity
    avgPx: float         # Average filled price
    fee: float           # Fee
    ts: int              # Timestamp

@dataclass
class MarketData:
    """Complete market data for an instrument"""
    ticker: Ticker
    order_book: OrderBook
    candles: List[Candle]
    timestamp: int

@dataclass
class AccountData:
    """Complete account data"""
    balances: List[Balance]
    orders: List[Order]
    timestamp: int


@dataclass
class Instrument:
    """Instrument information data"""
    instId: str                    # Instrument ID
    uly: str = ""                   # Underlying asset
    category: str = ""             # Instrument category
    baseCcy: str = ""              # Base currency
    quoteCcy: str = ""             # Quote currency
    settleCcy: str = ""            # Settlement currency
    ctVal: float = 0.0             # Contract value
    ctMult: float = 0.0            # Contract multiplier
    ctValCcy: str = ""             # Contract value currency
    optType: str = ""              # Option type (C: call, P: put)
    stk: float = 0.0               # Strike price
    listTime: int = 0              # Listing time
    expTime: int = 0               # Expiry time
    lever: float = 0.0             # Leverage
    tickSz: float = 0.0            # Tick size
    lotSz: float = 0.0             # Lot size
    minSz: float = 0.0             # Minimum order size
    ctType: str = ""               # Contract type
    alias: str = ""                # Alias
    state: str = ""                # Instrument state
    maxLmtSz: float = 0.0          # Maximum limit order size
    maxMktSz: float = 0.0          # Maximum market order size
    maxTwapSz: float = 0.0         # Maximum TWAP order size
    maxIcebergSz: float = 0.0      # Maximum iceberg order size
    maxTriggerSz: float = 0.0      # Maximum trigger order size
    maxStopSz: float = 0.0         # Maximum stop order size