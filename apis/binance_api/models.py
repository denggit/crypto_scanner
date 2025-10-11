#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : models.py
@Description: Binance API data models
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    """Order data model"""
    symbol: str = ""
    orderId: str = ""
    clientOrderId: str = ""
    price: float = 0.0
    origQty: float = 0.0
    executedQty: float = 0.0
    cummulativeQuoteQty: float = 0.0
    status: str = ""
    timeInForce: str = ""
    type: str = ""
    side: str = ""
    stopPrice: float = 0.0
    icebergQty: float = 0.0
    time: int = 0
    updateTime: int = 0
    isWorking: bool = False
    origQuoteOrderQty: float = 0.0


@dataclass
class Position:
    """Position data model"""
    symbol: str = ""
    positionAmt: float = 0.0
    entryPrice: float = 0.0
    markPrice: float = 0.0
    unRealizedProfit: float = 0.0
    liquidationPrice: float = 0.0
    leverage: int = 1
    maxNotionalValue: float = 0.0
    marginType: str = ""
    isolatedMargin: float = 0.0
    isAutoAddMargin: bool = False
    positionSide: str = ""
    notional: float = 0.0
    isolatedWallet: float = 0.0
    updateTime: int = 0


@dataclass
class Ticker:
    """Ticker data model"""
    symbol: str = ""
    priceChange: float = 0.0
    priceChangePercent: float = 0.0
    weightedAvgPrice: float = 0.0
    prevClosePrice: float = 0.0
    lastPrice: float = 0.0
    lastQty: float = 0.0
    bidPrice: float = 0.0
    bidQty: float = 0.0
    askPrice: float = 0.0
    askQty: float = 0.0
    openPrice: float = 0.0
    highPrice: float = 0.0
    lowPrice: float = 0.0
    volume: float = 0.0
    quoteVolume: float = 0.0
    openTime: int = 0
    closeTime: int = 0
    firstId: int = 0
    lastId: int = 0
    count: int = 0


@dataclass
class Kline:
    """Kline data model"""
    open_time: int = 0
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    close_time: int = 0
    quote_asset_volume: float = 0.0
    number_of_trades: int = 0
    taker_buy_base_asset_volume: float = 0.0
    taker_buy_quote_asset_volume: float = 0.0
    ignore: int = 0


@dataclass
class AccountBalance:
    """Account balance data model"""
    asset: str = ""
    free: float = 0.0
    locked: float = 0.0


@dataclass
class ExchangeInfo:
    """Exchange information data model"""
    symbol: str = ""
    status: str = ""
    baseAsset: str = ""
    baseAssetPrecision: int = 0
    quoteAsset: str = ""
    quotePrecision: int = 0
    quoteAssetPrecision: int = 0
    baseCommissionPrecision: int = 0
    quoteCommissionPrecision: int = 0
    orderTypes: list = None
    icebergAllowed: bool = False
    ocoAllowed: bool = False
    quoteOrderQtyMarketAllowed: bool = False
    allowTrailingStop: bool = False
    cancelReplaceAllowed: bool = False
    isSpotTradingAllowed: bool = False
    isMarginTradingAllowed: bool = False
    filters: list = None
    permissions: list = None
    defaultSelfTradePreventionMode: str = ""
    allowedSelfTradePreventionModes: list = None

    def __post_init__(self):
        if self.orderTypes is None:
            self.orderTypes = []
        if self.filters is None:
            self.filters = []
        if self.permissions is None:
            self.permissions = []
        if self.allowedSelfTradePreventionModes is None:
            self.allowedSelfTradePreventionModes = []