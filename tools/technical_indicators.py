"""
Technical Indicators Calculator for OKX API Data
==============================================

This module provides implementations of common technical indicators used in
financial market analysis, specifically designed to work with OKX API kline data.

The module accepts pandas DataFrame with the following OKX API kline structure:
- 'timestamp': Datetime timestamp
- 'open': Opening price
- 'high': Highest price
- 'low': Lowest price
- 'close': Closing price
- 'volume': Trading volume
- 'volume_currency': Trading volume in currency

Indicators included:
- MA (Moving Average)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- MACD (Moving Average Convergence Divergence)
- KDJ (Stochastic Oscillator)
- SAR (Parabolic Stop and Reverse)
- RSI (Relative Strength Index)
- Bollinger Bands

All functions accept pandas DataFrame (from OKX API) or Series as input and return calculated indicators.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional


def sma(data: Union[pd.Series, pd.DataFrame], period: int) -> pd.Series:
    """
    Simple Moving Average (SMA)

    Formula: SMA = (P1 + P2 + ... + Pn) / n

    Args:
        data (pd.Series or pd.DataFrame): Price data (typically closing prices)
                 For DataFrame, expects 'close' column from OKX API kline data
        period (int): Number of periods for moving average

    Returns:
        pd.Series: Simple moving average values
    """
    if isinstance(data, pd.DataFrame):
        # Handle OKX API DataFrame structure
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column from OKX API kline data")
        data = data['close']

    return data.rolling(window=period).mean()


def ema(data: Union[pd.Series, pd.DataFrame], period: int) -> pd.Series:
    """
    Exponential Moving Average (EMA)

    Formula: EMA_t = (Price_t * multiplier) + (EMA_{t-1} * (1 - multiplier))
             where multiplier = 2 / (period + 1)

    Args:
        data (pd.Series or pd.DataFrame): Price data (typically closing prices)
                 For DataFrame, expects 'close' column from OKX API kline data
        period (int): Number of periods for moving average

    Returns:
        pd.Series: Exponential moving average values
    """
    if isinstance(data, pd.DataFrame):
        # Handle OKX API DataFrame structure
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column from OKX API kline data")
        data = data['close']

    return data.ewm(span=period, adjust=False).mean()


def macd(data: Union[pd.Series, pd.DataFrame],
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD)

    Formula:
    - MACD Line = EMA(fast_period) - EMA(slow_period)
    - Signal Line = EMA(MACD Line, signal_period)
    - Histogram = MACD Line - Signal Line

    Args:
        data (pd.Series or pd.DataFrame): Price data (typically closing prices)
                 For DataFrame, expects 'close' column from OKX API kline data
        fast_period (int): Fast EMA period (default: 12)
        slow_period (int): Slow EMA period (default: 26)
        signal_period (int): Signal line EMA period (default: 9)

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    if isinstance(data, pd.DataFrame):
        # Handle OKX API DataFrame structure
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column from OKX API kline data")
        data = data['close']

    # Calculate fast and slow EMAs
    ema_fast = ema(data, fast_period)
    ema_slow = ema(data, slow_period)

    # Calculate MACD line
    macd_line = ema_fast - ema_slow

    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate histogram
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def kdj(data: pd.DataFrame,
        period: int = 9,
        k_period: int = 3,
        d_period: int = 3) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    KDJ Indicator (Stochastic Oscillator)

    Formula:
    - RSV = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    - K = SMA(RSV, k_period)
    - D = SMA(K, d_period)
    - J = 3*K - 2*D

    Args:
        data (pd.DataFrame): OKX API kline DataFrame with 'high', 'low', 'close' columns
        period (int): Period for calculating highest high and lowest low (default: 9)
        k_period (int): Period for K line smoothing (default: 3)
        d_period (int): Period for D line smoothing (default: 3)

    Returns:
        tuple: (k_line, d_line, j_line)
    """
    # Validate OKX API DataFrame structure
    required_columns = ['high', 'low', 'close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns from OKX API kline data")

    # Calculate highest high and lowest low over the period
    highest_high = data['high'].rolling(window=period).max()
    lowest_low = data['low'].rolling(window=period).min()

    # Calculate RSV (Raw Stochastic Value)
    rsv = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100

    # Calculate K line (fast stochastic)
    k_line = rsv.rolling(window=k_period).mean()

    # Calculate D line (slow stochastic)
    d_line = k_line.rolling(window=d_period).mean()

    # Calculate J line (difference between 3*K and 2*D)
    j_line = 3 * k_line - 2 * d_line

    return k_line, d_line, j_line


def sar(data: pd.DataFrame,
        acceleration: float = 0.02,
        maximum: float = 0.2) -> pd.Series:
    """
    Parabolic SAR (Stop and Reverse)

    Formula:
    - SAR_t = SAR_{t-1} + acceleration * (EP - SAR_{t-1})
    - EP (Extreme Point) = Maximum/Minimum price during the trend
    - Acceleration factor increases by acceleration step each period until maximum

    Args:
        data (pd.DataFrame): OKX API kline DataFrame with 'high', 'low' columns
        acceleration (float): Acceleration factor (default: 0.02)
        maximum (float): Maximum acceleration factor (default: 0.2)

    Returns:
        pd.Series: SAR values
    """
    # Validate OKX API DataFrame structure
    required_columns = ['high', 'low']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns from OKX API kline data")

    # Initialize SAR array
    sar_values = np.zeros(len(data))
    sar_values[0] = data['low'].iloc[0]  # Start with first low

    # Initialize trend variables
    is_up_trend = True
    ep = data['high'].iloc[0]  # Extreme point
    accel = acceleration

    for i in range(1, len(data)):
        # Calculate SAR for current period
        sar_values[i] = sar_values[i-1] + accel * (ep - sar_values[i-1])

        if is_up_trend:
            # Up trend
            # Check for new extreme point
            if data['high'].iloc[i] > ep:
                ep = data['high'].iloc[i]
                accel = min(accel + acceleration, maximum)

            # Check for trend reversal
            if data['low'].iloc[i] < sar_values[i]:
                is_up_trend = False
                sar_values[i] = ep
                ep = data['low'].iloc[i]
                accel = acceleration
        else:
            # Down trend
            # Check for new extreme point
            if data['low'].iloc[i] < ep:
                ep = data['low'].iloc[i]
                accel = min(accel + acceleration, maximum)

            # Check for trend reversal
            if data['high'].iloc[i] > sar_values[i]:
                is_up_trend = True
                sar_values[i] = ep
                ep = data['high'].iloc[i]
                accel = acceleration

    return pd.Series(sar_values, index=data.index)


def rsi(data: Union[pd.Series, pd.DataFrame], period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI)

    Formula:
    - RS = Average Gain / Average Loss
    - RSI = 100 - (100 / (1 + RS))

    Args:
        data (pd.Series or pd.DataFrame): Price data (typically closing prices)
                 For DataFrame, expects 'close' column from OKX API kline data
        period (int): Number of periods for RSI calculation (default: 14)

    Returns:
        pd.Series: RSI values (0-100)
    """
    if isinstance(data, pd.DataFrame):
        # Handle OKX API DataFrame structure
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column from OKX API kline data")
        data = data['close']

    # Calculate price changes
    delta = data.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def bollinger_bands(data: Union[pd.Series, pd.DataFrame],
                   period: int = 20,
                   std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Bollinger Bands

    Formula:
    - Middle Band = SMA(period)
    - Upper Band = Middle Band + (std_dev * STD(Middle Band))
    - Lower Band = Middle Band - (std_dev * STD(Middle Band))

    Args:
        data (pd.Series or pd.DataFrame): Price data (typically closing prices)
                 For DataFrame, expects 'close' column from OKX API kline data
        period (int): Number of periods for moving average (default: 20)
        std_dev (int): Number of standard deviations (default: 2)

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    if isinstance(data, pd.DataFrame):
        # Handle OKX API DataFrame structure
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column from OKX API kline data")
        data = data['close']

    # Calculate middle band (SMA)
    middle_band = sma(data, period)

    # Calculate standard deviation
    std_deviation = data.rolling(window=period).std()

    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * std_deviation)
    lower_band = middle_band - (std_dev * std_deviation)

    return upper_band, middle_band, lower_band


# Convenience aliases
ma = sma  # Default MA is SMA


def __getattr__(name):
    """Provide backward compatibility for older function names"""
    if name == 'simple_moving_average':
        return sma
    elif name == 'exponential_moving_average':
        return ema
    elif name == 'moving_average_convergence_divergence':
        return macd
    elif name == 'stochastic_oscillator':
        return kdj
    elif name == 'stop_and_reverse':
        return sar
    elif name == 'relative_strength_index':
        return rsi
    elif name == 'bollinger_bands_indicator':
        return bollinger_bands
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")