#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File       : trader.py
@Description: Strategy 2 specific trader implementation
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sdk.base_trader import BaseTrader


class Strategy2Trader(BaseTrader):
    """High Frequency Strategy Trader"""

    def __init__(self, client, trade_amount: float = 10.0, trade_mode: int = 3, leverage: int = 3):
        """
        Initialize Strategy 2 Trader
        
        Args:
            client: OKX client instance
            trade_amount: USDT amount for each trade
            trade_mode: Trading mode (1=现货, 2=全仓杠杆, 3=逐仓杠杆)
            leverage: Leverage multiplier (default: 3x)
        """
        super().__init__(client, trade_amount, trade_mode, leverage)
