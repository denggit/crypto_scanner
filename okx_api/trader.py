from typing import Optional
from .client import OKXClient
from .models import Order

class Trader:
    """
    Trading functionality for OKX
    """

    def __init__(self, client: OKXClient):
        """
        Initialize Trader

        Args:
            client: OKXClient instance
        """
        self.client = client

    def place_market_order(self, instId: str, side: str, sz: str, tdMode: str = 'cash') -> Optional[Order]:
        """
        Place a market order

        Args:
            instId: Instrument ID (e.g., BTC-USDT)
            side: Order side ('buy' or 'sell')
            sz: Quantity
            tdMode: Trade mode ('cash', 'cross', 'isolated')

        Returns:
            Order object if successful, None otherwise
        """
        response = self.client.place_order(
            instId=instId,
            tdMode=tdMode,
            side=side,
            ordType='market',
            sz=sz
        )

        if response.get('code') == '0' and 'data' in response:
            order_data = response['data'][0]
            return Order(
                instId=order_data.get('instId', ''),
                ordId=order_data.get('ordId', ''),
                clOrdId=order_data.get('clOrdId', ''),
                px=float(order_data.get('px', 0)) if order_data.get('px') else 0,
                sz=float(order_data.get('sz', 0)),
                ordType=order_data.get('ordType', ''),
                side=order_data.get('side', ''),
                state=order_data.get('state', ''),
                accFillSz=float(order_data.get('accFillSz', 0)),
                avgPx=float(order_data.get('avgPx', 0)) if order_data.get('avgPx') else 0,
                fee=float(order_data.get('fee', 0)) if order_data.get('fee') else 0,
                ts=int(order_data.get('ts', 0)) if order_data.get('ts') else 0
            )
        return None

    def place_limit_order(self, instId: str, side: str, sz: str, px: str,
                         tdMode: str = 'cash', reduceOnly: bool = False) -> Optional[Order]:
        """
        Place a limit order

        Args:
            instId: Instrument ID
            side: Order side ('buy' or 'sell')
            sz: Quantity
            px: Price
            tdMode: Trade mode
            reduceOnly: Reduce only flag

        Returns:
            Order object if successful, None otherwise
        """
        response = self.client.place_order(
            instId=instId,
            tdMode=tdMode,
            side=side,
            ordType='limit',
            sz=sz,
            px=px,
            reduceOnly=reduceOnly
        )

        if response.get('code') == '0' and 'data' in response:
            order_data = response['data'][0]
            return Order(
                instId=order_data.get('instId', ''),
                ordId=order_data.get('ordId', ''),
                clOrdId=order_data.get('clOrdId', ''),
                px=float(order_data.get('px', 0)),
                sz=float(order_data.get('sz', 0)),
                ordType=order_data.get('ordType', ''),
                side=order_data.get('side', ''),
                state=order_data.get('state', ''),
                accFillSz=float(order_data.get('accFillSz', 0)),
                avgPx=float(order_data.get('avgPx', 0)) if order_data.get('avgPx') else 0,
                fee=float(order_data.get('fee', 0)) if order_data.get('fee') else 0,
                ts=int(order_data.get('ts', 0)) if order_data.get('ts') else 0
            )
        return None

    def get_pending_orders(self, instId: str = None) -> list:
        """
        Get pending orders

        Args:
            instId: Instrument ID (optional)

        Returns:
            List of pending orders
        """
        response = self.client.get_orders(instId=instId)

        orders = []
        if response.get('code') == '0' and 'data' in response:
            for order_data in response['data']:
                order = Order(
                    instId=order_data.get('instId', ''),
                    ordId=order_data.get('ordId', ''),
                    clOrdId=order_data.get('clOrdId', ''),
                    px=float(order_data.get('px', 0)) if order_data.get('px') else 0,
                    sz=float(order_data.get('sz', 0)),
                    ordType=order_data.get('ordType', ''),
                    side=order_data.get('side', ''),
                    state=order_data.get('state', ''),
                    accFillSz=float(order_data.get('accFillSz', 0)),
                    avgPx=float(order_data.get('avgPx', 0)) if order_data.get('avgPx') else 0,
                    fee=float(order_data.get('fee', 0)) if order_data.get('fee') else 0,
                    ts=int(order_data.get('ts', 0)) if order_data.get('ts') else 0
                )
                orders.append(order)

        return orders

    def cancel_order(self, instId: str, ordId: str) -> bool:
        """
        Cancel an order

        Args:
            instId: Instrument ID
            ordId: Order ID

        Returns:
            True if successful, False otherwise
        """
        endpoint = "/api/v5/trade/cancel-order"
        params = {
            'instId': instId,
            'ordId': ordId
        }

        response = self.client._make_request('POST', endpoint, params, signed=True)
        return response.get('code') == '0'

    def get_account_balance(self):
        """
        Get account balance

        Returns:
            Account balance data
        """
        return self.client.get_account_balance()