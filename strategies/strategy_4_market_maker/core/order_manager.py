#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 11/3/25 8:31 PM
@File       : order_manager.py
@Description:
"""
from apis.okx_api.client import get_okx_client


class OrderMgmt:
    def __init__(self):
        self.client = get_okx_client()
    
    def place_limit_order(self, instId: str, side: str, size: float, price: float) -> str:
        """
        Place a limit order

        :param symbol: 交易对 (如 "BTC-USDT" 或 "BTC-USDT-SWAP")
        :param side: "buy" 或 "sell"
        :param price: 限价价格
        :param size: 数量（以币为单位）

        Returns:
            Order ID
        """
        pass

    def cancel_order(self, orderId: str) -> bool:
        """撤销订单。

        :param symbol: 交易对
        :param order_id: 交易所返回的订单ID
        :param client_id: 客户端生成的ID（可选）
        :return: True=成功, False=失败"""
        pass

    def get_order_status(self, instId: str, orderId: str=None) -> dict:
        """
        Get order status

        :param symbol: 交易对
        :param order_id: 交易所订单ID
        :param client_id: 客户端订单ID
        
        Returns:
            Order status
            {
                "status": "live" | "filled" | "canceled" | "partially_filled",
                "filled_size": "0.002",
                "avg_price": "10005.0"
            }
        """
        pass

    def get_open_orders(self, instId: str=None) -> list:
        """
        Get open orders
        
        Returns:
            List of open orders
            [
                {
                    "instId": "BTC-USDT-SWAP",
                    "orderId": "1234567890",
                    "side": "buy",
                    "size": "0.001",
                    "price": "10000.0"
                }
            ]
        """
        pass

    def get_position_info(self, instId: str) -> dict:
        """
        Get position info
        
        Returns:
            Position info
            {"instId": "BTC-USDT-SWAP", "pos_side": "long"|"short", "size": float, "avg_price": float, "unrealized_pnl": float}
        """
        pass

    def get_account_balance(self) -> dict:
        """
        Get account balance
        
        Returns:
            Account balance
        """
        pass
        
