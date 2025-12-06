#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 11/3/25 8:31 PM
@File       : order_manager.py
@Description:
"""
from apis.okx_api.client import get_okx_client, OKXAPIException
from apis.okx_api.trader import Trader


class OrderMgmt:
    def __init__(self):
        self.client = get_okx_client()
        self.trader = Trader(self.client)
        self._pos_mode_cache = None  # 缓存账户持仓模式：'long_short_mode' 或 'net_mode'

    def _get_position_mode(self) -> str:
        """
        查询并缓存账户持仓模式。

        Returns:
            'long_short_mode' 或 'net_mode'（未知时回退为 'net_mode' 以减少报错概率）
        """
        if self._pos_mode_cache:
            return self._pos_mode_cache
        try:
            resp = self.client._make_request('GET', '/api/v5/account/config', {}, signed=True)
            if resp.get('code') == '0' and resp.get('data'):
                cfg = resp['data'][0]
                pos_mode = cfg.get('posMode') or 'net_mode'
                self._pos_mode_cache = pos_mode
                return pos_mode
        except Exception:
            pass
        self._pos_mode_cache = 'net_mode'
        return self._pos_mode_cache
    
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
        try:
            is_swap = instId.endswith('-SWAP') or instId.endswith('-FUTURES') or instId.endswith('-USD-SWAP') or instId.endswith('-USDT-SWAP')
            td_mode = 'isolated' if is_swap else 'cash'

            # OKX 衍生品在净持仓模式下不能传 posSide；仅双向持仓下传入正确方向
            params = {
                'instId': instId,
                'tdMode': td_mode,
                'side': side,
                'ordType': 'limit',
                'sz': str(size),
                'px': str(price)
            }

            if not is_swap:
                # 仅现货使用 tgtCcy 参数（衍生品忽略该参数以避免报错）
                params['tgtCcy'] = 'quote_ccy'
            else:
                pos_mode = self._get_position_mode()
                if pos_mode == 'long_short_mode':
                    params['posSide'] = 'long' if side == 'buy' else 'short'

            resp = self.client.place_order(**params)
            if resp.get('code') == '0' and resp.get('data'):
                return resp['data'][0].get('ordId', '')
            return ""
        except OKXAPIException:
            return ""

    def cancel_order(self, orderId: str) -> bool:
        """撤销订单。

        :param symbol: 交易对
        :param order_id: 交易所返回的订单ID
        :param client_id: 客户端生成的ID（可选）
        :return: True=成功, False=失败"""
        # 由于接口需要 instId，这里先从未完成订单中定位对应 instId
        try:
            pending = self.client.get_orders()
            target_inst = None
            if pending.get('code') == '0' and 'data' in pending:
                for od in pending['data']:
                    if od.get('ordId') == orderId:
                        target_inst = od.get('instId')
                        break
            if not target_inst:
                # 未在挂单找到，直接尝试撤单（若不提供 instId 会失败），本地返回 False
                return False

            endpoint = "/api/v5/trade/cancel-order"
            params = {"instId": target_inst, "ordId": orderId}
            resp = self.client._make_request('POST', endpoint, params, signed=True)
            return resp.get('code') == '0'
        except OKXAPIException:
            return False

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
        result = {"status": "unknown", "filled_size": 0.0, "avg_price": 0.0}
        if not orderId:
            return result
        try:
            # 1) 优先使用交易所查询单笔订单（若可用）
            try:
                endpoint = "/api/v5/trade/order"
                params = {"instId": instId, "ordId": orderId}
                resp = self.client._make_request('GET', endpoint, params, signed=True)
                if resp.get('code') == '0' and resp.get('data'):
                    od = resp['data'][0]
                    state = od.get('state', '')
                    accFillSz = float(od.get('accFillSz', 0) or 0)
                    avgPx = float(od.get('avgPx', 0) or 0)
                    result["status"] = state or "unknown"
                    result["filled_size"] = accFillSz
                    result["avg_price"] = avgPx
                    return result
            except OKXAPIException:
                # 忽略，降级到本地挂单列表
                pass

            # 2) 降级：从未完成订单列表判断是否 live
            pending = self.client.get_orders(instId=instId)
            if pending.get('code') == '0' and 'data' in pending:
                for od in pending['data']:
                    if od.get('ordId') == orderId:
                        accFillSz = float(od.get('accFillSz', 0) or 0)
                        avgPx = float(od.get('avgPx', 0) or 0)
                        state = od.get('state', 'live') or 'live'
                        result["status"] = state
                        result["filled_size"] = accFillSz
                        result["avg_price"] = avgPx
                        return result

            # 3) 未在挂单列表，可能为 filled / canceled（由于未拉取历史，这里保持 unknown）
            return result
        except Exception:
            return result

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
        try:
            resp = self.client.get_orders(instId=instId)
            orders = []
            if resp.get('code') == '0' and 'data' in resp:
                for od in resp['data']:
                    orders.append({
                        "instId": od.get('instId', ''),
                        "orderId": od.get('ordId', ''),
                        "side": od.get('side', ''),
                        "size": float(od.get('sz', 0) or 0),
                        "price": float(od.get('px', 0) or 0)
                    })
            return orders
        except OKXAPIException:
            return []

    def get_position_info(self, instId: str) -> dict:
        """
        Get position info
        
        Returns:
            Position info
            {"instId": "BTC-USDT-SWAP", "pos_side": "long"|"short", "size": float, "avg_price": float, "unrealized_pnl": float}
        """
        try:
            resp = self.client.get_positions()
            if resp.get('code') != '0' or 'data' not in resp:
                return {"instId": instId, "pos_side": "flat", "size": 0.0, "avg_price": 0.0, "unrealized_pnl": 0.0}

            positions = [p for p in resp['data'] if p.get('instId') == instId]
            if not positions:
                return {"instId": instId, "pos_side": "flat", "size": 0.0, "avg_price": 0.0, "unrealized_pnl": 0.0}

            # 如果多空双持，选择绝对持仓较大的一个（做市常见，简化处理）
            def abs_sz(p):
                try:
                    return abs(float(p.get('pos', 0) or 0))
                except Exception:
                    return 0.0

            pos = max(positions, key=abs_sz)
            pos_side = pos.get('posSide', 'net') or 'net'
            size = float(pos.get('pos', 0) or 0)
            avg_price = float(pos.get('avgPx', 0) or 0)
            upl = float(pos.get('upl', 0) or 0)

            # 规范化 pos_side：当 size>0 视为 long，size<0 视为 short（若返回为 net 模式）
            if pos_side == 'net':
                if size > 0:
                    pos_side = 'long'
                elif size < 0:
                    pos_side = 'short'
                else:
                    pos_side = 'flat'

            return {"instId": instId, "pos_side": pos_side, "size": abs(size), "avg_price": avg_price, "unrealized_pnl": upl}
        except Exception:
            return {"instId": instId, "pos_side": "flat", "size": 0.0, "avg_price": 0.0, "unrealized_pnl": 0.0}

    def get_account_balance(self) -> float:
        """
        Get account balance
        
        Returns:
            USDT available balance (float)
        """
        try:
            resp = self.client.get_account_balance()
            if resp.get('code') != '0' or 'data' not in resp:
                return 0.0

            usdt_avail = 0.0
            for acc in resp['data']:
                for d in acc.get('details', []):
                    if d.get('ccy') == 'USDT':
                        usdt_avail += float(d.get('availBal', 0) or 0)

            return usdt_avail
        except OKXAPIException:
            return 0.0
        
