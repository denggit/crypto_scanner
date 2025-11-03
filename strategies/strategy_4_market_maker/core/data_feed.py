#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 11/3/25 8:31 PM
@File       : data_feed.py
@Description: 
"""
from apis.okx_api.client import get_okx_client
import asyncio
import json
import threading
import time
from typing import Dict, List, Optional, Tuple, Union
from queue import SimpleQueue

from utils.logger import logger


class DataFeed:
    def __init__(self):
        """OKX订单簿数据源。

        功能：
        - 维护基于 asyncio + websockets 的订单簿轻量级缓存（独立线程+事件循环）。
        - 主动订阅 `books5` 频道（5档），可扩展为 `books` 或 `books-l2-tbt`。
        - 提供同步读取接口 `get_best_bid_ask`，策略层可直接调用。

        线程/并发设计：
        - 后台线程内创建独立事件循环，负责与 OKX 建立/重连/接收数据并更新缓存。
        - 读写缓存通过细粒度锁保护，读操作尽量无阻塞（复制最小必要字段）。
        """
        self.client = get_okx_client()

        # 订单簿缓存结构：
        # {
        #   "BTC-USDT-SWAP": {
        #       "timestamp": 1730612345678,
        #       "bids": [(price, size), ...],
        #       "asks": [(price, size), ...],
        #       "best_bid": price,
        #       "best_ask": price,
        #       "spread": ((best_ask - best_bid) / ((best_ask + best_bid)/2)) * 100  # 百分比
        #   }
        # }
        self.orderbook_cache: Dict[str, Dict] = {}

        # 订阅命令队列（无锁），以及初始订阅列表
        self._cmd_queue: SimpleQueue[Tuple[str, str]] = SimpleQueue()  # (op, inst_id) op ∈ {"sub","unsub"}
        self._initial_subs: List[str] = []

        # 停止事件
        self._stop_event = threading.Event()

        # WebSocket 配置（订单簿使用公共频道）
        self._ws_url = "wss://ws.okx.com:8443/ws/v5/public"

        # 事件循环与线程
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws: Optional[object] = None

    def start(self, inst_ids: Optional[Union[str, List[str]]] = None):
        """启动后台线程与事件循环，并订阅指定合约。

        Args:
            inst_ids: 初始需订阅的合约（支持单个字符串或字符串列表），例如 "BTC-USDT-SWAP" 或 ["BTC-USDT-SWAP"].
        """
        if inst_ids:
            if isinstance(inst_ids, str):
                self._initial_subs.append(inst_ids)
            else:
                # 过滤掉可能的空字符串
                self._initial_subs.extend([iid for iid in inst_ids if isinstance(iid, str) and iid])
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop_in_thread, name="DataFeedWS")
        self._thread.daemon = True
        self._thread.start()

    def stop(self):
        """停止后台线程与事件循环。"""
        self._stop_event.set()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown_ws(), self._loop)
        if self._thread:
            self._thread.join(timeout=5)
        self._thread = None
        self._loop = None
        self._ws = None

    def subscribe(self, inst_id: str):
        """动态订阅合约。"""
        self._cmd_queue.put(("sub", inst_id))
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._drain_cmds(), self._loop)

    def unsubscribe(self, inst_id: str):
        """取消订阅合约。"""
        self._cmd_queue.put(("unsub", inst_id))
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._drain_cmds(), self._loop)

    def get_order_book(self, inst_id):
        """获取对应虚拟货币的盘口价格

        返回值：
            asks: 卖方深度
            bids: 买方深度
            ts: 深度产生的时间戳

        说明： ["411.8", "10", "0", "4"]
            - 411.8为深度价格
            - 10为此价格的数量 （合约交易为张数，现货/币币杠杆为交易币的数量）
            - 0该字段已弃用(始终为0)
            - 4为此价格的订单数量
        """
        orders = self.client.get_order_book(inst_id)
        if orders.get("code") != '0':
            raise KeyError("该 get_order_book 接口返回值 code != '0'")

        data = orders.get("data", [])
        if not data:
            return None, None
        return data[0]

    def get_best_bid_ask(self, inst_id: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """读取缓存中的最优买一/卖一与价差。

        若缓存暂无该合约数据，将回退一次 REST 调用填充缓存，随后返回。

        Returns:
            (best_bid, best_ask, spread)；若不可用则返回 (None, None, None)
            bid: 买方
            ask: 卖方
            spread: 相对中间价的百分比（乘以100），例如 0.07 表示 0.07%
        """
        entry = self.orderbook_cache.get(inst_id)
        if entry and entry.get("best_bid") is not None and entry.get("best_ask") is not None:
            return entry["best_bid"], entry["best_ask"], entry.get("spread")

        # 缓存没有则同步回退一次 REST（仅首缺时触发）
        try:
            raw = self.get_order_book(inst_id)
            if not raw:
                return None, None, None
            asks = tuple((float(p), float(sz)) for p, sz, *_ in raw.get("asks", []))
            bids = tuple((float(p), float(sz)) for p, sz, *_ in raw.get("bids", []))
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            if best_bid is not None and best_ask is not None and (best_bid + best_ask) != 0:
                mid = 0.5 * (best_bid + best_ask)
                spread = (best_ask - best_bid) / mid * 100.0
            else:
                spread = None
            self.orderbook_cache[inst_id] = {
                "timestamp": int(raw.get("ts")) if raw.get("ts") else int(time.time() * 1000),
                "bids": bids,
                "asks": asks,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
            }
            return best_bid, best_ask, spread
        except Exception as e:
            logger.error(f"REST 回退获取 {inst_id} orderbook 失败: {e}")
            return None, None, None

    # ------------------------- 内部：线程与事件循环 -------------------------
    def _run_loop_in_thread(self):
        """在线程中创建并运行事件循环，负责WS连接与重连。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_ws_forever())
        except Exception as e:
            logger.error(f"DataFeed 事件循环异常: {e}")
        finally:
            try:
                pending = asyncio.all_tasks(loop=self._loop)
                for t in pending:
                    t.cancel()
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            self._loop.close()

    async def _run_ws_forever(self):
        """保持与 OKX 的 WebSocket 连接与订阅，自动重连。"""
        backoff = 1
        while not self._stop_event.is_set():
            try:
                # 延迟导入，避免在未安装 websockets 时的模块级错误
                try:
                    import websockets  # type: ignore
                except Exception as ie:
                    logger.error(f"缺少依赖 websockets，请安装：pip install websockets；错误: {ie}")
                    await asyncio.sleep(5)
                    continue

                async with websockets.connect(self._ws_url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
                    self._ws = ws
                    logger.info("DataFeed WS 已连接")
                    # 初始订阅
                    if self._initial_subs:
                        logger.info(f"DataFeed 初始订阅: {self._initial_subs}")
                        await self._send_subscribe(self._initial_subs)
                    # 刷新命令队列
                    await self._drain_cmds()
                    # 读取循环
                    backoff = 1
                    async for msg in ws:
                        await self._drain_cmds()
                        await self._on_message(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"WS 连接异常，将在 {backoff}s 后重试: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
        await self._shutdown_ws()

    async def _shutdown_ws(self):
        """关闭WS连接。"""
        try:
            if self._ws:
                await self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None

    async def _on_message(self, message: str):
        """处理业务频道消息，关注 books5 数据。"""
        try:
            data = json.loads(message)
        except Exception:
            return

        # 订阅/事件回执
        if isinstance(data, dict) and data.get("event") in {"subscribe", "unsubscribe", "error"}:
            event = data.get("event")
            ch = data.get("arg", {}).get("channel")
            inst = data.get("arg", {}).get("instId")
            if event == "error":
                logger.error(f"WS 错误: code={data.get('code')} msg={data.get('msg')} arg={data.get('arg')}")
            else:
                # OKX 成功回执通常不带 code 字段
                logger.info(f"WS {event} 成功: {ch} {inst}")
            return

        # 订单簿数据（增量或快照，books5 为快照/刷新）
        if isinstance(data, dict) and data.get("arg", {}).get("channel") == "books5" and data.get("data"):
            arg = data.get("arg", {})
            inst_id = arg.get("instId")
            for ob in data.get("data", []):
                asks = [(float(p), float(sz)) for p, sz, *_ in ob.get("asks", [])]
                bids = [(float(p), float(sz)) for p, sz, *_ in ob.get("bids", [])]
                ts = int(ob.get("ts")) if ob.get("ts") else int(time.time() * 1000)
                best_bid = float(bids[0][0]) if bids else None
                best_ask = float(asks[0][0]) if asks else None
                if best_bid is not None and best_ask is not None and (best_bid + best_ask) != 0:
                    mid = 0.5 * (best_bid + best_ask)
                    spread = (best_ask - best_bid) / mid * 100.0
                else:
                    spread = None
                self.orderbook_cache[inst_id] = {
                    "timestamp": ts,
                    "bids": bids,
                    "asks": asks,
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "spread": spread,
                }

    async def _drain_cmds(self):
        """在 WS 线程内消费订阅命令队列并批量下发。"""
        subs: List[str] = []
        unsubs: List[str] = []
        while True:
            try:
                op, iid = self._cmd_queue.get_nowait()
            except Exception:
                break
            if op == "sub":
                subs.append(iid)
            elif op == "unsub":
                unsubs.append(iid)
        if subs:
            await self._send_subscribe(subs)
        if unsubs:
            await self._send_unsubscribe(unsubs)

    async def _send(self, payload: Dict):
        if self._ws is None:
            return
        try:
            await self._ws.send(json.dumps(payload))
        except Exception as e:
            logger.error(f"WS 发送失败: {e}")

    async def _send_subscribe(self, inst_ids: List[str]):
        args = [{"channel": "books5", "instId": i} for i in inst_ids]
        await self._send({"op": "subscribe", "args": args})

    async def _send_unsubscribe(self, inst_ids: List[str]):
        args = [{"channel": "books5", "instId": i} for i in inst_ids]
        await self._send({"op": "unsubscribe", "args": args})

