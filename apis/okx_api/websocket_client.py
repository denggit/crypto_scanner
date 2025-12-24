#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author     : Zijun Deng
@Date       : 10/12/2025 3:49 PM
@File       : websocket_client.py
@Description: OKX WebSocket Client for K-line channel
"""

import json
import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Any
import websocket
from websocket import WebSocketApp

from .models import Candle
from utils.logger import logger


class OKXWebSocketException(Exception):
    """Custom exception for OKX WebSocket errors"""
    pass


class OKXWebSocketClient:
    """
    OKX WebSocket Client for real-time K-line data
    """

    def __init__(self, demo: bool = False):
        """
        Initialize OKX WebSocket Client

        Args:
            demo: Use demo trading endpoint
        """
        self.demo = demo
        
        if demo:
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/business"
        else:
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/business"

        self.ws: Optional[WebSocketApp] = None
        self.is_connected = False
        self.is_authenticated = False
        
        # Callbacks
        self.on_kline_callback: Optional[Callable[[Candle], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None
        self.on_connect_callback: Optional[Callable[[], None]] = None
        self.on_disconnect_callback: Optional[Callable[[], None]] = None
        
        # Subscriptions
        self.subscriptions: List[Dict] = []
        
        # Threading
        self.ws_thread: Optional[threading.Thread] = None
        self.keep_running = False

    def _on_open(self, ws: WebSocketApp):
        """WebSocket connection opened"""
        logger.info("OKX WebSocket connected")
        self.is_connected = True
        
        # Resubscribe to previous channels
        if self.subscriptions:
            for sub in self.subscriptions:
                self._send_message(sub)
        
        if self.on_connect_callback:
            self.on_connect_callback()

    def _on_message(self, ws: WebSocketApp, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle ping-pong
            if data.get('event') == 'ping':
                self._send_pong()
                return
            
            # Handle subscription response
            if data.get('event') == 'subscribe':
                if data.get('code') == '0':
                    logger.info(f"Successfully subscribed to channel: {data.get('arg', {}).get('channel')}")
                else:
                    logger.error(f"Subscription failed: {data.get('msg')}")
                return
            
            # Handle kline data
            if 'data' in data and data.get('arg', {}).get('channel') == 'candle1s':
                self._handle_kline_data(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def _on_error(self, ws: WebSocketApp, error: Any):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
        if self.on_error_callback:
            self.on_error_callback(str(error))

    def _on_close(self, ws: WebSocketApp, close_status_code: int, close_msg: str):
        """WebSocket connection closed"""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_connected = False
        if self.on_disconnect_callback:
            self.on_disconnect_callback()

    def _handle_kline_data(self, data: Dict):
        """Process incoming kline data"""
        if not self.on_kline_callback:
            return
            
        try:
            for candle_data in data['data']:
                candle = Candle(
                    ts=int(candle_data[0]),
                    o=float(candle_data[1]),
                    h=float(candle_data[2]),
                    l=float(candle_data[3]),
                    c=float(candle_data[4]),
                    vol=float(candle_data[5]),
                    volCcy=float(candle_data[6])
                )
                self.on_kline_callback(candle)
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse kline data: {e}")

    def _send_message(self, message: Dict):
        """Send message through WebSocket"""
        if self.ws and self.is_connected:
            try:
                self.ws.send(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")

    def _send_pong(self):
        """Send pong response to ping"""
        pong_message = {
            "op": "pong"
        }
        self._send_message(pong_message)

    def connect(self):
        """Connect to WebSocket"""
        if self.is_connected:
            logger.warning("WebSocket already connected")
            return

        try:
            self.ws = WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            self.keep_running = True
            self.ws_thread = threading.Thread(target=self._run_websocket)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection
            timeout = 10
            start_time = time.time()
            while not self.is_connected and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            if not self.is_connected:
                raise OKXWebSocketException("Failed to connect to WebSocket within timeout")
                
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket: {e}")
            raise OKXWebSocketException(f"Connection failed: {e}")

    def _run_websocket(self):
        """Run WebSocket in a separate thread"""
        try:
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket run_forever error: {e}")
        finally:
            self.is_connected = False

    def disconnect(self):
        """Disconnect from WebSocket"""
        self.keep_running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread:
            self.ws_thread.join(timeout=5)
        self.is_connected = False
        logger.info("WebSocket disconnected")

    def subscribe_kline(self, instId: str):
        """
        Subscribe to K-line channel (1-second interval)
        
        Args:
            instId: Instrument ID (e.g., "BTC-USDT")
        """
        subscribe_message = {
            "op": "subscribe",
            "args": [{
                "channel": "candle1s",
                "instId": instId
            }]
        }
        
        self.subscriptions.append(subscribe_message)
        
        if self.is_connected:
            self._send_message(subscribe_message)

    def unsubscribe_kline(self, instId: str):
        """
        Unsubscribe from K-line channel
        
        Args:
            instId: Instrument ID (e.g., "BTC-USDT")
        """
        unsubscribe_message = {
            "op": "unsubscribe",
            "args": [{
                "channel": "candle1s",
                "instId": instId
            }]
        }
        
        # Remove from subscriptions
        self.subscriptions = [sub for sub in self.subscriptions 
                            if not (sub.get('op') == 'subscribe' and 
                                   sub.get('args', [{}])[0].get('instId') == instId)]
        
        if self.is_connected:
            self._send_message(unsubscribe_message)

    def set_on_kline_callback(self, callback: Callable[[Candle], None]):
        """Set callback for kline data"""
        self.on_kline_callback = callback

    def set_on_error_callback(self, callback: Callable[[str], None]):
        """Set callback for errors"""
        self.on_error_callback = callback

    def set_on_connect_callback(self, callback: Callable[[], None]):
        """Set callback for connection"""
        self.on_connect_callback = callback

    def set_on_disconnect_callback(self, callback: Callable[[], None]):
        """Set callback for disconnection"""
        self.on_disconnect_callback = callback

    def is_alive(self) -> bool:
        """Check if WebSocket connection is alive"""
        return self.is_connected and self.keep_running