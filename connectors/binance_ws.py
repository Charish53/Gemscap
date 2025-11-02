"""Binance WebSocket connector for live tick data."""

import asyncio
import json
from typing import AsyncIterator, Dict, Any, Optional
import websockets
from datetime import datetime
import time

from connectors.base import BaseConnector
from utils.exceptions import ConnectorError
from utils.logger import setup_logger


class BinanceWebSocketConnector(BaseConnector):
    """Binance WebSocket connector for real-time trade data."""
    
    BASE_URL = "wss://stream.binance.com:9443/ws/"
    
    def __init__(self, symbols: list[str], url: Optional[str] = None):
        """
        Initialize Binance WebSocket connector.
        
        Args:
            symbols: List of symbols (e.g., ['btcusdt', 'ethusdt'])
            url: Optional custom WebSocket URL
        """
        super().__init__(symbols)
        self.url = url or self.BASE_URL
        self.websocket = None
        self._reconnect_delay = 5
        self._max_reconnect_attempts = 10
    
    async def connect(self) -> None:
        """Establish WebSocket connection to Binance."""
        try:
            streams = [f"{symbol}@trade" for symbol in self.symbols]
            stream_url = f"{self.url}stream?streams={'/'.join(streams)}"
            
            self.logger.info(f"Connecting to Binance WebSocket: {stream_url}")
            self.websocket = await websockets.connect(stream_url)
            self._running = True
            self.logger.info("Connected to Binance WebSocket")
        except Exception as e:
            self._running = False
            raise ConnectorError(f"Failed to connect to Binance: {e}") from e
    
    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._running = False
        if self.websocket:
            await self.websocket.close()
            self.logger.info("Disconnected from Binance WebSocket")
    
    async def stream_ticks(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream normalized tick data from Binance.
        
        Yields:
            Normalized tick dictionaries
        """
        if not self._running or not self.websocket:
            await self.connect()
        
        reconnect_attempts = 0
        
        while self._running:
            try:
                async for message in self.websocket:
                    if not self._running:
                        break
                    
                    try:
                        data = json.loads(message)
                        if "data" in data:
                            tick_data = data["data"]
                            normalized = self._normalize_binance_tick(tick_data)
                            if normalized:
                                yield normalized
                            reconnect_attempts = 0
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse message: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing tick: {e}")
                        continue
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("WebSocket connection closed, attempting reconnect...")
                if reconnect_attempts < self._max_reconnect_attempts:
                    reconnect_attempts += 1
                    await asyncio.sleep(self._reconnect_delay)
                    try:
                        await self.connect()
                    except Exception as e:
                        self.logger.error(f"Reconnection attempt {reconnect_attempts} failed: {e}")
                else:
                    raise ConnectorError("Max reconnection attempts reached")
            except Exception as e:
                self.logger.error(f"Unexpected error in stream: {e}")
                if reconnect_attempts < self._max_reconnect_attempts:
                    reconnect_attempts += 1
                    await asyncio.sleep(self._reconnect_delay)
                    try:
                        await self.connect()
                    except:
                        pass
                else:
                    raise ConnectorError(f"Stream error: {e}") from e
    
    def _normalize_binance_tick(self, raw_tick: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize Binance trade tick to standard format.
        
        Args:
            raw_tick: Raw Binance trade data
            
        Returns:
            Normalized tick or None if invalid
        """
        try:
            symbol = raw_tick.get("s", "").lower()
            price = float(raw_tick.get("p", 0))
            size = float(raw_tick.get("q", 0))
            timestamp_ms = raw_tick.get("T", raw_tick.get("t", 0))
            
            if not symbol or price <= 0 or size <= 0:
                return None
            
            return {
                "symbol": symbol,
                "timestamp": timestamp_ms,
                "price": price,
                "size": size,
            }
        except (ValueError, KeyError) as e:
            self.logger.debug(f"Failed to normalize tick: {e}")
            return None

