"""Base connector interface for data ingestion."""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime
from dateutil import parser as date_parser
from utils.logger import setup_logger
from utils.exceptions import ConnectorError


class BaseConnector(ABC):
    """Abstract base class for data connectors."""
    
    def __init__(self, symbols: list[str]):
        """
        Initialize connector.
        
        Args:
            symbols: List of trading symbol pairs (e.g., ['btcusdt', 'ethusdt'])
        """
        self.symbols = symbols
        self.logger = setup_logger(f"{self.__class__.__name__}")
        self._running = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass
    
    @abstractmethod
    async def stream_ticks(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream normalized tick data.
        
        Yields:
            Dictionary with keys: symbol, timestamp, price, size
        """
        pass
    
    def is_connected(self) -> bool:
        """Check if connector is active."""
        return self._running
    
    def normalize_tick(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize raw tick data to standard format.
        
        Args:
            raw_data: Raw tick data from source
            
        Returns:
            Normalized tick: {symbol, timestamp, price, size}
        """
        symbol = raw_data.get("symbol", "").lower()
        
        timestamp = raw_data.get("timestamp", 0)
        ts = raw_data.get("ts", None)
        
        if ts is not None:
            try:
                if isinstance(ts, str):
                    dt = date_parser.parse(ts)
                    timestamp = int(dt.timestamp() * 1000)
                else:
                    timestamp = int(ts)
            except (ValueError, TypeError, AttributeError) as e:
                self.logger.warning(f"Failed to parse timestamp: {ts}, error: {e}")
                timestamp = 0
        
        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "price": float(raw_data.get("price", 0)),
            "size": float(raw_data.get("size", 0)),
        }

