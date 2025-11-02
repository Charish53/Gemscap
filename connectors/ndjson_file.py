"""NDJSON file connector for offline replay."""

import json
import asyncio
from pathlib import Path
from typing import AsyncIterator, Dict, Any, Optional
from datetime import datetime

from connectors.base import BaseConnector
from utils.exceptions import ConnectorError
from utils.logger import setup_logger


class NDJSONFileConnector(BaseConnector):
    """File-based connector for replaying historical tick data from NDJSON."""
    
    def __init__(self, file_path: str, symbols: Optional[list[str]] = None, replay_speed: float = 1.0):
        """
        Initialize NDJSON file connector.
        
        Args:
            file_path: Path to NDJSON file
            symbols: Optional filter for specific symbols (None = all)
            replay_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x, 0.5 = half)
        """
        super().__init__(symbols or [])
        self.file_path = Path(file_path)
        self.replay_speed = replay_speed
        self.symbol_filter = set(s.lower() for s in symbols) if symbols else None
        
        if not self.file_path.exists():
            raise ConnectorError(f"File not found: {file_path}")
    
    async def connect(self) -> None:
        """Validate file exists."""
        if not self.file_path.exists():
            raise ConnectorError(f"File not found: {self.file_path}")
        self._running = True
        self.logger.info(f"Connected to file: {self.file_path}")
    
    async def disconnect(self) -> None:
        """Close file connection."""
        self._running = False
        self.logger.info("Disconnected from file")
    
    async def stream_ticks(self) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream ticks from NDJSON file with optional replay timing.
        
        Yields:
            Normalized tick dictionaries
        """
        if not self._running:
            await self.connect()
        
        last_timestamp = None
        
        try:
            with open(self.file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if not self._running:
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        tick_data = json.loads(line)
                        normalized = self.normalize_tick(tick_data)
                        
                        if self.symbol_filter and normalized["symbol"] not in self.symbol_filter:
                            continue
                        
                        if last_timestamp is not None and self.replay_speed > 0:
                            delay_ms = normalized["timestamp"] - last_timestamp
                            delay_seconds = (delay_ms / 1000.0) / self.replay_speed
                            if delay_seconds > 0:
                                await asyncio.sleep(delay_seconds)
                        
                        last_timestamp = normalized["timestamp"]
                        yield normalized
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.error(f"Error processing line {line_num}: {e}")
                        continue
                        
        except FileNotFoundError:
            raise ConnectorError(f"File not found: {self.file_path}")
        except Exception as e:
            raise ConnectorError(f"Error reading file: {e}") from e
        
        self.logger.info(f"Finished replaying file: {self.file_path}")

