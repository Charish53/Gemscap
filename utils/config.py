"""Configuration management."""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""
    
    db_path: str = "data/market_data.db"
    
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/"
    binance_symbols: list = None
    
    data_dir: str = "sample_data"
    
    supported_timeframes: list = None
    
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    frontend_port: int = 8501
    
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize default values."""
        if self.binance_symbols is None:
            self.binance_symbols = ["btcusdt", "ethusdt"]
        
        if self.supported_timeframes is None:
            self.supported_timeframes = ["1s", "1m", "5m"]
        
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            db_path=os.getenv("DB_PATH", "data/market_data.db"),
            binance_ws_url=os.getenv("BINANCE_WS_URL", "wss://stream.binance.com:9443/ws/"),
            binance_symbols=os.getenv("BINANCE_SYMBOLS", "btcusdt,ethusdt").split(","),
            api_host=os.getenv("API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

