"""Candle resampler for converting ticks to OHLCV candles."""

import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

from utils.logger import setup_logger
from utils.exceptions import ResamplingError


class CandleResampler:
    """Resamples ticks into OHLCV candles."""
    
    TIMEFRAME_MAPPING = {
        "1s": "1s",
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1H",
        "1d": "1D",
    }
    
    def __init__(self, timeframes: List[str] = None):
        """
        Initialize candle resampler.
        
        Args:
            timeframes: List of supported timeframes (default: ['1s', '1m', '5m'])
        """
        self.timeframes = timeframes or ["1s", "1m", "5m"]
        self.logger = setup_logger(f"{self.__class__.__name__}")
        
        for tf in self.timeframes:
            if tf not in self.TIMEFRAME_MAPPING:
                raise ResamplingError(f"Unsupported timeframe: {tf}")
    
    def resample_ticks(
        self,
        ticks: List[Dict[str, Any]],
        timeframe: str,
        symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Resample ticks into candles for a specific timeframe.
        
        Args:
            ticks: List of tick dictionaries
            timeframe: Target timeframe (e.g., '1m', '5m')
            symbol: Optional symbol (extracted from ticks if not provided)
            
        Returns:
            List of candle dictionaries
        """
        if not ticks:
            return []
        
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ResamplingError(f"Unsupported timeframe: {timeframe}")
        
        try:
            df = pd.DataFrame(ticks)
            
            if df.empty:
                return []
            
            if symbol is None:
                symbols = df["symbol"].unique()
                if len(symbols) > 1:
                    raise ResamplingError("Multiple symbols found, specify symbol parameter")
                symbol = symbols[0]
            
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            
            pandas_freq = self.TIMEFRAME_MAPPING[timeframe]
            
            if "symbol" in df.columns:
                grouped = df.groupby("symbol")
            else:
                grouped = [(symbol, df)]
            
            candles = []
            
            for sym, sym_df in grouped:
                ohlcv = sym_df["price"].resample(pandas_freq).ohlc()
                volume = sym_df["size"].resample(pandas_freq).sum()
                
                result = pd.DataFrame({
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"],
                    "volume": volume,
                })
                
                result = result.dropna()
                
                for idx, row in result.iterrows():
                    candles.append({
                        "symbol": sym,
                        "timeframe": timeframe,
                        "ts": int(idx.timestamp() * 1000),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    })
            
            return candles
            
        except Exception as e:
            self.logger.error(f"Failed to resample ticks: {e}")
            raise ResamplingError(f"Resampling failed: {e}") from e
    
    def resample_dataframe(
        self,
        df: pd.DataFrame,
        timeframe: str,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Resample tick DataFrame into candle DataFrame.
        
        Args:
            df: DataFrame with columns: symbol, timestamp, price, size
            timeframe: Target timeframe
            symbol: Optional symbol filter
            
        Returns:
            DataFrame with OHLCV candles
        """
        if df.empty:
            return pd.DataFrame()
        
        if symbol:
            df = df[df["symbol"] == symbol].copy()
        
        if df.empty:
            return pd.DataFrame()
        
        try:
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("datetime", inplace=True)
            
            pandas_freq = self.TIMEFRAME_MAPPING[timeframe]
            
            candles_list = []
            
            for sym, sym_df in df.groupby("symbol"):
                ohlcv = sym_df["price"].resample(pandas_freq).ohlc()
                volume = sym_df["size"].resample(pandas_freq).sum()
                
                result = pd.DataFrame({
                    "symbol": sym,
                    "timeframe": timeframe,
                    "open": ohlcv["open"],
                    "high": ohlcv["high"],
                    "low": ohlcv["low"],
                    "close": ohlcv["close"],
                    "volume": volume,
                })
                
                result = result.dropna()
                
                result.reset_index(inplace=True)
                result["ts"] = result["datetime"].astype(np.int64) // 10**6
                result.drop("datetime", axis=1, inplace=True)
                
                candles_list.append(result)
            
            if not candles_list:
                return pd.DataFrame()
            
            candles_df = pd.concat(candles_list, ignore_index=True)
            return candles_df
            
        except Exception as e:
            self.logger.error(f"Failed to resample DataFrame: {e}")
            raise ResamplingError(f"Resampling failed: {e}") from e

