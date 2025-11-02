"""Price statistics analytics module."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class PriceStats:
    """Calculate rolling price statistics."""
    
    def __init__(self, window: int = 60):
        """
        Initialize price statistics calculator.
        
        Args:
            window: Rolling window size (default: 60)
        """
        self.window = window
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        prices: pd.Series,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate rolling price statistics.
        
        Args:
            prices: Price series
            window: Optional window size (overrides default)
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if prices.empty:
            raise AnalyticsError("Empty price series")
        
        window = window or self.window
        
        try:
            rolling = prices.rolling(window=window, min_periods=1)
            
            return {
                "mean": float(rolling.mean().iloc[-1]),
                "std": float(rolling.std().iloc[-1]) if len(prices) > 1 else 0.0,
                "min": float(rolling.min().iloc[-1]),
                "max": float(rolling.max().iloc[-1]),
                "current": float(prices.iloc[-1]),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate price stats: {e}")
            raise AnalyticsError(f"Price stats calculation failed: {e}") from e
    
    def calculate_series(
        self,
        prices: pd.Series,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics as time series.
        
        Args:
            prices: Price series
            window: Optional window size
            
        Returns:
            DataFrame with rolling statistics columns
        """
        if prices.empty:
            return pd.DataFrame()
        
        window = window or self.window
        
        try:
            rolling = prices.rolling(window=window, min_periods=1)
            
            df = pd.DataFrame({
                "price": prices,
                "mean": rolling.mean(),
                "std": rolling.std(),
                "min": rolling.min(),
                "max": rolling.max(),
            })
            
            return df
        except Exception as e:
            self.logger.error(f"Failed to calculate price stats series: {e}")
            raise AnalyticsError(f"Price stats series calculation failed: {e}") from e

