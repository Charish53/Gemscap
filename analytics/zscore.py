"""Z-score calculator for spread normalization."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class ZScoreCalculator:
    """Calculate Z-score series for spread normalization."""
    
    def __init__(self, window: int = 60):
        """
        Initialize Z-score calculator.
        
        Args:
            window: Rolling window size for mean/std calculation (default: 60)
        """
        self.window = window
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        series: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate Z-score: (value - rolling_mean) / rolling_std.
        
        Args:
            series: Input series (e.g., spread)
            window: Optional window size (overrides default)
            
        Returns:
            Z-score series
        """
        if series.empty:
            raise AnalyticsError("Empty series")
        
        window = window or self.window
        
        try:
            rolling_mean = series.rolling(window=window, min_periods=1).mean()
            rolling_std = series.rolling(window=window, min_periods=1).std()
            
            rolling_std = rolling_std.replace(0, np.nan)
            
            zscore = (series - rolling_mean) / rolling_std
            
            return zscore.fillna(0.0)
        except Exception as e:
            self.logger.error(f"Failed to calculate Z-score: {e}")
            raise AnalyticsError(f"Z-score calculation failed: {e}") from e
    
    def get_current_zscore(
        self,
        series: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """
        Get current Z-score value.
        
        Args:
            series: Input series
            window: Optional window size
            
        Returns:
            Current Z-score value
        """
        zscore_series = self.calculate(series, window)
        if zscore_series.empty:
            return 0.0
        return float(zscore_series.iloc[-1])

