"""Rolling correlation calculator."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class RollingCorrelation:
    """Calculate rolling correlation between two series."""
    
    def __init__(self, window: int = 60):
        """
        Initialize rolling correlation calculator.
        
        Args:
            window: Rolling window size (default: 60)
        """
        self.window = window
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate rolling correlation between two series.
        
        Args:
            series1: First series
            series2: Second series
            window: Optional window size (overrides default)
            
        Returns:
            Rolling correlation series
        """
        if series1.empty or series2.empty:
            raise AnalyticsError("Empty series")
        
        if len(series1) != len(series2):
            raise AnalyticsError("Series must have equal length")
        
        window = window or self.window
        
        try:
            aligned = pd.DataFrame({
                "series1": series1,
                "series2": series2,
            }).dropna()
            
            if len(aligned) < window:
                return pd.Series(dtype=float, index=aligned.index)
            
            rolling_corr = aligned["series1"].rolling(window=window, min_periods=window).corr(
                aligned["series2"]
            )
            
            return rolling_corr
        except Exception as e:
            self.logger.error(f"Failed to calculate rolling correlation: {e}")
            raise AnalyticsError(f"Rolling correlation calculation failed: {e}") from e
    
    def get_current_correlation(
        self,
        series1: pd.Series,
        series2: pd.Series,
        window: Optional[int] = None
    ) -> float:
        """
        Get current rolling correlation value.
        
        Args:
            series1: First series
            series2: Second series
            window: Optional window size
            
        Returns:
            Current correlation value
        """
        corr_series = self.calculate(series1, series2, window)
        if corr_series.empty:
            return 0.0
        
        last_value = corr_series.iloc[-1]
        return float(last_value) if not pd.isna(last_value) else 0.0

