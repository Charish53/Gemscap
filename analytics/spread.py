"""Spread calculator between two price series."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class SpreadCalculator:
    """Calculate spread between two price series."""
    
    def __init__(self, hedge_ratio: float = 1.0):
        """
        Initialize spread calculator.
        
        Args:
            hedge_ratio: Hedge ratio to use for spread calculation (default: 1.0)
        """
        self.hedge_ratio = hedge_ratio
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate spread: price1 - hedge_ratio * price2.
        
        Args:
            prices1: First price series
            prices2: Second price series
            hedge_ratio: Optional hedge ratio (overrides default)
            
        Returns:
            Spread series
        """
        if prices1.empty or prices2.empty:
            raise AnalyticsError("Empty price series")
        
        if len(prices1) != len(prices2):
            raise AnalyticsError("Price series must have equal length")
        
        hedge_ratio = hedge_ratio or self.hedge_ratio
        
        try:
            aligned = pd.DataFrame({
                "price1": prices1,
                "price2": prices2,
            }).dropna()
            
            if aligned.empty:
                raise AnalyticsError("No overlapping data after alignment")
            
            spread = aligned["price1"] - hedge_ratio * aligned["price2"]
            
            spread = spread.replace([np.inf, -np.inf], np.nan)
            
            return spread
        except Exception as e:
            self.logger.error(f"Failed to calculate spread: {e}")
            raise AnalyticsError(f"Spread calculation failed: {e}") from e
    
    def calculate_with_hedge_ratio(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        hedge_ratio_calculator
    ) -> pd.Series:
        """
        Calculate spread using dynamically computed hedge ratio.
        
        Args:
            prices1: First price series
            prices2: Second price series
            hedge_ratio_calculator: HedgeRatioCalculator instance
            
        Returns:
            Spread series
        """
        try:
            hedge_result = hedge_ratio_calculator.calculate(prices1, prices2)
            hedge_ratio = hedge_result["hedge_ratio"]
            
            return self.calculate(prices1, prices2, hedge_ratio)
        except Exception as e:
            self.logger.error(f"Failed to calculate spread with hedge ratio: {e}")
            raise AnalyticsError(f"Spread calculation failed: {e}") from e

