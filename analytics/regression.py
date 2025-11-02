"""OLS regression for hedge ratio calculation."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Optional, Tuple

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class HedgeRatioCalculator:
    """Calculate hedge ratio via OLS regression between two symbols."""
    
    def __init__(self, window: int = 100):
        """
        Initialize hedge ratio calculator.
        
        Args:
            window: Rolling window size for regression (default: 100)
        """
        self.window = window
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate hedge ratio via OLS regression: price1 = alpha + beta * price2.
        
        Args:
            prices1: First price series (dependent variable)
            prices2: Second price series (independent variable)
            window: Optional window size (overrides default)
            
        Returns:
            Dictionary with hedge_ratio (beta), alpha, r_squared
        """
        if prices1.empty or prices2.empty:
            raise AnalyticsError("Empty price series")
        
        if len(prices1) != len(prices2):
            raise AnalyticsError("Price series must have equal length")
        
        window = window or self.window
        min_samples = min(window, len(prices1))
        
        try:
            aligned = pd.DataFrame({
                "price1": prices1,
                "price2": prices2,
            }).dropna()
            
            if len(aligned) < 2:
                raise AnalyticsError("Insufficient data for regression")
            
            recent = aligned.tail(min_samples)
            
            X = recent["price2"].values.reshape(-1, 1)
            y = recent["price1"].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            hedge_ratio = float(model.coef_[0])
            alpha = float(model.intercept_)
            r_squared = float(model.score(X, y))
            
            return {
                "hedge_ratio": hedge_ratio,
                "alpha": alpha,
                "r_squared": r_squared,
                "n_samples": len(recent),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate hedge ratio: {e}")
            raise AnalyticsError(f"Hedge ratio calculation failed: {e}") from e
    
    def calculate_rolling(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling hedge ratio.
        
        Args:
            prices1: First price series
            prices2: Second price series
            window: Optional window size
            
        Returns:
            DataFrame with rolling hedge ratio, alpha, r_squared
        """
        if prices1.empty or prices2.empty:
            return pd.DataFrame()
        
        if len(prices1) != len(prices2):
            raise AnalyticsError("Price series must have equal length")
        
        window = window or self.window
        
        try:
            aligned = pd.DataFrame({
                "price1": prices1,
                "price2": prices2,
            }).dropna()
            
            if len(aligned) < window:
                return pd.DataFrame()
            
            hedge_ratios = []
            alphas = []
            r_squareds = []
            
            for i in range(window, len(aligned) + 1):
                window_data = aligned.iloc[i - window:i]
                
                X = window_data["price2"].values.reshape(-1, 1)
                y = window_data["price1"].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                hedge_ratios.append(model.coef_[0])
                alphas.append(model.intercept_)
                r_squareds.append(model.score(X, y))
            
            result = pd.DataFrame({
                "hedge_ratio": hedge_ratios,
                "alpha": alphas,
                "r_squared": r_squareds,
            }, index=aligned.index[window - 1:])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to calculate rolling hedge ratio: {e}")
            raise AnalyticsError(f"Rolling hedge ratio calculation failed: {e}") from e

