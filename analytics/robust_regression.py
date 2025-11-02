"""Robust regression methods for hedge ratio calculation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import HuberRegressor, TheilSenRegressor

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class RobustRegression:
    """Robust regression methods for hedge ratio calculation."""
    
    def __init__(self, method: str = "huber", epsilon: float = 1.35):
        """
        Initialize robust regression calculator.
        
        Args:
            method: Regression method ('huber' or 'theilsen')
            epsilon: Epsilon parameter for Huber (default: 1.35)
        """
        self.method = method.lower()
        self.epsilon = epsilon
        self.logger = setup_logger(f"{self.__class__.__name__}")
        
        if self.method not in ["huber", "theilsen"]:
            raise AnalyticsError(f"Unknown robust regression method: {method}")
    
    def calculate(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        window: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate hedge ratio using robust regression.
        
        Args:
            prices1: First price series (dependent variable)
            prices2: Second price series (independent variable)
            window: Optional window size (uses all data if None)
            
        Returns:
            Dictionary with hedge_ratio, alpha, r_squared, n_samples
        """
        if prices1.empty or prices2.empty:
            raise AnalyticsError("Empty price series")
        
        if len(prices1) != len(prices2):
            raise AnalyticsError("Price series must have equal length")
        
        try:
            aligned = pd.DataFrame({
                "price1": prices1,
                "price2": prices2,
            }).dropna()
            
            if len(aligned) < 2:
                raise AnalyticsError("Insufficient data for regression")
            
            if window:
                aligned = aligned.tail(window)
            
            X = aligned["price2"].values.reshape(-1, 1)
            y = aligned["price1"].values
            
            if self.method == "huber":
                model = HuberRegressor(epsilon=self.epsilon)
            elif self.method == "theilsen":
                model = TheilSenRegressor(random_state=42, max_subpopulation=1000)
            
            model.fit(X, y)
            
            hedge_ratio = float(model.coef_[0])
            alpha = float(model.intercept_)
            
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            return {
                "hedge_ratio": hedge_ratio,
                "alpha": alpha,
                "r_squared": float(r_squared),
                "n_samples": len(aligned),
                "method": self.method,
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate robust regression: {e}")
            raise AnalyticsError(f"Robust regression calculation failed: {e}") from e

