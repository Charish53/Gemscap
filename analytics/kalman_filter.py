"""Kalman Filter for dynamic hedge ratio estimation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from filterpy.kalman import KalmanFilter
    from filterpy.common import Q_discrete_white_noise
    FILTERPY_AVAILABLE = True
except ImportError:
    FILTERPY_AVAILABLE = False
    KalmanFilter = None

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class KalmanFilterHedge:
    """Dynamic hedge ratio estimation using Kalman Filter."""
    
    def __init__(self, initial_hedge_ratio: float = 1.0, process_noise: float = 0.001, measurement_noise: float = 0.1):
        """
        Initialize Kalman Filter for hedge ratio.
        
        Args:
            initial_hedge_ratio: Initial hedge ratio estimate
            process_noise: Process noise variance
            measurement_noise: Measurement noise variance
        """
        self.initial_hedge_ratio = initial_hedge_ratio
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        initial_hedge_ratio: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate dynamic hedge ratio using Kalman Filter.
        
        Args:
            prices1: First price series (dependent variable)
            prices2: Second price series (independent variable)
            initial_hedge_ratio: Optional initial hedge ratio
            
        Returns:
            Dictionary with dynamic hedge ratio series and statistics
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
                raise AnalyticsError("Insufficient data for Kalman Filter")
            
            initial_hedge = initial_hedge_ratio or self.initial_hedge_ratio
            
            if not FILTERPY_AVAILABLE:
                self.logger.warning("KalmanFilter (filterpy) not available, using OLS fallback")
                from sklearn.linear_model import LinearRegression
                X = aligned["price2"].values.reshape(-1, 1)
                y = aligned["price1"].values
                model = LinearRegression()
                model.fit(X, y)
                hedge_ratio_series = pd.Series([model.coef_[0]] * len(aligned), index=aligned.index)
                uncertainty_series = pd.Series([0.0] * len(aligned), index=aligned.index)
                
                return {
                    "hedge_ratio_series": hedge_ratio_series.tolist(),
                    "uncertainty_series": uncertainty_series.tolist(),
                    "current_hedge_ratio": float(model.coef_[0]),
                    "current_uncertainty": 0.0,
                    "mean_hedge_ratio": float(model.coef_[0]),
                    "std_hedge_ratio": 0.0,
                    "min_hedge_ratio": float(model.coef_[0]),
                    "max_hedge_ratio": float(model.coef_[0]),
                    "n_samples": len(aligned),
                }
            
            kf = KalmanFilter(dim_x=1, dim_z=1)
            
            kf.x = np.array([[initial_hedge]])
            
            kf.F = np.array([[1.0]])
            
            kf.H = np.array([[1.0]])
            
            kf.R = np.array([[self.measurement_noise]])
            
            kf.Q = np.array([[self.process_noise]])
            
            kf.P = np.array([[1.0]])
            
            hedge_ratios = []
            hedge_ratio_uncertainties = []
            
            for i in range(len(aligned)):
                p1 = aligned.iloc[i]["price1"]
                p2 = aligned.iloc[i]["price2"]
                
                if p2 == 0:
                    continue
                
                observed_hedge = p1 / p2
                
                kf.predict()
                kf.update(np.array([observed_hedge]))
                
                hedge_ratios.append(float(kf.x[0, 0]))
                hedge_ratio_uncertainties.append(float(kf.P[0, 0]))
            
            if not hedge_ratios:
                raise AnalyticsError("No valid hedge ratios calculated")
            
            hedge_ratio_series = pd.Series(hedge_ratios, index=aligned.index[:len(hedge_ratios)])
            uncertainty_series = pd.Series(hedge_ratio_uncertainties, index=aligned.index[:len(hedge_ratio_uncertainties)])
            
            return {
                "hedge_ratio_series": hedge_ratio_series.tolist(),
                "uncertainty_series": uncertainty_series.tolist(),
                "current_hedge_ratio": float(hedge_ratio_series.iloc[-1]),
                "current_uncertainty": float(uncertainty_series.iloc[-1]),
                "mean_hedge_ratio": float(hedge_ratio_series.mean()),
                "std_hedge_ratio": float(hedge_ratio_series.std()),
                "min_hedge_ratio": float(hedge_ratio_series.min()),
                "max_hedge_ratio": float(hedge_ratio_series.max()),
                "n_samples": len(hedge_ratios),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate Kalman Filter hedge ratio: {e}")
            raise AnalyticsError(f"Kalman Filter calculation failed: {e}") from e
    
    def get_current_hedge_ratio(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        initial_hedge_ratio: Optional[float] = None
    ) -> float:
        """
        Get current dynamic hedge ratio.
        
        Args:
            prices1: First price series
            prices2: Second price series
            initial_hedge_ratio: Optional initial hedge ratio
            
        Returns:
            Current hedge ratio
        """
        result = self.calculate(prices1, prices2, initial_hedge_ratio)
        return result["current_hedge_ratio"]

