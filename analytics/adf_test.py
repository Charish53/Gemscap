"""Augmented Dickey-Fuller test for stationarity."""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class ADFTester:
    """Perform Augmented Dickey-Fuller test for spread stationarity."""
    
    def __init__(self, maxlag: Optional[int] = None):
        """
        Initialize ADF tester.
        
        Args:
            maxlag: Maximum lag order (None = auto-select)
        """
        self.maxlag = maxlag
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def test(
        self,
        series: pd.Series,
        maxlag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform ADF test for stationarity.
        
        Args:
            series: Time series to test (e.g., spread)
            maxlag: Optional max lag (overrides default)
            
        Returns:
            Dictionary with test results:
            - adf_statistic: ADF test statistic
            - p_value: p-value
            - critical_values: Dict of critical values at 1%, 5%, 10%
            - is_stationary: Boolean (p-value < 0.05)
            - n_observations: Number of observations used
        """
        if series.empty or len(series) < 10:
            raise AnalyticsError("Insufficient data for ADF test (minimum 10 observations)")
        
        maxlag = maxlag or self.maxlag
        
        try:
            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(clean_series) < 10:
                raise AnalyticsError(f"Insufficient clean data for ADF test: {len(clean_series)} valid observations (minimum 10 required)")
            
            if clean_series.nunique() <= 1:
                raise AnalyticsError("Cannot perform ADF test on constant series (all values are the same)")
            
            try:
                result = adfuller(clean_series, maxlag=maxlag, autolag="AIC")
            except Exception as e:
                raise AnalyticsError(f"ADF test computation failed: {str(e)}. Series may contain invalid values.")
            
            adf_statistic = float(result[0])
            p_value = float(result[1])
            critical_values = {
                "1%": float(result[4]["1%"]),
                "5%": float(result[4]["5%"]),
                "10%": float(result[4]["10%"]),
            }
            n_observations = int(result[3])
            
            is_stationary = p_value < 0.05
            
            return {
                "adf_statistic": adf_statistic,
                "p_value": p_value,
                "critical_values": critical_values,
                "is_stationary": is_stationary,
                "n_observations": n_observations,
            }
        except Exception as e:
            self.logger.error(f"Failed to perform ADF test: {e}")
            raise AnalyticsError(f"ADF test failed: {e}") from e

