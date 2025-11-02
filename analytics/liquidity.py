"""Liquidity filters and metrics."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


class LiquidityAnalyzer:
    """Analyze liquidity metrics and filters."""
    
    def __init__(self, min_volume_threshold: float = 0.0, min_trade_count: int = 1):
        """
        Initialize liquidity analyzer.
        
        Args:
            min_volume_threshold: Minimum volume threshold
            min_trade_count: Minimum number of trades
        """
        self.min_volume_threshold = min_volume_threshold
        self.min_trade_count = min_trade_count
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def calculate_metrics(
        self,
        volume_series: pd.Series,
        price_series: Optional[pd.Series] = None,
        window: int = 60
    ) -> Dict[str, Any]:
        """
        Calculate liquidity metrics.
        
        Args:
            volume_series: Volume series
            price_series: Optional price series for VWAP calculation
            window: Rolling window size
            
        Returns:
            Dictionary with liquidity metrics
        """
        if volume_series.empty:
            raise AnalyticsError("Empty volume series")
        
        try:
            rolling_volume = volume_series.rolling(window=window, min_periods=1)
            
            metrics = {
                "current_volume": float(volume_series.iloc[-1]) if len(volume_series) > 0 else 0.0,
                "mean_volume": float(rolling_volume.mean().iloc[-1]),
                "median_volume": float(rolling_volume.median().iloc[-1]),
                "std_volume": float(rolling_volume.std().iloc[-1]) if len(volume_series) > 1 else 0.0,
                "min_volume": float(rolling_volume.min().iloc[-1]),
                "max_volume": float(rolling_volume.max().iloc[-1]),
                "volume_percentile_25": float(volume_series.rolling(window).quantile(0.25).iloc[-1]),
                "volume_percentile_75": float(volume_series.rolling(window).quantile(0.75).iloc[-1]),
            }
            
            if len(volume_series) >= 2:
                recent_trend = volume_series.tail(window).diff().mean()
                metrics["volume_trend"] = float(recent_trend)
            else:
                metrics["volume_trend"] = 0.0
            
            if price_series is not None and len(price_series) == len(volume_series):
                vwap = (price_series * volume_series).rolling(window=window, min_periods=1).sum() / \
                       volume_series.rolling(window=window, min_periods=1).sum()
                metrics["current_vwap"] = float(vwap.iloc[-1]) if len(vwap) > 0 else 0.0
                metrics["vwap_spread"] = float(price_series.iloc[-1] - vwap.iloc[-1]) if len(price_series) > 0 and len(vwap) > 0 else 0.0
            
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to calculate liquidity metrics: {e}")
            raise AnalyticsError(f"Liquidity calculation failed: {e}") from e
    
    def filter_liquidity(
        self,
        df: pd.DataFrame,
        volume_column: str = "volume",
        min_volume: Optional[float] = None,
        min_percentile: float = 0.1
    ) -> pd.DataFrame:
        """
        Filter DataFrame based on liquidity criteria.
        
        Args:
            df: DataFrame to filter
            volume_column: Name of volume column
            min_volume: Minimum volume threshold
            min_percentile: Minimum percentile threshold
            
        Returns:
            Filtered DataFrame
        """
        if df.empty or volume_column not in df.columns:
            return df
        
        try:
            filtered = df.copy()
            
            if min_volume is not None:
                filtered = filtered[filtered[volume_column] >= min_volume]
            else:
                volume_threshold = filtered[volume_column].quantile(min_percentile)
                filtered = filtered[filtered[volume_column] >= volume_threshold]
            
            self.logger.debug(f"Filtered {len(df)} rows to {len(filtered)} rows based on liquidity")
            
            return filtered
        except Exception as e:
            self.logger.error(f"Failed to filter liquidity: {e}")
            raise AnalyticsError(f"Liquidity filtering failed: {e}") from e
    
    def calculate_bid_ask_spread_estimate(
        self,
        high_series: pd.Series,
        low_series: pd.Series,
        close_series: pd.Series,
        window: int = 60
    ) -> Dict[str, Any]:
        """
        Estimate bid-ask spread from OHLC data.
        
        Args:
            high_series: High price series
            low_series: Low price series
            close_series: Close price series
            window: Rolling window size
            
        Returns:
            Dictionary with spread estimates
        """
        if high_series.empty or low_series.empty or close_series.empty:
            raise AnalyticsError("Empty series")
        
        try:
            hl_spread = high_series - low_series
            hl_spread_pct = (hl_spread / close_series) * 100
            
            rolling_spread = hl_spread.rolling(window=window, min_periods=1)
            rolling_spread_pct = hl_spread_pct.rolling(window=window, min_periods=1)
            
            return {
                "current_spread": float(hl_spread.iloc[-1]) if len(hl_spread) > 0 else 0.0,
                "current_spread_pct": float(hl_spread_pct.iloc[-1]) if len(hl_spread_pct) > 0 else 0.0,
                "mean_spread": float(rolling_spread.mean().iloc[-1]),
                "mean_spread_pct": float(rolling_spread_pct.mean().iloc[-1]),
                "median_spread": float(rolling_spread.median().iloc[-1]),
                "max_spread": float(rolling_spread.max().iloc[-1]),
                "min_spread": float(rolling_spread.min().iloc[-1]),
            }
        except Exception as e:
            self.logger.error(f"Failed to calculate bid-ask spread: {e}")
            raise AnalyticsError(f"Bid-ask spread calculation failed: {e}") from e

