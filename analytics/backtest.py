"""Mean-reversion backtest strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import setup_logger
from utils.exceptions import AnalyticsError


@dataclass
class Trade:
    """Trade record."""
    entry_time: int
    exit_time: Optional[int]
    entry_zscore: float
    exit_zscore: Optional[float]
    entry_price1: float
    entry_price2: float
    exit_price1: Optional[float]
    exit_price2: Optional[float]
    pnl: Optional[float]
    status: str


class MeanReversionBacktest:
    """Mean-reversion backtest strategy: enter when z>2, exit when z<0."""
    
    def __init__(self, entry_threshold: float = 2.0, exit_threshold: float = 0.0):
        """
        Initialize backtest strategy.
        
        Args:
            entry_threshold: Z-score threshold for entry (default: 2.0)
            exit_threshold: Z-score threshold for exit (default: 0.0)
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.logger = setup_logger(f"{self.__class__.__name__}")
    
    def run(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        zscores: pd.Series,
        hedge_ratio: float = 1.0,
        ts: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Run mean-reversion backtest.
        
        Strategy:
        - Enter long spread (short price1, long price2) when z-score > entry_threshold
        - Exit when z-score < exit_threshold
        - PnL = (exit_spread - entry_spread) where spread = price1 - hedge_ratio * price2
        
        Args:
            prices1: First price series
            prices2: Second price series
            zscores: Z-score series
            hedge_ratio: Hedge ratio for spread calculation
            ts: Optional timestamp series
            
        Returns:
            Dictionary with backtest results
        """
        if prices1.empty or prices2.empty or zscores.empty:
            raise AnalyticsError("Empty series")
        
        if len(prices1) != len(prices2) or len(prices1) != len(zscores):
            raise AnalyticsError("Series must have equal length")
        
        try:
            aligned = pd.DataFrame({
                "price1": prices1,
                "price2": prices2,
                "zscore": zscores,
            }).dropna()
            
            if ts is not None and len(ts) == len(aligned):
                aligned["ts"] = ts.values[:len(aligned)]
            else:
                aligned["ts"] = aligned.index
            
            if len(aligned) < 2:
                raise AnalyticsError("Insufficient data for backtest")
            
            aligned["spread"] = aligned["price1"] - hedge_ratio * aligned["price2"]
            
            trades: List[Trade] = []
            current_trade: Optional[Trade] = None
            
            for i in range(len(aligned)):
                row = aligned.iloc[i]
                zscore = row["zscore"]
                timestamp = row["ts"]
                price1 = row["price1"]
                price2 = row["price2"]
                spread = row["spread"]
                
                if current_trade is None:
                    if zscore > self.entry_threshold:
                        current_trade = Trade(
                            entry_time=timestamp,
                            exit_time=None,
                            entry_zscore=zscore,
                            exit_zscore=None,
                            entry_price1=price1,
                            entry_price2=price2,
                            exit_price1=None,
                            exit_price2=None,
                            pnl=None,
                            status="open"
                        )
                        self.logger.debug(f"Trade opened at z={zscore:.2f}")
                
                elif current_trade.status == "open":
                    if zscore < self.exit_threshold:
                        exit_spread = spread
                        entry_spread = current_trade.entry_price1 - hedge_ratio * current_trade.entry_price2
                        
                        pnl = -(exit_spread - entry_spread)
                        
                        current_trade.exit_time = timestamp
                        current_trade.exit_zscore = zscore
                        current_trade.exit_price1 = price1
                        current_trade.exit_price2 = price2
                        current_trade.pnl = pnl
                        current_trade.status = "closed"
                        
                        trades.append(current_trade)
                        self.logger.debug(f"Trade closed at z={zscore:.2f}, PnL={pnl:.2f}")
                        current_trade = None
            
            if current_trade and current_trade.status == "open":
                last_row = aligned.iloc[-1]
                exit_spread = last_row["spread"]
                entry_spread = current_trade.entry_price1 - hedge_ratio * current_trade.entry_price2
                pnl = -(exit_spread - entry_spread)
                
                current_trade.exit_time = last_row["ts"]
                current_trade.exit_zscore = last_row["zscore"]
                current_trade.exit_price1 = last_row["price1"]
                current_trade.exit_price2 = last_row["price2"]
                current_trade.pnl = pnl
                current_trade.status = "closed"
                
                trades.append(current_trade)
            
            closed_trades = [t for t in trades if t.status == "closed"]
            
            if not closed_trades:
                return {
                    "total_trades": 0,
                    "closed_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_pnl": 0.0,
                    "average_pnl": 0.0,
                    "win_rate": 0.0,
                    "max_drawdown": 0.0,
                    "trades": [],
                }
            
            pnls = [t.pnl for t in closed_trades if t.pnl is not None]
            winning_trades = len([p for p in pnls if p > 0])
            losing_trades = len([p for p in pnls if p < 0])
            
            total_pnl = sum(pnls)
            average_pnl = np.mean(pnls) if pnls else 0.0
            win_rate = winning_trades / len(closed_trades) if closed_trades else 0.0
            
            cumulative_pnl = np.cumsum(pnls)
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = running_max - cumulative_pnl
            max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            
            return {
                "total_trades": len(trades),
                "closed_trades": len(closed_trades),
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_pnl": float(total_pnl),
                "average_pnl": float(average_pnl),
                "win_rate": float(win_rate),
                "max_drawdown": float(max_drawdown),
                "sharpe_ratio": self._calculate_sharpe_ratio(pnls) if len(pnls) > 1 else 0.0,
                "trades": [
                    {
                        "entry_time": t.entry_time,
                        "exit_time": t.exit_time,
                        "entry_zscore": t.entry_zscore,
                        "exit_zscore": t.exit_zscore,
                        "pnl": t.pnl,
                        "status": t.status,
                    }
                    for t in trades
                ],
            }
        except Exception as e:
            self.logger.error(f"Failed to run backtest: {e}")
            raise AnalyticsError(f"Backtest failed: {e}") from e
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        
        if std_return == 0:
            return 0.0
        
        return float(mean_return / std_return)

