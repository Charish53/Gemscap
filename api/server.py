"""FastAPI server for market analytics API."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import json
import io

from storage.sqlite_adapter import SQLiteAdapter
from analytics.price_stats import PriceStats
from analytics.regression import HedgeRatioCalculator
from analytics.spread import SpreadCalculator
from analytics.zscore import ZScoreCalculator
from analytics.adf_test import ADFTester
from analytics.rolling_corr import RollingCorrelation
from analytics.kalman_filter import KalmanFilterHedge
from analytics.robust_regression import RobustRegression
from analytics.backtest import MeanReversionBacktest
from analytics.liquidity import LiquidityAnalyzer
from alerts.rule_engine import RuleEngine
from utils.logger import setup_logger
from utils.config import Config

app = FastAPI(title="Market Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = setup_logger("API")
config = Config()
storage = SQLiteAdapter(config.db_path)

price_stats = PriceStats()
hedge_calc = HedgeRatioCalculator()
spread_calc = SpreadCalculator()
zscore_calc = ZScoreCalculator()
adf_tester = ADFTester()
rolling_corr = RollingCorrelation()
kalman_hedge = KalmanFilterHedge()
huber_reg = RobustRegression(method="huber")
theilsen_reg = RobustRegression(method="theilsen")
backtest = MeanReversionBacktest()
liquidity_analyzer = LiquidityAnalyzer()

rule_engine = RuleEngine()

active_connections: List[WebSocket] = []


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    logger.info("API server starting up...")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    storage.close()
    logger.info("API server shutting down...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Market Analytics API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": int(datetime.now().timestamp() * 1000)}


@app.get("/api/ticks")
async def get_ticks(
    symbol: Optional[str] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = 1000
):
    """Get ticks from database."""
    try:
        df = storage.get_ticks(symbol=symbol, start_ts=start_ts, end_ts=end_ts, limit=limit)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get ticks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/candles")
async def get_candles(
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    limit: Optional[int] = 1000
):
    """Get candles from database."""
    try:
        df = storage.get_candles(
            symbol=symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit
        )
        if not df.empty:
            df = df.sort_values("ts").reset_index(drop=True)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to get candles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/price-stats")
async def get_price_stats(
    symbol: str,
    timeframe: str = "1m",
    window: int = 60
):
    """Get rolling price statistics."""
    try:
        df = storage.get_candles(symbol=symbol, timeframe=timeframe, limit=1000)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for symbol {symbol}")
        
        prices = df["close"]
        stats = price_stats.calculate(prices, window=window)
        
        return stats
    except Exception as e:
        logger.error(f"Failed to calculate price stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/hedge-ratio")
async def get_hedge_ratio(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    window: int = 100
):
    """Get hedge ratio between two symbols."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=1000)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=1000)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        result = hedge_calc.calculate(prices1, prices2, window=window)
        
        return result
    except Exception as e:
        logger.error(f"Failed to calculate hedge ratio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/spread")
async def get_spread(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    hedge_ratio: Optional[float] = None,
    limit: int = 1000
):
    """Get spread series between two symbols."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        if hedge_ratio is None:
            hedge_result = hedge_calc.calculate(prices1, prices2)
            hedge_ratio = hedge_result["hedge_ratio"]
        
        spread_series = spread_calc.calculate(prices1, prices2, hedge_ratio)
        spread_series = spread_series.dropna()
        
        if spread_series.empty:
            raise HTTPException(status_code=400, detail="Spread calculation resulted in no valid data")
        
        valid_indices = spread_series.index
        valid_ts = merged.loc[valid_indices, "ts"] if len(valid_indices) > 0 else pd.Series([], dtype=int)
        
        result = pd.DataFrame({
            "ts": valid_ts.values,
            "spread": spread_series.values,
        }).to_dict(orient="records")
        
        return result
    except Exception as e:
        logger.error(f"Failed to calculate spread: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/zscore")
async def get_zscore(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    window: int = 60,
    limit: int = 1000
):
    """Get Z-score series."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        hedge_result = hedge_calc.calculate(prices1, prices2)
        spread_series = spread_calc.calculate(prices1, prices2, hedge_result["hedge_ratio"])
        spread_series = spread_series.dropna()
        
        if spread_series.empty:
            raise HTTPException(status_code=400, detail="Spread calculation resulted in no valid data for Z-score")
        
        zscore_series = zscore_calc.calculate(spread_series, window=window)
        zscore_series = zscore_series.dropna()
        
        if zscore_series.empty:
            raise HTTPException(status_code=400, detail="Z-score calculation resulted in no valid data")
        
        valid_indices = zscore_series.index
        valid_ts = merged.loc[valid_indices, "ts"] if len(valid_indices) > 0 else pd.Series([], dtype=int)
        
        result = pd.DataFrame({
            "ts": valid_ts.values,
            "zscore": zscore_series.values,
        }).to_dict(orient="records")
        
        return result
    except Exception as e:
        logger.error(f"Failed to calculate Z-score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/rolling-correlation")
async def get_rolling_correlation(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    window: int = 60,
    limit: int = 1000
):
    """Get rolling correlation between two symbols."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        available_points = len(merged)
        if window > available_points:
            adjusted_window = max(2, available_points)
            if adjusted_window < 2:
                return []
        else:
            adjusted_window = window
        
        corr_series = rolling_corr.calculate(prices1, prices2, window=adjusted_window)
        
        if corr_series.empty:
            return []
        
        result_df = pd.DataFrame({"ts": merged["ts"].values})
        
        if len(corr_series) <= len(result_df):
            corr_values = corr_series.values
            if len(corr_values) < len(result_df):
                padding = len(result_df) - len(corr_values)
                corr_values = np.concatenate([np.full(padding, np.nan), corr_values])
            result_df["correlation"] = corr_values[:len(result_df)]
        else:
            result_df["correlation"] = corr_series.values[:len(result_df)]
        
        result = result_df.dropna().to_dict(orient="records")
        
        return result
    except Exception as e:
        logger.error(f"Failed to calculate rolling correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/adf-test")
@app.post("/api/analytics/adf-test")
async def run_adf_test(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    limit: int = 1000
):
    """Run ADF test on spread."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        hedge_result = hedge_calc.calculate(prices1, prices2)
        spread_series = spread_calc.calculate(prices1, prices2, hedge_result["hedge_ratio"])
        spread_series = spread_series.dropna()
        
        if len(spread_series) < 10:
            available_count = len(spread_series)
            df1_count = len(df1)
            df2_count = len(df2)
            merged_count = len(merged)
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for ADF test: {available_count} valid observations after spread calculation (minimum 10 required). "
                       f"Available: {df1_count} candles for {symbol1}, {df2_count} candles for {symbol2}, {merged_count} overlapping timestamps. "
                       f"Please ensure both symbols have overlapping data in the same timeframe."
            )
        
        try:
            result = adf_tester.test(spread_series)
            return result
        except Exception as e:
            logger.error(f"ADF test calculation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"ADF test calculation failed: {str(e)}. Ensure spread series has valid numeric values."
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run ADF test: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run ADF test: {str(e)}")


@app.post("/api/alerts/rules")
async def create_alert_rule(
    rule_id: str,
    rule_name: str,
    metric: str,
    operator: str,
    threshold: float,
    enabled: bool = True
):
    """Create a new alert rule."""
    try:
        rule_engine.register_rule(rule_id, rule_name, metric, operator, threshold, enabled)
        return {"message": "Rule created", "rule_id": rule_id}
    except Exception as e:
        logger.error(f"Failed to create rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts/rules")
async def get_alert_rules():
    """Get all alert rules."""
    return rule_engine.get_rules()


@app.delete("/api/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str):
    """Delete an alert rule."""
    try:
        rule_engine.unregister_rule(rule_id)
        return {"message": "Rule deleted", "rule_id": rule_id}
    except Exception as e:
        logger.error(f"Failed to delete rule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/alerts")
async def get_alerts(rule_id: Optional[str] = None, limit: Optional[int] = 100):
    """Get alert history."""
    alerts = rule_engine.get_alerts(rule_id=rule_id, limit=limit)
    return [alert.__dict__ for alert in alerts]


@app.post("/api/alerts/evaluate")
async def evaluate_alerts():
    """Evaluate all rules against current metrics."""
    try:
        symbols = config.binance_symbols[:2] if len(config.binance_symbols) >= 2 else ["btcusdt", "ethusdt"]
        
        df1 = storage.get_candles(symbol=symbols[0], timeframe="1m", limit=1000)
        df2 = storage.get_candles(symbol=symbols[1], timeframe="1m", limit=1000)
        
        if df1.empty or df2.empty:
            return []
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            return []
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        hedge_result = hedge_calc.calculate(prices1, prices2)
        spread_series = spread_calc.calculate(prices1, prices2, hedge_result["hedge_ratio"])
        zscore_value = zscore_calc.get_current_zscore(spread_series)
        corr_value = rolling_corr.get_current_correlation(prices1, prices2)
        
        metrics = {
            "zscore": zscore_value,
            "spread": float(spread_series.iloc[-1]),
            "correlation": corr_value,
            "hedge_ratio": hedge_result["hedge_ratio"],
        }
        
        alerts = rule_engine.evaluate(metrics)
        
        return [alert.__dict__ for alert in alerts]
    except Exception as e:
        logger.error(f"Failed to evaluate alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest/ticks")
async def ingest_ticks(ticks: List[Dict[str, Any]]):
    """Receive tick data from browser and process in real-time."""
    try:
        from resampler.candle_resampler import CandleResampler
        from collections import defaultdict
        from dateutil import parser as date_parser
        
        resampler = CandleResampler(timeframes=config.supported_timeframes)
        
        normalized_ticks = []
        tick_buffer = defaultdict(list)
        
        for tick_data in ticks:
            try:
                if "ts" in tick_data:
                    ts_str = tick_data["ts"]
                    if isinstance(ts_str, str):
                        dt_obj = date_parser.parse(ts_str)
                        timestamp_ms = int(dt_obj.timestamp() * 1000)
                    else:
                        timestamp_ms = int(ts_str)
                else:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)
                
                tick = {
                    "symbol": tick_data.get("symbol", "").lower(),
                    "timestamp": timestamp_ms,
                    "price": float(tick_data.get("price", 0)),
                    "size": float(tick_data.get("size", 0)),
                }
                
                if not tick["symbol"] or tick["price"] <= 0 or tick["size"] <= 0:
                    continue
                
                normalized_ticks.append(tick)
                tick_buffer[tick["symbol"]].append(tick)
                
            except Exception as e:
                logger.warning(f"Failed to normalize tick: {e}")
                continue
        
        if normalized_ticks:
            await storage.insert_ticks_batch(normalized_ticks)
            
            for symbol, symbol_ticks in tick_buffer.items():
                for timeframe in config.supported_timeframes:
                    candles = resampler.resample_ticks(symbol_ticks, timeframe, symbol)
                    if candles:
                        await storage.insert_candles_batch(candles)
            
            logger.info(f"Processed {len(normalized_ticks)} ticks from browser")
        
        return {
            "status": "ok",
            "processed": len(normalized_ticks),
            "symbols": list(tick_buffer.keys())
        }
    except Exception as e:
        logger.error(f"Failed to ingest ticks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload/ohlc")
async def upload_ohlc_csv(
    file: UploadFile = File(...),
    symbol: Optional[str] = Form(None),
    timeframe: str = Form("1m")
):
    """
    Upload OHLC CSV file and import into database.
    
    CSV format should have columns: timestamp, open, high, low, close, volume
    Or: ts, open, high, low, close, volume
    Timestamp can be in milliseconds (int) or ISO format (str)
    Symbol can be provided as form field or extracted from filename
    """
    try:
        contents = await file.read()
        
        try:
            df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        df.columns = df.columns.str.lower().str.strip()
        
        if symbol is None:
            filename_lower = file.filename.lower()
            for possible_symbol in config.binance_symbols:
                if possible_symbol in filename_lower:
                    symbol = possible_symbol
                    break
            
            if symbol is None:
                if 'symbol' in df.columns:
                    symbols = df['symbol'].unique()
                    if len(symbols) == 1:
                        symbol = symbols[0]
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail="Multiple symbols found in CSV. Please specify symbol parameter."
                        )
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Symbol not found in filename or CSV. Please provide symbol parameter."
                    )
        
        symbol = symbol.lower()
        
        ts_cols = ['timestamp', 'ts', 'time', 'datetime', 'date']
        ts_col = None
        for col in ts_cols:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col is None:
            raise HTTPException(
                status_code=400,
                detail=f"Timestamp column not found. Expected one of: {', '.join(ts_cols)}"
            )
        
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_cols)}"
            )
        
        ts_series = df[ts_col]
        if ts_series.dtype == 'object' or ts_series.dtype == 'int64':
            try:
                ts_series = pd.to_numeric(ts_series, errors='coerce')
                if ts_series.max() < 1e12:
                    ts_series = ts_series * 1000
            except:
                from dateutil import parser as date_parser
                ts_series = pd.to_datetime(ts_series, errors='coerce')
                ts_series = ts_series.astype(np.int64) // 10**6
        else:
            ts_series = pd.to_datetime(ts_series, errors='coerce')
            ts_series = ts_series.astype(np.int64) // 10**6
        
        df = df[ts_series.notna()].copy()
        df['ts'] = ts_series[ts_series.notna()].astype(int)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No valid timestamps found in CSV")
        
        candles = []
        for _, row in df.iterrows():
            try:
                candle = {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "ts": int(row['ts']),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']) if pd.notna(row['volume']) else 0.0
                }
                
                if candle['open'] <= 0 or candle['high'] <= 0 or candle['low'] <= 0 or candle['close'] <= 0:
                    continue
                if candle['high'] < candle['low']:
                    continue
                if not (candle['low'] <= candle['open'] <= candle['high'] and 
                        candle['low'] <= candle['close'] <= candle['high']):
                    continue
                
                candles.append(candle)
            except Exception as e:
                logger.warning(f"Skipping invalid row: {e}")
                continue
        
        if not candles:
            raise HTTPException(status_code=400, detail="No valid candles found after validation")
        
        await storage.insert_candles_batch(candles)
        
        logger.info(f"Uploaded {len(candles)} candles for {symbol} (timeframe: {timeframe})")
        
        return {
            "status": "ok",
            "message": f"Successfully uploaded {len(candles)} candles",
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(candles),
            "first_ts": candles[0]["ts"],
            "last_ts": candles[-1]["ts"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload OHLC CSV: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload OHLC CSV: {str(e)}")


@app.post("/api/export/csv")
async def export_csv(
    data_type: str,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    filename: Optional[str] = None
):
    """Export data as CSV."""
    try:
        if data_type == "ticks":
            df = storage.get_ticks(symbol=symbol)
        elif data_type == "candles":
            df = storage.get_candles(symbol=symbol, timeframe=timeframe)
        else:
            raise HTTPException(status_code=400, detail="Invalid data_type")
        
        if filename is None:
            filename = f"{data_type}_{symbol or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        csv_path = f"exports/{filename}"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        
        return {"message": "Export successful", "filename": filename, "path": csv_path}
    except Exception as e:
        logger.error(f"Failed to export CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/api/ingest")
async def ingest_websocket(websocket: WebSocket):
    """WebSocket endpoint to receive tick data from browser and process in real-time."""
    await websocket.accept()
    logger.info("Tick ingestion WebSocket connected")
    
    from resampler.candle_resampler import CandleResampler
    from collections import defaultdict
    from dateutil import parser as date_parser
    
    resampler = CandleResampler(timeframes=config.supported_timeframes)
    tick_buffer = defaultdict(list)
    buffer_size = 50
    
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                if "ts" in data:
                    ts_str = data["ts"]
                    if isinstance(ts_str, str):
                        dt_obj = date_parser.parse(ts_str)
                        timestamp_ms = int(dt_obj.timestamp() * 1000)
                    else:
                        timestamp_ms = int(ts_str)
                else:
                    timestamp_ms = int(datetime.now().timestamp() * 1000)
                
                tick = {
                    "symbol": data.get("symbol", "").lower(),
                    "timestamp": timestamp_ms,
                    "price": float(data.get("price", 0)),
                    "size": float(data.get("size", 0)),
                }
                
                if not tick["symbol"] or tick["price"] <= 0 or tick["size"] <= 0:
                    continue
                
                symbol = tick["symbol"]
                await storage.insert_tick(tick)
                
                tick_buffer[symbol].append(tick)
                
                if len(tick_buffer[symbol]) >= buffer_size:
                    ticks_to_process = tick_buffer[symbol].copy()
                    tick_buffer[symbol] = []
                    
                    for timeframe in config.supported_timeframes:
                        candles = resampler.resample_ticks(ticks_to_process, timeframe, symbol)
                        if candles:
                            await storage.insert_candles_batch(candles)
                    
                    logger.debug(f"Processed {len(ticks_to_process)} ticks for {symbol}")
                
                await websocket.send_json({
                    "status": "ok",
                    "tick": tick,
                    "buffered": len(tick_buffer[symbol])
                })
                
            except Exception as e:
                logger.error(f"Error processing tick: {e}")
                await websocket.send_json({
                    "status": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        for symbol, ticks in tick_buffer.items():
            if ticks:
                for timeframe in config.supported_timeframes:
                    candles = resampler.resample_ticks(ticks, timeframe, symbol)
                    if candles:
                        await storage.insert_candles_batch(candles)
        logger.info("Tick ingestion WebSocket disconnected")
    except Exception as e:
        logger.error(f"Ingestion WebSocket error: {e}")


@app.get("/api/analytics/cross-correlation")
async def get_cross_correlation(
    symbols: str,
    timeframe: str = "1m",
    limit: int = 1000
):
    """Get cross-correlation matrix for multiple symbols."""
    try:
        symbol_list = [s.strip().lower() for s in symbols.split(",")]
        if len(symbol_list) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required")
        
        price_data = {}
        for symbol in symbol_list:
            df = storage.get_candles(symbol=symbol, timeframe=timeframe, limit=limit)
            if not df.empty:
                price_data[symbol] = df[["ts", "close"]].rename(columns={"close": symbol})
        
        if len(price_data) < 2:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = price_data[list(price_data.keys())[0]]
        for symbol in list(price_data.keys())[1:]:
            merged = pd.merge(merged, price_data[symbol][["ts", symbol]], on="ts", how="inner")
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        price_cols = [s for s in symbol_list if s in merged.columns]
        corr_matrix = merged[price_cols].corr()
        
        result = {
            "symbols": price_cols,
            "correlation_matrix": corr_matrix.to_dict(),
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate cross-correlation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/time-series")
async def get_analytics_time_series(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    window: int = 60,
    limit: int = 1000
):
    """Get comprehensive time-series analytics."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close", "volume"]].rename(columns={"close": f"{symbol1}_price", "volume": f"{symbol1}_volume"}),
            df2[["ts", "close", "volume"]].rename(columns={"close": f"{symbol2}_price", "volume": f"{symbol2}_volume"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged[f"{symbol1}_price"]
        prices2 = merged[f"{symbol2}_price"]
        
        hedge_result = hedge_calc.calculate(prices1, prices2)
        spread_series = spread_calc.calculate(prices1, prices2, hedge_result["hedge_ratio"])
        
        zscore_series = zscore_calc.calculate(spread_series, window=window)
        
        corr_series = rolling_corr.calculate(prices1, prices2, window=window)
        
        price_stats1 = price_stats.calculate(prices1, window=window)
        price_stats2 = price_stats.calculate(prices2, window=window)
        
        result_df = pd.DataFrame({
            "ts": merged["ts"],
            f"{symbol1}_price": prices1.values,
            f"{symbol1}_volume": merged[f"{symbol1}_volume"].values,
            f"{symbol2}_price": prices2.values,
            f"{symbol2}_volume": merged[f"{symbol2}_volume"].values,
            "spread": spread_series.values,
            "zscore": zscore_series.values if len(zscore_series) == len(merged) else [None] * len(merged),
            "correlation": corr_series.values if len(corr_series) == len(merged) else [None] * len(merged),
            "hedge_ratio": hedge_result["hedge_ratio"],
        })
        
        result_df = result_df.where(pd.notnull(result_df), None)
        
        return result_df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate time-series analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/kalman-hedge")
async def get_kalman_hedge(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    limit: int = 1000
):
    """Get dynamic hedge ratio using Kalman Filter."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        if len(merged) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for Kalman Filter: {len(merged)} overlapping observations available (minimum 10 required). "
                       f"Please ensure both symbols have overlapping data in the same timeframe."
            )
        
        result = kalman_hedge.calculate(prices1, prices2)
        
        result["ts"] = merged["ts"].tolist()[:len(result["hedge_ratio_series"])]
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate Kalman Filter hedge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate Kalman Filter hedge: {str(e)}")


@app.get("/api/analytics/robust-hedge")
async def get_robust_hedge(
    symbol1: str,
    symbol2: str,
    method: str = "huber",
    timeframe: str = "1m",
    window: int = 100
):
    """Get hedge ratio using robust regression (Huber or Theil-Sen)."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=1000)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=1000)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        if len(merged) < window:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for robust regression: {len(merged)} overlapping observations available (minimum {window} required). "
                       f"Available: {len(df1)} candles for {symbol1}, {len(df2)} candles for {symbol2}. "
                       f"Please ensure both symbols have overlapping data in the same timeframe."
            )
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        
        if method.lower() == "huber":
            reg = huber_reg
        elif method.lower() == "theilsen":
            reg = theilsen_reg
        else:
            raise HTTPException(status_code=400, detail="Method must be 'huber' or 'theilsen'")
        
        result = reg.calculate(prices1, prices2, window=window)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate robust hedge: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate robust hedge: {str(e)}")


@app.get("/api/analytics/backtest")
async def run_backtest(
    symbol1: str,
    symbol2: str,
    timeframe: str = "1m",
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.0,
    window: int = 60,
    limit: int = 1000
):
    """Run mean-reversion backtest."""
    try:
        df1 = storage.get_candles(symbol=symbol1, timeframe=timeframe, limit=limit)
        df2 = storage.get_candles(symbol=symbol2, timeframe=timeframe, limit=limit)
        
        if df1.empty or df2.empty:
            raise HTTPException(status_code=404, detail="Insufficient data")
        
        merged = pd.merge(
            df1[["ts", "close"]].rename(columns={"close": "price1"}),
            df2[["ts", "close"]].rename(columns={"close": "price2"}),
            on="ts",
            how="inner"
        )
        
        if merged.empty:
            raise HTTPException(status_code=404, detail="No overlapping data")
        
        prices1 = merged["price1"]
        prices2 = merged["price2"]
        ts = merged["ts"]
        
        hedge_result = hedge_calc.calculate(prices1, prices2)
        hedge_ratio = hedge_result["hedge_ratio"]
        
        spread_series = spread_calc.calculate(prices1, prices2, hedge_ratio)
        zscore_series = zscore_calc.calculate(spread_series, window=window)
        
        backtest_instance = MeanReversionBacktest(
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold
        )
        
        result = backtest_instance.run(prices1, prices2, zscore_series, hedge_ratio, ts)
        
        return result
    except Exception as e:
        logger.error(f"Failed to run backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics/liquidity")
async def get_liquidity_metrics(
    symbol: str,
    timeframe: str = "1m",
    window: int = 60,
    limit: int = 1000
):
    """Get liquidity metrics for a symbol."""
    try:
        df = storage.get_candles(symbol=symbol, timeframe=timeframe, limit=limit)
        
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data for symbol {symbol} in timeframe {timeframe}. "
                       f"Please ensure data is being collected for this symbol."
            )
        
        if len(df) < window:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data for liquidity metrics: {len(df)} candles available (minimum {window} required). "
                       f"Please collect more data for {symbol} in {timeframe} timeframe."
            )
        
        volume_series = df["volume"]
        price_series = df["close"]
        
        metrics = liquidity_analyzer.calculate_metrics(
            volume_series,
            price_series=price_series,
            window=window
        )
        
        spread_estimate = liquidity_analyzer.calculate_bid_ask_spread_estimate(
            df["high"],
            df["low"],
            df["close"],
            window=window
        )
        
        metrics.update(spread_estimate)
        
        return metrics
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate liquidity metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate liquidity metrics: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analytics updates."""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"WebSocket connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            await asyncio.sleep(5)
            
            symbols = config.binance_symbols[:2] if len(config.binance_symbols) >= 2 else ["btcusdt", "ethusdt"]
            
            df1 = storage.get_candles(symbol=symbols[0], timeframe="1m", limit=100)
            df2 = storage.get_candles(symbol=symbols[1], timeframe="1m", limit=100)
            
            if df1.empty or df2.empty:
                continue
            
            merged = pd.merge(
                df1[["ts", "close"]].rename(columns={"close": "price1"}),
                df2[["ts", "close"]].rename(columns={"close": "price2"}),
                on="ts",
                how="inner"
            )
            
            if merged.empty:
                continue
            
            prices1 = merged["price1"]
            prices2 = merged["price2"]
            
            hedge_result = hedge_calc.calculate(prices1, prices2)
            spread_series = spread_calc.calculate(prices1, prices2, hedge_result["hedge_ratio"])
            zscore_value = zscore_calc.get_current_zscore(spread_series)
            corr_value = rolling_corr.get_current_correlation(prices1, prices2)
            
            update = {
                "type": "metrics_update",
                "timestamp": int(datetime.now().timestamp() * 1000),
                "symbol1": symbols[0],
                "symbol2": symbols[1],
                "metrics": {
                    "zscore": zscore_value,
                    "spread": float(spread_series.iloc[-1]),
                    "correlation": corr_value,
                    "hedge_ratio": hedge_result["hedge_ratio"],
                }
            }
            
            metrics = {
                "zscore": zscore_value,
                "spread": float(spread_series.iloc[-1]),
                "correlation": corr_value,
            }
            alerts = rule_engine.evaluate(metrics)
            if alerts:
                update["alerts"] = [alert.__dict__ for alert in alerts]
            
            await websocket.send_json(update)
            
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)

