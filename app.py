"""Main entry point for market analytics system."""

import asyncio
import signal
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional
import threading

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from connectors.base import BaseConnector
from connectors.binance_ws import BinanceWebSocketConnector
from connectors.ndjson_file import NDJSONFileConnector
from storage.sqlite_adapter import SQLiteAdapter
from resampler.candle_resampler import CandleResampler
from utils.logger import setup_logger
from utils.config import Config
from utils.exceptions import ConnectorError, StorageError

logger = setup_logger("app")


class MarketAnalyticsEngine:
    """Main engine for market analytics system."""
    
    def __init__(self, config: Config):
        """
        Initialize market analytics engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.storage = SQLiteAdapter(config.db_path)
        self.resampler = CandleResampler(timeframes=config.supported_timeframes)
        self.connector: Optional[BaseConnector] = None
        self.running = False
        
        self.tick_buffer = []
        self.batch_size = 100
        
        logger.info("Market Analytics Engine initialized")
    
    def setup_connector(self, connector_type: str = "file", **kwargs) -> None:
        """
        Setup data connector.
        
        Args:
            connector_type: Type of connector ('binance' or 'file')
            **kwargs: Additional connector parameters
        """
        if connector_type == "binance":
            symbols = kwargs.get("symbols", self.config.binance_symbols)
            url = kwargs.get("url", self.config.binance_ws_url)
            self.connector = BinanceWebSocketConnector(symbols=symbols, url=url)
            logger.info(f"Using Binance WebSocket connector for symbols: {symbols}")
        elif connector_type == "file":
            file_path = kwargs.get("file_path", f"{self.config.data_dir}/ticks_example.ndjson")
            symbols = kwargs.get("symbols", None)
            replay_speed = kwargs.get("replay_speed", 1.0)
            self.connector = NDJSONFileConnector(
                file_path=file_path,
                symbols=symbols,
                replay_speed=replay_speed
            )
            logger.info(f"Using NDJSON file connector: {file_path}")
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")
    
    async def process_tick(self, tick: dict) -> None:
        """
        Process a single tick.
        
        Args:
            tick: Normalized tick dictionary
        """
        try:
            self.tick_buffer.append(tick)
            
            await self.storage.insert_tick(tick)
            
            if len(self.tick_buffer) >= self.batch_size:
                await self.resample_buffer()
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    async def resample_buffer(self) -> None:
        """Resample tick buffer into candles."""
        if not self.tick_buffer:
            return
        
        try:
            from collections import defaultdict
            ticks_by_symbol = defaultdict(list)
            
            for tick in self.tick_buffer:
                ticks_by_symbol[tick["symbol"]].append(tick)
            
            candles_batch = []
            
            for symbol, ticks in ticks_by_symbol.items():
                for timeframe in self.config.supported_timeframes:
                    candles = self.resampler.resample_ticks(ticks, timeframe, symbol)
                    candles_batch.extend(candles)
            
            if candles_batch:
                await self.storage.insert_candles_batch(candles_batch)
                logger.debug(f"Resampled {len(self.tick_buffer)} ticks into {len(candles_batch)} candles")
            
            self.tick_buffer = []
            
        except Exception as e:
            logger.error(f"Error resampling buffer: {e}")
    
    async def run(self) -> None:
        """Run the main data ingestion loop."""
        if not self.connector:
            raise RuntimeError("Connector not set up. Call setup_connector() first.")
        
        logger.info("Starting data ingestion...")
        self.running = True
        
        try:
            await self.connector.connect()
            
            async for tick in self.connector.stream_ticks():
                if not self.running:
                    break
                
                await self.process_tick(tick)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            raise
        finally:
            if self.tick_buffer:
                await self.resample_buffer()
            
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.running = False
        
        if self.connector:
            await self.connector.disconnect()
        
        self.storage.close()
        logger.info("Engine shutdown complete")


def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    import os
    
    os.environ.setdefault("UVICORN_HOST", host)
    os.environ.setdefault("UVICORN_PORT", str(port))
    logger.info(f"Starting API server on {host}:{port}")
    try:
        uvicorn.run(
            "api.server:app",
            host=host,
            port=port,
            log_level="info",
            reload=False,
            loop="asyncio"
        )
    except Exception as e:
        logger.error(f"API server failed to start: {e}")
        raise


def run_frontend(port: int = 8501):
    """Run the Streamlit frontend."""
    import subprocess
    import os
    logger.info(f"Starting Streamlit frontend on port {port}")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.join(PROJECT_ROOT, "frontend", "dashboard.py"),
        "--server.port", str(port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--logger.level", "warning"
    ]
    
    process = None
    try:
        process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        logger.info(f"Streamlit process started (PID: {process.pid})")
        process.wait()
        logger.info("Streamlit frontend exited")
    except KeyboardInterrupt:
        logger.info("Streamlit frontend stopped by user")
        if process:
            process.terminate()
            process.wait(timeout=5)
    except Exception as e:
        logger.error(f"Streamlit frontend error: {e}")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except:
                pass
        raise


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Market Analytics System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py --mode api

  python app.py --mode frontend

  python app.py --mode full

  python app.py --mode api --connector binance --symbols btcusdt ethusdt

  python app.py --mode api --connector file --file sample_data/ticks_example.ndjson
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["api", "frontend", "full", "ingest"],
        default="full",
        help="Run mode: 'api' (API only), 'frontend' (frontend only), 'full' (API + frontend), 'ingest' (data ingestion only)"
    )
    
    parser.add_argument(
        "--connector",
        choices=["binance", "file"],
        default=None,
        help="Connector type for data ingestion (only used with --mode ingest)"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        default="sample_data/ticks_example.ndjson",
        help="NDJSON file path (for file connector)"
    )
    
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Trading symbols (e.g., btcusdt ethusdt)"
    )
    
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (for file connector)"
    )
    
    parser.add_argument(
        "--api-host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port"
    )
    
    parser.add_argument(
        "--frontend-port",
        type=int,
        default=8501,
        help="Frontend server port"
    )
    
    args = parser.parse_args()
    
    config = Config.from_env()
    
    if args.symbols:
        config.binance_symbols = args.symbols
    
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("exports").mkdir(exist_ok=True)
    Path("sample_data").mkdir(exist_ok=True)
    
    if args.mode == "ingest":
        if not args.connector:
            logger.error("--connector required for ingest mode")
            sys.exit(1)
        
        engine = MarketAnalyticsEngine(config)
        
        if args.connector == "binance":
            engine.setup_connector("binance", symbols=config.binance_symbols)
        else:
            engine.setup_connector("file", file_path=args.file, replay_speed=args.replay_speed)
        
        def signal_handler(sig, frame):
            logger.info("Interrupt received, shutting down...")
            engine.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            await engine.run()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            sys.exit(1)
    
    elif args.mode == "api":
        run_api_server(host=args.api_host, port=args.api_port)
    
    elif args.mode == "frontend":
        run_frontend(port=args.frontend_port)
    
    elif args.mode == "full":
        logger.info("Starting in full mode (API + Frontend)")
        
        api_thread = threading.Thread(
            target=run_api_server,
            args=(args.api_host, args.api_port),
            daemon=True
        )
        api_thread.start()
        
        import time
        time.sleep(3)
        
        try:
            import requests
            response = requests.get(f"http://localhost:{args.api_port}/health", timeout=2)
            if response.status_code == 200:
                logger.info("✓ API server is ready")
            else:
                logger.warning("API server may not be fully ready")
        except:
            logger.warning("Could not verify API health, continuing anyway...")
        
        logger.info("Starting Streamlit frontend...")
        run_frontend(port=args.frontend_port)


if __name__ == "__main__":
    asyncio.run(main())
