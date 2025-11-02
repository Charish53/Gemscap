"""SQLite storage adapter for ticks and candles."""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.logger import setup_logger
from utils.exceptions import StorageError


class SQLiteAdapter:
    """SQLite adapter for persisting ticks and candles."""
    
    def __init__(self, db_path: str):
        """
        Initialize SQLite adapter.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(f"{self.__class__.__name__}")
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                ts TIMESTAMP NOT NULL,
                price REAL NOT NULL,
                size REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts ON ticks(symbol, ts)
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                ts TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, ts)
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_candles_symbol_timeframe_ts ON candles(symbol, timeframe, ts)
        """)
        
        conn.commit()
        conn.close()
        self.logger.info(f"Database initialized: {self.db_path}")
    
    async def insert_tick(self, tick: Dict[str, Any]) -> None:
        """
        Insert a single tick asynchronously.
        
        Args:
            tick: Tick dictionary with symbol, timestamp, price, size
        """
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._insert_tick_sync,
            tick
        )
    
    def _insert_tick_sync(self, tick: Dict[str, Any]) -> None:
        """Synchronous tick insertion."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ticks (symbol, ts, price, size)
                VALUES (?, ?, ?, ?)
            """, (
                tick["symbol"],
                tick["timestamp"],
                tick["price"],
                tick["size"],
            ))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert tick: {e}")
            raise StorageError(f"Failed to insert tick: {e}") from e
    
    async def insert_ticks_batch(self, ticks: List[Dict[str, Any]]) -> None:
        """
        Insert multiple ticks in batch.
        
        Args:
            ticks: List of tick dictionaries
        """
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._insert_ticks_batch_sync,
            ticks
        )
    
    def _insert_ticks_batch_sync(self, ticks: List[Dict[str, Any]]) -> None:
        """Synchronous batch tick insertion."""
        if not ticks:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.executemany("""
                INSERT INTO ticks (symbol, ts, price, size)
                VALUES (?, ?, ?, ?)
            """, [
                (tick["symbol"], tick["timestamp"], tick["price"], tick["size"])
                for tick in ticks
            ])
            
            conn.commit()
            conn.close()
            self.logger.debug(f"Inserted {len(ticks)} ticks in batch")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert ticks batch: {e}")
            raise StorageError(f"Failed to insert ticks batch: {e}") from e
    
    async def insert_candle(self, candle: Dict[str, Any]) -> None:
        """
        Insert or update a candle.
        
        Args:
            candle: Candle dictionary with symbol, timeframe, ts, open, high, low, close, volume
        """
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._insert_candle_sync,
            candle
        )
    
    def _insert_candle_sync(self, candle: Dict[str, Any]) -> None:
        """Synchronous candle insertion."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candle["symbol"],
                candle["timeframe"],
                candle["ts"],
                candle["open"],
                candle["high"],
                candle["low"],
                candle["close"],
                candle["volume"],
            ))
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert candle: {e}")
            raise StorageError(f"Failed to insert candle: {e}") from e
    
    async def insert_candles_batch(self, candles: List[Dict[str, Any]]) -> None:
        """
        Insert multiple candles in batch.
        
        Args:
            candles: List of candle dictionaries
        """
        await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._insert_candles_batch_sync,
            candles
        )
    
    def _insert_candles_batch_sync(self, candles: List[Dict[str, Any]]) -> None:
        """Synchronous batch candle insertion."""
        if not candles:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.executemany("""
                INSERT OR REPLACE INTO candles 
                (symbol, timeframe, ts, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c["symbol"],
                    c["timeframe"],
                    c["ts"],
                    c["open"],
                    c["high"],
                    c["low"],
                    c["close"],
                    c["volume"],
                )
                for c in candles
            ])
            
            conn.commit()
            conn.close()
            self.logger.debug(f"Inserted {len(candles)} candles in batch")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert candles batch: {e}")
            raise StorageError(f"Failed to insert candles batch: {e}") from e
    
    def get_ticks(
        self,
        symbol: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query ticks from database.
        
        Args:
            symbol: Optional symbol filter
            start_ts: Optional start timestamp
            end_ts: Optional end timestamp
            limit: Optional result limit
            
        Returns:
            DataFrame with tick data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT symbol, ts, price, size FROM ticks WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_ts:
                query += " AND ts >= ?"
                params.append(start_ts)
            
            if end_ts:
                query += " AND ts <= ?"
                params.append(end_ts)
            
            query += " ORDER BY ts ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query ticks: {e}")
            raise StorageError(f"Failed to query ticks: {e}") from e
    
    def get_candles(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query candles from database.
        
        Args:
            symbol: Optional symbol filter
            timeframe: Optional timeframe filter
            start_ts: Optional start timestamp
            end_ts: Optional end timestamp
            limit: Optional result limit
            
        Returns:
            DataFrame with candle data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT symbol, timeframe, ts, open, high, low, close, volume FROM candles WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe)
            
            if start_ts:
                query += " AND ts >= ?"
                params.append(start_ts)
            
            if end_ts:
                query += " AND ts <= ?"
                params.append(end_ts)
            
            if limit:
                query += " ORDER BY ts DESC"
                query += " LIMIT ?"
                params.append(limit)
            else:
                query += " ORDER BY ts ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if not df.empty and limit:
                df = df.iloc[::-1].reset_index(drop=True)
                self.logger.debug(f"Returned {len(df)} candles, timestamp range: {df['ts'].min()} to {df['ts'].max()}")
            
            return df
        except sqlite3.Error as e:
            self.logger.error(f"Failed to query candles: {e}")
            raise StorageError(f"Failed to query candles: {e}") from e
    
    def close(self) -> None:
        """Close database connection and cleanup."""
        self._executor.shutdown(wait=True)
        self.logger.info("Storage adapter closed")

