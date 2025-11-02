"""Quick script to view database contents."""

import sqlite3
import pandas as pd
from datetime import datetime

db_path = "data/market_data.db"

conn = sqlite3.connect(db_path)

print("=" * 80)
print("DATABASE TABLES")
print("=" * 80)
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
for table in tables:
    print(f"  - {table[0]}")

print("\n" + "=" * 80)
print("TICKS SUMMARY")
print("=" * 80)
ticks_df = pd.read_sql_query("SELECT * FROM ticks LIMIT 0", conn)
total_ticks = pd.read_sql_query("SELECT COUNT(*) as count FROM ticks", conn).iloc[0]['count']
symbols = pd.read_sql_query("SELECT DISTINCT symbol FROM ticks ORDER BY symbol", conn)

print(f"Total ticks: {total_ticks}")
print(f"Symbols: {', '.join(symbols['symbol'].tolist()) if not symbols.empty else 'None'}")

if total_ticks > 0:
    tick_cols = pd.read_sql_query("PRAGMA table_info(ticks)", conn)
    print("\nTicks table columns:")
    print(tick_cols.to_string(index=False))
    
    latest_ticks = pd.read_sql_query("""
        SELECT symbol, ts, price, size, 
               datetime(ts/1000, 'unixepoch') as dt
        FROM ticks 
        ORDER BY ts DESC 
        LIMIT 10
    """, conn)
    print("\nLatest 10 ticks:")
    print(latest_ticks.to_string(index=False))
    
    time_range = pd.read_sql_query("""
        SELECT 
            datetime(MIN(ts)/1000, 'unixepoch') as earliest,
            datetime(MAX(ts)/1000, 'unixepoch') as latest
        FROM ticks
    """, conn)
    print("\nTime range:")
    print(time_range.to_string(index=False))

print("\n" + "=" * 80)
print("CANDLES SUMMARY")
print("=" * 80)
total_candles = pd.read_sql_query("SELECT COUNT(*) as count FROM candles", conn).iloc[0]['count']
print(f"Total candles: {total_candles}")

if total_candles > 0:
    candle_summary = pd.read_sql_query("""
        SELECT symbol, timeframe, COUNT(*) as count,
               datetime(MIN(ts)/1000, 'unixepoch') as earliest,
               datetime(MAX(ts)/1000, 'unixepoch') as latest
        FROM candles 
        GROUP BY symbol, timeframe
        ORDER BY latest DESC
    """, conn)
    print("\nCandles by symbol/timeframe:")
    print(candle_summary.to_string(index=False))
    
    latest_candles = pd.read_sql_query("""
        SELECT symbol, timeframe, ts, open, high, low, close, volume,
               datetime(ts/1000, 'unixepoch') as dt
        FROM candles 
        ORDER BY ts DESC 
        LIMIT 10
    """, conn)
    print("\nLatest 10 candles:")
    print(latest_candles.to_string(index=False))

print("\n" + "=" * 80)
print("DATA FRESHNESS CHECK")
print("=" * 80)
now_ms = int(datetime.now().timestamp() * 1000)

if total_ticks > 0:
    latest_tick_ts = pd.read_sql_query("SELECT MAX(ts) as max_ts FROM ticks", conn).iloc[0]['max_ts']
    tick_age_seconds = (now_ms - latest_tick_ts) / 1000
    latest_tick_time = datetime.fromtimestamp(latest_tick_ts / 1000)
    print(f"Latest tick: {latest_tick_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(tick_age_seconds)}s ago)")

if total_candles > 0:
    latest_candle_ts = pd.read_sql_query("SELECT MAX(ts) as max_ts FROM candles WHERE timeframe='1m'", conn).iloc[0]['max_ts']
    candle_age_seconds = (now_ms - latest_candle_ts) / 1000
    latest_candle_time = datetime.fromtimestamp(latest_candle_ts / 1000)
    print(f"Latest 1m candle: {latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')} ({int(candle_age_seconds)}s ago)")

conn.close()

