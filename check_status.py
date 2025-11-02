"""Quick status check for Market Analytics System."""

import sys
import requests
from pathlib import Path

print("=" * 60)
print("Market Analytics System - Status Check")
print("=" * 60)

db_path = Path("data/market_data.db")
if db_path.exists():
    print("✓ Database exists")
    db_size = db_path.stat().st_size / 1024
    print(f"  Size: {db_size:.2f} KB")
else:
    print("⚠ Database does not exist (will be created on first run)")

print("\nChecking API server (http://localhost:8000)...")
try:
    response = requests.get("http://localhost:8000/health", timeout=2)
    if response.status_code == 200:
        print("✓ API server is running")
        health = response.json()
        print(f"  Status: {health.get('status', 'unknown')}")
    else:
        print(f"⚠ API server responded with status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ API server is NOT running")
    print("  Start it with: ./start.sh or python3 app.py --mode full")
except Exception as e:
    print(f"✗ Error checking API: {e}")

print("\nChecking Frontend (http://localhost:8501)...")
try:
    response = requests.get("http://localhost:8501", timeout=2)
    if response.status_code == 200:
        print("✓ Frontend is running")
    else:
        print(f"⚠ Frontend responded with status {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Frontend is NOT running")
    print("  Start it with: ./start.sh or python3 app.py --mode full")
except Exception as e:
    print(f"✗ Error checking frontend: {e}")

print("\n" + "=" * 60)
print("Quick Start:")
print("  1. Run: ./start.sh")
print("  2. Open: data_collector.html in your browser")
print("  3. Click: Start button to begin collecting data")
print("  4. View dashboard at: http://localhost:8501")
print("=" * 60)

