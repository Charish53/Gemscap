#!/bin/bash
# Startup script for Market Analytics System
# Run this to start the application

set -e

echo "=== Market Analytics System Startup ==="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if API server is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  API server is already running on port 8000"
    echo "   Stop it first or use different ports"
fi

if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "⚠️  Frontend server is already running on port 8501"
    echo "   Stop it first or use different ports"
fi

echo ""
echo "Starting application in full mode (API + Frontend)..."
echo "  - API: http://localhost:8000"
echo "  - Frontend: http://localhost:8501"
echo "  - Data Collector: Open data_collector.html in your browser"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the application
python3 app.py --mode full

