#!/bin/bash
# Setup script for Market Analytics System
# Run this after cloning the repository

set -e

echo "=== Market Analytics System Setup ==="
echo ""

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found: $PYTHON_CMD version $PYTHON_VERSION"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
    echo ""
else
    echo "Virtual environment already exists."
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Could not find virtual environment activation script"
    exit 1
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Dependencies installed"
echo ""

# Create required directories
echo "Creating required directories..."
mkdir -p data logs exports sample_data
echo "✓ Directories created"
echo ""

# Verify installation
echo "Verifying installation..."
$PYTHON_CMD -c "import fastapi, streamlit, pandas, numpy, sklearn, statsmodels; print('✓ Core packages verified')" 2>/dev/null || echo "⚠ Some packages may need verification"
echo ""

echo "=== Setup Complete ==="
echo ""
echo "To run the application:"
echo "  ./start.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python3 app.py --mode full"
echo ""
echo "Then open data_collector.html in your browser to start collecting data."

