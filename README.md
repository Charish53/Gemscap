# Market Analytics System

Real-time cryptocurrency market analytics platform with live data ingestion, quantitative analytics, and interactive visualization dashboard.

## Features

- **Real-time Data Ingestion**: Live tick data from Binance WebSocket streams
- **Data Storage**: SQLite database for ticks and resampled OHLCV candles
- **Quantitative Analytics**:
  - Price statistics (rolling mean, std, min, max)
  - OLS Regression for hedge ratio calculation
  - Spread and Z-score calculation
  - Augmented Dickey-Fuller (ADF) test for stationarity
  - Rolling correlation analysis
  - Dynamic hedge ratio via Kalman Filter
  - Robust regression (Huber, Theil-Sen)
  - Mean-reversion backtesting
  - Liquidity metrics analysis
- **Interactive Dashboard**: Streamlit-based frontend with real-time updates
- **Custom Alerts**: Rule-based alerting system
- **Data Export**: CSV export functionality

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Binance   │────▶│  Data       │────▶│  SQLite    │
│  WebSocket  │     │  Collector  │     │  Database  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  Resampler   │
                    │  (1s,1m,5m)  │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐     ┌─────────────┐
                    │   Analytics  │────▶│  FastAPI    │
                    │   Engine     │     │  Server     │
                    └──────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌──────────────┐     ┌─────────────┐
                    │  Streamlit   │◀────│  REST API   │
                    │  Dashboard   │     └─────────────┘
                    └──────────────┘
```

## Requirements

- Python 3.10+
- pip

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Charish53/Gemscap.git
cd Gemscap
```

2. **Set up virtual environment:**
```bash
chmod +x setup.sh
./setup.sh
```

Or manually:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
chmod +x start.sh
./start.sh
```

Or manually:
```bash
source venv/bin/activate
python3 app.py --mode full
```

This starts:
- **API Server**: http://localhost:8000
- **Frontend Dashboard**: http://localhost:8501

### Start Data Collection

1. Open `data_collector.html` in your browser
2. Enter symbols (e.g., `btcusdt,ethusdt`)
3. Click "Start" to begin collecting data
4. Click "Send to API" to send buffered ticks to the API

### Running Modes

```bash
# Full mode (API + Frontend)
python3 app.py --mode full

# API only
python3 app.py --mode api

# Frontend only
python3 app.py --mode frontend

# Data ingestion only
python3 app.py --mode ingest --connector binance --symbols btcusdt ethusdt
```

## API Endpoints

### Health Check
- `GET /health` - Health check endpoint

### Data Endpoints
- `GET /api/ticks` - Get tick data
- `GET /api/candles` - Get OHLCV candles
- `POST /api/ingest/ticks` - Ingest tick data from browser
- `POST /api/upload/ohlc` - Upload OHLC CSV file

### Analytics Endpoints
- `GET /api/analytics/price-stats` - Price statistics
- `GET /api/analytics/hedge-ratio` - Hedge ratio calculation
- `GET /api/analytics/spread` - Spread calculation
- `GET /api/analytics/zscore` - Z-score series
- `GET /api/analytics/rolling-correlation` - Rolling correlation
- `GET /api/analytics/adf-test` - ADF test for stationarity
- `GET /api/analytics/kalman-hedge` - Dynamic hedge ratio (Kalman Filter)
- `GET /api/analytics/robust-hedge` - Robust regression hedge ratio
- `GET /api/analytics/backtest` - Mean-reversion backtest
- `GET /api/analytics/liquidity` - Liquidity metrics

### Alerts
- `GET /api/alerts/rules` - Get all alert rules
- `POST /api/alerts/rules` - Create alert rule
- `DELETE /api/alerts/rules/{rule_id}` - Delete alert rule
- `GET /api/alerts` - Get alert history
- `POST /api/alerts/evaluate` - Evaluate alerts

## Dashboard Features

- **Price Charts**: Interactive price and volume charts with zoom, pan, and hover
- **Spread & Z-Score**: Real-time spread and z-score visualization
- **Correlation Analysis**: Rolling correlation between symbols
- **ADF Test**: Stationarity testing interface
- **Alert Management**: Create and manage custom alerts
- **Data Upload**: Upload OHLC CSV files
- **Time-Series Stats**: Comprehensive statistics table
- **Advanced Analytics**: Kalman Filter, Robust Regression, Backtesting, Liquidity Metrics

## Configuration

Default configuration in `utils/config.py`:
- Database: `data/market_data.db`
- Supported timeframes: `1s`, `1m`, `5m`
- Default symbols: `btcusdt`, `ethusdt`
- API port: `8000`
- Frontend port: `8501`

Environment variables:
- `DB_PATH`: Database path
- `API_HOST`: API server host (default: `0.0.0.0`)
- `API_PORT`: API server port (default: `8000`)
- `BINANCE_SYMBOLS`: Comma-separated symbols

## Project Structure

```
Gemscap/
├── alerts/          # Alert rule engine
├── analytics/       # Analytics modules
├── api/             # FastAPI server
├── connectors/      # Data connectors (Binance, File)
├── frontend/        # Streamlit dashboard
├── resampler/       # OHLCV resampling
├── storage/         # Database adapters
├── utils/           # Utilities (config, logger, exceptions)
├── data/            # Database files (gitignored)
├── logs/            # Log files (gitignored)
├── exports/         # Exported CSV files (gitignored)
├── app.py           # Main entry point
├── requirements.txt # Python dependencies
├── setup.sh         # Setup script
├── start.sh         # Start script
└── data_collector.html # Browser-based data collector
```

## Dependencies

- **fastapi**: Web framework for API
- **streamlit**: Frontend framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning (regression)
- **statsmodels**: Statistical models (ADF test)
- **filterpy**: Kalman Filter implementation
- **plotly**: Interactive charts

## Development

```bash
# Check status
python3 check_status.py

# View database
python3 view_db.py

# Test API endpoints
python3 app.py --mode api
# Then visit: http://localhost:8000/docs for API documentation
```

## License

MIT License

## Author

Charish53

