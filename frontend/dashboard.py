"""Streamlit dashboard for market analytics with real-time updates."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, Any
import time
import threading

API_BASE_URL = "http://localhost:8000"

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False


def fetch_data(endpoint: str, params: Optional[Dict] = None, cache_bust: bool = True) -> Any:
    """Fetch data from API endpoint with cache busting."""
    try:
        if params is None:
            params = {}
        if cache_bust:
            params['_t'] = int(datetime.now().timestamp() * 1000)
        
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get('detail', str(e))
            if e.response.status_code == 400:
                st.warning(f"⚠️ {error_detail}")
            elif e.response.status_code == 404:
                st.warning(f"⚠️ {error_detail}")
            else:
                st.error(f"API Error: {error_detail}")
        except:
            st.error(f"API Error: {e}")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Is the API server running?")
        st.info("💡 Run: `python app.py --mode full` or `uvicorn api.server:app --host 0.0.0.0 --port 8000`")
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ API request timed out. The server may be busy.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


def post_data(endpoint: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None, files: Optional[Dict] = None) -> Any:
    """Post data to API endpoint."""
    try:
        if files:
            response = requests.post(f"{API_BASE_URL}{endpoint}", params=params, files=files, timeout=30)
        else:
            response = requests.post(f"{API_BASE_URL}{endpoint}", params=params, json=json_data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = e.response.json().get('detail', str(e))
            st.error(f"API Error: {error_detail}")
        except:
            st.error(f"API Error: {e}")
        return None
    except requests.exceptions.Timeout:
        st.error(f"⏱️ API request timed out. The server may be busy.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None


st.set_page_config(
    page_title="Market Analytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'realtime_metrics' not in st.session_state:
    st.session_state.realtime_metrics = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

st.title("📈 Real-Time Market Analytics Dashboard")

current_time = datetime.now()
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = current_time

st.session_state.last_update_time = current_time

st.sidebar.header("⚙️ Configuration")

if st.sidebar.button("🔍 Check API Connection"):
    health = fetch_data("/health")
    if health:
        st.sidebar.success("✅ API is connected")
    else:
        st.sidebar.error("❌ API is not responding")

symbol1 = st.sidebar.text_input("Symbol 1", value="btcusdt")
symbol2 = st.sidebar.text_input("Symbol 2", value="ethusdt")

timeframe = st.sidebar.selectbox("Timeframe", ["1s", "1m", "5m"], index=1)

window = st.sidebar.slider("Rolling Window", min_value=10, max_value=200, value=60, step=10)

data_limit = st.sidebar.slider("Data Points to Show", min_value=50, max_value=2000, value=500, step=50)

auto_refresh = st.sidebar.checkbox("Auto-refresh (Real-time)", value=True)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=60, value=30, step=1)

if auto_refresh:
    st.sidebar.success(f"🟢 Auto-refresh active ({refresh_interval}s)")
else:
    st.sidebar.warning("⚠️ Auto-refresh disabled")

st.caption(f"🕐 Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {'ON' if auto_refresh else 'OFF'} ({refresh_interval}s)")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Price Charts", "📉 Spread & Z-Score", "🔗 Correlation", "🧪 ADF Test", 
    "⚠️ Alerts", "📤 Data Upload", "📋 Time-Series Stats", "🚀 Advanced Analytics"
])

with tab1:
    st.header("Price Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{symbol1.upper()} Price")
        
        candles1 = fetch_data("/api/candles", {
            "symbol": symbol1, 
            "timeframe": timeframe, 
            "limit": data_limit,
            "_refresh": int(datetime.now().timestamp() * 1000)
        })
        
        if candles1 is None:
            st.warning(f"⚠️ Could not fetch data for {symbol1}. Check API connection.")
        elif candles1:
            df1 = pd.DataFrame(candles1)
            if not df1.empty:
                df1 = df1.sort_values("ts").reset_index(drop=True)
                
                latest_ts = df1["ts"].max()
                latest_time = pd.to_datetime(latest_ts, unit="ms")
                current_time = datetime.now()
                time_diff = (current_time - latest_time).total_seconds()
                
                if time_diff < 60:
                    status_color = "🟢"
                elif time_diff < 300:
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                
                st.caption(f"{status_color} Latest data: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({len(df1)} candles) | Age: {int(time_diff)}s")
                
                st.markdown("---")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    chart_tf = st.select_slider("Timeframe", options=["1s", "1m", "5m"], value=timeframe, key="tf1_chart")
                with col_t2:
                    use_custom_range = st.checkbox("Custom Time Range", value=False, key="range1")
                
                if chart_tf != timeframe:
                    candles1 = fetch_data("/api/candles", {
                        "symbol": symbol1, 
                        "timeframe": chart_tf, 
                        "limit": data_limit,
                        "_refresh": int(datetime.now().timestamp() * 1000)
                    })
                    if candles1:
                        df1 = pd.DataFrame(candles1)
                        df1 = df1.sort_values("ts").reset_index(drop=True)
                
                df1_filtered = df1.copy()
                if use_custom_range and not df1.empty:
                    min_time = pd.to_datetime(df1["ts"].min(), unit="ms")
                    max_time = pd.to_datetime(df1["ts"].max(), unit="ms")
                    time_range = st.slider(
                        "Select Time Range",
                        min_value=min_time,
                        max_value=max_time,
                        value=(min_time, max_time),
                        format="YYYY-MM-DD HH:mm:ss",
                        key="timerange1"
                    )
                    min_ts = int(time_range[0].timestamp() * 1000)
                    max_ts = int(time_range[1].timestamp() * 1000)
                    df1_filtered = df1[(df1["ts"] >= min_ts) & (df1["ts"] <= max_ts)].copy()
                    if df1_filtered.empty:
                        st.warning(f"⚠️ No data in selected time range. Showing all data.")
                        df1_filtered = df1.copy()
                    else:
                        st.info(f"📊 Showing {len(df1_filtered)} candles out of {len(df1)} total")
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=pd.to_datetime(df1_filtered["ts"], unit="ms"),
                    y=df1_filtered["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="blue", width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Price: $%{y:.4f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="blue", font_size=12)
                ))
                fig1.add_trace(go.Bar(
                    x=pd.to_datetime(df1_filtered["ts"], unit="ms"),
                    y=df1_filtered["volume"],
                    name="Volume",
                    yaxis="y2",
                    marker_color="rgba(0,0,255,0.2)",
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Volume: %{y:.2f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="blue", font_size=12)
                ))
                fig1.update_layout(
                    title=f"{symbol1.upper()} Price & Volume",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    yaxis2=dict(title="Volume", overlaying="y", side="right"),
                    height=450,
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            x=0, xanchor="left", y=1.02, yanchor="top"
                        ),
                        showspikes=True,
                        spikecolor="blue",
                        spikethickness=1,
                        spikedash="solid",
                        spikemode="across+toaxis"
                    ),
                    yaxis=dict(showspikes=True, spikecolor="blue", spikethickness=1, spikedash="solid"),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    dragmode='pan',
                    hoverdistance=100
                )
                st.plotly_chart(fig1, width='stretch', config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'scrollZoom': True,
                    'doubleClick': 'reset'
                })
                
                st.markdown("---")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    chart_opacity = st.slider("Line Opacity", min_value=0.1, max_value=1.0, value=1.0, step=0.1, key="op1")
                with col_s2:
                    volume_opacity = st.slider("Volume Opacity", min_value=0.1, max_value=1.0, value=0.2, step=0.1, key="vol_op1")
                with col_s3:
                    show_volume_chart = st.checkbox("Show Volume", value=True, key="show_vol1")
                
                price_stats1 = fetch_data("/api/analytics/price-stats", {
                    "symbol": symbol1,
                    "timeframe": timeframe,
                    "window": window
                })
                
                if price_stats1:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Current Price", f"${price_stats1['current']:.2f}")
                    col_b.metric("Mean", f"${price_stats1['mean']:.2f}")
                    col_c.metric("Std Dev", f"${price_stats1['std']:.2f}")
                    col_d.metric("Min/Max", f"${price_stats1['min']:.2f} / ${price_stats1['max']:.2f}")
    
    with col2:
        st.subheader(f"{symbol2.upper()} Price")
        
        candles2 = fetch_data("/api/candles", {
            "symbol": symbol2, 
            "timeframe": timeframe, 
            "limit": data_limit,
            "_refresh": int(datetime.now().timestamp() * 1000)
        })
        
        if candles2 is None:
            st.warning(f"⚠️ Could not fetch data for {symbol2}. Check API connection.")
        elif candles2:
            df2 = pd.DataFrame(candles2)
            if not df2.empty:
                df2 = df2.sort_values("ts").reset_index(drop=True)
                
                latest_ts = df2["ts"].max()
                latest_time = pd.to_datetime(latest_ts, unit="ms")
                current_time = datetime.now()
                time_diff = (current_time - latest_time).total_seconds()
                
                if time_diff < 60:
                    status_color = "🟢"
                elif time_diff < 300:
                    status_color = "🟡"
                else:
                    status_color = "🔴"
                
                st.caption(f"{status_color} Latest data: {latest_time.strftime('%Y-%m-%d %H:%M:%S')} ({len(df2)} candles) | Age: {int(time_diff)}s")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=pd.to_datetime(df2["ts"], unit="ms"),
                    y=df2["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="green", width=2)
                ))
                fig2.add_trace(go.Bar(
                    x=pd.to_datetime(df2["ts"], unit="ms"),
                    y=df2["volume"],
                    name="Volume",
                    yaxis="y2",
                    marker_color="rgba(0,255,0,0.2)"
                ))
                st.markdown("---")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    chart_tf = st.select_slider("Timeframe", options=["1s", "1m", "5m"], value=timeframe, key="tf2_chart")
                with col_t2:
                    use_custom_range = st.checkbox("Custom Time Range", value=False, key="range2")
                
                if chart_tf != timeframe:
                    candles2 = fetch_data("/api/candles", {
                        "symbol": symbol2, 
                        "timeframe": chart_tf, 
                        "limit": data_limit,
                        "_refresh": int(datetime.now().timestamp() * 1000)
                    })
                    if candles2:
                        df2 = pd.DataFrame(candles2)
                        df2 = df2.sort_values("ts").reset_index(drop=True)
                
                df2_filtered = df2.copy()
                if use_custom_range and not df2.empty:
                    min_time = pd.to_datetime(df2["ts"].min(), unit="ms")
                    max_time = pd.to_datetime(df2["ts"].max(), unit="ms")
                    time_range = st.slider(
                        "Select Time Range",
                        min_value=min_time,
                        max_value=max_time,
                        value=(min_time, max_time),
                        format="YYYY-MM-DD HH:mm:ss",
                        key="timerange2"
                    )
                    min_ts = int(time_range[0].timestamp() * 1000)
                    max_ts = int(time_range[1].timestamp() * 1000)
                    df2_filtered = df2[(df2["ts"] >= min_ts) & (df2["ts"] <= max_ts)].copy()
                    if df2_filtered.empty:
                        st.warning(f"⚠️ No data in selected time range. Showing all data.")
                        df2_filtered = df2.copy()
                    else:
                        st.info(f"📊 Showing {len(df2_filtered)} candles out of {len(df2)} total")
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=pd.to_datetime(df2_filtered["ts"], unit="ms"),
                    y=df2_filtered["close"],
                    mode="lines",
                    name="Close",
                    line=dict(color="green", width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Price: $%{y:.4f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="green", font_size=12)
                ))
                fig2.add_trace(go.Bar(
                    x=pd.to_datetime(df2_filtered["ts"], unit="ms"),
                    y=df2_filtered["volume"],
                    name="Volume",
                    yaxis="y2",
                    marker_color="rgba(0,255,0,0.2)",
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Volume: %{y:.2f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="green", font_size=12)
                ))
                fig2.update_layout(
                    title=f"{symbol2.upper()} Price & Volume",
                    xaxis_title="Time",
                    yaxis_title="Price",
                    yaxis2=dict(title="Volume", overlaying="y", side="right"),
                    height=450,
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            x=0, xanchor="left", y=1.02, yanchor="top"
                        ),
                        showspikes=True,
                        spikecolor="green",
                        spikethickness=1,
                        spikedash="solid",
                        spikemode="across+toaxis"
                    ),
                    yaxis=dict(showspikes=True, spikecolor="green", spikethickness=1, spikedash="solid"),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    dragmode='pan',
                    hoverdistance=100
                )
                st.plotly_chart(fig2, width='stretch', config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
                    'scrollZoom': True,
                    'doubleClick': 'reset'
                })
                
                st.markdown("---")
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    chart_opacity = st.slider("Line Opacity", min_value=0.1, max_value=1.0, value=1.0, step=0.1, key="op2")
                with col_s2:
                    volume_opacity = st.slider("Volume Opacity", min_value=0.1, max_value=1.0, value=0.2, step=0.1, key="vol_op2")
                with col_s3:
                    show_volume_chart = st.checkbox("Show Volume", value=True, key="show_vol2")
                
                price_stats2 = fetch_data("/api/analytics/price-stats", {
                    "symbol": symbol2,
                    "timeframe": timeframe,
                    "window": window
                })
                
                if price_stats2:
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Current Price", f"${price_stats2['current']:.2f}")
                    col_b.metric("Mean", f"${price_stats2['mean']:.2f}")
                    col_c.metric("Std Dev", f"${price_stats2['std']:.2f}")
                    col_d.metric("Min/Max", f"${price_stats2['min']:.2f} / ${price_stats2['max']:.2f}")

with tab2:
    st.header("Spread & Z-Score Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hedge Ratio")
        
        hedge_ratio = fetch_data("/api/analytics/hedge-ratio", {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": timeframe,
            "window": 100
        })
        
        if hedge_ratio:
            st.metric("Hedge Ratio (β)", f"{hedge_ratio['hedge_ratio']:.6f}")
            st.metric("Alpha (α)", f"{hedge_ratio['alpha']:.6f}")
            st.metric("R²", f"{hedge_ratio['r_squared']:.4f}")
            st.metric("Samples", hedge_ratio["n_samples"])
    
    with col2:
        st.subheader("Spread Series")
        
        spread_data = fetch_data("/api/analytics/spread", {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": timeframe,
            "limit": data_limit
        })
        
        if spread_data:
            df_spread = pd.DataFrame(spread_data)
            if not df_spread.empty:
                fig_spread = go.Figure()
                fig_spread.add_trace(go.Scatter(
                    x=pd.to_datetime(df_spread["ts"], unit="ms"),
                    y=df_spread["spread"],
                    mode="lines",
                    name="Spread",
                    line=dict(color="purple", width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Spread: %{y:.6f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="purple", font_size=12)
                ))
                fig_spread.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_spread.update_layout(
                    title="Spread Series",
                    xaxis_title="Time",
                    yaxis_title="Spread",
                    height=300,
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            x=0, xanchor="left", y=1.02, yanchor="top"
                        ),
                        showspikes=True,
                        spikecolor="purple",
                        spikethickness=1,
                        spikedash="solid",
                        spikemode="across+toaxis"
                    ),
                    yaxis=dict(showspikes=True, spikecolor="purple", spikethickness=1, spikedash="solid"),
                    hovermode='x unified',
                    dragmode='pan',
                    hoverdistance=100
                )
                st.plotly_chart(fig_spread, width='stretch', config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'doubleClick': 'reset',
                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
                })
                
                with st.expander("📊 Chart Controls", expanded=False):
                    col_s1, col_s2 = st.columns(2)
                    with col_s1:
                        spread_smoothing = st.slider("Spread Smoothing", min_value=1, max_value=20, value=1, step=1, key="smooth")
                    with col_s2:
                        spread_yaxis_range = st.slider("Y-Axis Range Multiplier", min_value=1.0, max_value=3.0, value=1.5, step=0.1, key="yrange")
    
    st.subheader("Z-Score Series")
    
    zscore_data = fetch_data("/api/analytics/zscore", {
        "symbol1": symbol1,
        "symbol2": symbol2,
        "timeframe": timeframe,
        "window": window,
        "limit": data_limit
    })
    
    if zscore_data:
        df_zscore = pd.DataFrame(zscore_data)
        if not df_zscore.empty:
            fig_zscore = go.Figure()
            fig_zscore.add_trace(go.Scatter(
                x=pd.to_datetime(df_zscore["ts"], unit="ms"),
                y=df_zscore["zscore"],
                mode="lines",
                name="Z-Score",
                line=dict(color="orange", width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                            'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                            'Z-Score: %{y:.4f}<extra></extra>',
                hoverlabel=dict(bgcolor="white", bordercolor="orange", font_size=12)
            ))
            fig_zscore.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="+2σ")
            fig_zscore.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="-2σ")
            fig_zscore.add_hline(y=0, line_dash="dot", line_color="gray")
            fig_zscore.update_layout(
                title="Z-Score Series",
                xaxis_title="Time",
                yaxis_title="Z-Score",
                height=400,
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date",
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(count=6, label="6h", step="hour", stepmode="backward"),
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        x=0, xanchor="left", y=1.02, yanchor="top"
                    ),
                    showspikes=True,
                    spikecolor="orange",
                    spikethickness=1,
                    spikedash="solid",
                    spikemode="across+toaxis"
                ),
                yaxis=dict(showspikes=True, spikecolor="orange", spikethickness=1, spikedash="solid"),
                hovermode='x unified',
                dragmode='pan',
                hoverdistance=100
            )
            st.plotly_chart(fig_zscore, width='stretch', config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,
                'doubleClick': 'reset'
            })
            
            st.markdown("---")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                zscore_threshold = st.slider("Z-Score Threshold", min_value=0.0, max_value=5.0, value=2.0, step=0.1, key="zthresh")
            with col_s2:
                zscore_line_width = st.slider("Line Width", min_value=1, max_value=5, value=2, step=1, key="zscore_width")
            
            current_zscore = df_zscore["zscore"].iloc[-1] if len(df_zscore) > 0 else 0
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Current Z-Score", f"{current_zscore:.4f}")
            if abs(current_zscore) > 2:
                col_b.warning("⚠️ Z-Score > 2σ")
            col_c.metric("Mean Z-Score", f"{df_zscore['zscore'].mean():.4f}")

with tab3:
    st.header("Rolling Correlation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rolling Correlation Series")
        
        corr_data = fetch_data("/api/analytics/rolling-correlation", {
            "symbol1": symbol1,
            "symbol2": symbol2,
            "timeframe": timeframe,
            "window": window,
            "limit": data_limit
        })
        
        if corr_data:
            df_corr = pd.DataFrame(corr_data)
            if not df_corr.empty:
                fig_corr = go.Figure()
                fig_corr.add_trace(go.Scatter(
                    x=pd.to_datetime(df_corr["ts"], unit="ms"),
                    y=df_corr["correlation"],
                    mode="lines",
                    name="Correlation",
                    line=dict(color="teal", width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Time: %{x|%Y-%m-%d %H:%M:%S.%L}<br>' +
                                'Correlation: %{y:.4f}<extra></extra>',
                    hoverlabel=dict(bgcolor="white", bordercolor="teal", font_size=12)
                ))
                fig_corr.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_corr.update_layout(
                    title="Rolling Correlation",
                    xaxis_title="Time",
                    yaxis_title="Correlation",
                    height=400,
                    yaxis_range=[-1, 1],
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date",
                        rangeselector=dict(
                            buttons=list([
                                dict(count=1, label="1h", step="hour", stepmode="backward"),
                                dict(count=6, label="6h", step="hour", stepmode="backward"),
                                dict(count=1, label="1d", step="day", stepmode="backward"),
                                dict(step="all", label="All")
                            ]),
                            x=0, xanchor="left", y=1.02, yanchor="top"
                        ),
                        showspikes=True,
                        spikecolor="teal",
                        spikethickness=1,
                        spikedash="solid",
                        spikemode="across+toaxis"
                    ),
                    yaxis=dict(showspikes=True, spikecolor="teal", spikethickness=1, spikedash="solid"),
                    hovermode='x unified',
                    dragmode='pan',
                    hoverdistance=100
                )
                st.plotly_chart(fig_corr, width='stretch', config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': True,
                    'doubleClick': 'reset'
                })
                
                st.markdown("---")
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    corr_window_adjust = st.slider("Correlation Window", min_value=10, max_value=200, value=60, step=10, key="corrwin")
                with col_s2:
                    corr_line_width = st.slider("Line Width", min_value=1, max_value=5, value=2, step=1, key="corr_width")
                
                current_corr = df_corr["correlation"].iloc[-1] if len(df_corr) > 0 else 0
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Current Correlation", f"{current_corr:.4f}")
                col_b.metric("Mean Correlation", f"{df_corr['correlation'].mean():.4f}")
                col_c.metric("Std Dev", f"{df_corr['correlation'].std():.4f}")
    
    with col2:
        st.subheader("Cross-Correlation Heatmap")
        
        additional_symbols = st.text_input("Additional Symbols (comma-separated)", value="")
        
        if st.button("Generate Heatmap"):
            symbols_list = [symbol1, symbol2]
            if additional_symbols:
                symbols_list.extend([s.strip() for s in additional_symbols.split(",")])
            
            symbols_str = ",".join(symbols_list)
            
            heatmap_data = fetch_data("/api/analytics/cross-correlation", {
                "symbols": symbols_str,
                "timeframe": timeframe,
                "limit": data_limit
            })
            
            if heatmap_data:
                symbols_heat = heatmap_data["symbols"]
                corr_matrix = pd.DataFrame(heatmap_data["correlation_matrix"])
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdYlBu',
                    zmid=0,
                    text=corr_matrix.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorbar=dict(title="Correlation")
                ))
                
                fig_heatmap.update_layout(
                    title="Cross-Correlation Heatmap",
                    xaxis_title="Symbol",
                    yaxis_title="Symbol",
                    height=400
                )
                
                st.plotly_chart(fig_heatmap, width='stretch', config={'displayModeBar': True, 'displaylogo': False})

with tab4:
    st.header("Augmented Dickey-Fuller Test")
    
    st.write("Test spread stationarity using ADF test.")
    
    if st.button("Run ADF Test"):
        with st.spinner("Running ADF test..."):
            try:
                adf_result = fetch_data("/api/analytics/adf-test", params={
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "timeframe": timeframe,
                    "limit": 1000
                })
                
                if not adf_result:
                    st.warning("⚠️ Could not run ADF test. Please check if there's enough data (minimum 10 observations required).")
                    st.info("💡 Make sure you have overlapping data for both symbols. Start the data collector and wait for data to arrive.")
                    st.info(f"📊 Required: At least 10 overlapping candles for {symbol1} and {symbol2} in {timeframe} timeframe")
                    st.stop()
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error running ADF test: {error_msg}")
                if "Insufficient data" in error_msg or "minimum 10" in error_msg.lower():
                    st.info("💡 Tip: You need at least 10 overlapping candle observations for the ADF test.")
                    st.info(f"📊 Action: Wait for more data to accumulate for {symbol1} and {symbol2}")
                elif "constant series" in error_msg.lower():
                    st.info("💡 Tip: The spread is constant (all values are the same). This may indicate insufficient price variation.")
                    st.info("📊 Action: Ensure both symbols have sufficient price movement")
                else:
                    st.info("💡 Tip: Check that both symbols have valid overlapping data in the same timeframe.")
                st.stop()
            
            if adf_result:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Test Statistics")
                    st.metric("ADF Statistic", f"{adf_result['adf_statistic']:.6f}")
                    st.metric("p-value", f"{adf_result['p_value']:.6f}")
                    st.metric("Observations", adf_result["n_observations"])
                    
                    if adf_result["is_stationary"]:
                        st.success("✅ Spread is stationary (p < 0.05)")
                    else:
                        st.warning("⚠️ Spread is not stationary (p >= 0.05)")
                
                with col2:
                    st.subheader("Critical Values")
                    critical_values = adf_result["critical_values"]
                    st.metric("1%", f"{critical_values['1%']:.6f}")
                    st.metric("5%", f"{critical_values['5%']:.6f}")
                    st.metric("10%", f"{critical_values['10%']:.6f}")
                    
                    st.info(
                        f"""
                        **Interpretation:**
                        - If ADF statistic < critical value, the series is stationary
                        - Current ADF: {adf_result['adf_statistic']:.6f}
                        - Critical (5%): {critical_values['5%']:.6f}
                        """
                    )
                
                if st.button("Export Results to CSV"):
                    df_export = pd.DataFrame([adf_result])
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"adf_test_{symbol1}_{symbol2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

with tab5:
    st.header("Alert Management")
    
    with st.expander("➕ Create New Alert"):
        alert_id = st.text_input("Alert ID", value=f"alert_{int(time.time())}")
        alert_name = st.text_input("Alert Name", value="Z-Score Threshold")
        alert_metric = st.selectbox("Metric", ["zscore", "spread", "correlation"])
        alert_operator = st.selectbox("Operator", [">", ">=", "<", "<=", "==", "!="])
        alert_threshold = st.number_input("Threshold", value=2.0)
        
        if st.button("Create Alert"):
            result = post_data("/api/alerts/rules", params={
                "rule_id": alert_id,
                "rule_name": alert_name,
                "metric": alert_metric,
                "operator": alert_operator,
                "threshold": alert_threshold,
                "enabled": True
            })
            
            if result:
                st.success(f"Alert created: {alert_id}")
                st.rerun()
    
    st.subheader("Active Alert Rules")
    
    rules = fetch_data("/api/alerts/rules")
    
    if rules:
        rules_df = pd.DataFrame.from_dict(rules, orient="index")
        if not rules_df.empty:
            st.dataframe(rules_df[["rule_name", "metric", "operator", "threshold", "enabled"]])
            
            rule_to_delete = st.selectbox("Delete Rule", ["Select rule..."] + list(rules.keys()))
            if rule_to_delete != "Select rule..." and st.button("Delete Rule"):
                result = requests.delete(f"{API_BASE_URL}/api/alerts/rules/{rule_to_delete}")
                if result.status_code == 200:
                    st.success(f"Rule {rule_to_delete} deleted")
                    st.rerun()
        else:
            st.info("No alert rules configured.")
    else:
        st.info("No alert rules configured.")
    
    st.subheader("Evaluate Alerts")
    
    if st.button("Evaluate Now"):
        with st.spinner("Evaluating alerts..."):
            alerts = post_data("/api/alerts/evaluate")
            
            if alerts:
                if len(alerts) > 0:
                    st.warning(f"⚠️ {len(alerts)} alert(s) triggered!")
                    for alert in alerts:
                        st.error(f"**{alert['rule_name']}**: {alert['message']}")
                else:
                    st.success("✅ No alerts triggered.")
            else:
                st.info("No alerts triggered.")
    
    st.subheader("Alert History")
    
    history = fetch_data("/api/alerts", {"limit": 50})
    
    if history:
        if len(history) > 0:
            history_df = pd.DataFrame(history)
            history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], unit="ms")
            st.dataframe(history_df[["rule_name", "message", "timestamp", "severity"]])
            
            if st.button("Export Alert History to CSV"):
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("No alert history.")
    else:
        st.info("No alert history.")

with tab6:
    st.header("📤 Upload OHLC Data")
    
    st.write("Upload a CSV file with OHLC data. The file should have columns: timestamp, open, high, low, close, volume")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            symbol_upload = st.text_input("Symbol (or leave empty to extract from filename)", value="")
        
        with col2:
            timeframe_upload = st.selectbox("Timeframe", ["1s", "1m", "5m"], index=1)
        
        if st.button("Upload OHLC CSV"):
            with st.spinner("Uploading and processing CSV..."):
                try:
                    uploaded_file.seek(0)
                    
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    params = {"timeframe": timeframe_upload}
                    
                    if symbol_upload:
                        params["symbol"] = symbol_upload
                    
                    result = post_data("/api/upload/ohlc", params=params, files=files)
                    
                    if result:
                        st.success(f"✅ Successfully uploaded {result['count']} candles for {result['symbol']}")
                        st.json(result)
                    else:
                        st.error("Failed to upload CSV")
                except Exception as e:
                    st.error(f"Error uploading CSV: {e}")

with tab7:
    st.header("📋 Time-Series Statistics Table")
    
    st.write("View detailed statistics for each time period with all features")
    
    candles1 = fetch_data("/api/candles", {"symbol": symbol1, "timeframe": timeframe, "limit": data_limit})
    candles2 = fetch_data("/api/candles", {"symbol": symbol2, "timeframe": timeframe, "limit": data_limit})
    
    if candles1 and candles2:
        df1 = pd.DataFrame(candles1)
        df2 = pd.DataFrame(candles2)
        
        if not df1.empty and not df2.empty:
            merged = pd.merge(
                df1[["ts", "close", "volume"]].rename(columns={"close": f"{symbol1}_price", "volume": f"{symbol1}_volume"}),
                df2[["ts", "close", "volume"]].rename(columns={"close": f"{symbol2}_price", "volume": f"{symbol2}_volume"}),
                on="ts",
                how="inner"
            )
            
            if not merged.empty:
                stats_rows = []
                
                spread_data = fetch_data("/api/analytics/spread", {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "timeframe": timeframe,
                    "limit": data_limit
                })
                
                zscore_data = fetch_data("/api/analytics/zscore", {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "timeframe": timeframe,
                    "window": window,
                    "limit": data_limit
                })
                
                corr_data = fetch_data("/api/analytics/rolling-correlation", {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "timeframe": timeframe,
                    "window": window,
                    "limit": data_limit
                })
                
                df_spread = pd.DataFrame(spread_data) if spread_data else pd.DataFrame()
                df_zscore = pd.DataFrame(zscore_data) if zscore_data else pd.DataFrame()
                df_corr = pd.DataFrame(corr_data) if corr_data else pd.DataFrame()
                
                merged["datetime"] = pd.to_datetime(merged["ts"], unit="ms")
                
                if not df_spread.empty:
                    df_spread["datetime"] = pd.to_datetime(df_spread["ts"], unit="ms")
                    merged = pd.merge(merged, df_spread[["datetime", "spread"]], on="datetime", how="left")
                else:
                    merged["spread"] = None
                
                if not df_zscore.empty:
                    df_zscore["datetime"] = pd.to_datetime(df_zscore["ts"], unit="ms")
                    merged = pd.merge(merged, df_zscore[["datetime", "zscore"]], on="datetime", how="left")
                else:
                    merged["zscore"] = None
                
                if not df_corr.empty:
                    df_corr["datetime"] = pd.to_datetime(df_corr["ts"], unit="ms")
                    merged = pd.merge(merged, df_corr[["datetime", "correlation"]], on="datetime", how="left")
                else:
                    merged["correlation"] = None
                
                merged[f"{symbol1}_mean"] = merged[f"{symbol1}_price"].rolling(window=window).mean()
                merged[f"{symbol1}_std"] = merged[f"{symbol1}_price"].rolling(window=window).std()
                merged[f"{symbol2}_mean"] = merged[f"{symbol2}_price"].rolling(window=window).mean()
                merged[f"{symbol2}_std"] = merged[f"{symbol2}_price"].rolling(window=window).std()
                
                stats_df = merged[[
                    "datetime",
                    f"{symbol1}_price",
                    f"{symbol1}_volume",
                    f"{symbol1}_mean",
                    f"{symbol1}_std",
                    f"{symbol2}_price",
                    f"{symbol2}_volume",
                    f"{symbol2}_mean",
                    f"{symbol2}_std",
                    "spread",
                    "zscore",
                    "correlation"
                ]].copy()
                
                numeric_cols = stats_df.select_dtypes(include=['float64']).columns
                stats_df[numeric_cols] = stats_df[numeric_cols].round(6)
                
                st.dataframe(stats_df, width='stretch', height=400)
                
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics CSV",
                    data=csv,
                    file_name=f"timeseries_stats_{symbol1}_{symbol2}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                st.subheader("Summary Statistics")
                st.dataframe(stats_df.describe(), width='stretch')
            else:
                st.warning("No overlapping data between symbols")
        else:
            st.warning("Could not fetch data for both symbols")
    else:
        st.warning("Could not fetch candle data. Check API connection.")

with tab8:
    st.header("🚀 Advanced Analytics")
    
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        st.subheader("Dynamic Hedge Ratio (Kalman Filter)")
        
        if st.button("Calculate Kalman Filter Hedge Ratio"):
            with st.spinner("Calculating Kalman Filter hedge ratio..."):
                try:
                    kalman_data = fetch_data("/api/analytics/kalman-hedge", {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "timeframe": timeframe,
                        "limit": data_limit
                    })
                    
                    if not kalman_data:
                        st.warning("⚠️ Could not calculate Kalman Filter hedge ratio. Ensure you have overlapping data for both symbols.")
                        st.info(f"💡 Required: Overlapping candles for {symbol1} and {symbol2} in {timeframe} timeframe")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure you have enough overlapping data for both symbols")
                    st.stop()
                
                if kalman_data:
                    st.metric("Current Hedge Ratio", f"{kalman_data['current_hedge_ratio']:.6f}")
                    st.metric("Uncertainty", f"{kalman_data['current_uncertainty']:.6f}")
                    st.metric("Mean Hedge Ratio", f"{kalman_data['mean_hedge_ratio']:.6f}")
                    st.metric("Std Hedge Ratio", f"{kalman_data['std_hedge_ratio']:.6f}")
                    
                    if "ts" in kalman_data and "hedge_ratio_series" in kalman_data:
                        df_kalman = pd.DataFrame({
                            "ts": kalman_data["ts"],
                            "hedge_ratio": kalman_data["hedge_ratio_series"],
                            "uncertainty": kalman_data.get("uncertainty_series", [])
                        })
                        
                        fig_kalman = go.Figure()
                        fig_kalman.add_trace(go.Scatter(
                            x=pd.to_datetime(df_kalman["ts"], unit="ms"),
                            y=df_kalman["hedge_ratio"],
                            mode="lines",
                            name="Hedge Ratio",
                            line=dict(color="blue", width=2)
                        ))
                        fig_kalman.update_layout(
                            title="Dynamic Hedge Ratio (Kalman Filter)",
                            xaxis_title="Time",
                            yaxis_title="Hedge Ratio",
                            height=400
                        )
                        st.plotly_chart(fig_kalman, width='stretch')
        
        st.subheader("Robust Regression Hedge Ratio")
        
        reg_method = st.selectbox("Method", ["huber", "theilsen"], index=0)
        
        if st.button("Calculate Robust Hedge Ratio"):
            with st.spinner("Calculating robust hedge ratio..."):
                try:
                    robust_data = fetch_data("/api/analytics/robust-hedge", {
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "method": reg_method,
                        "timeframe": timeframe,
                        "window": window
                    })
                    
                    if not robust_data:
                        st.warning(f"⚠️ Could not calculate {reg_method.upper()} robust hedge ratio. Ensure you have overlapping data.")
                        st.info(f"💡 Required: Overlapping candles for {symbol1} and {symbol2} in {timeframe} timeframe (minimum {window} observations)")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info(f"💡 Make sure you have enough overlapping data (minimum {window} observations)")
                    st.stop()
                
                if robust_data:
                    st.metric("Hedge Ratio", f"{robust_data['hedge_ratio']:.6f}")
                    st.metric("Alpha", f"{robust_data['alpha']:.6f}")
                    st.metric("R²", f"{robust_data['r_squared']:.4f}")
                    st.metric("Method", robust_data["method"].upper())
        
        st.subheader("Liquidity Metrics")
        
        if st.button("Calculate Liquidity"):
            with st.spinner("Calculating liquidity metrics..."):
                try:
                    liquidity1 = fetch_data("/api/analytics/liquidity", {
                        "symbol": symbol1,
                        "timeframe": timeframe,
                        "window": window,
                        "limit": data_limit
                    })
                    
                    liquidity2 = fetch_data("/api/analytics/liquidity", {
                        "symbol": symbol2,
                        "timeframe": timeframe,
                        "window": window,
                        "limit": data_limit
                    })
                    
                    if not liquidity1 and not liquidity2:
                        st.warning("⚠️ Could not calculate liquidity metrics. Ensure you have data for the symbols.")
                        st.info(f"💡 Required: Candle data for {symbol1} and {symbol2} in {timeframe} timeframe")
                        st.stop()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("💡 Make sure you have candle data for both symbols")
                    st.stop()
                
                if liquidity1:
                    st.write(f"**{symbol1.upper()} Liquidity:**")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Current Volume", f"{liquidity1['current_volume']:.2f}")
                    col_b.metric("Mean Volume", f"{liquidity1['mean_volume']:.2f}")
                    col_a.metric("VWAP Spread", f"{liquidity1.get('vwap_spread', 0):.4f}")
                    col_b.metric("Bid-Ask Est.", f"{liquidity1.get('mean_spread_pct', 0):.4f}%")
                
                if liquidity2:
                    st.write(f"**{symbol2.upper()} Liquidity:**")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Current Volume", f"{liquidity2['current_volume']:.2f}")
                    col_b.metric("Mean Volume", f"{liquidity2['mean_volume']:.2f}")
                    col_a.metric("VWAP Spread", f"{liquidity2.get('vwap_spread', 0):.4f}")
                    col_b.metric("Bid-Ask Est.", f"{liquidity2.get('mean_spread_pct', 0):.4f}%")
    
    with adv_col2:
        st.subheader("Mean-Reversion Backtest")
        
        entry_threshold = st.number_input("Entry Threshold (Z-Score)", value=2.0, step=0.1)
        exit_threshold = st.number_input("Exit Threshold (Z-Score)", value=0.0, step=0.1)
        
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                backtest_data = fetch_data("/api/analytics/backtest", {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "timeframe": timeframe,
                    "entry_threshold": entry_threshold,
                    "exit_threshold": exit_threshold,
                    "window": window,
                    "limit": data_limit
                })
                
                if backtest_data:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Total Trades", backtest_data["total_trades"])
                    col_b.metric("Closed Trades", backtest_data["closed_trades"])
                    col_a.metric("Winning Trades", backtest_data["winning_trades"])
                    col_b.metric("Losing Trades", backtest_data["losing_trades"])
                    col_a.metric("Total PnL", f"{backtest_data['total_pnl']:.2f}")
                    col_b.metric("Avg PnL", f"{backtest_data['average_pnl']:.2f}")
                    col_a.metric("Win Rate", f"{backtest_data['win_rate']*100:.1f}%")
                    col_b.metric("Max Drawdown", f"{backtest_data['max_drawdown']:.2f}")
                    
                    if backtest_data.get("sharpe_ratio"):
                        st.metric("Sharpe Ratio", f"{backtest_data['sharpe_ratio']:.2f}")
                    
                    if backtest_data.get("trades"):
                        trades_df = pd.DataFrame(backtest_data["trades"])
                        st.dataframe(trades_df, width='stretch')
                        
                        csv = trades_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Backtest Trades CSV",
                            data=csv,
                            file_name=f"backtest_trades_{symbol1}_{symbol2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
    
    st.subheader("📡 Real-Time Metrics")
    
    if st.checkbox("Enable Real-Time Updates", value=False):
        realtime_placeholder = st.empty()
        
        if st.button("Refresh Real-Time Metrics"):
            metrics = fetch_data("/api/analytics/price-stats", {
                "symbol": symbol1,
                "timeframe": timeframe,
                "window": window
            })
            
            if metrics:
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Current Price", f"${metrics['current']:.2f}")
                col_b.metric("Mean", f"${metrics['mean']:.2f}")
                col_c.metric("Std Dev", f"${metrics['std']:.2f}")
                col_d.metric("Min/Max", f"${metrics['min']:.2f} / ${metrics['max']:.2f}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Status")
if st.sidebar.button("Check API Health"):
    health = fetch_data("/health")
    if health:
        st.sidebar.success("✅ API is healthy")
    else:
        st.sidebar.error("❌ API is not responding")

if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
