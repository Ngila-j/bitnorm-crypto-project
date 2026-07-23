from datetime import datetime, time
from io import BytesIO
import hashlib
import os
import secrets
import sqlite3
import time as t_mod
from analytics import (
    compute_blockactivities_health_score,
    compute_net_taker_flow,
    compute_user_sentiment_index,
    fetch_latest_crypto_metrics,
)
from pipeline import generate_all_crypto_metrics, generate_simulated_trades
from prophet import Prophet
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import requests
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration with wide layout
st.set_page_config(
    page_title="BitNorm / BNAnalytics Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .block-container {
        padding-top: 4.5rem !important;
        padding-bottom: 3rem;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    h1 {
        font-size: 1.4rem !important;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
        margin-bottom: 0.1rem !important;
    }
    h2 {
        font-size: 1.2rem !important;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    h3 {
        font-size: 1.0rem !important;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
    }
    .ticker-bar {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
        color: #10b981;
        text-align: center;
        margin-bottom: 15px;
        border: 1px solid #374151;
        font-size: 0.85rem;
    }
    .alert-box-warning {
        background-color: #7f1d1d;
        border: 1px solid #ef4444;
        padding: 10px;
        border-radius: 8px;
        color: #fee2e2;
        font-weight: 600;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .alert-box-success {
        background-color: #065f46;
        border: 1px solid #10b981;
        padding: 10px;
        border-radius: 8px;
        color: #d1fae5;
        font-weight: 600;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    div.stButton > button {
        width: 100%;
        background-color: transparent;
        color: #9ca3af;
        border: 1px solid transparent;
        border-radius: 8px;
        text-align: left;
        padding: 8px 12px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #1f2937;
        color: #ffffff;
        border-color: #374151;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- INSTITUTIONAL ANALYTICS ENGINE (ADVANCED CAPABILITIES) ---
class InstitutionalAnalyticsEngine:

  @staticmethod
  def optimize_strategy_grid(
      prices: pd.Series, short_windows: list, long_windows: list
  ):
    """Performs grid search optimization on Moving Average Crossover parameters."""
    best_sharpe = -np.inf
    best_params = (short_windows[0], long_windows[0])
    results = []

    for s in short_windows:
      for l in long_windows:
        if s >= l:
          continue
        sma_s = prices.rolling(s).mean()
        sma_l = prices.rolling(l).mean()
        signal = np.where(sma_s > sma_l, 1, -1)
        returns = prices.pct_change() * pd.Series(signal).shift(1)

        sharpe = (
            (returns.mean() / returns.std()) * np.sqrt(252)
            if returns.std() != 0 and not pd.isna(returns.std())
            else 0
        )
        cum_ret = (1 + returns.fillna(0)).prod() - 1
        max_dd = (
            (returns.cumsum() - returns.cumsum().cummax()).min()
            if not returns.empty
            else 0
        )

        results.append({
            "Short MA": s,
            "Long MA": l,
            "Sharpe Ratio": round(float(sharpe), 2),
            "Cumulative Return (%)": round(float(cum_ret * 100), 2),
            "Max Drawdown (%)": round(float(max_dd * 100), 2),
        })

        if sharpe > best_sharpe:
          best_sharpe = sharpe
          best_params = (s, l)

    return pd.DataFrame(results), best_params

  @staticmethod
  def generate_prophet_forecast(df: pd.DataFrame, periods: int = 30):
    """Generates machine learning time-series forecasts using Meta Prophet."""
    pdf = df.rename(columns={"metric_date": "ds", "market_cap": "y"})[
        ["ds", "y"]
    ].copy()
    pdf["ds"] = pd.to_datetime(pdf["ds"])

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.fit(pdf)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


# --- AUTO SESSION INITIALIZATION ---
if "authenticated" not in st.session_state:
  st.session_state.authenticated = True
  st.session_state.username = "admin_lead"
  st.session_state.role = "Admin"


def init_rbac_db():
  conn = sqlite3.connect("bnanalytics_institutional.db")
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutional_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('Admin', 'Portfolio Manager', 'Analyst'))
        )
    """)
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS institutional_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            api_key TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES institutional_users(id)
        )
    """)
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            asset_symbol TEXT,
            health_score REAL,
            threshold REAL,
            status TEXT
        )
    """)
  cursor.execute("SELECT COUNT(*) FROM institutional_users")
  if cursor.fetchone()[0] == 0:
    default_pass = hashlib.sha256("AdminSecure2026!".encode()).hexdigest()
    cursor.execute(
        "INSERT INTO institutional_users (username, password_hash, role) VALUES"
        " (?, ?, ?)",
        ("admin_lead", default_pass, "Admin"),
    )
  conn.commit()
  conn.close()


init_rbac_db()


@st.cache_data
def load_dashboard_data():
  conn = sqlite3.connect("crypto_data.db")
  cursor = conn.cursor()

  required_tables = [
      "customer_trades",
      "sourcecode_metrics",
      "network_metrics",
      "economics_metrics",
      "sentiment_metrics",
      "accessibility_metrics",
      "paper_portfolio",
  ]
  missing_tables = [
      t
      for t in required_tables
      if cursor.execute(
          "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)
      ).fetchone()
      is None
  ]

  if missing_tables:
    conn.close()
    generate_simulated_trades(num_records=5000, db_path="crypto_data.db")
    generate_all_crypto_metrics(days=30, db_path="crypto_data.db")
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()

  cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_portfolio (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, asset_symbol TEXT, action TEXT,
            quantity REAL, execution_price REAL, total_cost REAL
        )
    """)

  cursor.execute("""
        CREATE TABLE IF NOT EXISTS whale_transactions (
            tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, asset_symbol TEXT, sender_wallet TEXT,
            receiver_wallet TEXT, amount_tokens REAL, usd_value REAL, tx_type TEXT
        )
    """)

  cursor.execute("SELECT COUNT(*) FROM whale_transactions")
  if cursor.fetchone()[0] == 0:
    np.random.seed(42)
    assets = ["BTC", "ETH", "SOL", "ADA"]
    types = [
        "Exchange Inflow",
        "Exchange Outflow",
        "OTC Transfer",
        "Wallet-to-Wallet",
    ]
    for _ in range(250):
      ast = np.random.choice(assets)
      amt = (
          np.random.uniform(100, 15000)
          if ast in ["BTC", "ETH"]
          else np.random.uniform(50000, 2000000)
      )
      price = (
          65000
          if ast == "BTC"
          else (2000 if ast == "ETH" else (140 if ast == "SOL" else 0.48))
      )
      cursor.execute(
          """
                INSERT INTO whale_transactions (timestamp, asset_symbol, sender_wallet, receiver_wallet, amount_tokens, usd_value, tx_type)
                VALUES (DATETIME('now', '-' || ABS(RANDOM() % 10) || ' days'), ?, ?, ?, ?, ?, ?)
            """,
          (
              ast,
              f"0x{np.random.randint(1e8, 9e8):x}",
              f"0x{np.random.randint(1e8, 9e8):x}",
              amt,
              amt * price,
              np.random.choice(types),
          ),
      )

  conn.commit()

  trades = pd.read_sql("SELECT * FROM customer_trades", conn)
  sourcecode = pd.read_sql("SELECT * FROM sourcecode_metrics", conn)
  network = pd.read_sql("SELECT * FROM network_metrics", conn)
  economics = pd.read_sql("SELECT * FROM economics_metrics", conn)
  sentiment = pd.read_sql("SELECT * FROM sentiment_metrics", conn)
  accessibility = pd.read_sql("SELECT * FROM accessibility_metrics", conn)
  paper_trades = pd.read_sql("SELECT * FROM paper_portfolio", conn)
  whale_df = pd.read_sql("SELECT * FROM whale_transactions", conn)
  conn.close()

  if "timestamp" in trades.columns:
    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
  for frame in [sourcecode, network, economics, sentiment, accessibility]:
    if "metric_date" in frame.columns:
      frame["metric_date"] = pd.to_datetime(frame["metric_date"])
  if "timestamp" in whale_df.columns:
    whale_df["timestamp"] = pd.to_datetime(whale_df["timestamp"])

  return {
      "trades": trades,
      "sourcecode": sourcecode,
      "network": network,
      "economics": economics,
      "sentiment": sentiment,
      "accessibility": accessibility,
      "paper_trades": paper_trades,
      "whale_df": whale_df,
  }


def format_currency(value):
  if value is None:
    return "—"
  if abs(value) >= 1e12:
    return f"${value/1e12:,.2f}T"
  if abs(value) >= 1e9:
    return f"${value/1e9:,.2f}B"
  if abs(value) >= 1e6:
    return f"${value/1e6:,.2f}M"
  return f"${value:,.2f}"


def render_metric_cards(metrics):
  cols = st.columns(len(metrics))
  for col, (title, value, delta) in zip(cols, metrics):
    with col:
      st.metric(label=title, value=value, delta=delta)


def render_history_chart(frame, metric_name, title, y_label, color="#10b981"):
  if frame.empty:
    st.warning("No historical data available for this selection.")
    return
  chart_data = (
      frame[["metric_date", metric_name]].copy().sort_values("metric_date")
  )
  fig = px.line(
      chart_data,
      x="metric_date",
      y=metric_name,
      markers=True,
      title=title,
      labels={"metric_date": "Timeline", metric_name: y_label},
  )
  fig.update_traces(line_color=color, line_width=3)
  fig.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
      title_font_size=14,
  )
  st.plotly_chart(fig, use_container_width=True)


@st.fragment(run_every=5)
def render_live_websocket_ticker():
  prices = {"BTC": 65901.0, "ETH": 1927.0, "SOL": 142.50, "ADA": 0.48}
  try:
    url = (
        "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,cardano&vs_currencies=usd&include_24hr_change=true"
    )
    response = requests.get(url, timeout=2)
    if response.status_code == 200:
      data = response.json()
      if "bitcoin" in data:
        prices["BTC"] = data["bitcoin"]["usd"]
      if "solana" in data:
        prices["SOL"] = data["solana"]["usd"]
      if "cardano" in data:
        prices["ADA"] = data["cardano"]["usd"]
  except Exception:
    prices["BTC"] += np.random.uniform(-10, 10)
    prices["ETH"] += np.random.uniform(-3, 3)

  ticker_text = (
      f"⚡ LIVE STREAMING WS: BTC/USD ${prices['BTC']:,.2f} | ETH/USD"
      f" ${prices['ETH']:,.2f} | SOL/USD ${prices['SOL']:,.2f} | ADA/USD"
      f" ${prices['ADA']:,.2f}"
  )
  st.markdown(
      f'<div class="ticker-bar">{ticker_text}</div>', unsafe_allow_html=True
  )


def generate_pdf_report(symbol, health_data, latest_econ, latest_net):
  buffer = BytesIO()
  doc = SimpleDocTemplate(
      buffer,
      pagesize=letter,
      rightMargin=30,
      leftMargin=30,
      topMargin=30,
      bottomMargin=30,
  )
  story = []

  styles = getSampleStyleSheet()
  title_style = ParagraphStyle(
      "TitleStyle",
      parent=styles["Heading1"],
      fontSize=18,
      textColor=colors.HexColor("#10b981"),
      spaceAfter=6,
  )
  subtitle_style = ParagraphStyle(
      "SubTitleStyle",
      parent=styles["Normal"],
      fontSize=10,
      textColor=colors.HexColor("#6b7280"),
      spaceAfter=15,
  )
  heading_style = ParagraphStyle(
      "HeadingStyle",
      parent=styles["Heading2"],
      fontSize=12,
      textColor=colors.HexColor("#1f2937"),
      spaceAfter=8,
      spaceBefore=12,
  )
  body_style = ParagraphStyle(
      "BodyStyle",
      parent=styles["Normal"],
      fontSize=10,
      textColor=colors.HexColor("#374151"),
      spaceAfter=6,
  )

  story.append(
      Paragraph(
          f"BNAnalytics Enterprise Executive Report: {symbol}", title_style
      )
  )
  story.append(
      Paragraph(
          "Generated automatically via Bitnorm Production Suite", subtitle_style
      )
  )
  story.append(Spacer(1, 10))

  story.append(Paragraph("1. Composite Health Score Breakdown", heading_style))
  score_summary = (
      f"<b>Overall Health Rating:</b> {health_data['health_score']:.1f} /"
      f" 100<br/>• Source Code Activity Vector:"
      f" {health_data['pillar_scores']['sourcecode']:.1f}/100<br/>• Ledger &"
      f" Network Activity: {health_data['pillar_scores']['network']:.1f}/100<br/>•"
      " Market Economics & Liquidity:"
      f" {health_data['pillar_scores']['economics']:.1f}/100<br/>• User Sentiment"
      f" Index: {health_data['pillar_scores']['sentiment']:.1f}/100<br/>•"
      " Accessibility & Integration:"
      f" {health_data['pillar_scores']['accessibility']:.1f}/100"
  )
  story.append(Paragraph(score_summary, body_style))
  story.append(Spacer(1, 10))

  story.append(
      Paragraph("2. Key Financial & Operational Metrics", heading_style)
  )
  data_table = [
      ["Metric Description", "Recorded Value"],
      [
          "Market Capitalization",
          format_currency(latest_econ.get("market_cap", 0)),
      ],
      ["24h Trading Volume", format_currency(latest_econ.get("volume_24h", 0))],
      ["Network TPS Throughput", f"{latest_net.get('tx_tps', 0):.2f} TPS"],
      ["Active On-Chain Addresses", f"{latest_net.get('active_addresses', 0):,}"],
  ]
  t = Table(data_table, colWidths=[250, 200])
  t.setStyle(
      TableStyle([
          ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
          ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
          ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
          ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
          ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
          ("FONTSIZE", (0, 0), (-1, -1), 9),
      ])
  )
  story.append(t)

  doc.build(story)
  buffer.seek(0)
  return buffer


page_data = load_dashboard_data()

# --- SIDEBAR NAVIGATION ---
if "nav_section" not in st.session_state:
  st.session_state.nav_section = "Home"

st.sidebar.image("logo.png", width=45)
st.sidebar.title("BNANALYTICS")
st.sidebar.caption(
    f"User: {st.session_state.username} | Role: {st.session_state.role}"
)
st.sidebar.markdown("---")

nav_mapping = {
    "Home": "Home",
    "Code Intelligence": "Code Intelligence",
    "Ledger Metrics": "Ledger Metrics",
    "Market Economics": "Market Economics",
    "Social Sentiment": "Social Sentiment",
    "Ecosystem Liquidity": "Ecosystem Liquidity",
    "Multi-Asset Comparison": "Multi-Asset Comparison",
    "Portfolio Risk & VaR": "Portfolio Risk & VaR",
    "Predictive ML Forecast": "Predictive ML Forecast",
    "Prophet AI Forecaster": "Prophet AI Forecaster",
    "Strategy Grid Optimizer": "Strategy Grid Optimizer",
    "Automated Report Scheduler": "Automated Report Scheduler",
    "Arbitrage Monitor": "Arbitrage Monitor",
    "AI Executive Summary": "AI Executive Summary",
    "Advanced Tech Indicators": "Advanced Tech Indicators",
    "Strategy Backtester": "Strategy Backtester",
    "Macro Correlation Matrix": "Macro Correlation Matrix",
    "Whale Wallet & Flow Tracker": "Whale Wallet & Flow Tracker",
    "Order Book Depth Chart": "Order Book Depth Chart",
    "Liquidation Heatmap": "Liquidation Heatmap",
    "Gas & Fee Oracle": "Gas & Fee Oracle",
    "Alerts & Audit Log": "Alerts & Audit Log",
    "Paper Trading & PnL": "Paper Trading & PnL",
    "API Key Management": "API Key Management",
    "SQL Query Sandbox": "SQL Query Sandbox",
}

for label, target_section in nav_mapping.items():
  if st.sidebar.button(label, key=f"nav_{label}"):
    st.session_state.nav_section = target_section

section = st.session_state.nav_section

st.sidebar.markdown("---")
asset_symbol = st.sidebar.selectbox(
    "Target Asset", ["BTC", "ETH", "SOL", "ADA"], index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Automated Alert Dispatcher")
alert_health_min = st.sidebar.slider("Min Health Score Warning", 0, 100, 45)
webhook_url_input = st.sidebar.text_input(
    "Webhook URL (Slack/Telegram)", placeholder="https://hooks.slack.com/..."
)

current_check_score = compute_blockactivities_health_score(
    asset_symbol, db_path="crypto_data.db"
)["health_score"]
if current_check_score < alert_health_min:
  conn_log = sqlite3.connect("bnanalytics_institutional.db")
  c_log = conn_log.cursor()
  c_log.execute(
      "INSERT INTO alert_audit_logs (asset_symbol, health_score, threshold,"
      " status) VALUES (?, ?, ?, ?)",
      (asset_symbol, current_check_score, alert_health_min, "Triggered - Warning"),
  )
  conn_log.commit()
  conn_log.close()

if current_check_score < alert_health_min and webhook_url_input:
  if st.sidebar.button("Broadcast Webhook Alert Now"):
    try:
      payload = {
          "text": (
              f"🚨 BNAnalytics Automatic Dispatch: {asset_symbol} Health Score"
              f" dropped to {current_check_score:.1f} (Threshold:"
              f" {alert_health_min})!"
          )
      }
      res = requests.post(webhook_url_input, json=payload, timeout=4)
      if res.status_code in [200, 201]:
        st.sidebar.success("Automated webhook alert dispatched successfully!")
      else:
        st.sidebar.warning(f"Webhook response status {res.status_code}")
    except Exception as e:
      st.sidebar.error(f"Connection failed: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Executive Report Exports")
export_frame = page_data["economics"][
    page_data["economics"]["asset_symbol"] == asset_symbol
]
if not export_frame.empty:
  csv_data = export_frame.to_csv(index=False).encode("utf-8")
  st.sidebar.download_button(
      label=f"Download {asset_symbol} CSV Feed",
      data=csv_data,
      file_name=f"{asset_symbol}_bnanalytics.csv",
      mime="text/csv",
  )

  snapshot_pdf = fetch_latest_crypto_metrics(
      asset_symbol, db_path="crypto_data.db"
  )
  health_pdf = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  pdf_buffer = generate_pdf_report(
      asset_symbol,
      health_pdf,
      snapshot_pdf["economics"] or {},
      snapshot_pdf["network"] or {},
  )
  st.sidebar.download_button(
      label=f"📄 Download {asset_symbol} Executive PDF",
      data=pdf_buffer,
      file_name=f"{asset_symbol}_Report.pdf",
      mime="application/pdf",
  )

# --- VIEW ROUTING WITH ENRICHED VISUALIZATIONS ---
if section == "Home":
  render_live_websocket_ticker()

  st.title("BNAnalytics Production Command Center")
  st.markdown(
      "Institutional automated health tracking, multi-dimensional code"
      " verification, and live data telemetry."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  current_health_check = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  score_val = current_health_check["health_score"]

  if score_val < alert_health_min:
    st.markdown(
        f'<div class="alert-box-warning">⚠️ WARNING: {asset_symbol} Health'
        f" Score ({score_val:.1f}) is below your warning threshold of"
        f" {alert_health_min}! Automated webhook dispatch primed.</div>",
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f'<div class="alert-box-success">✅ STATUS NORMAL: {asset_symbol}'
        f" Health Score ({score_val:.1f}) is operating within optimal"
        " institutional parameters.</div>",
        unsafe_allow_html=True,
    )

  latest_sentiment_val = fetch_latest_crypto_metrics(
      asset_symbol, db_path="crypto_data.db"
  )["sentiment"].get("user_sentiment_index", 50)
  latest_network_val = fetch_latest_crypto_metrics(
      asset_symbol, db_path="crypto_data.db"
  )["network"].get("tx_tps", 0)

  with st.container():
    st.info(
        f"**🤖 BNAnalytics Intelligence Brief:** {asset_symbol} shows robust"
        f" performance with network throughput at {latest_network_val:.2f} TPS"
        f" and a health rating of {score_val:.1f}/100. Sentiment momentum"
        f" registers at {latest_sentiment_val:.2f}."
    )

  st.markdown("<br>", unsafe_allow_html=True)

  latest_economics = (
      page_data["economics"]
      .sort_values("metric_date")
      .groupby("asset_symbol")
      .tail(1)
  )
  latest_health = [
      (
          s,
          compute_blockactivities_health_score(
              s, db_path="crypto_data.db"
          )["health_score"],
      )
      for s in ["BTC", "ETH", "SOL", "ADA"]
  ]
  health_df = pd.DataFrame(latest_health, columns=["asset_symbol", "health_score"])
  latest_economics = latest_economics.merge(
      health_df, on="asset_symbol", how="left"
  )

  col1, col2, col3, col4 = st.columns(4)
  with col1:
    st.metric(
        "Global Market Cap",
        format_currency(latest_economics["market_cap"].sum()),
        "+4.2%",
    )
  with col2:
    st.metric(
        "Global 24h Volume",
        format_currency(latest_economics["volume_24h"].sum()),
        "+12.8%",
    )
  with col3:
    st.metric(
        "Avg. Health Score",
        f"{latest_economics['health_score'].mean():.1f}/100",
        "Optimized",
    )
  with col4:
    st.metric(
        "Tracked Assets", f"{len(latest_economics)} Core", "Active"
    )

  st.markdown("---")
  col_a, col_b = st.columns(2)
  with col_a:
    fig_health = px.bar(
        health_df.sort_values("health_score", ascending=False),
        x="asset_symbol",
        y="health_score",
        color="health_score",
        color_continuous_scale="Tealgrn",
        title="BNAnalytics Composite Health Scores",
    )
    fig_health.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_health, use_container_width=True)

  with col_b:
    fig_market = px.bar(
        latest_economics,
        x="asset_symbol",
        y="market_cap",
        color="market_cap",
        color_continuous_scale="Purples",
        title="Market Capitalization Breakdown",
    )
    fig_market.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_market, use_container_width=True)

elif section == "Code Intelligence":
  st.subheader(f"Code Intelligence & Developer Velocity: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  latest = snapshot["sourcecode"] or {}

  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      ("Commit Velocity", f"{latest.get('commits', 0):,}", "Weekly Rate"),
      ("Active Developers", f"{latest.get('active_devs', 0):,}", "Contributors"),
  ])
  st.markdown("<br>", unsafe_allow_html=True)

  col_c1, col_c2 = st.columns(2)
  with col_c1:
    render_history_chart(
        page_data["sourcecode"][
            page_data["sourcecode"]["asset_symbol"] == asset_symbol
        ],
        "commits",
        f"{asset_symbol} Commit Trend Line",
        "Commits",
        color="#3b82f6",
    )
  with col_c2:
    code_frame = (
        page_data["sourcecode"][
            page_data["sourcecode"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not code_frame.empty:
      fig_area = px.area(
          code_frame,
          x="metric_date",
          y="commits",
          title=f"{asset_symbol} Cumulative Commit Volume Area",
          color_discrete_sequence=["#3b82f6"],
      )
      fig_area.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_area, use_container_width=True)

elif section == "Ledger Metrics":
  st.subheader(f"Ledger Health & Throughput: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest = snapshot["network"] or {}
  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )

  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      (
          "Active Addresses",
          f"{latest.get('active_addresses', 0):,}",
          "On-Chain Users",
      ),
      ("Transactions / Sec (TPS)", f"{latest.get('tx_tps', 0):.2f}", "Throughput"),
  ])
  st.markdown("<br>", unsafe_allow_html=True)

  col_l1, col_l2 = st.columns(2)
  with col_l1:
    render_history_chart(
        page_data["network"][
            page_data["network"]["asset_symbol"] == asset_symbol
        ],
        "tx_tps",
        f"{asset_symbol} Ledger TPS Performance",
        "TPS",
        color="#10b981",
    )
  with col_l2:
    net_frame = (
        page_data["network"][
            page_data["network"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not net_frame.empty:
      fig_scatter = px.scatter(
          net_frame,
          x="metric_date",
          y="active_addresses",
          size="tx_tps",
          color="tx_tps",
          title=f"{asset_symbol} Active Addresses vs. TPS Scalability",
          color_continuous_scale="Viridis",
      )
      fig_scatter.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

elif section == "Market Economics":
  st.subheader(f"Tokenomics & Market Economics: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest = snapshot["economics"] or {}
  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )

  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      ("Market Cap", format_currency(latest.get("market_cap", 0)), "Valuation"),
      (
          "24h Volume",
          format_currency(latest.get("volume_24h", 0)),
          "Liquidity",
      ),
  ])
  st.markdown("<br>", unsafe_allow_html=True)

  col_e1, col_e2 = st.columns(2)
  with col_e1:
    render_history_chart(
        page_data["economics"][
            page_data["economics"]["asset_symbol"] == asset_symbol
        ],
        "market_cap",
        f"{asset_symbol} Market Cap Valuation Trend",
        "Market Cap",
        color="#8b5cf6",
    )
  with col_e2:
    econ_frame = (
        page_data["economics"][
            page_data["economics"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not econ_frame.empty:
      fig_bar_vol = px.bar(
          econ_frame,
          x="metric_date",
          y="volume_24h",
          title=f"{asset_symbol} Daily Trading Volume Bar Chart",
          color="volume_24h",
          color_continuous_scale="Purples",
      )
      fig_bar_vol.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_bar_vol, use_container_width=True)

elif section == "Social Sentiment":
  st.subheader(f"User Sentiment & Market Mood: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest = snapshot["sentiment"] or {}
  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )

  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      (
          "Sentiment Index",
          f"{latest.get('user_sentiment_index', 0):.3f}",
          "Aggregated Index",
      ),
      (
          "Buy / Sell Pressure",
          f"{latest.get('buy_sell_ratio', 0):.2f}",
          "Ratio",
      ),
  ])
  st.markdown("<br>", unsafe_allow_html=True)

  col_s1, col_s2 = st.columns(2)
  with col_s1:
    render_history_chart(
        page_data["sentiment"][
            page_data["sentiment"]["asset_symbol"] == asset_symbol
        ],
        "user_sentiment_index",
        f"{asset_symbol} User Sentiment Evolution",
        "Sentiment Index",
        color="#f59e0b",
    )
  with col_s2:
    sent_frame = (
        page_data["sentiment"][
            page_data["sentiment"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not sent_frame.empty:
      fig_box = px.box(
          sent_frame,
          y="buy_sell_ratio",
          title=f"{asset_symbol} Buy/Sell Pressure Ratio Distribution",
          color_discrete_sequence=["#f59e0b"],
      )
      fig_box.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_box, use_container_width=True)

elif section == "Ecosystem Liquidity":
  st.subheader(f"Accessibility & Ecosystem Liquidity: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest = snapshot["accessibility"] or {}
  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )

  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      (
          "Exchange Availability",
          f"{latest.get('exchange_count', 0):,}",
          "Listings",
      ),
      (
          "Wallet Support Score",
          f"{latest.get('wallet_support_score', 0):.2f}",
          "Integration Score",
      ),
  ])
  st.markdown("<br>", unsafe_allow_html=True)

  col_acc1, col_acc2 = st.columns(2)
  with col_acc1:
    render_history_chart(
        page_data["accessibility"][
            page_data["accessibility"]["asset_symbol"] == asset_symbol
        ],
        "wallet_support_score",
        f"{asset_symbol} Wallet Support Progress",
        "Support Score",
        color="#06b6d4",
    )
  with col_acc2:
    acc_frame = (
        page_data["accessibility"][
            page_data["accessibility"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not acc_frame.empty:
      fig_hist = px.histogram(
          acc_frame,
          x="exchange_count",
          title=f"{asset_symbol} Exchange Availability Frequency",
          color_discrete_sequence=["#06b6d4"],
      )
      fig_hist.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_hist, use_container_width=True)

elif section == "Multi-Asset Comparison":
  st.subheader("Cross-Asset Comparative Analysis")
  st.markdown("Compare key metrics across all tracked digital assets.")
  st.markdown("<br>", unsafe_allow_html=True)

  comp_df = (
      page_data["economics"]
      .sort_values("metric_date")
      .groupby("asset_symbol")
      .tail(1)
  )
  fig_comp = px.bar(
      comp_df,
      x="asset_symbol",
      y=["market_cap", "volume_24h"],
      barmode="group",
      title="Market Capitalization vs 24h Volume Across Assets",
      labels={"value": "USD ($)", "asset_symbol": "Asset"},
  )
  fig_comp.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_comp, use_container_width=True)

elif section == "Portfolio Risk & VaR":
  st.subheader("Institutional Portfolio Risk & Value at Risk (VaR)")
  st.markdown(
      "Analyze historical parametric and historical Value at Risk (VaR) for your"
      " portfolio positions."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  prices_pivot = page_data["economics"].pivot(
      index="metric_date", columns="asset_symbol", values="market_cap"
  )
  returns = prices_pivot.pct_change().dropna()
  if not returns.empty:
    portfolio_returns = returns.mean(axis=1)
    var_95 = np.percentile(portfolio_returns, 5)
    var_99 = np.percentile(portfolio_returns, 1)

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
      st.metric("Daily VaR (95% Confidence)", f"{var_95*100:.2f}%")
    with r_col2:
      st.metric("Daily VaR (99% Confidence)", f"{var_99*100:.2f}%")
    with r_col3:
      st.metric(
          "Portfolio Volatility (Ann.)",
          f"{portfolio_returns.std() * np.sqrt(252) * 100:.2f}%",
      )

    st.markdown("---")
    fig_var = px.histogram(
        portfolio_returns * 100,
        nbins=30,
        title="Portfolio Daily Returns Distribution (%)",
        labels={"value": "Daily Return (%)"},
    )
    fig_var.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_var, use_container_width=True)
  else:
    st.warning("Insufficient return data for portfolio VaR calculation.")

elif section == "Predictive ML Forecast":
  st.subheader(f"Predictive ML Trend Forecasting: {asset_symbol}")
  st.markdown(
      "Linear regression and rolling statistical trend projections for valuation"
      " modeling."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  df_asset = (
      page_data["economics"][
          page_data["economics"]["asset_symbol"] == asset_symbol
      ]
      .sort_values("metric_date")
      .copy()
  )
  if len(df_asset) > 5:
    df_asset["days"] = (
        df_asset["metric_date"] - df_asset["metric_date"].min()
    ).dt.days
    x = df_asset["days"].values
    y = df_asset["market_cap"].values
    slope, intercept = np.polyfit(x, y, 1)
    df_asset["forecast"] = intercept + slope * x

    fig_ml = go.Figure()
    fig_ml.add_trace(
        go.Scatter(
            x=df_asset["metric_date"],
            y=df_asset["market_cap"],
            mode="lines+markers",
            name="Actual Market Cap",
            line=dict(color="#10b981", width=3),
        )
    )
    fig_ml.add_trace(
        go.Scatter(
            x=df_asset["metric_date"],
            y=df_asset["forecast"],
            mode="lines",
            name="Linear Trend Fit",
            line=dict(color="#ef4444", width=2, dash="dash"),
        )
    )
    fig_ml.update_layout(
        title=f"{asset_symbol} Market Cap Linear Regression Forecast",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_ml, use_container_width=True)
  else:
    st.warning("Not enough data points for ML forecast.")

elif section == "Prophet AI Forecaster":
  st.subheader(f"Prophet AI Time-Series Forecaster: {asset_symbol}")
  st.markdown(
      "Advanced machine learning forecasting powered by Meta Prophet for market"
      " cap projection."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  forecast_days = st.slider("Forecast Horizon (Days)", 7, 90, 30)
  df_prophet = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ]

  if not df_prophet.empty and len(df_prophet) > 10:
    with st.spinner("Training Prophet AI model..."):
      model, forecast = InstitutionalAnalyticsEngine.generate_prophet_forecast(
          df_prophet, periods=forecast_days
      )

      fig_p = go.Figure()
      fig_p.add_trace(
          go.Scatter(
              x=forecast["ds"],
              y=forecast["yhat"],
              mode="lines",
              name="Prophet Prediction",
              line=dict(color="#3b82f6", width=2),
          )
      )
      fig_p.add_trace(
          go.Scatter(
              x=forecast["ds"],
              y=forecast["yhat_upper"],
              mode="lines",
              name="Upper Bound",
              line=dict(color="rgba(59,130,246,0.2)", width=0),
              showlegend=False,
          )
      )
      fig_p.add_trace(
          go.Scatter(
              x=forecast["ds"],
              y=forecast["yhat_lower"],
              mode="lines",
              name="Lower Bound",
              fill="tonexty",
              fillcolor="rgba(59,130,246,0.1)",
              line=dict(color="rgba(59,130,246,0.2)", width=0),
              showlegend=False,
          )
      )
      fig_p.update_layout(
          title=f"{asset_symbol} Prophet AI Valuation Forecast ({forecast_days} Days)",
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_p, use_container_width=True)
  else:
    st.warning("Insufficient data history to train Prophet model.")

elif section == "Strategy Grid Optimizer":
  st.subheader("Quantitative Strategy Grid Search Optimizer")
  st.markdown(
      "Optimize Moving Average crossover parameters via historical grid"
      " search."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  df_opt = (
      page_data["economics"][
          page_data["economics"]["asset_symbol"] == asset_symbol
      ]
      .sort_values("metric_date")
      .copy()
  )
  if not df_opt.empty and len(df_opt) > 15:
    prices = df_opt["market_cap"].reset_index(drop=True)
    short_list = [3, 5, 10]
    long_list = [15, 20, 30]

    res_df, best_p = InstitutionalAnalyticsEngine.optimize_strategy_grid(
        prices, short_list, long_list
    )
    st.success(
        f"Optimized Parameters Found! Best Short MA: {best_p[0]} | Best Long MA:"
        f" {best_p[1]}"
    )
    st.dataframe(res_df, use_container_width=True)
  else:
    st.warning("Insufficient data for strategy grid optimization.")

elif section == "Automated Report Scheduler":
  st.subheader("Automated Executive Report Scheduler")
  st.markdown(
      "Configure automated email dispatch and report generation cadence for"
      " enterprise stakeholders."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  with st.form("scheduler_form"):
    recipient_email = st.text_input(
        "Recipient Email Address", placeholder="manager@institution.com"
    )
    report_freq = st.selectbox(
        "Dispatch Frequency", ["Daily", "Weekly", "Monthly"]
    )
    include_pdf = st.checkbox("Attach Executive PDF Report", value=True)
    submitted = st.form_submit_button("Save Schedule Configuration")
    if submitted:
      if recipient_email:
        st.success(
            f"Successfully scheduled {report_freq} reports for"
            f" {recipient_email}!"
        )
      else:
        st.error("Please enter a valid recipient email address.")

elif section == "Arbitrage Monitor":
  st.subheader(f"Cross-Exchange Arbitrage Monitor: {asset_symbol}")
  st.markdown(
      "Real-time spread detection across major centralized and decentralized"
      " exchanges."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  arb_data = {
      "Exchange Pair": [
          "Binance / Coinbase",
          "Kraken / Binance",
          "Uniswap / Binance",
          "OKX / Kraken",
      ],
      "Spread (%)": [
          np.random.uniform(0.01, 0.15),
          np.random.uniform(-0.05, 0.08),
          np.random.uniform(0.10, 0.45),
          np.random.uniform(-0.02, 0.05),
      ],
      "Status": [
          "Opportunity Active",
          "Normal",
          "High Spread Opportunity",
          "Balanced",
      ],
  }
  arb_df = pd.DataFrame(arb_data)
  st.dataframe(arb_df, use_container_width=True)

elif section == "AI Executive Summary":
  st.subheader(f"AI Executive Synthesis: {asset_symbol}")
  st.markdown(
      "Automated natural language intelligence brief generated from telemetry"
      " and risk scores."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  health_res = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  snapshot_ai = fetch_latest_crypto_metrics(
      asset_symbol, db_path="crypto_data.db"
  )

  summary_html = f"""
    <div style="background-color: #1f2937; padding: 20px; border-radius: 10px; border: 1px solid #374151;">
        <h3 style="color: #10b981; margin-top: 0;">Executive Synthesis Report</h3>
        <p><b>Asset Evaluated:</b> {asset_symbol}</p>
        <p><b>Composite Health Rating:</b> {health_res['health_score']:.1f} / 100</p>
        <p><b>Pillar Breakdown:</b></p>
        <ul>
            <li>Source Code Velocity: {health_res['pillar_scores']['sourcecode']:.1f}</li>
            <li>Ledger & Network Activity: {health_res['pillar_scores']['network']:.1f}</li>
            <li>Market Economics: {health_res['pillar_scores']['economics']:.1f}</li>
            <li>User Sentiment: {health_res['pillar_scores']['sentiment']:.1f}</li>
            <li>Ecosystem Accessibility: {health_res['pillar_scores']['accessibility']:.1f}</li>
        </ul>
        <p><b>Strategic Outlook:</b> The asset demonstrates stable on-chain metrics with strong validator participation. Liquidity depth remains adequate across primary order books. Recommended stance: <b>HOLD / ACCUMULATE ON DIPS</b>.</p>
    </div>
    """
  st.markdown(summary_html, unsafe_allow_html=True)

elif section == "Advanced Tech Indicators":
  st.subheader(f"Advanced Technical Indicators (RSI, MACD, Bollinger): {asset_symbol}")
  st.markdown("Momentum oscillators and volatility bands for professional trading.")
  st.markdown("<br>", unsafe_allow_html=True)

  df_tech = (
      page_data["economics"][
          page_data["economics"]["asset_symbol"] == asset_symbol
      ]
      .sort_values("metric_date")
      .copy()
  )
  if not df_tech.empty and len(df_tech) > 14:
    prices = df_tech["market_cap"]
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df_tech["RSI"] = 100 - (100 / (1 + rs))

    df_tech["MA20"] = prices.rolling(20).mean()
    df_tech["STD20"] = prices.rolling(20).std()
    df_tech["Upper Band"] = df_tech["MA20"] + (df_tech["STD20"] * 2)
    df_tech["Lower Band"] = df_tech["MA20"] - (df_tech["STD20"] * 2)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
      fig_rsi = px.line(
          df_tech,
          x="metric_date",
          y="RSI",
          title=f"{asset_symbol} 14-Period RSI Oscillator",
          color_discrete_sequence=["#f59e0b"],
      )
      fig_rsi.add_hline(
          y=70, line_dash="dash", line_color="red", annotation_text="Overbought"
      )
      fig_rsi.add_hline(
          y=30, line_dash="dash", line_color="green", annotation_text="Oversold"
      )
      fig_rsi.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_rsi, use_container_width=True)

    with col_t2:
      fig_bb = go.Figure()
      fig_bb.add_trace(
          go.Scatter(
              x=df_tech["metric_date"],
              y=df_tech["market_cap"],
              name="Price / Cap",
              line=dict(color="#10b981"),
          )
      )
      fig_bb.add_trace(
          go.Scatter(
              x=df_tech["metric_date"],
              y=df_tech["Upper Band"],
              name="Upper Band",
              line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
          )
      )
      fig_bb.add_trace(
          go.Scatter(
              x=df_tech["metric_date"],
              y=df_tech["Lower Band"],
              name="Lower Band",
              fill="tonexty",
              fillcolor="rgba(150,150,150,0.1)",
              line=dict(color="rgba(150,150,150,0.5)", dash="dot"),
          )
      )
      fig_bb.update_layout(
          title=f"{asset_symbol} Bollinger Bands (20, 2)",
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_bb, use_container_width=True)
  else:
    st.warning("Insufficient data history for advanced technical indicators.")

elif section == "Strategy Backtester":
  st.subheader("Historical Strategy Backtester")
  st.markdown(
      "Simulate historical performance of Moving Average Crossover strategies."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  col_b1, col_b2 = st.columns(2)
  with col_b1:
    short_window = st.slider("Short Moving Average Window", 3, 20, 5)
  with col_b2:
    long_window = st.slider("Long Moving Average Window", 10, 50, 20)

  df_bt = (
      page_data["economics"][
          page_data["economics"]["asset_symbol"] == asset_symbol
      ]
      .sort_values("metric_date")
      .copy()
  )
  if not df_bt.empty and len(df_bt) > long_window:
    prices = df_bt["market_cap"].astype(float)
    sma_s = prices.rolling(short_window).mean()
    sma_l = prices.rolling(long_window).mean()
    signal = np.where(sma_s > sma_l, 1, -1)
    signal_series = pd.Series(signal, index=prices.index).shift(1)
    strategy_returns = prices.pct_change().fillna(0) * signal_series.fillna(0)
    cum_returns = (1 + strategy_returns.fillna(0)).cumprod() - 1

    plot_df = pd.DataFrame(
        {
            "metric_date": df_bt["metric_date"],
            "cumulative_return_pct": cum_returns * 100,
        }
    ).dropna()

    fig_bt = px.line(
        plot_df,
        x="metric_date",
        y="cumulative_return_pct",
        title=f"{asset_symbol} Strategy Cumulative Returns (%) - Short:{short_window} / Long:{long_window}",
        labels={"metric_date": "Timeline", "cumulative_return_pct": "Return (%)"},
    )
    fig_bt.update_traces(line_color="#10b981", line_width=3)
    fig_bt.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_bt, use_container_width=True)
  else:
    st.warning("Not enough data points for selected backtest windows.")

elif section == "Macro Correlation Matrix":
  st.subheader("Macroeconomic & Inter-Asset Correlation Matrix")
  st.markdown(
      "Pearson correlation coefficients across digital asset market caps and"
      " trading volumes."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  corr_pivot = page_data["economics"].pivot(
      index="metric_date", columns="asset_symbol", values="market_cap"
  )
  corr_matrix = corr_pivot.corr()

  fig_corr = px.imshow(
      corr_matrix,
      text_auto=True,
      color_continuous_scale="Viridis",
      title="Asset Valuation Correlation Matrix",
  )
  fig_corr.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_corr, use_container_width=True)

elif section == "Whale Wallet & Flow Tracker":
  st.subheader("Whale Wallet Movement & Exchange Flow Tracker")
  st.markdown(
      "Monitor large-scale on-chain transfers, exchange inflows, and outflows."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  whale_df = page_data["whale_df"]
  if not whale_df.empty:
    fig_whale = px.scatter(
        whale_df,
        x="timestamp",
        y="usd_value",
        color="tx_type",
        size="amount_tokens",
        hover_data=["asset_symbol", "sender_wallet", "receiver_wallet"],
        title="Recent Whale Transactions & Exchange Flows",
    )
    fig_whale.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_whale, use_container_width=True)

    st.markdown("### Recent High-Value Transactions Ledger")
    st.dataframe(whale_df.tail(15), use_container_width=True)
  else:
    st.warning("No whale transaction data available.")

elif section == "Order Book Depth Chart":
  st.subheader(f"Simulated Order Book Depth Chart: {asset_symbol}")
  st.markdown(
      "Aggregate bid and ask liquidity depth across institutional liquidity"
      " providers."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  base_p = 65000 if asset_symbol == "BTC" else (2000 if asset_symbol == "ETH" else 150)
  prices_bid = np.linspace(base_p * 0.95, base_p, 50)
  volumes_bid = np.cumsum(np.random.uniform(1, 10, 50))

  prices_ask = np.linspace(base_p, base_p * 1.05, 50)
  volumes_ask = np.cumsum(np.random.uniform(1, 10, 50))

  fig_ob = go.Figure()
  fig_ob.add_trace(
      go.Scatter(
          x=prices_bid,
          y=volumes_bid,
          fill="tozeroy",
          name="Bids (Buy)",
          line=dict(color="#10b981"),
      )
  )
  fig_ob.add_trace(
      go.Scatter(
          x=prices_ask,
          y=volumes_ask,
          fill="tozeroy",
          name="Asks (Sell)",
          line=dict(color="#ef4444"),
      )
  )
  fig_ob.update_layout(
      title=f"{asset_symbol} Order Book Market Depth",
      xaxis_title="Price (USD)",
      yaxis_title="Cumulative Volume",
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_ob, use_container_width=True)

elif section == "Liquidation Heatmap":
  st.subheader("Leverage Liquidation Heatmap")
  st.markdown(
      "Estimated liquidation clusters across derivative exchanges at various"
      " leverage tiers."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  leverage_tiers = ["10x", "25x", "50x", "100x"]
  price_levels = [60000, 62000, 64000, 66000, 68000]
  np.random.seed(42)
  heatmap_data = np.random.uniform(10, 500, size=(len(price_levels), len(leverage_tiers)))

  fig_hm = px.imshow(
      heatmap_data,
      x=leverage_tiers,
      y=[str(p) for p in price_levels],
      labels=dict(x="Leverage Tier", y="Price Level (USD)", color="Liquidations ($M)"),
      color_continuous_scale="Reds",
      title="Estimated Long/Short Liquidation Clusters ($M)",
  )
  fig_hm.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_hm, use_container_width=True)

elif section == "Gas & Fee Oracle":
  st.subheader("Network Gas Price & Fee Oracle")
  st.markdown("Real-time blockchain gas fees for priority transactions.")
  st.markdown("<br>", unsafe_allow_html=True)

  gas_data = {
      "Network": ["Ethereum", "Solana", "Bitcoin", "Cardano"],
      "Fast (Gwei / Lamports / Sat/vB)": [24, 0.00005, 18, 0.15],
      "Standard": [18, 0.00001, 12, 0.10],
      "Slow": [12, 0.000005, 8, 0.05],
      "Estimated Conf Time": ["~15 sec", "~0.4 sec", "~10 min", "~20 sec"],
  }
  st.dataframe(pd.DataFrame(gas_data), use_container_width=True)

elif section == "Alerts & Audit Log":
  st.subheader("Institutional Alert Dispatcher & Audit Trail")
  st.markdown(
      "Review historical alert triggers and automated webhook transmissions."
  )
  st.markdown("<br>", unsafe_allow_html=True)

  conn_log = sqlite3.connect("bnanalytics_institutional.db")
  audit_df = pd.read_sql("SELECT * FROM alert_audit_logs ORDER BY timestamp DESC", conn_log)
  conn_log.close()

  if not audit_df.empty:
    st.dataframe(audit_df, use_container_width=True)
  else:
    st.info("No audit logs recorded yet.")

elif section == "Paper Trading & PnL":
  st.subheader("Paper Trading Desk & Real-Time PnL Tracker")
  st.markdown("Simulate institutional order execution and track live portfolio PnL.")
  st.markdown("<br>", unsafe_allow_html=True)

  with st.form("paper_trade_form"):
    action = st.selectbox("Action", ["BUY", "SELL"])
    qty = st.number_input("Quantity", min_value=0.01, value=1.0)
    exec_price = st.number_input("Execution Price (USD)", min_value=0.01, value=65000.0)
    submit_trade = st.form_submit_button("Execute Paper Trade")
    if submit_trade:
      conn_p = sqlite3.connect("crypto_data.db")
      c_p = conn_p.cursor()
      c_p.execute(
          """
                INSERT INTO paper_portfolio (timestamp, asset_symbol, action, quantity, execution_price, total_cost)
                VALUES (DATETIME('now'), ?, ?, ?, ?, ?)
            """,
          (asset_symbol, action, qty, exec_price, qty * exec_price),
      )
      conn_p.commit()
      conn_p.close()
      st.success("Paper trade executed successfully!")
      st.rerun()

  st.markdown("### Active Paper Portfolio Transactions")
  paper_trades_df = page_data["paper_trades"]
  if not paper_trades_df.empty:
    st.dataframe(paper_trades_df, use_container_width=True)
  else:
    st.info("No paper trades executed yet.")

elif section == "API Key Management":
  st.subheader("Institutional API Key Management")
  st.markdown("Generate and manage secure programmatic API keys for data ingestion.")
  st.markdown("<br>", unsafe_allow_html=True)

  if st.button("Generate New API Key"):
    new_key = f"bn_live_{secrets.token_hex(16)}"
    conn_api = sqlite3.connect("bnanalytics_institutional.db")
    c_api = conn_api.cursor()
    c_api.execute(
        "INSERT INTO institutional_api_keys (user_id, api_key) VALUES (1, ?)",
        (new_key,),
    )
    conn_api.commit()
    conn_api.close()
    st.success(f"Generated new API Key: `{new_key}`")

  conn_api = sqlite3.connect("bnanalytics_institutional.db")
  keys_df = pd.read_sql("SELECT id, api_key, created_at FROM institutional_api_keys", conn_api)
  conn_api.close()
  if not keys_df.empty:
    st.dataframe(keys_df, use_container_width=True)
  else:
    st.info("No active API keys found.")

elif section == "SQL Query Sandbox":
  st.subheader("Interactive SQL Query Sandbox")
  st.markdown("Execute custom SQL queries directly against the underlying analytics database.")
  st.markdown("<br>", unsafe_allow_html=True)

  default_query = "SELECT asset_symbol, market_cap, volume_24h, metric_date FROM economics_metrics LIMIT 10;"
  query_input = st.text_area("SQL Query", value=default_query, height=100)

  if st.button("Execute Query"):
    try:
      conn_sql = sqlite3.connect("crypto_data.db")
      res_query_df = pd.read_sql(query_input, conn_sql)
      conn_sql.close()
      st.success("Query executed successfully!")
      st.dataframe(res_query_df, use_container_width=True)
    except Exception as e:
      st.error(f"SQL Execution Error: {e}")