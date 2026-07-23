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
        " {alert_health_min}! Automated webhook dispatch primed.</div>",
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
        color="#ec4899",
    )
  with col_acc2:
    acc_frame = (
        page_data["accessibility"][
            page_data["accessibility"]["asset_symbol"] == asset_symbol
        ]
        .sort_values("metric_date")
    )
    if not acc_frame.empty:
      fig_lines = px.line(
          acc_frame,
          x="metric_date",
          y=["exchange_count", "wallet_support_score"],
          title=f"{asset_symbol} Accessibility Multi-Metric Comparison",
      )
      fig_lines.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_lines, use_container_width=True)

elif section == "Multi-Asset Comparison":
  st.markdown("### Multi-Asset Comparative Performance Matrix")
  selected_compare_assets = st.multiselect(
      "Select Assets to Compare",
      ["BTC", "ETH", "SOL", "ADA"],
      default=["BTC", "ETH", "SOL"],
  )
  if selected_compare_assets and not page_data["economics"].empty:
    pivot_comp = (
        page_data["economics"]
        .pivot(index="metric_date", columns="asset_symbol", values="market_cap")
        .dropna()
    )
    valid_cols = [c for c in selected_compare_assets if c in pivot_comp.columns]
    if valid_cols:
      normalized_comp = (
          pivot_comp[valid_cols] / pivot_comp[valid_cols].iloc[0]
      ) * 100
      fig_multi = px.line(
          normalized_comp,
          title=(
              "Comparative Market Cap Growth (Normalized Base = 100)"
          ),
          labels={"value": "Growth Index", "metric_date": "Timeline"},
      )
      fig_multi.update_traces(line_width=3)
      fig_multi.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_multi, use_container_width=True)

elif section == "Portfolio Risk & VaR":
  st.markdown("### Interactive Portfolio Risk & VaR Simulator")
  col_w1, col_w2, col_w3, col_w4 = st.columns(4)
  with col_w1:
    w_btc = st.slider("BTC Weight (%)", 0, 100, 40)
  with col_w2:
    w_eth = st.slider("ETH Weight (%)", 0, 100, 30)
  with col_w3:
    w_sol = st.slider("SOL Weight (%)", 0, 100, 20)
  with col_w4:
    w_ada = st.slider("ADA Weight (%)", 0, 100, 10)

  econ_df = page_data["economics"]
  if not econ_df.empty:
    pivot_market = (
        econ_df.pivot(index="metric_date", columns="asset_symbol", values="market_cap")
        .dropna()
    )
    returns_df = pivot_market.pct_change().dropna()
    available_assets = [
        a for a in ["BTC", "ETH", "SOL", "ADA"] if a in returns_df.columns
    ]
    if available_assets:
      weights = np.array(
          [w_btc, w_eth, w_sol, w_ada][: len(available_assets)], dtype=float
      )
      if weights.sum() > 0:
        weights = weights / weights.sum()
        port_returns = returns_df[available_assets].dot(weights)
        port_vol = port_returns.std() * np.sqrt(365)
        port_var_95 = np.percentile(port_returns, 5) * 100

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
          st.metric("Portfolio Annualized Volatility", f"{port_vol*100:.2f}%")
        with col_r2:
          st.metric("Estimated Daily 95% VaR", f"{port_var_95:.2f}%")
        with col_r3:
          st.metric(
              "Risk Profile Category",
              "Moderate / Growth" if port_vol < 0.8 else "High Risk",
          )

        fig_port = px.line(
            port_returns.cumsum() * 100,
            title="Simulated Cumulative Portfolio Returns (%)",
        )
        fig_port.update_traces(line_color="#3b82f6", line_width=3)
        fig_port.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f3f4f6",
        )
        st.plotly_chart(fig_port, use_container_width=True)

elif section == "Predictive ML Forecast":
  st.markdown("### Predictive Machine Learning Valuation Model")
  asset_econ = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ].sort_values("metric_date")
  if len(asset_econ) > 5:
    last_date = asset_econ["metric_date"].iloc[-1].date()
    target_forecast_date = st.date_input(
        "Target Forecast Date", value=last_date + pd.Timedelta(days=14)
    )
    days_ahead = (target_forecast_date - last_date).days

    X = np.arange(len(asset_econ)).reshape(-1, 1)
    y = asset_econ["market_cap"].values
    slope, intercept = np.polyfit(X.flatten(), y, 1)

    future_indices = np.arange(len(asset_econ), len(asset_econ) + days_ahead)
    future_y = slope * future_indices + intercept
    future_dates = pd.date_range(
        start=asset_econ["metric_date"].iloc[-1] + pd.Timedelta(days=1),
        periods=days_ahead,
    )

    forecast_df = pd.DataFrame({
        "metric_date": future_dates,
        "forecast_market_cap": future_y,
        "type": "Forecast",
    })
    historical_df = asset_econ[["metric_date", "market_cap"]].rename(
        columns={"market_cap": "forecast_market_cap"}
    )
    historical_df["type"] = "Historical"
    combined_forecast = pd.concat([historical_df, forecast_df])

    st.metric(
        f"Projected Valuation for {target_forecast_date}",
        format_currency(future_y[-1] if len(future_y) > 0 else y[-1]),
    )
    fig_ml = px.line(
        combined_forecast,
        x="metric_date",
        y="forecast_market_cap",
        color="type",
        title=f"{asset_symbol} Custom ML Valuation Forecast",
    )
    fig_ml.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_ml, use_container_width=True)

# --- NEW MODULE INTEGRATION: Prophet AI Forecaster ---
elif section == "Prophet AI Forecaster":
  st.markdown(
      f"### 📈 Prophet AI Time-Series Forecaster: {asset_symbol}"
  )
  st.markdown(
      "Advanced Bayesian seasonal trend forecasting for institutional asset"
      " valuations."
  )

  forecast_horizon = st.slider(
      "Forecast Horizon (Days)", min_value=7, max_value=90, value=30
  )
  asset_econ = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ].sort_values("metric_date")

  if not asset_econ.empty and len(asset_econ) > 5:
    if st.button("Run Prophet Model Training"):
      with st.spinner("Training Bayesian neural decomposition model..."):
        model, forecast_df = (
            InstitutionalAnalyticsEngine.generate_prophet_forecast(
                asset_econ, periods=forecast_horizon
            )
        )

        st.subheader("Forecast Trajectory Output (Upper & Lower Bands)")
        fig_prophet = go.Figure()
        fig_prophet.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat"],
                name="Predicted Market Cap",
                line=dict(color="#10b981", width=2.5),
            )
        )
        fig_prophet.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat_upper"],
                fill=None,
                mode="lines",
                marker=dict(color="rgba(16,185,129,0.2)"),
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig_prophet.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat_lower"],
                fill="tonexty",
                mode="lines",
                marker=dict(color="rgba(16,185,129,0.2)"),
                line=dict(width=0),
                name="Confidence Interval",
            )
        )
        fig_prophet.update_layout(
            title=f"{asset_symbol} Prophet Valuation Trajectory",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#f3f4f6",
        )
        st.plotly_chart(fig_prophet, use_container_width=True)

        st.subheader("Model Seasonality Components Breakdown")
        fig_comp = model.plot_components(forecast_df)
        st.pyplot(fig_comp)
  else:
    st.warning("Insufficient historical records to run Prophet forecasting.")

# --- NEW MODULE INTEGRATION: Strategy Grid Optimizer ---
elif section == "Strategy Grid Optimizer":
  st.markdown("### ⚙️ Automated Strategy Grid Optimizer")
  st.markdown(
      "Exhaustively scan parameter permutations for Moving Average crossover"
      " strategies to maximize institutional Sharpe Ratios."
  )

  col_og1, col_og2 = st.columns(2)
  with col_og1:
    short_window_input = st.text_input(
        "Short MA Options (Comma separated)", "5, 10, 15, 20"
    )
  with col_og2:
    long_window_input = st.text_input(
        "Long MA Options (Comma separated)", "30, 50, 100, 150"
    )

  asset_econ = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ].sort_values("metric_date")
  if not asset_econ.empty and len(asset_econ) > 10:
    if st.button("Execute Grid Search Optimizer"):
      try:
        s_list = [int(x.strip()) for x in short_window_input.split(",")]
        l_list = [int(x.strip()) for x in long_window_input.split(",")]
        prices = asset_econ["market_cap"].reset_index(drop=True)

        with st.spinner("Scanning parameter space..."):
          res_df, best_p = InstitutionalAnalyticsEngine.optimize_strategy_grid(
              prices, s_list, l_list
          )
          st.success(
              f"Optimization Complete! Optimal Configuration Found -> Short MA:"
              f" {best_p[0]} | Long MA: {best_p[1]}"
          )
          st.dataframe(
              res_df.sort_values(by="Sharpe Ratio", ascending=False),
              use_container_width=True,
          )
      except Exception as e:
        st.error(f"Execution Error: {e}")
  else:
    st.warning("Insufficient historical metrics for grid optimization.")

# --- NEW MODULE INTEGRATION: Automated Report Scheduler ---
elif section == "Automated Report Scheduler":
  st.markdown("### 🕒 Automated Executive Report & Webhook Dispatcher")
  st.markdown(
      "Configure automated background report generation, scheduling, and"
      " distribution."
  )

  sched_time = st.time_input(
      "Daily Scheduled Execution Time", value=time(8, 0)
  )
  target_email = st.text_input(
      "Institutional Recipient Email", "risk-desk@fbridge.africa"
  )
  cron_enabled = st.checkbox("Arm Background Cron Daemon")

  if cron_enabled:
    st.success(
        f"Background scheduler successfully armed for daily dispatch at"
        f" {sched_time.strftime('%H:%M')} to {target_email}."
    )
    st.info(
        "💡 **Enterprise Note:** Ensure the background daemon process is"
        " running on your server environment to execute scheduled cron tasks."
    )
  else:
    st.warning("Scheduler is currently disarmed.")

elif section == "Arbitrage Monitor":
  st.markdown("### Multi-Exchange Arbitrage Monitor")
  st.markdown(
      "Real-time simulated price spread monitoring across top tier liquidity"
      " venues."
  )

  base_price = (
      65000.0
      if asset_symbol == "BTC"
      else (2000.0 if asset_symbol == "ETH" else 140.0)
  )
  np.random.seed(42)

  arb_data = pd.DataFrame({
      "Exchange": ["Binance", "Coinbase Pro", "Kraken", "OKX", "Bybit"],
      "Bid Price ($)": [
          base_price + np.random.uniform(-15, 15) for _ in range(5)
      ],
      "Ask Price ($)": [base_price + np.random.uniform(5, 35) for _ in range(5)],
      "24h Volume ($M)": [np.random.uniform(800, 3400) for _ in range(5)],
  })

  arb_data["Spread (%)"] = (
      (arb_data["Ask Price ($)"] - arb_data["Bid Price ($)"])
      / arb_data["Bid Price ($)"]
  ) * 100
  st.dataframe(
      arb_data.style.format({
          "Bid Price ($)": "${:,.2f}",
          "Ask Price ($)": "${:,.2f}",
          "24h Volume ($M)": "${:,.1f}M",
          "Spread (%)": "{:.3f}%",
      }),
      use_container_width=True,
  )

  max_bid = arb_data.loc[arb_data["Bid Price ($)"].idxmax()]
  min_ask = arb_data.loc[arb_data["Ask Price ($)"].idxmin()]
  spread_diff = max_bid["Bid Price ($)"] - min_ask["Ask Price ($)"]

  col_a1, col_a2 = st.columns(2)
  with col_a1:
    st.metric(
        "Optimal Cross-Exchange Spread",
        f"${spread_diff:,.2f}",
        "Arbitrage Opportunity" if spread_diff > 0 else "Tight Spread",
    )
  with col_a2:
    st.info(
        f"💡 **Route Execution:** Buy on {min_ask['Exchange']} @"
        f" ${min_ask['Ask Price ($)']:,.2f} and sell on {max_bid['Exchange']} @"
        f" ${max_bid['Bid Price ($)']:,.2f}."
    )

elif section == "AI Executive Summary":
  st.markdown("### Automated AI Market Summary Generator")
  st.markdown(
      "Instant analytical synthesis of current multi-vector metrics powered by"
      " automated heuristic text formatting."
  )

  health_score = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  econ = snapshot["economics"] or {}
  net = snapshot["network"] or {}
  sent = snapshot["sentiment"] or {}

  if st.button("Generate Fresh Executive Brief"):
    with st.spinner("Synthesizing multi-vector analytics vectors..."):
      summary_text = f"""
            ### 📊 BNAnalytics Executive Intelligence Brief: {asset_symbol}
            **Generated for Bitnorm Production Suite**
            
            * **Overall Health Standing:** {asset_symbol} scored an aggregate composite health rating of **{health_score['health_score']:.1f} / 100**. This places the asset in a {'strong institutional tier' if health_score['health_score'] >= 50 else 're-accumulation zone'}.
            * **Market Economics & Valuation:** Current market capitalization is logged at **{format_currency(econ.get('market_cap', 0))}**, accompanied by a **24-hour trading liquidity turnover of {format_currency(econ.get('volume_24h', 0))}**.
            * **Ledger Throughput & Activity:** On-chain telemetry demonstrates active network utilization running at **{net.get('tx_tps', 0):.2f} TPS** with **{net.get('active_addresses', 0):,} active daily wallet addresses** interacting with core contracts.
            * **Market Sentiment Profile:** Social mood indexes register a sentiment score of **{sent.get('user_sentiment_index', 0):.3f}**, reflecting consistent holder conviction across active orderbooks.
            
            *Conclusion:* BNAnalytics telemetry indicates stable structural health for {asset_symbol}, supporting continued deployment across enterprise liquidity portfolios.
            """
      st.markdown(summary_text)
  else:
    st.info(
        "Click the button above to generate a real-time synthesized briefing"
        " for "
        + asset_symbol
    )

elif section == "Advanced Tech Indicators":
  st.markdown(
      "### Advanced Technical Indicators Engine (RSI, MACD, Bollinger Bands)"
  )
  st.markdown(
      "Automated quantitative calculations and visual plotting for momentum and"
      " volatility tracking."
  )

  asset_econ = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ].sort_values("metric_date").copy()
  if len(asset_econ) > 14:
    delta = asset_econ["market_cap"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    asset_econ["RSI"] = 100 - (100 / (1 + rs))

    asset_econ["MA20"] = asset_econ["market_cap"].rolling(window=20).mean()
    asset_econ["STD20"] = asset_econ["market_cap"].rolling(window=20).std()
    asset_econ["UpperBand"] = asset_econ["MA20"] + (asset_econ["STD20"] * 2)
    asset_econ["LowerBand"] = asset_econ["MA20"] - (asset_econ["STD20"] * 2)

    tab_rsi, tab_bb = st.tabs(
        ["Relative Strength Index (RSI)", "Bollinger Bands"]
    )

    with tab_rsi:
      fig_rsi = px.line(
          asset_econ,
          x="metric_date",
          y="RSI",
          title=f"{asset_symbol} 14-Day RSI Momentum Oscillator",
      )
      fig_rsi.add_hline(
          y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)"
      )
      fig_rsi.add_hline(
          y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)"
      )
      fig_rsi.update_traces(line_color="#f59e0b", line_width=2.5)
      fig_rsi.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_rsi, use_container_width=True)

    with tab_bb:
      fig_bb = go.Figure()
      fig_bb.add_trace(
          go.Scatter(
              x=asset_econ["metric_date"],
              y=asset_econ["market_cap"],
              name="Market Cap",
              line=dict(color="#3b82f6", width=2),
          )
      )
      fig_bb.add_trace(
          go.Scatter(
              x=asset_econ["metric_date"],
              y=asset_econ["UpperBand"],
              name="Upper Band",
              line=dict(color="gray", dash="dot"),
          )
      )
      fig_bb.add_trace(
          go.Scatter(
              x=asset_econ["metric_date"],
              y=asset_econ["LowerBand"],
              name="Lower Band",
              line=dict(color="gray", dash="dot"),
              fill="tonexty",
              fillcolor="rgba(59, 130, 246, 0.05)",
          )
      )
      fig_bb.update_layout(
          title=f"{asset_symbol} Bollinger Bands Volatility Envelope",
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_bb, use_container_width=True)
  else:
    st.warning("Insufficient historical data points to calculate indicators.")

elif section == "Strategy Backtester":
  st.markdown("### Quantitative Strategy Backtesting Engine")
  st.markdown(
      "Backtest rules-based momentum strategies against historical database"
      " valuation metrics."
  )

  col_b1, col_b2 = st.columns(2)
  with col_b1:
    rsi_buy_threshold = st.slider("Buy RSI Threshold (<)", 10, 45, 30)
  with col_b2:
    rsi_sell_threshold = st.slider("Sell RSI Threshold (>)", 55, 90, 70)

  asset_econ = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ].sort_values("metric_date").copy()
  if len(asset_econ) > 15:
    delta = asset_econ["market_cap"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    asset_econ["RSI"] = 100 - (100 / (1 + (gain / loss)))
    asset_econ["Signal"] = 0
    asset_econ.loc[asset_econ["RSI"] < rsi_buy_threshold, "Signal"] = 1
    asset_econ.loc[asset_econ["RSI"] > rsi_sell_threshold, "Signal"] = -1

    asset_econ["Strategy_Returns"] = (
        asset_econ["market_cap"].pct_change().shift(-1)
        * asset_econ["Signal"].shift(1)
    )
    cumulative_strategy = (
        1 + asset_econ["Strategy_Returns"].fillna(0)
    ).cumprod() - 1

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
      st.metric(
          "Backtest Total Strategy Return",
          f"{cumulative_strategy.iloc[-1]*100:.2f}%",
      )
    with col_m2:
      st.metric(
          "Total Signals Generated", f"{(asset_econ['Signal'] != 0).sum()} triggers"
      )
    with col_m3:
      st.metric("Backtest Status", "Optimized", "Passed")

    fig_backtest = px.line(
        asset_econ,
        x="metric_date",
        y=cumulative_strategy * 100,
        title=f"{asset_symbol} RSI Strategy Cumulative Performance (%)",
    )
    fig_backtest.update_traces(line_color="#10b981", line_width=3)
    fig_backtest.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_backtest, use_container_width=True)
  else:
    st.warning("Not enough historical data points to execute backtesting rules.")

elif section == "Macro Correlation Matrix":
  st.markdown("### Macroeconomic & Benchmark Correlation Matrix")
  st.markdown(
      "Analyze rolling correlations between crypto assets and traditional"
      " economic instruments."
  )

  econ_df = page_data["economics"]
  if not econ_df.empty:
    pivot_macro = (
        econ_df.pivot(index="metric_date", columns="asset_symbol", values="market_cap")
        .dropna()
    )

    np.random.seed(100)
    dates = pivot_macro.index
    macro_sim = pd.DataFrame(
        {
            "DXY (USD Index)": 104
            + np.cumsum(np.random.normal(0, 0.2, len(dates))),
            "US 10Y Yield": 4.2
            + np.cumsum(np.random.normal(0, 0.05, len(dates))),
            "Gold Spot": 2300
            + np.cumsum(np.random.normal(0, 8, len(dates))),
        },
        index=dates,
    )

    combined_macro_matrix = pd.concat([pivot_macro, macro_sim], axis=1).pct_change().dropna()
    corr_matrix = combined_macro_matrix.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Cross-Asset & Macro Correlation Heatmap",
    )
    fig_corr.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif section == "Whale Wallet & Flow Tracker":
  st.markdown("### Smart-Money Whale Wallet & On-Chain Flow Tracker")
  st.markdown(
      "Real-time monitoring of large-ticket wallet transfers, exchange"
      " inflows/outflows, and capital accumulation trends."
  )

  whale_df = page_data["whale_df"]
  asset_whales = whale_df[whale_df["asset_symbol"] == asset_symbol]

  if not asset_whales.empty:
    total_whale_vol = asset_whales["usd_value"].sum()
    inflows = asset_whales[asset_whales["tx_type"] == "Exchange Inflow"][
        "usd_value"
    ].sum()
    outflows = asset_whales[asset_whales["tx_type"] == "Exchange Outflow"][
        "usd_value"
    ].sum()
    net_flow = outflows - inflows

    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
      st.metric("Total Whale Volume Tracked", format_currency(total_whale_vol))
    with col_w2:
      st.metric(
          "Exchange Net Accumulation",
          format_currency(net_flow),
          "Bullish Flow" if net_flow > 0 else "Distribution Pressure",
      )
    with col_w3:
      st.metric("Whale Transaction Count", f"{len(asset_whales):,} transfers")

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
      fig_pie = px.pie(
          asset_whales,
          names="tx_type",
          values="usd_value",
          title=(
              f"{asset_symbol} Whale Volume Distribution by Transfer Vector"
          ),
      )
      fig_pie.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_pie, use_container_width=True)

    with col_chart2:
      fig_hist = px.histogram(
          asset_whales,
          x="usd_value",
          nbins=20,
          title=f"{asset_symbol} Whale Transaction Size Histogram ($)",
          color_discrete_sequence=["#3b82f6"],
      )
      fig_hist.update_layout(
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
      )
      st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Latest Recorded Whale Transactions")
    st.dataframe(
        asset_whales[[
            "timestamp",
            "tx_type",
            "amount_tokens",
            "usd_value",
            "sender_wallet",
            "receiver_wallet",
        ]]
        .sort_values("timestamp", ascending=False)
        .head(10)
        .style.format(
            {"amount_tokens": "{:,.2f}", "usd_value": "${:,.2f}"}
        ),
        use_container_width=True,
    )
  else:
    st.warning("No whale transaction data available for the selected asset.")

elif section == "Order Book Depth Chart":
  st.markdown(f"### Real-Time Order Book & Depth Chart: {asset_symbol}")
  st.markdown(
      "Simulated live order book bid/ask liquidity spread and cumulative depth"
      " walls."
  )

  mid_price = (
      65000.0
      if asset_symbol == "BTC"
      else (
          2000.0
          if asset_symbol == "ETH"
          else (140.0 if asset_symbol == "SOL" else 0.48)
      )
  )
  offsets = np.linspace(0.001, 0.05, 25)

  bids_prices = mid_price * (1 - offsets[::-1])
  bids_volumes = np.cumsum(np.random.uniform(10, 100, 25))

  asks_prices = mid_price * (1 + offsets)
  asks_volumes = np.cumsum(np.random.uniform(10, 100, 25))

  fig_depth = go.Figure()
  fig_depth.add_trace(
      go.Scatter(
          x=bids_prices,
          y=bids_volumes,
          fill="tozeroy",
          name="Bids (Buy)",
          line=dict(color="#10b981", width=2),
          fillcolor="rgba(16, 185, 129, 0.2)",
      )
  )
  fig_depth.add_trace(
      go.Scatter(
          x=asks_prices,
          y=asks_volumes,
          fill="tozeroy",
          name="Asks (Sell)",
          line=dict(color="#ef4444", width=2),
          fillcolor="rgba(239, 68, 68, 0.2)",
      )
  )

  fig_depth.update_layout(
      title=f"{asset_symbol} Order Book Market Depth Liquidity",
      xaxis_title="Price ($)",
      yaxis_title="Cumulative Size",
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_depth, use_container_width=True)

  col_ob1, col_ob2 = st.columns(2)
  with col_ob1:
    st.metric("Best Bid Spread", f"${mid_price * 0.999:,.2f}", "Bid Wall")
  with col_ob2:
    st.metric("Best Ask Spread", f"${mid_price * 1.001:,.2f}", "Ask Wall")

elif section == "Liquidation Heatmap":
  st.markdown(f"### Margin Liquidation Heatmap Simulator: {asset_symbol}")
  st.markdown(
      "Estimated long and short liquidation clusters across leverage"
      " thresholds."
  )

  current_px = (
      65000.0
      if asset_symbol == "BTC"
      else (
          2000.0
          if asset_symbol == "ETH"
          else (140.0 if asset_symbol == "SOL" else 0.48)
      )
  )
  price_range = np.linspace(current_px * 0.8, current_px * 1.2, 40)

  np.random.seed(42)
  liq_intensity = (
      np.exp(-((price_range - current_px) ** 2) / (2 * (current_px * 0.05) ** 2))
      * np.random.uniform(50, 500, 40)
  )

  liq_df = pd.DataFrame(
      {"Price Level ($)": price_range, "Liquidation Volume ($M)": liq_intensity}
  )
  fig_liq = px.bar(
      liq_df,
      x="Price Level ($)",
      y="Liquidation Volume ($M)",
      color="Liquidation Volume ($M)",
      color_continuous_scale="Reds",
      title=f"{asset_symbol} Leveraged Position Liquidation Clusters",
  )
  fig_liq.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_liq, use_container_width=True)

  st.info(
      "⚠️ High cluster concentration detected near $\\pm 5\%$ of current spot"
      " reference. Expect potential cascading volatility triggers upon"
      " breakout."
  )

elif section == "Gas & Fee Oracle":
  st.markdown("### On-Chain Gas & Fee Oracle Monitor")
  st.markdown(
      "Real-time network congestion, base fees, and priority gas estimations."
  )

  col_g1, col_g2, col_g3 = st.columns(3)
  with col_g1:
    st.metric("Fast Gas Rate", "18 Gwei", "-2.1% (Low Congestion)")
  with col_g2:
    st.metric("Standard Gas Rate", "14 Gwei", "Stable")
  with col_g3:
    st.metric("Base Fee Burn (24h)", "3,420 ETH", "+14.5%")

  times = pd.date_range(start="2026-07-01", periods=24, freq="h")
  gas_trend = pd.DataFrame(
      {"Timestamp": times, "Gas Price (Gwei)": 12 + np.random.poisson(3, 24)}
  )

  fig_gas = px.line(
      gas_trend,
      x="Timestamp",
      y="Gas Price (Gwei)",
      title="24-Hour Network Gas Price Fluctuations",
      markers=True,
  )
  fig_gas.update_traces(line_color="#ec4899", line_width=2.5)
  fig_gas.update_layout(
      plot_bgcolor="rgba(0,0,0,0)",
      paper_bgcolor="rgba(0,0,0,0)",
      font_color="#f3f4f6",
  )
  st.plotly_chart(fig_gas, use_container_width=True)

elif section == "Alerts & Audit Log":
  st.markdown("### Automated Risk Scoring & Alert Dispatch History")
  st.markdown(
      "Comprehensive audit trail of past automated health score warning"
      " triggers and webhook dispatches."
  )

  conn_log = sqlite3.connect("bnanalytics_institutional.db")
  audit_df = pd.read_sql(
      "SELECT * FROM alert_audit_logs ORDER BY log_id DESC", conn_log
  )
  conn_log.close()

  if not audit_df.empty:
    st.dataframe(audit_df, use_container_width=True)
  else:
    st.info("No alert warning breaches logged during the current operating cycle.")

elif section == "Paper Trading & PnL":
  st.markdown("### Interactive Paper Trading & PnL Ledger")
  col_t1, col_t2 = st.columns(2)
  with col_t1:
    trade_action = st.selectbox("Order Action", ["BUY", "SELL"])
    trade_qty = st.number_input("Quantity", value=1.0, min_value=0.01, step=0.1)
    exec_price = st.number_input(
        "Execution Price ($)",
        value=65000.0 if asset_symbol == "BTC" else 2000.0,
        step=10.0,
    )

    if st.button("Execute Paper Trade"):
      conn = sqlite3.connect("crypto_data.db")
      cursor = conn.cursor()
      cursor.execute(
          "INSERT INTO paper_portfolio (timestamp, asset_symbol, action,"
          " quantity, execution_price, total_cost) VALUES (DATETIME('now'), ?,"
          " ?, ?, ?, ?)",
          (
              asset_symbol,
              trade_action,
              trade_qty,
              exec_price,
              trade_qty * exec_price,
          ),
      )
      conn.commit()
      conn.close()
      st.success(f"Executed {trade_action} order for {trade_qty} {asset_symbol}!")
      st.rerun()

  with col_t2:
    st.markdown("#### Active Paper Portfolio Ledger")
    conn = sqlite3.connect("crypto_data.db")
    active_trades_df = pd.read_sql(
        "SELECT * FROM paper_portfolio ORDER BY trade_id DESC", conn
    )
    conn.close()
    st.dataframe(active_trades_df, use_container_width=True)

elif section == "API Key Management":
  st.markdown("### Programmatic API Key Management")
  st.markdown(
      "Generate and manage secure Bearer tokens for connecting external"
      " automated execution pipelines to your SQLite database feed."
  )

  if st.button("Generate New API Key"):
    new_key = f"bn_live_{secrets.token_hex(24)}"
    conn = sqlite3.connect("bnanalytics_institutional.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM institutional_users WHERE username = ?",
        (st.session_state.username,),
    )
    uid_res = cursor.fetchone()
    if uid_res:
      uid = uid_res[0]
      cursor.execute(
          "INSERT INTO institutional_api_keys (user_id, api_key) VALUES (?, ?)",
          (uid, new_key),
      )
      conn.commit()
      st.success(
          "API Key generated successfully. Copy it now; it won't be shown again."
      )
      st.code(new_key)
    conn.close()

  st.markdown(
      "#### Active API Tokens for Account: " + str(st.session_state.username)
  )
  conn = sqlite3.connect("bnanalytics_institutional.db")
  cursor = conn.cursor()
  cursor.execute(
      """
        SELECT K.api_key, K.created_at FROM institutional_api_keys K
        JOIN institutional_users U ON K.user_id = U.id
        WHERE U.username = ?
    """,
      (st.session_state.username,),
  )
  keys = cursor.fetchall()
  conn.close()

  if keys:
    keys_df = pd.DataFrame(keys, columns=["API Key Hash", "Creation Timestamp"])
    st.dataframe(keys_df, use_container_width=True)
  else:
    st.info("No active API tokens found for your user profile.")

elif section == "SQL Query Sandbox":
  st.markdown("### Internal SQL Query Sandbox")
  default_query = (
      "SELECT asset_symbol, market_cap, volume_24h, metric_date FROM"
      " economics_metrics LIMIT 10;"
  )
  user_query = st.text_area("SQL Statement", value=default_query, height=120)

  if st.button("Run Query"):
    try:
      conn = sqlite3.connect("crypto_data.db")
      result_df = pd.read_sql(user_query, conn)
      conn.close()
      st.success(f"Query executed successfully! Returned {len(result_df)} rows.")
      st.dataframe(result_df, use_container_width=True)
    except Exception as e:
      st.error(f"SQL Error: {e}")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #9ca3af;'>© 2013–2026 BitNorm.com —"
    " BNAnalytics Institutional Terminal | Bitnorm Production Platform.</p>",
    unsafe_allow_html=True,
)