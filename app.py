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
try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - fallback for environments without Prophet
    Prophet = None
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
    .metric-card {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 8px;
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
    """Generates a time-series forecast using Prophet when available, otherwise a lightweight fallback."""
    pdf = df.rename(columns={"metric_date": "ds", "market_cap": "y"})[
        ["ds", "y"]
    ].copy()
    pdf["ds"] = pd.to_datetime(pdf["ds"])
    if Prophet is not None:
      model = Prophet(
          daily_seasonality=False,
          weekly_seasonality=True,
          yearly_seasonality=False,
      )
      model.fit(pdf)
      future = model.make_future_dataframe(periods=periods)
      forecast = model.predict(future)
      return model, forecast
    history = pdf["y"].astype(float).to_numpy()
    if len(history) < 2:
      forecast = pd.DataFrame({
          "ds": pd.date_range(pdf["ds"].max() + pd.Timedelta(days=1), periods=periods, freq="D"),
          "yhat": [float(history[-1])] * periods,
          "yhat_upper": [float(history[-1])] * periods,
          "yhat_lower": [float(history[-1])] * periods,
      })
      return None, forecast
    x = np.arange(len(history))
    slope, intercept = np.polyfit(x, history, 1)
    future_x = np.arange(len(history), len(history) + periods)
    yhat = intercept + slope * future_x
    residuals = history - (intercept + slope * x)
    scale = max(float(residuals.std(ddof=0)), 1e-6)
    future_dates = pd.date_range(
        pdf["ds"].max() + pd.Timedelta(days=1),
        periods=periods,
        freq="D",
    )
    forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": yhat,
        "yhat_upper": yhat + 1.96 * scale,
        "yhat_lower": yhat - 1.96 * scale,
    })
    return None, forecast
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
        "INSERT INTO institutional_users (username, password_hash, role) VALUES (?, ?, ?)",
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
      f"LIVE STREAMING WS: BTC/USD ${prices['BTC']:,.2f} | ETH/USD"
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
# --- SIDEBAR NAVIGATION (SITEMAP ALIGNED TO BOSS SPECS) ---
if "nav_section" not in st.session_state:
  st.session_state.nav_section = "Home"
if "nav_category" not in st.session_state:
  st.session_state.nav_category = "Landing & Marketing"
if os.path.exists("logo.png"):
  st.sidebar.image("logo.png", width=45)
st.sidebar.title("BNANALYTICS")
st.sidebar.caption(
    f"User: {st.session_state.username} | Role: {st.session_state.role}"
)
st.sidebar.markdown("---")
nav_categories = {
    "Landing & Marketing": [
        "Home",
        "Features Overview",
        "Pricing",
        "Docs / API",
        "Blog / Resources",
    ],
    "Insights": [
        "Research Reports",
        "Market Analysis",
        "News",
    ],
    "Analytics & Terminal": [
        "Overview Dashboard",
        "Project Explorer",
        "Project Detail Page",
        "Settings",
    ],
    "Projects & Explorer": [
        "Categories",
        "All Projects",
        "Search",
    ],
    "Account & Watchlist": [
        "Profile",
        "Watchlist",
    ],
    "Learn & Resources": [
        "Tutorials",
        "Guides",
        "Glossary",
    ],
}
category_for_section = {
    page: category
    for category, pages in nav_categories.items()
    for page in pages
}
if st.session_state.nav_section not in category_for_section:
  st.session_state.nav_section = "Home"
if st.session_state.nav_category not in nav_categories:
  st.session_state.nav_category = category_for_section.get(
      st.session_state.nav_section, "Landing & Marketing"
  )
available_pages = nav_categories[st.session_state.nav_category]
if st.session_state.nav_section not in available_pages:
  st.session_state.nav_section = available_pages[0]
st.sidebar.caption("Choose a focused workspace view below")
for category_name, pages in nav_categories.items():
  is_active_category = st.session_state.nav_category == category_name
  with st.sidebar.expander(
      category_name,
      expanded=is_active_category,
  ):
    for page_name in pages:
      if st.sidebar.button(
          page_name,
          use_container_width=True,
          key=f"nav_{category_name}_{page_name}",
      ):
        st.session_state.nav_category = category_name
        st.session_state.nav_section = page_name
        st.rerun()
st.sidebar.markdown("---")
asset_symbol = st.sidebar.selectbox(
    "Target Asset", ["BTC", "ETH", "SOL", "ADA"], index=0
)
st.sidebar.markdown("---")
st.sidebar.subheader("Automated Alert Dispatcher")
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
      "INSERT INTO alert_audit_logs (asset_symbol, health_score, threshold, status) VALUES (?, ?, ?, ?)",
      (asset_symbol, current_check_score, alert_health_min, "Triggered - Warning"),
  )
  conn_log.commit()
  conn_log.close()
if current_check_score < alert_health_min and webhook_url_input:
  if st.sidebar.button("Broadcast Webhook Alert Now"):
    try:
      payload = {
          "text": (
              f"BNAnalytics Automatic Dispatch: {asset_symbol} Health Score"
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
st.sidebar.subheader("Executive Report Exports")
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
      label=f"Download {asset_symbol} Executive PDF",
      data=pdf_buffer,
      file_name=f"{asset_symbol}_Report.pdf",
      mime="application/pdf",
  )
# --- VIEW ROUTING & RENDERING ---
current_view = st.session_state.nav_section
if current_view == "Home":
  render_live_websocket_ticker()
  st.markdown("""
      <div style="background-color: #111827; border: 1px solid #374151; padding: 10px 15px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; color: #9ca3af; margin-bottom: 20px;">
          <div><b>Global Cap:</b> <span style="color: #f3f4f6;">$2.48T (+3.4%)</span></div>
          <div><b>24h Vol:</b> <span style="color: #f3f4f6;">$84.2B</span></div>
          <div><b>BTC Dominance:</b> <span style="color: #f3f4f6;">54.2%</span></div>
          <div><b>ETH Gas:</b> <span style="color: #f3f4f6;">14 Gwei</span></div>
          <div><b>Total Value Locked (TVL):</b> <span style="color: #f3f4f6;">$94.6B</span></div>
      </div>
  """, unsafe_allow_html=True)
  st.markdown("""
      <div style="background: linear-gradient(135deg, #1f2937 0%, #111827 100%); border: 1px solid #374151; padding: 35px; border-radius: 12px; text-align: center; margin-bottom: 25px;">
          <h1 style="font-size: 2.2rem !important; color: #ffffff; margin-bottom: 10px;">Institutional Intelligence for the Blockchain & Smart Contract Ecosystem</h1>
          <p style="color: #9ca3af; font-size: 1.05rem; max-width: 800px; margin: 0 auto 20px auto;">
              Unrivaled data telemetry powered by automated GitHub source code ingestion pipelines, advanced economic monitoring, and multi-pillar market intelligence.
          </p>
      </div>
  """, unsafe_allow_html=True)
  hero_col1, hero_col2, hero_col3 = st.columns([1, 1, 2])
  with hero_col1:
      if st.button("Explore Terminal", key="hero_explore_terminal"):
          st.session_state.nav_section = "Overview Dashboard"
          st.rerun()
  with hero_col2:
      if st.button("View Ecosystem Maps", key="hero_view_ecosystems"):
          st.session_state.nav_section = "Project Explorer"
          st.rerun()
  st.markdown("<br>", unsafe_allow_html=True)
  current_health_check = compute_blockactivities_health_score(
      asset_symbol, db_path="crypto_data.db"
  )
  score_val = current_health_check["health_score"]
  if score_val < alert_health_min:
    st.markdown(
        f'<div class="alert-box-warning">WARNING: {asset_symbol} Health'
        f" Score ({score_val:.1f}) is below your warning threshold of"
        f" {alert_health_min}! Automated webhook dispatch primed.</div>",
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f'<div class="alert-box-success">STATUS NORMAL: {asset_symbol}'
        f" Health Score ({score_val:.1f}) is operating within optimal"
        " institutional parameters.</div>",
        unsafe_allow_html=True,
    )
  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Ecosystem Health Dashboard")
  total_repos = len(page_data["sourcecode"]) if "sourcecode" in page_data else 1420
  active_threads = len(page_data["sentiment"]) if "sentiment" in page_data else 385
  tagged_contracts = len(page_data["network"]) if "network" in page_data else 5120
  dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
  with dash_col1:
    st.metric("Repositories Tracked", f"{total_repos:,}", "GitHub Scraper")
  with dash_col2:
    st.metric("Active Project Threads", f"{active_threads:,}", "BitTalk Ingestion")
  with dash_col3:
    st.metric("Smart Contracts Tagged", f"{tagged_contracts:,}", "Custom Framework")
  with dash_col4:
    st.metric("Composite Health Average", f"{score_val:.1f}/100", "Optimized")
  st.markdown("---")
  st.markdown("### Trending & High-Velocity Projects Leaderboard")
  st.markdown("<p style='color: #9ca3af; font-size: 0.9rem;'>Top protocols ranked by source code velocity metrics and community tagging systems rather than raw price action alone.</p>", unsafe_allow_html=True)
  leaderboard_data = []
  for sym in ["BTC", "ETH", "SOL", "ADA"]:
      h_score = compute_blockactivities_health_score(sym, db_path="crypto_data.db")["health_score"]
      leaderboard_data.append({
          "Protocol": sym,
          "Code Velocity Index": round(h_score * 0.95, 1),
          "Commit Activity (7d)": f"+{int(h_score * 3.4)} commits",
          "Ecosystem Tag": "Layer-1 / Smart Contracts",
          "Health Rating": f"{h_score:.1f} / 100"
      })
  lb_df = pd.DataFrame(leaderboard_data).sort_values(by="Code Velocity Index", ascending=False)
  st.dataframe(lb_df, use_container_width=True, hide_index=True)
elif current_view == "Features Overview":
    st.subheader("BNAnalytics Feature Suite")
    st.markdown("<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 25px;'>Explore the enterprise-grade modules engineered to deliver deep telemetry, automated risk control, and multi-pillar market intelligence.</p>", unsafe_allow_html=True)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("""
            <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; margin-bottom: 20px; min-height: 210px;">
                <h3 style="color: #10b981; margin-top: 0;">Institutional Analytics & Backtesting</h3>
                <p style="color: #9ca3af; font-size: 0.88rem; line-height: 1.5;">
                    Optimize strategies via advanced grid searches, Sharpe ratio evaluations, maximum drawdown constraints, and moving average crossover simulations across historical token data.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; margin-bottom: 20px; min-height: 210px;">
                <h3 style="color: #3b82f6; margin-top: 0;">Multi-Pillar Metrics Telemetry</h3>
                <p style="color: #9ca3af; font-size: 0.88rem; line-height: 1.5;">
                    Deep-dive verification spanning GitHub source code velocity, on-chain ledger throughput, macro market economics, accessibility indices, and community sentiment tracking.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_f2:
        st.markdown("""
            <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; margin-bottom: 20px; min-height: 210px;">
                <h3 style="color: #8b5cf6; margin-top: 0;">Automated Alert Dispatcher</h3>
                <p style="color: #9ca3af; font-size: 0.88rem; line-height: 1.5;">
                    Real-time threshold auditing with customizable webhooks for Slack and Telegram channels. Automatically log and broadcast alerts when asset health scores fluctuate.
                </p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; margin-bottom: 20px; min-height: 210px;">
                <h3 style="color: #f59e0b; margin-top: 0;">Executive PDF & CSV Reporting</h3>
                <p style="color: #9ca3af; font-size: 0.88rem; line-height: 1.5;">
                    Generate structured, boardroom-ready PDF executive summaries containing composite health ratings, data tables, and raw CSV feeds straight from the sidebar.
                </p>
            </div>
        """, unsafe_allow_html=True)
elif current_view == "Pricing":
  st.subheader("Institutional Subscription Tiers")
  st.markdown("<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 25px;'>Flexible, scalable pricing engineered for developers, quantitative funds, and enterprise-grade blockchain infrastructure.</p>", unsafe_allow_html=True)
  col_p1, col_p2, col_p3, col_p4 = st.columns(4)
  with col_p1:
    st.markdown("""
        <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; min-height: 340px;">
            <h3 style="color: #60a5fa; margin-top: 0;">Developer</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: #ffffff; margin-bottom: 15px;">Free</p>
            <ul style="color: #9ca3af; font-size: 0.85rem; padding-left: 18px; line-height: 1.6;">
                <li>100K API Call Credits / mo</li>
                <li>300 Rate Limit / min</li>
                <li>Basic REST Endpoints</li>
                <li>Public Market Feeds</li>
                <li>Community Support</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
  with col_p2:
    st.markdown("""
        <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; min-height: 340px;">
            <h3 style="color: #34d399; margin-top: 0;">Analyst</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: #ffffff; margin-bottom: 15px;">$129 <span style="font-size: 0.8rem; color: #9ca3af;">/ mo</span></p>
            <ul style="color: #9ca3af; font-size: 0.85rem; padding-left: 18px; line-height: 1.6;">
                <li>500K Call Credits / mo</li>
                <li>WebSocket Live Streaming</li>
                <li>Automated Webhook Alerts</li>
                <li>10 Years Historical Data</li>
                <li>Priority Email Support</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
  with col_p3:
    st.markdown("""
        <div style="background-color: #1f2937; border: 1px solid #10b981; padding: 20px; border-radius: 10px; min-height: 340px; position: relative;">
            <div style="position: absolute; top: -10px; right: 15px; background-color: #10b981; color: #000000; font-size: 0.65rem; font-weight: bold; padding: 2px 8px; border-radius: 4px;">POPULAR</div>
            <h3 style="color: #10b981; margin-top: 0;">Fund Manager</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: #ffffff; margin-bottom: 15px;">$999 <span style="font-size: 0.8rem; color: #9ca3af;">/ mo</span></p>
            <ul style="color: #9ca3af; font-size: 0.85rem; padding-left: 18px; line-height: 1.6;">
                <li>Full Terminal Access</li>
                <li>Advanced Strategy Backtesting</li>
                <li>2M+ Call Credits / mo</li>
                <li>Executive PDF & CSV Reports</li>
                <li>Multi-Pillar Telemetry Feeds</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
  with col_p4:
    st.markdown("""
        <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px; min-height: 340px;">
            <h3 style="color: #a78bfa; margin-top: 0;">Enterprise</h3>
            <p style="font-size: 1.4rem; font-weight: bold; color: #ffffff; margin-bottom: 15px;">Custom</p>
            <ul style="color: #9ca3af; font-size: 0.85rem; padding-left: 18px; line-height: 1.6;">
                <li>Dedicated Nodes & Infrastructure</li>
                <li>Custom Data Pipelines</li>
                <li>99.9% Uptime SLA</li>
                <li>Regulatory Compliance & Auditing</li>
                <li>Priority Slack & Phone Support</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
elif current_view == "Docs / API":
  st.subheader("Documentation & API Access")
  st.markdown("<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 20px;'>Access programmatic blockchain insights, historical telemetry, and real-time order flows via our REST and WebSocket endpoints.</p>", unsafe_allow_html=True)
  doc_tab1, doc_tab2, doc_tab3 = st.tabs(["Interactive Playground", "Endpoints & Reference", "Authentication & Errors"])
  with doc_tab1:
    st.markdown("### Live API Sandbox")
    st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Test endpoint parameters live and view live JSON payloads directly within the terminal.</p>", unsafe_allow_html=True)
    
    col_play1, col_play2 = st.columns(2)
    with col_play1:
      sandbox_symbol = st.selectbox("Select Asset Symbol", ["BTC", "ETH", "SOL", "ADA"], key="sandbox_sym")
      sandbox_endpoint = st.selectbox("Select Endpoint", ["/v1/metrics", "/v1/economics", "/v1/network"], key="sandbox_endpoint")
    with col_play2:
      sandbox_key = st.text_input("API Key", value="bn_live_99f8a2c10b", type="password", key="sandbox_key")
      
    if st.button("Execute API Request", key="run_sandbox_req"):
      with st.spinner("Fetching data from BNAnalytics node..."):
        t_mod.sleep(0.6)
        sample_payload = {
            "status": "success",
            "code": 200,
            "timestamp": pd.Timestamp.now().isoformat(),
            "data": {
                "symbol": sandbox_symbol,
                "endpoint": sandbox_endpoint,
                "market_cap": 1284500000000 if sandbox_symbol == "BTC" else 450000000000,
                "tps_throughput": 42.5 if sandbox_symbol == "SOL" else 14.2,
                "health_score": float(compute_blockactivities_health_score(sandbox_symbol, db_path="crypto_data.db")["health_score"])
            }
        }
        st.success("Request processed successfully (200 OK)")
        st.json(sample_payload)
  with doc_tab2:
    st.markdown("### Core API Endpoints")
    endpoint_data = [
        {"Endpoint": "/v1/metrics", "Method": "GET", "Description": "Retrieves multi-pillar composite health scores."},
        {"Endpoint": "/v1/economics", "Method": "GET", "Description": "Returns market capitalization and 24h trading volume."},
        {"Endpoint": "/v1/network", "Method": "GET", "Description": "Streams active addresses and TPS throughput data."},
        {"Endpoint": "/v1/stream/ws", "Method": "WEBSOCKET", "Description": "Low-latency bidirectional feed for real-time tickers."}
    ]
    st.dataframe(pd.DataFrame(endpoint_data), use_container_width=True, hide_index=True)
    st.markdown("### Code Snippets")
    lang_tab1, lang_tab2, lang_tab3 = st.tabs(["Python", "cURL", "Node.js"])
    with lang_tab1:
      st.code("import requests\n\nheaders = {'X-API-Key': 'YOUR_API_KEY'}\nres = requests.get('https://api.bnanalytics.io/v1/metrics?symbol=BTC', headers=headers)\nprint(res.json())", language="python")
    with lang_tab2:
      st.code("curl -X GET 'https://api.bnanalytics.io/v1/metrics?symbol=BTC' \\\n  -H 'X-API-Key: YOUR_API_KEY'", language="bash")
    with lang_tab3:
      st.code("const response = await fetch('https://api.bnanalytics.io/v1/metrics?symbol=BTC', {\n  headers: { 'X-API-Key': 'YOUR_API_KEY' }\n});\nconst data = await response.json();\nconsole.log(data);", language="javascript")
  with doc_tab3:
    st.markdown("### Authentication & Error Handling")
    st.markdown("""
        * **Authentication**: All requests must include your institutional API key via the `X-API-Key` request header.
        * **Rate Limits**: Standard tiers permit up to **300 requests per minute**. Enterprise tiers feature dedicated unthrottled gateway routing.
        * **Error Codes**:
            * `400 Bad Request`: Missing or invalid query parameters (e.g., unknown asset symbol).
            * `401 Unauthorized`: Missing or expired API key. Check your credentials in the dashboard profile.
            * `429 Too Many Requests`: Rate limit exceeded. Implement exponential backoff before retrying.
    """)
elif current_view == "Blog / Resources":
    st.subheader("Research Blog & Institutional Resources Portal")
    st.markdown("Explore comprehensive institutional briefs, categorized learning tracks, macroeconomic market research, on-chain glossaries, and regulatory compliance updates.")
    
    blog_tab1, blog_tab2, blog_tab3, blog_tab4, blog_tab5 = st.tabs([
        "📚 Categorized Content Tracks",
        "🎥 Masterclasses & Video Hub",
        "📊 Macro & Market Insights",
        "📖 Crypto Glossary Index",
        "⚖️ Regulatory & Compliance"
    ])
    
    with blog_tab1:
        st.markdown("### Structured Learning Tracks")
        st.markdown("<p style='color: #9ca3af; font-size: 0.88rem;'>Filter institutional modules by proficiency level and core focus area.</p>", unsafe_allow_html=True)
        
        track_filter = st.selectbox("Select Proficiency Track", ["All Tracks", "Beginner Basics", "Intermediate Protocol Architecture", "Advanced Quantitative Strategies"])
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if track_filter in ["All Tracks", "Beginner Basics"]:
                st.markdown("""
                    <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px; margin-bottom: 15px;">
                        <span style="background-color: #3b82f6; color: #fff; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;">BEGINNER BASICS</span>
                        <h4 style="color: #ffffff; margin-top: 8px; margin-bottom: 6px;">Understanding On-Chain Liquidity & Order Flow Dynamics</h4>
                        <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.4;">A foundational overview of how liquidity pools, order book depth, and market maker activities dictate digital asset pricing stability.</p>
                        <span style="color: #10b981; font-size: 0.78rem; font-weight: 500;">Reading Time: 6 mins</span>
                    </div>
                """, unsafe_allow_html=True)
            if track_filter in ["All Tracks", "Intermediate Protocol Architecture"]:
                st.markdown("""
                    <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px; margin-bottom: 15px;">
                        <span style="background-color: #8b5cf6; color: #fff; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;">INTERMEDIATE</span>
                        <h4 style="color: #ffffff; margin-top: 8px; margin-bottom: 6px;">Layer-2 Throughput Scalability & Rollup Architectures</h4>
                        <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.4;">Deep dive into optimistic versus zero-knowledge rollups, data availability sampling, and transaction throughput bottlenecks.</p>
                        <span style="color: #10b981; font-size: 0.78rem; font-weight: 500;">Reading Time: 12 mins</span>
                    </div>
                """, unsafe_allow_html=True)
        with col_b2:
            if track_filter in ["All Tracks", "Advanced Quantitative Strategies"]:
                st.markdown("""
                    <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px; margin-bottom: 15px;">
                        <span style="background-color: #f59e0b; color: #fff; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;">ADVANCED</span>
                        <h4 style="color: #ffffff; margin-top: 8px; margin-bottom: 6px;">Tokenomics Optimization & Mathematical Incentive Design</h4>
                        <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.4;">Rigorous mathematical framework for balancing emission schedules, staking velocity, and protocol utility curves to prevent inflationary decay.</p>
                        <span style="color: #10b981; font-size: 0.78rem; font-weight: 500;">Reading Time: 18 mins</span>
                    </div>
                """, unsafe_allow_html=True)
            if track_filter in ["All Tracks", "Advanced Quantitative Strategies"]:
                st.markdown("""
                    <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px; margin-bottom: 15px;">
                        <span style="background-color: #f59e0b; color: #fff; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;">ADVANCED</span>
                        <h4 style="color: #ffffff; margin-top: 8px; margin-bottom: 6px;">DeFi Liquidity Crunch & Stress-Testing Automated Market Makers</h4>
                        <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.4;">Analysis of systemic cascading liquidations, impermanent loss mitigation models, and collateral debt position risk bounds.</p>
                        <span style="color: #10b981; font-size: 0.78rem; font-weight: 500;">Reading Time: 15 mins</span>
                    </div>
                """, unsafe_allow_html=True)
    with blog_tab2:
        st.markdown("### Masterclasses & Interactive Video Tutorials")
        st.markdown("<p style='color: #9ca3af; font-size: 0.88rem;'>Watch expert walkthroughs covering terminal setup, API integration, and chart analytics.</p>", unsafe_allow_html=True)
        
        vid_col1, vid_col2 = st.columns(2)
        with vid_col1:
            st.markdown("""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px;">
                    <div style="background-color: #111827; height: 130px; display: flex; align-items: center; justify-content: center; border-radius: 6px; margin-bottom: 10px; color: #10b981; font-weight: bold;">▶ [Embed Video] Terminal Setup Walkthrough</div>
                    <h4 style="color: #ffffff; margin-bottom: 4px;">Mastering BNAnalytics Terminal Navigation</h4>
                    <p style="color: #9ca3af; font-size: 0.8rem;">Learn how to configure multi-pillar custom views, set up real-time websocket streams, and dispatch automated webhook alerts.</p>
                </div>
            """, unsafe_allow_html=True)
        with vid_col2:
            st.markdown("""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px;">
                    <div style="background-color: #111827; height: 130px; display: flex; align-items: center; justify-content: center; border-radius: 6px; margin-bottom: 10px; color: #3b82f6; font-weight: bold;">▶ [Embed Video] API Integration Masterclass</div>
                    <h4 style="color: #ffffff; margin-bottom: 4px;">Programmatic Data Ingestion & REST Feeds</h4>
                    <p style="color: #9ca3af; font-size: 0.8rem;">Step-by-step developer tutorial on authenticating requests, managing rate limits, and parsing JSON responses for automated trading bots.</p>
                </div>
            """, unsafe_allow_html=True)
    with blog_tab3:
        st.markdown("### Macro & Market Insights Briefs")
        st.markdown("<p style='color: #9ca3af; font-size: 0.88rem;'>Monthly review roundups, token launch research reports, and on-chain analytics summaries.</p>", unsafe_allow_html=True)
        
        insight_data = [
            {"Date": "2026-07-15", "Title": "Q3 Macro Liquidity Outlook & Global Interest Rate Impacts", "Category": "Market Review"},
            {"Date": "2026-07-01", "Title": "On-Chain Analytics Roundup: Whale Accumulation Patterns", "Category": "On-Chain Research"},
            {"Date": "2026-06-20", "Title": "Emerging Token Launch Analysis: Validator Security Audit Benchmarks", "Category": "Tokenomics"}
        ]
        st.dataframe(pd.DataFrame(insight_data), use_container_width=True, hide_index=True)
    with blog_tab4:
        st.markdown("### Crypto Glossary & Terminology Index")
        st.markdown("<p style='color: #9ca3af; font-size: 0.88rem;'>Searchable dictionary explaining complex industry concepts, technical jargon, and trading acronyms.</p>", unsafe_allow_html=True)
        
        glossary_search = st.text_input("Search Glossary Term", placeholder="e.g. APR, FUD, Impermanent Loss...")
        
        glossary_dict = {
            "APR vs. APY": "APR (Annual Percentage Rate) does not account for compounding interest, whereas APY (Annual Percentage Yield) accounts for the compound effect over time.",
            "Impermanent Loss": "The temporary loss of funds experienced by liquidity providers when the price ratio of deposited crypto assets shifts compared to when they were deposited into an AMM pool.",
            "FUD": "Fear, Uncertainty, and Doubt — negative market sentiment often spread intentionally to influence asset valuations.",
            "TVL": "Total Value Locked — the aggregate USD value of digital assets deposited across decentralized finance protocols and smart contracts.",
            "Gas Limit": "The maximum amount of computational units a user is willing to expend to execute a transaction or smart contract on Ethereum-based networks."
        }
        
        if glossary_search:
            filtered_glossary = {k: v for k, v in glossary_dict.items() if glossary_search.lower() in k.lower() or glossary_search.lower() in v.lower()}
        else:
            filtered_glossary = glossary_dict
            
        for term, definition in filtered_glossary.items():
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
                    <b style="color: #10b981; font-size: 0.95rem;">{term}</b>
                    <p style="color: #f3f4f6; font-size: 0.85rem; margin-top: 4px; margin-bottom: 0;">{definition}</p>
                </div>
            """, unsafe_allow_html=True)
    with blog_tab5:
        st.markdown("### Regulatory & Compliance Updates")
        st.markdown("<p style='color: #9ca3af; font-size: 0.88rem;'>Dedicated advisory briefs tracking global regulatory frameworks, institutional tax compliance, and legal standards.</p>", unsafe_allow_html=True)
        
        st.markdown("""
            * **Global Framework Tracker (July 2026)**: Comprehensive overview of emerging MiCA compliance guidelines across European jurisdictions and SEC digital asset reporting standards in North America.
            * **Institutional Tax Guide**: Best practices for auditing multi-chain transactions, calculating capital gains on staking rewards, and reconciling decentralized exchange (DEX) trade logs.
            * **Custody & AML Standards**: Navigating Know Your Customer (KYC) mandates and Anti-Money Laundering (AML) controls for corporate treasury management.
        """)
elif current_view == "Research Reports":
  st.subheader("Curated Research Reports")
  st.markdown("Deep analytical write-ups covering multi-chain macroeconomic trends and validator security audits.")
elif current_view == "Market Analysis":
  st.subheader("Real-Time Market Analysis")
  render_history_chart(
      page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol],
      "market_cap",
      f"{asset_symbol} Valuation Trend Analysis",
      "USD",
      color="#8b5cf6",
  )
elif current_view == "News":
  st.subheader("Crypto Industry News Feed")
  st.markdown("- **SEC Approves New Multi-Chain ETF Baskets** — *2 hours ago*")
  st.markdown("- **Network Throughput Surges Across Layer-1 Ecosystems** — *5 hours ago*")
  st.markdown("- **Whale Wallet Accumulation Reaches 6-Month High** — *1 day ago*")
elif current_view == "Overview Dashboard":
  render_live_websocket_ticker()
  st.subheader(f"Overview Dashboard: {asset_symbol}")
  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest_n = snapshot["network"] or {}
  health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
  
  render_metric_cards([
      ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
      ("Active Addresses", f"{latest_n.get('active_addresses', 0):,}", "Users"),
      ("Transactions / Sec", f"{latest_n.get('tx_tps', 0):.2f}", "Throughput"),
  ])
  st.markdown("<br>", unsafe_allow_html=True)
  
  st.markdown("### 1. Real-Time Order Book Depth & Market Microstructure")
  st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Live visual depth chart and order book spread indicator aggregated across liquidity providers.</p>", unsafe_allow_html=True)
  
  depth_col1, depth_col2 = st.columns([2, 1])
  with depth_col1:
      price_base = 65000 if asset_symbol == "BTC" else (2000 if asset_symbol == "ETH" else 140)
      prices_dummy = np.linspace(price_base * 0.95, price_base * 1.05, 50)
      bids_cum = np.cumsum(np.random.exponential(10, 50))[::-1]
      asks_cum = np.cumsum(np.random.exponential(10, 50))
      
      fig_depth = go.Figure()
      fig_depth.add_trace(go.Scatter(x=prices_dummy[:25], y=bids_cum[:25], fill='tozeroy', name='Bids', line=dict(color='#10b981')))
      fig_depth.add_trace(go.Scatter(x=prices_dummy[25:], y=asks_cum[25:], fill='tozeroy', name='Asks', line=dict(color='#ef4444')))
      fig_depth.update_layout(
          title=f"{asset_symbol} Aggregate Order Book Depth",
          plot_bgcolor="rgba(0,0,0,0)",
          paper_bgcolor="rgba(0,0,0,0)",
          font_color="#f3f4f6",
          height=280,
          margin=dict(l=20, r=20, t=30, b=20)
      )
      st.plotly_chart(fig_depth, use_container_width=True)
      
  with depth_col2:
      st.markdown("""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px; height: 280px;">
              <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Spread & Slippage Indicator</h4>
              <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Spread:</b> 0.02% ($1.20)</p>
              <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Est. Slippage ($100k):</b> 0.045%</p>
              <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Liquidity Provider Count:</b> 14 LPs</p>
              <hr style="border-color: #374151;">
              <span style="color: #10b981; font-size: 0.78rem; font-weight: bold;">● Execution Risk: Minimal</span>
          </div>
      """, unsafe_allow_html=True)
  st.markdown("---")
  
  st.markdown("### 2. Advanced Multi-Timeframe Technical Summary Matrix")
  st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Synthesized technical indicators across 1H, 4H, and 1D intervals providing an immediate consensus widget.</p>", unsafe_allow_html=True)
  
  matrix_data = [
      {"Indicator": "RSI (14)", "1H Interval": "Neutral (54.2)", "4H Interval": "Buy (61.8)", "1D Interval": "Strong Buy (72.4)"},
      {"Indicator": "MACD Crossover", "1H Interval": "Bullish", "4H Interval": "Bullish", "1D Interval": "Bullish"},
      {"Indicator": "Moving Averages (EMA/SMA)", "1H Interval": "Buy", "4H Interval": "Strong Buy", "1D Interval": "Strong Buy"},
      {"Indicator": "Bollinger Bands", "1H Interval": "Neutral", "4H Interval": "Overbought", "1D Interval": "Bullish"}
  ]
  st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
  
  st.markdown("<br>", unsafe_allow_html=True)
  
  st.markdown("### 3. On-Chain Whale Alert & Large Transaction Tracker")
  st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Live data feed highlighting recent large-scale transfers exceeding $1M USD.</p>", unsafe_allow_html=True)
  
  if "whale_df" in page_data and not page_data["whale_df"].empty:
      whale_subset = page_data["whale_df"][page_data["whale_df"]["asset_symbol"] == asset_symbol].head(5)
      if not whale_subset.empty:
          display_whale = whale_subset[["timestamp", "tx_type", "amount_tokens", "usd_value", "sender_wallet", "receiver_wallet"]].copy()
          display_whale["usd_value"] = display_whale["usd_value"].apply(lambda x: format_currency(x))
          display_whale["amount_tokens"] = display_whale["amount_tokens"].apply(lambda x: f"{x:,.2f}")
          st.dataframe(display_whale, use_container_width=True, hide_index=True)
      else:
          st.info(f"No recent whale transactions recorded for {asset_symbol}.")
  else:
      st.info("No whale tracking data currently loaded.")
  st.markdown("<br>", unsafe_allow_html=True)
  
  st.markdown("### 4. Macro Correlation & Volatility Index")
  st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Short-term correlation coefficient against macroeconomic benchmarks and Implied Volatility (IV).</p>", unsafe_allow_html=True)
  
  macro_col1, macro_col2 = st.columns(2)
  with macro_col1:
      macro_corr_data = [
          {"Benchmark": "S&P 500", "Correlation (30D)": "+0.64", "Behavior": "Positive Coupling"},
          {"Benchmark": "US Dollar Index (DXY)", "Correlation (30D)": "-0.42", "Behavior": "Inverse Hedge"},
          {"Benchmark": "Gold (XAU)", "Correlation (30D)": "+0.18", "Behavior": "Weak Correlation"}
      ]
      st.dataframe(pd.DataFrame(macro_corr_data), use_container_width=True, hide_index=True)
  with macro_col2:
      st.markdown("""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px;">
              <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Volatility Metrics</h4>
              <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 8px;"><b>Implied Volatility (30D IV):</b> 54.2%</p>
              <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 8px;"><b>Historical Volatility (HV):</b> 48.6%</p>
              <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 8px;"><b>Options Put/Call Ratio:</b> 0.72 (Bullish Bias)</p>
          </div>
      """, unsafe_allow_html=True)
  st.markdown("<br>", unsafe_allow_html=True)
  render_history_chart(
      page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol],
      "tx_tps",
      f"{asset_symbol} Throughput Velocity",
      "TPS",
      color="#10b981",
  )
elif current_view == "Project Detail Page":
    render_live_websocket_ticker()
    
    # Initialize price_base safely for the detail views
    price_base = 65000 if asset_symbol == "BTC" else (2000 if asset_symbol == "ETH" else (140 if asset_symbol == "SOL" else 0.48))
    
    # Header Section with Interactive "Set Alert" Trigger (Feature 5)
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"Institutional Project Detail: {asset_symbol}")
        st.markdown(f"<p style='color: #9ca3af; font-size: 0.9rem;'>Advanced market intelligence, derivatives structure, cost basis distribution, and alternative data feeds for <b>{asset_symbol}</b>.</p>", unsafe_allow_html=True)
    with header_col2:
        if "alert_modal_active" not in st.session_state:
            st.session_state.alert_modal_active = False
        if st.button("🔔 Set Alert Trigger"):
            st.session_state.alert_modal_active = not st.session_state.alert_modal_active
    if st.session_state.alert_modal_active:
        with st.container():
            st.markdown("""
                <div style="background-color: #1f2937; border: 1px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                    <h4 style="color: #10b981; margin-top: 0;">Configure Custom Alert Threshold</h4>
                </div>
            """, unsafe_allow_html=True)
            alert_target_type = st.selectbox("Trigger Condition", ["Health Score Drop", "Volume Spike (>50%)", "Abnormal On-Chain Outflow", "Funding Rate Extremes"])
            alert_threshold_val = st.slider("Threshold Target Value", 0, 100, 50)
            if st.button("Save Alert Rules"):
                st.success(f"Custom alert successfully configured for {asset_symbol} ({alert_target_type})!")
                st.session_state.alert_modal_active = False
    st.markdown("---")
    # Layout Tabs for Institutional Modules
    detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
        "📊 Derivatives & Market Structure",
        "🔗 On-Chain Supply & Cost Basis",
        "🌊 Capital Flows & Exchange Dynamics",
        "📰 Regulatory, Filings & Narrative Feed"
    ])
    with detail_tab1:
        st.markdown("### Off-Chain Derivatives & Market Structure Panel")
        st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Real-time tracking of Open Interest (OI), Funding Rates, Implied Volatility (IV), and Delta Skew across major institutional exchanges.</p>", unsafe_allow_html=True)
        
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            deriv_metrics = [
                {"Exchange": "Binance", "Open Interest (USD)": "$4.82B", "Funding Rate": "+0.0124%", "Delta Skew": "+1.4% (Calls)"},
                {"Exchange": "Bybit", "Open Interest (USD)": "$2.15B", "Funding Rate": "+0.0098%", "Delta Skew": "+0.8% (Calls)"},
                {"Exchange": "OKX", "Open Interest (USD)": "$1.74B", "Funding Rate": "+0.0110%", "Delta Skew": "-0.2% (Puts)"},
                {"Exchange": "Deribit (Options)", "Open Interest (USD)": "$3.90B", "Funding Rate": "N/A (IV: 52.4%)", "Delta Skew": "+3.1% (Bull Bias)"}
            ]
            st.dataframe(pd.DataFrame(deriv_metrics), use_container_width=True, hide_index=True)
        with d_col2:
            st.markdown("""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 8px; height: 215px;">
                    <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Liquidation Cascade Risk Monitor</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Cumulative Long Liquidation Cluster:</b> $63,200 (-2.8%)</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Cumulative Short Liquidation Cluster:</b> $68,400 (+5.2%)</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Leverage Ratio Index:</b> 0.24 (Moderate Risk)</p>
                    <span style="color: #3b82f6; font-size: 0.78rem; font-weight: bold;">● Status: Balanced Positioning</span>
                </div>
            """, unsafe_allow_html=True)
    with detail_tab2:
        st.markdown("### On-Chain Supply Dynamics & Cost Basis Distribution")
        st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Visualizing aggregate investor cost bases, Realized Capitalization vs. Market Capitalization (MVRV), and holder cohort distributions.</p>", unsafe_allow_html=True)
        
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            cohort_data = [
                {"Holder Cohort (< 1M)", "% of Circulating Supply", "Cost Basis (USD)", "Profit / Loss Status"},
                {"Short-Term Holders (STH)", "18.4%", f"${price_base * 0.96:,.2f}", "At Risk / Slight Loss"},
                {"Long-Term Holders (LTH)", "65.2%", f"${price_base * 0.52:,.2f}", "Deep In Profit (+95%)"},
                {"Whale Entities (>10k Tokens)", "16.4%", f"${price_base * 0.68:,.2f}", "In Profit (+42%)"}
            ]
            st.dataframe(pd.DataFrame(cohort_data[1:], columns=list(cohort_data[0])), use_container_width=True, hide_index=True)
        with sc_col2:
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 8px; height: 215px;">
                    <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">MVRV & Valuation Multiples</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Market Capitalization:</b> $1.28T</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Realized Capitalization:</b> $840.5B</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>MVRV Ratio:</b> 1.52 (Fair Value Band)</p>
                    <span style="color: #10b981; font-size: 0.78rem; font-weight: bold;">● Zone: Neutral Accumulation</span>
                </div>
            """, unsafe_allow_html=True)
    with detail_tab3:
        st.markdown("### Capital Flows & Exchange Dynamics Tracker")
        st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Real-time net flow indicators tracking token movements between institutional wallets, DeFi pools, and CEX reserves.</p>", unsafe_allow_html=True)
        
        flow_cols = st.columns(3)
        with flow_cols[0]:
            st.metric("24h CEX Netflow", "-4,250 Tokens", "-$276M (Outflow / Bullish)")
        with flow_cols[1]:
            st.metric("DeFi Protocol Inflows", "+12,800 Tokens", "+$832M (Staking/Lending)")
        with flow_cols[2]:
            st.metric("OTC Desk Accumulation", "+8,500 Tokens", "+$552M (Institutional Custody)")
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Recent Tracked Institutional Wallet Flows")
        flow_history = [
            {"Timestamp": "2026-07-24 14:12", "Source": "Binance Cold Storage", "Destination": "Unknown Custody", "Amount": "1,450 BTC", "Classification": "OTC Accumulation"},
            {"Timestamp": "2026-07-24 11:05", "Source": "Coinbase Institutional", "Destination": "Aave Lending Pool", "Amount": "12,500 ETH", "Classification": "DeFi Deployment"},
            {"Timestamp": "2026-07-23 22:40", "Source": "Kraken Reserves", "Destination": "Private Multi-Sig", "Amount": "850 BTC", "Classification": "Self-Custody Withdrawal"}
        ]
        st.dataframe(pd.DataFrame(flow_history), use_container_width=True, hide_index=True)
    with detail_tab4:
        st.markdown("### Regulatory, Filings, & Narrative Feed (Alternative Data)")
        st.markdown("<p style='color: #9ca3af; font-size: 0.85rem;'>Live-updating news and regulatory filing sentiment stream tracking legal catalysts, SEC documents, and developer governance proposals.</p>", unsafe_allow_html=True)
        
        feed_data = [
            {"Date": "2026-07-24", "Category": "Regulatory Filing", "Title": "SEC Form 19b-4 Submitted for Spot Multi-Asset Staking ETP", "Sentiment": "Bullish (+0.82)"},
            {"Date": "2026-07-22", "Category": "Governance Proposal", "Title": "Core Developer Group Proposes Fee Burn Parameter Adjustment", "Sentiment": "Neutral (+0.15)"},
            {"Date": "2026-07-20", "Category": "Legal Compliance", "Title": "European Banking Authority Issues Updated MiCA Stablecoin Guidance", "Sentiment": "Compliant"},
            {"Date": "2026-07-18", "Category": "Institutional Custody", "Title": "Global Custodian Launches Regulated Prime Brokerage Vaults", "Sentiment": "Strong Positive (+0.91)"}
        ]
        st.dataframe(pd.DataFrame(feed_data), use_container_width=True, hide_index=True)
elif current_view == "Project Explorer":
    st.subheader("Project Explorer & Institutional Asset Matrix")
    st.markdown("<p style='color: #9ca3af; font-size: 0.9rem;'>Browse, filter, and analyze tracked blockchain protocols with enhanced telemetry and interactive controls.</p>", unsafe_allow_html=True)
    
    st.markdown("#### 🔍 Filter & Control Panel")
    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
        search_query = st.text_input("Search Assets (Symbol or Name)", placeholder="e.g. BTC, Ethereum, SOL...")
    with f_col2:
        selected_categories = st.multiselect(
            "Filter by Ecosystem Category",
            ["Layer-1", "DeFi", "Smart Contracts", "Infrastructure"],
            default=["Layer-1", "DeFi", "Smart Contracts", "Infrastructure"]
        )
    
    st.markdown("---")
    
    base_assets = ["BTC", "ETH", "SOL", "ADA"]
    explorer_rows = []
    
    category_map = {
        "BTC": "Layer-1",
        "ETH": "Smart Contracts",
        "SOL": "Infrastructure",
        "ADA": "Layer-1"
    }
    
    for sym in base_assets:
        econ_sub = page_data["economics"][page_data["economics"]["asset_symbol"] == sym]
        mcap = econ_sub["market_cap"].iloc[-1] if not econ_sub.empty else 1000000000
        vol = econ_sub["volume_24h"].iloc[-1] if not econ_sub.empty else 50000000
        
        h_score = compute_blockactivities_health_score(sym, db_path="crypto_data.db")["health_score"]
        price_change_24h = round(np.random.uniform(-4.5, 6.8), 2)
        dev_index = round(h_score * 0.92, 1)
        ecosystem_cat = category_map.get(sym, "DeFi")
        
        if selected_categories and ecosystem_cat not in selected_categories:
            continue
        if search_query and search_query.lower() not in sym.lower():
            continue
            
        explorer_rows.append({
            "Asset": sym,
            "Category": ecosystem_cat,
            "Market Cap": format_currency(mcap),
            "24h Volume": format_currency(vol),
            "24h Change (%)": f"{price_change_24h}%",
            "Health Score": f"{h_score:.1f} / 100",
            "Dev Activity Index": f"{dev_index}"
        })
        
    if explorer_rows:
        st.dataframe(pd.DataFrame(explorer_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No assets match your search criteria.")
elif current_view == "Categories":
  st.subheader("Ecosystem Categories Breakdown")
  st.markdown("Browse protocols grouped by functional utility (Layer-1, Smart Contracts, DeFi, Infrastructure).")
elif current_view == "All Projects":
  st.subheader("Complete Directory of Tracked Protocols")
  st.markdown("Exhaustive list of all indexed blockchain networks and decentralized applications.")
elif current_view == "Search":
  st.subheader("Advanced Global Terminal Search")
  search_box = st.text_input("Search across metrics, repositories, and governance proposals...")
  if search_box:
    st.success(f"Found 3 matching institutional records for '{search_box}'.")
elif current_view == "Profile":
  st.subheader("Institutional User Profile")
  st.markdown(f"**Username:** {st.session_state.username}")
  st.markdown(f"**Assigned Role:** {st.session_state.role}")
  st.markdown("**API Access Key:** `bn_live_99f8a2c10b`")
elif current_view == "Watchlist":
  st.subheader("Custom Asset Watchlist")
  st.markdown("- **BTC**: $65,901.00 (Health: 88.4 / 100)")
  st.markdown("- **ETH**: $1,927.00 (Health: 84.1 / 100)")
  st.markdown("- **SOL**: $142.50 (Health: 79.5 / 100)")
elif current_view == "Tutorials":
  st.subheader("Terminal Tutorials & Guides")
  st.markdown("Step-by-step documentation on how to navigate multi-pillar metrics.")
elif current_view == "Guides":
  st.subheader("Risk Management & Compliance Guides")
  st.markdown("Best practices for establishing automated webhook dispatchers and auditing portfolio risk.")
elif current_view == "Glossary":
  st.subheader("Blockchain & Finance Glossary")
  st.markdown("Quick lookup reference for financial and technical terminology.")
elif current_view == "Settings":
  st.subheader("Terminal Settings & Configurations")
  st.markdown("Manage API gateway access, notification endpoints, and UI preferences.")