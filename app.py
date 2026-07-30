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
            status TEXT,
            webhook_status TEXT DEFAULT 'Not Sent'
        )
    """)
  # Migration safety for older DBs that lack webhook_status
  try:
      cursor.execute("ALTER TABLE alert_audit_logs ADD COLUMN webhook_status TEXT DEFAULT 'Not Sent'")
  except Exception:
      pass
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
          else (3200 if ast == "ETH" else (145 if ast == "SOL" else 0.48))
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
  if frame is None or frame.empty or metric_name not in frame.columns:
    st.info(f"No historical data available for **{title}** yet. Try regenerating data from Settings.")
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
      if "ethereum" in data:
        prices["ETH"] = data["ethereum"]["usd"]
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

def log_alert_event(asset_symbol, health_score, threshold, status, webhook_status="Not Sent"):
    """Centralized audit logger for health-score alerts."""
    try:
        conn = sqlite3.connect("bnanalytics_institutional.db")
        c = conn.cursor()
        c.execute(
            """INSERT INTO alert_audit_logs 
               (asset_symbol, health_score, threshold, status, webhook_status) 
               VALUES (?, ?, ?, ?, ?)""",
            (asset_symbol, health_score, threshold, status, webhook_status),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

def get_alert_history(limit=50):
    """Fetch recent alert audit logs."""
    try:
        conn = sqlite3.connect("bnanalytics_institutional.db")
        df = pd.read_sql(
            f"""SELECT log_id, timestamp, asset_symbol, health_score, threshold, status, 
                       COALESCE(webhook_status, 'Not Sent') as webhook_status
                FROM alert_audit_logs 
                ORDER BY log_id DESC LIMIT {limit}""",
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()

page_data = load_dashboard_data()


def get_data_last_refreshed(db_path="crypto_data.db"):
  """Return a human-readable last-modified time for the metrics database."""
  try:
    if os.path.exists(db_path):
      ts = os.path.getmtime(db_path)
      return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
  except Exception:
    pass
  return "Unknown"


data_last_refreshed = get_data_last_refreshed()

# --- SIDEBAR NAVIGATION ---
if "nav_section" not in st.session_state:
  st.session_state.nav_section = "Home"
if "nav_category" not in st.session_state:
  st.session_state.nav_category = "Landing & Marketing"
if "asset_symbol" not in st.session_state:
  st.session_state.asset_symbol = "BTC"
if "nav_history" not in st.session_state:
  st.session_state.nav_history = []  # list of dicts: {section, category, asset}


def push_nav_history():
  """Save current location so the user can go back."""
  entry = {
      "section": st.session_state.nav_section,
      "category": st.session_state.nav_category,
      "asset": st.session_state.get("asset_symbol", "BTC"),
  }
  # Avoid consecutive duplicates
  if not st.session_state.nav_history or st.session_state.nav_history[-1] != entry:
    st.session_state.nav_history.append(entry)
    # Keep history bounded
    if len(st.session_state.nav_history) > 30:
      st.session_state.nav_history = st.session_state.nav_history[-30:]


def go_back():
  """Pop history and restore previous page (callback-safe)."""
  if st.session_state.nav_history:
    prev = st.session_state.nav_history.pop()
    st.session_state.nav_section = prev["section"]
    st.session_state.nav_category = prev["category"]
    st.session_state.asset_symbol = prev.get("asset", "BTC")


if os.path.exists("logo.png"):
  st.sidebar.image("logo.png", width=45)
st.sidebar.title("BNANALYTICS")
st.sidebar.caption(
    f"User: {st.session_state.username} | Role: {st.session_state.role}"
)
st.sidebar.caption(f"Data refreshed: {data_last_refreshed}")

# Go Back button (visible when history exists)
if st.session_state.nav_history:
  st.sidebar.button(
      "← Go Back",
      use_container_width=True,
      key="sidebar_go_back",
      on_click=go_back,
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
        # Record current page before leaving
        if page_name != st.session_state.nav_section or category_name != st.session_state.nav_category:
          push_nav_history()
        st.session_state.nav_category = category_name
        st.session_state.nav_section = page_name
        st.rerun()

st.sidebar.markdown("---")
asset_symbol = st.sidebar.selectbox(
    "Target Asset",
    ["BTC", "ETH", "SOL", "ADA"],
    key="asset_symbol",  # bound to st.session_state.asset_symbol
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

# --- HARDENED ALERT LOGIC ---
if current_check_score < alert_health_min:
  webhook_status = "Not Sent"
  if webhook_url_input:
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
              webhook_status = "Delivered"
          else:
              webhook_status = f"Failed ({res.status_code})"
      except Exception as e:
          webhook_status = f"Error: {str(e)[:40]}"
  log_alert_event(
      asset_symbol,
      current_check_score,
      alert_health_min,
      "Triggered - Warning",
      webhook_status,
  )
  st.sidebar.warning(f"{asset_symbol} below threshold ({current_check_score:.1f})")
else:
  # Optional: log healthy status occasionally for audit completeness (commented to avoid noise)
  pass

if current_check_score < alert_health_min and webhook_url_input:
  if st.sidebar.button("Broadcast Webhook Alert Now"):
    try:
      payload = {
          "text": (
              f"BNAnalytics Manual Dispatch: {asset_symbol} Health Score"
              f" is {current_check_score:.1f} (Threshold: {alert_health_min})!"
          )
      }
      res = requests.post(webhook_url_input, json=payload, timeout=4)
      if res.status_code in [200, 201]:
        log_alert_event(asset_symbol, current_check_score, alert_health_min, "Manual Broadcast", "Delivered")
        st.sidebar.success("Webhook alert dispatched & logged!")
      else:
        log_alert_event(asset_symbol, current_check_score, alert_health_min, "Manual Broadcast", f"Failed ({res.status_code})")
        st.sidebar.warning(f"Webhook response status {res.status_code}")
    except Exception as e:
      log_alert_event(asset_symbol, current_check_score, alert_health_min, "Manual Broadcast", f"Error: {str(e)[:30]}")
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

# In-page Go Back control (mirrors sidebar; useful on every view)
if st.session_state.nav_history:
  top_back_col, _ = st.columns([1, 5])
  with top_back_col:
    st.button(
        "← Go Back",
        key="main_go_back",
        on_click=go_back,
        use_container_width=True,
    )

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
          push_nav_history()
          st.session_state.nav_category = "Analytics & Terminal"
          st.session_state.nav_section = "Overview Dashboard"
          st.rerun()
  with hero_col2:
      if st.button("View Ecosystem Maps", key="hero_view_ecosystems"):
          push_nav_history()
          st.session_state.nav_category = "Analytics & Terminal"
          st.session_state.nav_section = "Project Explorer"
          st.rerun()
  st.markdown("<br>", unsafe_allow_html=True)

  # Cross-asset health for home summary
  home_health = {}
  for _sym in ["BTC", "ETH", "SOL", "ADA"]:
      home_health[_sym] = compute_blockactivities_health_score(_sym, db_path="crypto_data.db")

  score_val = home_health[asset_symbol]["health_score"]
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
  avg_all = sum(h["health_score"] for h in home_health.values()) / max(len(home_health), 1)

  dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)
  with dash_col1:
    st.metric("Repositories Tracked", f"{total_repos:,}", "GitHub Scraper")
  with dash_col2:
    st.metric("Active Project Threads", f"{active_threads:,}", "BitTalk Ingestion")
  with dash_col3:
    st.metric("Smart Contracts Tagged", f"{tagged_contracts:,}", "Custom Framework")
  with dash_col4:
    st.metric("Avg Health (All Assets)", f"{avg_all:.1f}/100", "Composite")

  # Per-asset health strip
  st.markdown("#### Live Asset Health")
  hcols = st.columns(4)
  for col, sym in zip(hcols, ["BTC", "ETH", "SOL", "ADA"]):
      hs = home_health[sym]["health_score"]
      with col:
          st.metric(sym, f"{hs:.1f}/100", "Alert" if hs < alert_health_min else "OK")

  st.markdown("---")
  st.markdown("### Trending & High-Velocity Leaderboard")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Protocols ranked by composite health and source-code velocity — not price alone."
      "</p>",
      unsafe_allow_html=True,
  )
  leaderboard_data = []
  cat_map = {"BTC": "Layer-1", "ETH": "Smart Contracts", "SOL": "Infrastructure", "ADA": "Layer-1"}
  for sym in ["BTC", "ETH", "SOL", "ADA"]:
      payload = home_health[sym]
      h_score = payload["health_score"]
      pillars = payload.get("pillar_scores", {})
      leaderboard_data.append({
          "Protocol": sym,
          "Category": cat_map.get(sym, "—"),
          "Health": round(h_score, 1),
          "Source Code": round(pillars.get("sourcecode", 0), 1),
          "Network": round(pillars.get("network", 0), 1),
          "Economics": round(pillars.get("economics", 0), 1),
          "Sentiment": round(pillars.get("sentiment", 0), 1),
          "Accessibility": round(pillars.get("accessibility", 0), 1),
          "Status": "Alert" if h_score < alert_health_min else "OK",
      })
  lb_df = pd.DataFrame(leaderboard_data).sort_values(by="Health", ascending=False)
  st.dataframe(lb_df, use_container_width=True, hide_index=True)

  # Alert audit summary (hardened)
  st.markdown("### Recent Alert Activity")
  alert_df = get_alert_history(limit=10)
  if not alert_df.empty:
      st.dataframe(alert_df, use_container_width=True, hide_index=True)
  else:
      st.caption("No alert events logged yet. Alerts appear when health falls below your sidebar threshold.")

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
        st.info("Full research library available under **Insights → Research Reports**.")
    with blog_tab4:
        st.markdown("### Quick Glossary Redirect")
        st.info("The complete institutional glossary lives under **Learn & Resources → Glossary**.")
    with blog_tab5:
        st.markdown("### Regulatory Snapshot")
        st.markdown("""
            - MiCA stablecoin guidance updates (EU)
            - SEC multi-asset ETF filings monitoring
            - Custody & prime-brokerage licensing trackers
        """)

# ===================== INSIGHTS – RESEARCH REPORTS (PHASE 3 POLISH) =====================
elif current_view == "Research Reports":
    st.subheader("Institutional Research Reports")
    st.markdown(
        "<p style='color: #9ca3af; font-size: 0.9rem;'>"
        "Board-ready research briefs covering protocol health, tokenomics, regulatory catalysts, and quantitative frameworks."
        "</p>",
        unsafe_allow_html=True,
    )

    # Filters
    rf_col1, rf_col2, rf_col3 = st.columns([2, 2, 1])
    with rf_col1:
        report_cat = st.selectbox(
            "Category",
            ["All", "Protocol Deep Dive", "Tokenomics", "Regulatory", "Quantitative", "Market Structure"],
            key="research_cat",
        )
    with rf_col2:
        report_search = st.text_input("Search reports", placeholder="e.g. Solana, MVRV, unlock...")
    with rf_col3:
        st.write("")  # spacing
        st.write("")

    research_library = [
        {
            "title": "Bitcoin Network Health & Long-Term Holder Dynamics – Q3 2026",
            "category": "Protocol Deep Dive",
            "date": "2026-07-22",
            "pages": 18,
            "summary": "Comprehensive review of active addresses, STH/LTH supply cohorts, realized price bands, and miner revenue sustainability under current difficulty epochs.",
            "tags": ["BTC", "On-Chain", "HODL"],
        },
        {
            "title": "Ethereum Post-Dencun: Throughput, Blob Economics & L2 Fee Compression",
            "category": "Protocol Deep Dive",
            "date": "2026-07-18",
            "pages": 24,
            "summary": "Analysis of blob gas markets, L2 settlement costs, and the impact on total value secured across the Ethereum rollup ecosystem.",
            "tags": ["ETH", "L2", "Fees"],
        },
        {
            "title": "Solana Token Unlock Calendar & Emission Stress Test",
            "category": "Tokenomics",
            "date": "2026-07-15",
            "pages": 12,
            "summary": "Detailed unlock schedule for the next 90 days with estimated USD impact, circulating supply dilution, and historical price reaction patterns.",
            "tags": ["SOL", "Unlocks", "Supply"],
        },
        {
            "title": "MVRV as a Regime Indicator Across Major L1s",
            "category": "Quantitative",
            "date": "2026-07-10",
            "pages": 15,
            "summary": "Cross-sectional study of Market-Value-to-Realized-Value ratios for BTC, ETH, SOL and ADA. Identifies accumulation and distribution thresholds with back-tested hit rates.",
            "tags": ["MVRV", "Valuation", "Quant"],
        },
        {
            "title": "SEC Multi-Asset Spot ETF Filings – Institutional Implications",
            "category": "Regulatory",
            "date": "2026-07-08",
            "pages": 9,
            "summary": "Summary of recent 19b-4 submissions, custody requirements, and projected AUM inflows under different approval scenarios.",
            "tags": ["SEC", "ETF", "Regulation"],
        },
        {
            "title": "Order-Book Microstructure & Net Taker Flow as Leading Indicators",
            "category": "Market Structure",
            "date": "2026-07-03",
            "pages": 14,
            "summary": "Empirical relationship between aggressive buy/sell imbalance and subsequent 4h–24h returns across the four core assets tracked by BN Analytics.",
            "tags": ["Order Flow", "Microstructure"],
        },
        {
            "title": "Cardano Governance & Treasury Dynamics After Chang Hard Fork",
            "category": "Protocol Deep Dive",
            "date": "2026-06-28",
            "pages": 11,
            "summary": "Review of on-chain governance participation, treasury spend velocity, and implications for long-term protocol sustainability.",
            "tags": ["ADA", "Governance"],
        },
        {
            "title": "Institutional Custody & Staking Yield Landscape 2026",
            "category": "Market Structure",
            "date": "2026-06-20",
            "pages": 16,
            "summary": "Comparative analysis of regulated custody providers, staking APYs, slashing risk, and insurance coverage for BTC, ETH and SOL.",
            "tags": ["Custody", "Staking", "Yield"],
        },
    ]

    filtered = []
    for r in research_library:
        if report_cat != "All" and r["category"] != report_cat:
            continue
        if report_search:
            q = report_search.lower()
            if not (q in r["title"].lower() or q in r["summary"].lower() or any(q in t.lower() for t in r["tags"])):
                continue
        filtered.append(r)

    if not filtered:
        st.info("No reports match the current filters.")
    else:
        st.caption(f"Showing {len(filtered)} report(s)")
        for r in filtered:
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px 20px; border-radius: 10px; margin-bottom: 14px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="background-color:#374151; color:#10b981; font-size:0.7rem; padding:2px 8px; border-radius:4px; font-weight:600;">{r['category']}</span>
                        <span style="color:#6b7280; font-size:0.78rem;">{r['date']} · {r['pages']} pages</span>
                    </div>
                    <h4 style="color:#ffffff; margin:6px 0 8px 0; font-size:1.05rem;">{r['title']}</h4>
                    <p style="color:#9ca3af; font-size:0.85rem; line-height:1.45; margin-bottom:10px;">{r['summary']}</p>
                    <div style="display:flex; gap:6px; flex-wrap:wrap;">
                        {"".join([f'<span style="background:#111827; color:#9ca3af; font-size:0.72rem; padding:2px 7px; border-radius:4px;">{t}</span>' for t in r['tags']])}
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ===================== INSIGHTS – MARKET ANALYSIS =====================
elif current_view == "Market Analysis":
    st.subheader(f"Market Analysis – {asset_symbol}")
    st.markdown(
        "<p style='color: #9ca3af; font-size: 0.9rem;'>"
        "Multi-dimensional market intelligence: overview bias, trading context, AI model outlook, and token unlock telemetry."
        "</p>",
        unsafe_allow_html=True,
    )

    m_tab_overview, m_tab_trading, m_tab_ai, m_tab_unlock = st.tabs([
        "Overview", "Trading Context", "AI Select", "Token Unlock"
    ])

    ma_health = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    ma_composite = ma_health.get("health_score", 0)
    ma_pillars = ma_health.get("pillar_scores", {})
    bias = "Accumulate" if ma_composite >= 65 else ("Hold / Monitor" if ma_composite >= 45 else "Reduce Exposure")

    with m_tab_overview:
        st.markdown("### Composite Market Bias")
        ov1, ov2, ov3 = st.columns(3)
        with ov1:
            st.metric("Health Score", f"{ma_composite:.1f}/100")
        with ov2:
            st.metric("30-Day Bias", "Bullish" if ma_composite >= 55 else "Cautious")
        with ov3:
            st.metric("Recommendation", bias)

        st.markdown("#### Pillar Contribution")
        pcols = st.columns(5)
        for col, (label, key) in zip(pcols, [
            ("Source Code", "sourcecode"),
            ("Network", "network"),
            ("Economics", "economics"),
            ("Sentiment", "sentiment"),
            ("Accessibility", "accessibility"),
        ]):
            with col:
                st.metric(label, f"{ma_pillars.get(key, 0):.1f}")

    with m_tab_trading:
        st.markdown("### Trading Context & Key Levels")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Reference levels derived from recent price action and on-chain cost basis."
            "</p>",
            unsafe_allow_html=True,
        )
        price_base = 65000 if asset_symbol == "BTC" else (3200 if asset_symbol == "ETH" else (145 if asset_symbol == "SOL" else 0.48))
        levels = [
            {"Level": "Strong Support", "Price": f"${price_base * 0.92:,.2f}", "Basis": "LTH cost basis cluster"},
            {"Level": "Near Support", "Price": f"${price_base * 0.97:,.2f}", "Basis": "Recent swing low"},
            {"Level": "Spot Reference", "Price": f"${price_base:,.2f}", "Basis": "Current mid"},
            {"Level": "Near Resistance", "Price": f"${price_base * 1.05:,.2f}", "Basis": "Funding + OI cluster"},
            {"Level": "Strong Resistance", "Price": f"${price_base * 1.12:,.2f}", "Basis": "Prior distribution zone"},
        ]
        st.dataframe(pd.DataFrame(levels), use_container_width=True, hide_index=True)

        st.markdown("#### Net Taker Flow Snapshot")
        try:
            flow_df = compute_net_taker_flow(db_path="crypto_data.db")
            flow_row = flow_df[flow_df["asset_symbol"] == asset_symbol]
            if not flow_row.empty:
                st.dataframe(flow_row[["asset_symbol", "Buy", "Sell", "Net_Flow", "Buy_Sell_Ratio"]], use_container_width=True, hide_index=True)
            else:
                st.caption("No taker-flow rows for this asset yet.")
        except Exception:
            st.caption("Taker-flow summary unavailable.")

    with m_tab_ai:
        st.markdown("### AI Model Outlook")
        ai_col1, ai_col2 = st.columns(2)
        with ai_col1:
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px;">
                    <h4 style="color: #10b981; margin-top: 0; font-size: 0.95rem;">Model Outlook ({asset_symbol})</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Composite Health:</b> {ma_composite:.1f}/100</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>30-Day Bias:</b> {"Bullish" if ma_composite >= 55 else "Cautious"}</p>
                    <span style="color: #10b981; font-size: 0.78rem; font-weight: bold;">● Recommendation: {bias}</span>
                </div>
            """, unsafe_allow_html=True)
        with ai_col2:
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px;">
                    <h4 style="color: #8b5cf6; margin-top: 0; font-size: 0.95rem;">Pillar Inputs</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 4px;">Source Code: {ma_pillars.get('sourcecode', 0):.1f}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 4px;">Network: {ma_pillars.get('network', 0):.1f}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 4px;">Economics: {ma_pillars.get('economics', 0):.1f}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 4px;">Sentiment: {ma_pillars.get('sentiment', 0):.1f}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 0;">Accessibility: {ma_pillars.get('accessibility', 0):.1f}</p>
                </div>
            """, unsafe_allow_html=True)

        econ_hist = page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol]
        if not econ_hist.empty and len(econ_hist) >= 5:
            try:
                _, forecast = InstitutionalAnalyticsEngine.generate_prophet_forecast(econ_hist, periods=14)
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=pd.to_datetime(econ_hist["metric_date"]),
                    y=econ_hist["market_cap"],
                    name="History",
                    line=dict(color="#8b5cf6"),
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast["ds"],
                    y=forecast["yhat"],
                    name="Forecast",
                    line=dict(color="#10b981", dash="dash"),
                ))
                fig_fc.update_layout(
                    title=f"{asset_symbol} Market Cap Forecast (14D)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#f3f4f6",
                    height=320,
                )
                st.plotly_chart(fig_fc, use_container_width=True)
            except Exception:
                st.caption("Forecast chart unavailable for this series.")

    with m_tab_unlock:
        st.markdown("### Token Unlock Schedules & Emission Telemetry")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Upcoming unlocks and estimated supply impact."
            "</p>",
            unsafe_allow_html=True,
        )
        unlock_data = [
            {"Protocol": "SOL", "Unlock Date": "2026-08-01", "Tokens Unlocked": "2,450,000 SOL", "USD Value": "$187.2M", "% Circulating": "0.55%"},
            {"Protocol": "ETH", "Unlock Date": "2026-08-05", "Tokens Unlocked": "45,000 ETH", "USD Value": "$88.1M", "% Circulating": "0.04%"},
            {"Protocol": "ADA", "Unlock Date": "2026-08-12", "Tokens Unlocked": "18,200,000 ADA", "USD Value": "$8.7M", "% Circulating": "0.05%"},
            {"Protocol": "BTC", "Unlock Date": "—", "Tokens Unlocked": "N/A (fixed supply)", "USD Value": "—", "% Circulating": "—"},
        ]
        st.dataframe(pd.DataFrame(unlock_data), use_container_width=True, hide_index=True)

# ===================== INSIGHTS – NEWS (PHASE 3 POLISH) =====================
elif current_view == "News":
  st.subheader("Crypto Industry News Feed")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Curated headlines relevant to institutional monitoring. Filter by category or search keywords.</p>",
      unsafe_allow_html=True,
  )

  news_items = [
      {"Time": "2 hours ago", "Headline": "SEC Approves New Multi-Chain ETF Baskets", "Tag": "Regulatory", "Source": "Bloomberg Crypto"},
      {"Time": "5 hours ago", "Headline": "Network Throughput Surges Across Layer-1 Ecosystems", "Tag": "Network", "Source": "The Block"},
      {"Time": "8 hours ago", "Headline": "Major Custody Provider Expands SOL Staking Support", "Tag": "Accessibility", "Source": "CoinDesk"},
      {"Time": "1 day ago", "Headline": "Whale Wallet Accumulation Reaches 6-Month High", "Tag": "On-Chain", "Source": "Glassnode Insights"},
      {"Time": "1 day ago", "Headline": "Developer Commit Velocity Rises on Major L1 Repos", "Tag": "Source Code", "Source": "Electric Capital"},
      {"Time": "2 days ago", "Headline": "Institutional Custody Providers Expand Staking Support", "Tag": "Accessibility", "Source": "Cointelegraph"},
      {"Time": "2 days ago", "Headline": "Funding Rates Turn Deeply Negative on Perpetual Futures", "Tag": "Derivatives", "Source": "Coinglass"},
      {"Time": "3 days ago", "Headline": "MiCA Stablecoin Guidance Updated by European Banking Authority", "Tag": "Regulatory", "Source": "EU Official Journal"},
      {"Time": "3 days ago", "Headline": "Solana Token Unlock of 2.45M SOL Scheduled for Early August", "Tag": "Tokenomics", "Source": "Token Unlocks"},
      {"Time": "4 days ago", "Headline": "Active Addresses on Bitcoin Hit Multi-Month Peak", "Tag": "Network", "Source": "Blockchain.com"},
      {"Time": "5 days ago", "Headline": "Open Interest on ETH Options Hits Record at Deribit", "Tag": "Derivatives", "Source": "Deribit Insights"},
      {"Time": "6 days ago", "Headline": "GitHub Activity for Core Ethereum Clients Remains Elevated", "Tag": "Source Code", "Source": "CryptoMiso"},
  ]

  n_col1, n_col2 = st.columns([2, 3])
  with n_col1:
      news_tag = st.selectbox("Filter by Tag", ["All"] + sorted(list(set(n["Tag"] for n in news_items))))
  with n_col2:
      news_q = st.text_input("Search headlines", placeholder="e.g. ETF, whale, unlock...")

  filtered_news = []
  for n in news_items:
      if news_tag != "All" and n["Tag"] != news_tag:
          continue
      if news_q and news_q.lower() not in n["Headline"].lower() and news_q.lower() not in n["Tag"].lower():
          continue
      filtered_news.append(n)

  if not filtered_news:
      st.info("No headlines match the current filters.")
  else:
      st.caption(f"{len(filtered_news)} headline(s)")
      for n in filtered_news:
          st.markdown(f"""
              <div style="background-color: #1f2937; border: 1px solid #374151; padding: 14px 16px; border-radius: 8px; margin-bottom: 10px;">
                  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                      <span style="color: #10b981; font-size: 0.75rem; font-weight: 600;">{n['Tag']}</span>
                      <span style="color: #6b7280; font-size: 0.75rem;">{n['Time']} · {n['Source']}</span>
                  </div>
                  <p style="color: #f3f4f6; font-size: 0.92rem; margin: 4px 0 0 0;">{n['Headline']}</p>
              </div>
          """, unsafe_allow_html=True)

elif current_view == "Overview Dashboard":
  render_live_websocket_ticker()
  st.subheader(f"Overview Dashboard: {asset_symbol}")
  st.markdown(
      f"<p style='color: #9ca3af; font-size: 0.85rem;'>"
      f"Institutional snapshot for <b>{asset_symbol}</b> — composite health, macro context, "
      f"order-book structure, technicals, and on-chain activity.</p>",
      unsafe_allow_html=True,
  )

  ov_sub_tab1, ov_sub_tab2, ov_sub_tab3, ov_sub_tab4 = st.tabs([
      "Macro Overview", "Order Flow & Depth", "Technical Indicators", "On-Chain Activity"
  ])

  snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
  latest_n = snapshot["network"] or {}
  latest_e = snapshot["economics"] or {}
  health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
  ov_pillars = health_score.get("pillar_scores", {})
  ov_composite = health_score.get("health_score", 0)

  with ov_sub_tab1:
      render_metric_cards([
          ("Health Score", f"{ov_composite:.1f}/100", "Composite"),
          ("Market Cap", format_currency(latest_e.get("market_cap", 0)), "Economics"),
          ("Active Addresses", f"{latest_n.get('active_addresses', 0):,}", "Network"),
          ("TPS", f"{latest_n.get('tx_tps', 0):.2f}", "Throughput"),
      ])

      st.markdown("#### 5-Pillar Snapshot")
      pcols = st.columns(5)
      for col, (label, key) in zip(pcols, [
          ("Source Code", "sourcecode"),
          ("Network", "network"),
          ("Economics", "economics"),
          ("Sentiment", "sentiment"),
          ("Accessibility", "accessibility"),
      ]):
          with col:
              st.metric(label, f"{ov_pillars.get(key, 0):.1f}")

      st.markdown("<br>", unsafe_allow_html=True)
      st.markdown("### Macro Correlation & Volatility Index")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Short-term correlation against macro benchmarks and implied volatility context."
          "</p>",
          unsafe_allow_html=True,
      )

      macro_col1, macro_col2 = st.columns(2)
      with macro_col1:
          macro_corr_data = [
              {"Benchmark": "S&P 500", "Correlation (30D)": "+0.64", "Behavior": "Positive Coupling"},
              {"Benchmark": "US Dollar Index (DXY)", "Correlation (30D)": "-0.42", "Behavior": "Inverse Hedge"},
              {"Benchmark": "Gold (XAU)", "Correlation (30D)": "+0.18", "Behavior": "Weak Correlation"},
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

      if st.button("Open full Project Detail →", key="ov_to_detail"):
          push_nav_history()
          st.session_state.nav_category = "Analytics & Terminal"
          st.session_state.nav_section = "Project Detail Page"
          st.rerun()

  with ov_sub_tab2:
      st.markdown("### Order Book Depth & Market Microstructure")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Aggregate depth visualization and execution-quality indicators. "
          "Will connect to proprietary exchange books when the exchange module is live."
          "</p>",
          unsafe_allow_html=True,
      )

      depth_col1, depth_col2 = st.columns([2, 1])
      with depth_col1:
          price_base = 65000 if asset_symbol == "BTC" else (3200 if asset_symbol == "ETH" else (145 if asset_symbol == "SOL" else 0.48))
          prices_dummy = np.linspace(price_base * 0.95, price_base * 1.05, 50)
          bids_cum = np.cumsum(np.random.exponential(10, 50))[::-1]
          asks_cum = np.cumsum(np.random.exponential(10, 50))

          fig_depth = go.Figure()
          fig_depth.add_trace(go.Scatter(
              x=prices_dummy[:25], y=bids_cum[:25], fill="tozeroy", name="Bids", line=dict(color="#10b981")
          ))
          fig_depth.add_trace(go.Scatter(
              x=prices_dummy[25:], y=asks_cum[25:], fill="tozeroy", name="Asks", line=dict(color="#ef4444")
          ))
          fig_depth.update_layout(
              title=f"{asset_symbol} Aggregate Order Book Depth",
              plot_bgcolor="rgba(0,0,0,0)",
              paper_bgcolor="rgba(0,0,0,0)",
              font_color="#f3f4f6",
              height=280,
              margin=dict(l=20, r=20, t=30, b=20),
          )
          st.plotly_chart(fig_depth, use_container_width=True)

      with depth_col2:
          st.markdown(f"""
              <div style="background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 8px; height: 280px;">
                  <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Spread & Slippage</h4>
                  <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Spread:</b> 0.02%</p>
                  <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Est. Slippage ($100k):</b> 0.045%</p>
                  <p style="color: #9ca3af; font-size: 0.8rem; margin-bottom: 8px;"><b>Liquidity Providers:</b> 14 LPs</p>
                  <hr style="border-color: #374151;">
                  <span style="color: #10b981; font-size: 0.78rem; font-weight: bold;">● Execution Risk: Minimal</span>
              </div>
          """, unsafe_allow_html=True)

      st.markdown("#### Net Taker Flow (from trade ledger)")
      try:
          flow_df = compute_net_taker_flow(db_path="crypto_data.db")
          flow_row = flow_df[flow_df["asset_symbol"] == asset_symbol]
          if not flow_row.empty:
              st.dataframe(flow_row[["asset_symbol", "Buy", "Sell", "Net_Flow", "Buy_Sell_Ratio"]], use_container_width=True, hide_index=True)
          else:
              st.caption("No taker-flow rows for this asset yet.")
      except Exception:
          st.caption("Taker-flow summary unavailable.")

  with ov_sub_tab3:
      st.markdown("### Multi-Timeframe Technical Summary")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Consensus view across 1H, 4H, and 1D intervals."
          "</p>",
          unsafe_allow_html=True,
      )
      matrix_data = [
          {"Indicator": "RSI (14)", "1H": "Neutral (54.2)", "4H": "Buy (61.8)", "1D": "Strong Buy (72.4)"},
          {"Indicator": "MACD Crossover", "1H": "Bullish", "4H": "Bullish", "1D": "Bullish"},
          {"Indicator": "Moving Averages", "1H": "Buy", "4H": "Strong Buy", "1D": "Strong Buy"},
          {"Indicator": "Bollinger Bands", "1H": "Neutral", "4H": "Overbought", "1D": "Bullish"},
      ]
      st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)
      st.success("Overall technical bias: **Bullish** across higher timeframes.")

  with ov_sub_tab4:
      st.markdown("### On-Chain Activity & Whale Tracker")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Large transfers and network throughput for the selected asset."
          "</p>",
          unsafe_allow_html=True,
      )

      oc1, oc2 = st.columns(2)
      with oc1:
          st.metric("Active Addresses", f"{latest_n.get('active_addresses', 0):,}")
      with oc2:
          st.metric("Gas (gwei)", f"{latest_n.get('gas_fee_gwei', 0):.2f}")

      if "whale_df" in page_data and not page_data["whale_df"].empty:
          whale_subset = page_data["whale_df"][page_data["whale_df"]["asset_symbol"] == asset_symbol].head(8)
          if not whale_subset.empty:
              display_whale = whale_subset[["timestamp", "tx_type", "amount_tokens", "usd_value", "sender_wallet", "receiver_wallet"]].copy()
              display_whale["usd_value"] = display_whale["usd_value"].apply(lambda x: format_currency(x))
              display_whale["amount_tokens"] = display_whale["amount_tokens"].apply(lambda x: f"{x:,.2f}")
              st.dataframe(display_whale, use_container_width=True, hide_index=True)
          else:
              st.info(f"No recent whale transactions recorded for **{asset_symbol}**.")
      else:
          st.info("No whale tracking data loaded. Try regenerating data from Settings.")

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
    price_base = 65000 if asset_symbol == "BTC" else (3200 if asset_symbol == "ETH" else (145 if asset_symbol == "SOL" else 0.48))

    detail_health = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    detail_snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    detail_econ = detail_snapshot.get("economics") or {}
    detail_net = detail_snapshot.get("network") or {}
    detail_src = detail_snapshot.get("sourcecode") or {}
    detail_sent = detail_snapshot.get("sentiment") or {}
    detail_acc = detail_snapshot.get("accessibility") or {}
    pillar = detail_health.get("pillar_scores", {})
    composite = detail_health.get("health_score", 0)

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.subheader(f"Institutional Project Detail: {asset_symbol}")
        st.markdown(
            f"<p style='color: #9ca3af; font-size: 0.9rem;'>"
            f"Core multi-pillar intelligence, market structure, on-chain supply, capital flows, "
            f"and alternative data for <b>{asset_symbol}</b>.</p>",
            unsafe_allow_html=True,
        )
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
            alert_target_type = st.selectbox(
                "Trigger Condition",
                ["Health Score Drop", "Volume Spike (>50%)", "Abnormal On-Chain Outflow", "Funding Rate Extremes"],
            )
            alert_threshold_val = st.slider("Threshold Target Value", 0, 100, 50)
            if st.button("Save Alert Rules"):
                st.success(f"Custom alert successfully configured for {asset_symbol} ({alert_target_type})!")
                st.session_state.alert_modal_active = False

    render_metric_cards([
        ("Composite Health", f"{composite:.1f}/100", "5-Pillar Score"),
        ("Market Cap", format_currency(detail_econ.get("market_cap", 0)), "Economics"),
        ("24h Volume", format_currency(detail_econ.get("volume_24h", 0)), "Liquidity"),
        ("Network TPS", f"{detail_net.get('tx_tps', 0):.2f}", "Throughput"),
    ])
    st.markdown("---")

    detail_tab0, detail_tab1, detail_tab2, detail_tab3, detail_tab4 = st.tabs([
        "🏛️ 5 Pillars",
        "📊 Derivatives & Market Structure",
        "🔗 On-Chain Supply & Cost Basis",
        "🌊 Capital Flows & Exchange",
        "📰 Regulatory & Narrative",
    ])

    with detail_tab0:
        st.markdown("### Multi-Pillar Health Breakdown")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "The composite Health Score is a weighted blend of five institutional pillars: "
            "Source Code, Network, Economics, Sentiment, and Accessibility."
            "</p>",
            unsafe_allow_html=True,
        )

        pillar_rows = [
            {
                "Pillar": "Source Code",
                "Score": f"{pillar.get('sourcecode', 0):.1f}/100",
                "Weight": "25%",
                "Key Signals": f"Commits {detail_src.get('commits', '—')} · Active devs {detail_src.get('active_devs', '—')} · Repo score {detail_src.get('repo_score', '—')}",
            },
            {
                "Pillar": "Network",
                "Score": f"{pillar.get('network', 0):.1f}/100",
                "Weight": "20%",
                "Key Signals": f"Active addresses {detail_net.get('active_addresses', 0):,} · TPS {detail_net.get('tx_tps', 0):.2f} · Gas {detail_net.get('gas_fee_gwei', 0):.2f} gwei",
            },
            {
                "Pillar": "Economics",
                "Score": f"{pillar.get('economics', 0):.1f}/100",
                "Weight": "20%",
                "Key Signals": f"MCap {format_currency(detail_econ.get('market_cap', 0))} · Vol {format_currency(detail_econ.get('volume_24h', 0))} · Tokenomics {detail_econ.get('tokenomics_score', '—')}",
            },
            {
                "Pillar": "Sentiment",
                "Score": f"{pillar.get('sentiment', 0):.1f}/100",
                "Weight": "15%",
                "Key Signals": f"User sentiment {detail_sent.get('user_sentiment_index', '—')} · Buy/Sell ratio {detail_sent.get('buy_sell_ratio', '—')}",
            },
            {
                "Pillar": "Accessibility",
                "Score": f"{pillar.get('accessibility', 0):.1f}/100",
                "Weight": "20%",
                "Key Signals": f"Exchanges {detail_acc.get('exchange_count', '—')} · Wallet support {detail_acc.get('wallet_support_score', '—')}",
            },
        ]
        st.dataframe(pd.DataFrame(pillar_rows), use_container_width=True, hide_index=True)

        pcols = st.columns(5)
        pillar_labels = [
            ("Source Code", pillar.get("sourcecode", 0), "#10b981"),
            ("Network", pillar.get("network", 0), "#3b82f6"),
            ("Economics", pillar.get("economics", 0), "#8b5cf6"),
            ("Sentiment", pillar.get("sentiment", 0), "#f59e0b"),
            ("Accessibility", pillar.get("accessibility", 0), "#ec4899"),
        ]
        for col, (label, score, _color) in zip(pcols, pillar_labels):
            with col:
                st.metric(label, f"{score:.1f}")

        st.markdown("<br>", unsafe_allow_html=True)
        econ_hist = page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol]
        net_hist = page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol]
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            render_history_chart(econ_hist, "market_cap", f"{asset_symbol} Market Cap", "USD", color="#8b5cf6")
        with chart_c2:
            render_history_chart(net_hist, "tx_tps", f"{asset_symbol} Network TPS", "TPS", color="#3b82f6")

    with detail_tab1:
        st.markdown("### Off-Chain Derivatives & Market Structure")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Open Interest, funding rates, implied volatility, and skew across major venues. "
            "Prepared for future exchange-native feeds."
            "</p>",
            unsafe_allow_html=True,
        )
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            deriv_metrics = [
                {"Exchange": "Binance", "Open Interest (USD)": "$4.82B", "Funding Rate": "+0.0124%", "Delta Skew": "+1.4% (Calls)"},
                {"Exchange": "Bybit", "Open Interest (USD)": "$2.15B", "Funding Rate": "+0.0098%", "Delta Skew": "+0.8% (Calls)"},
                {"Exchange": "OKX", "Open Interest (USD)": "$1.74B", "Funding Rate": "+0.0110%", "Delta Skew": "-0.2% (Puts)"},
                {"Exchange": "Deribit (Options)", "Open Interest (USD)": "$3.90B", "Funding Rate": "N/A (IV: 52.4%)", "Delta Skew": "+3.1% (Bull Bias)"},
            ]
            st.dataframe(pd.DataFrame(deriv_metrics), use_container_width=True, hide_index=True)
        with d_col2:
            risk_status = "Balanced Positioning" if composite >= 55 else "Elevated Risk Watch"
            risk_color = "#3b82f6" if composite >= 55 else "#f59e0b"
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 8px; height: 215px;">
                    <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Liquidation Cascade Risk Monitor</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Long Liquidation Cluster:</b> ${price_base * 0.97:,.0f} (−2.8%)</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Short Liquidation Cluster:</b> ${price_base * 1.05:,.0f} (+5.2%)</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Leverage Ratio Index:</b> 0.24</p>
                    <span style="color: {risk_color}; font-size: 0.78rem; font-weight: bold;">● Status: {risk_status}</span>
                </div>
            """, unsafe_allow_html=True)

    with detail_tab2:
        st.markdown("### On-Chain Supply Dynamics & Cost Basis")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Holder cohorts, realized vs market capitalization, and valuation context linked to Economics pillar data."
            "</p>",
            unsafe_allow_html=True,
        )
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            cohort_data = [
                {"Holder Cohort": "Short-Term Holders (STH)", "% Supply": "18.4%", "Cost Basis": f"${price_base * 0.96:,.2f}", "P/L Status": "At Risk / Slight Loss"},
                {"Holder Cohort": "Long-Term Holders (LTH)", "% Supply": "65.2%", "Cost Basis": f"${price_base * 0.52:,.2f}", "P/L Status": "Deep In Profit (+95%)"},
                {"Holder Cohort": "Whale Entities (>10k)", "% Supply": "16.4%", "Cost Basis": f"${price_base * 0.68:,.2f}", "P/L Status": "In Profit (+42%)"},
            ]
            st.dataframe(pd.DataFrame(cohort_data), use_container_width=True, hide_index=True)
        with sc_col2:
            mcap_val = detail_econ.get("market_cap", 0) or 0
            realized_est = mcap_val / 1.52 if mcap_val else 0
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 8px; height: 215px;">
                    <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">MVRV & Valuation Multiples</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Market Cap:</b> {format_currency(mcap_val)}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Est. Realized Cap:</b> {format_currency(realized_est)}</p>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>MVRV Ratio:</b> 1.52 (Fair Value Band)</p>
                    <span style="color: #10b981; font-size: 0.78rem; font-weight: bold;">● Zone: Neutral Accumulation</span>
                </div>
            """, unsafe_allow_html=True)

        render_history_chart(
            page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol],
            "active_addresses",
            f"{asset_symbol} Active Addresses",
            "Addresses",
            color="#10b981",
        )

    with detail_tab3:
        st.markdown("### Capital Flows & Exchange Dynamics")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Net flow indicators across CEX, DeFi, and OTC. "
            "<b>Ready to integrate proprietary exchange order-book, volume, and user-flow data</b> when the exchange module is connected."
            "</p>",
            unsafe_allow_html=True,
        )
        flow_cols = st.columns(3)
        with flow_cols[0]:
            st.metric("24h CEX Netflow", "−4,250 Tokens", "−$276M (Outflow / Bullish)")
        with flow_cols[1]:
            st.metric("DeFi Protocol Inflows", "+12,800 Tokens", "+$832M (Staking/Lending)")
        with flow_cols[2]:
            st.metric("OTC Desk Accumulation", "+8,500 Tokens", "+$552M (Institutional Custody)")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Recent Institutional Wallet Flows")
        flow_history = [
            {"Timestamp": "2026-07-24 14:12", "Source": "Binance Cold Storage", "Destination": "Unknown Custody", "Amount": f"1,450 {asset_symbol}", "Classification": "OTC Accumulation"},
            {"Timestamp": "2026-07-24 11:05", "Source": "Coinbase Institutional", "Destination": "Aave Lending Pool", "Amount": f"12,500 {asset_symbol}", "Classification": "DeFi Deployment"},
            {"Timestamp": "2026-07-23 22:40", "Source": "Kraken Reserves", "Destination": "Private Multi-Sig", "Amount": f"850 {asset_symbol}", "Classification": "Self-Custody Withdrawal"},
        ]
        st.dataframe(pd.DataFrame(flow_history), use_container_width=True, hide_index=True)

        st.markdown("#### Exchange Integration Roadmap")
        st.info(
            "Future: connect acquired exchange feeds (spot/perp volume, order-book depth, internal user flows) "
            "into this panel and the Accessibility pillar for a unified analytics + exchange view."
        )

    with detail_tab4:
        st.markdown("### Regulatory, Filings & Narrative Feed")
        st.markdown(
            "<p style='color: #9ca3af; font-size: 0.85rem;'>"
            "Legal catalysts, SEC/regulatory filings, and governance proposals affecting institutional posture."
            "</p>",
            unsafe_allow_html=True,
        )
        feed_data = [
            {"Date": "2026-07-24", "Category": "Regulatory Filing", "Title": "SEC Form 19b-4 Submitted for Spot Multi-Asset Staking ETP", "Sentiment": "Bullish (+0.82)"},
            {"Date": "2026-07-22", "Category": "Governance Proposal", "Title": "Core Developer Group Proposes Fee Burn Parameter Adjustment", "Sentiment": "Neutral (+0.15)"},
            {"Date": "2026-07-20", "Category": "Legal Compliance", "Title": "European Banking Authority Issues Updated MiCA Stablecoin Guidance", "Sentiment": "Compliant"},
            {"Date": "2026-07-18", "Category": "Institutional Custody", "Title": "Global Custodian Launches Regulated Prime Brokerage Vaults", "Sentiment": "Strong Positive (+0.91)"},
        ]
        st.dataframe(pd.DataFrame(feed_data), use_container_width=True, hide_index=True)

elif current_view == "Project Explorer":
    st.subheader("Project Explorer & Institutional Asset Matrix")
    st.markdown(
        "<p style='color: #9ca3af; font-size: 0.9rem;'>"
        "Browse, filter, and open tracked protocols. Scores use the live 5-pillar health engine."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🔍 Filter & Control Panel")
    f_col1, f_col2, f_col3 = st.columns([2, 2, 1])
    with f_col1:
        search_query = st.text_input("Search Assets (Symbol or Name)", placeholder="e.g. BTC, Ethereum, SOL...")
    with f_col2:
        selected_categories = st.multiselect(
            "Filter by Ecosystem Category",
            ["Layer-1", "DeFi", "Smart Contracts", "Infrastructure"],
            default=["Layer-1", "DeFi", "Smart Contracts", "Infrastructure"],
        )
    with f_col3:
        min_health = st.slider("Min Health", 0, 100, 0)

    st.markdown("---")

    base_assets = ["BTC", "ETH", "SOL", "ADA"]
    category_map = {
        "BTC": "Layer-1",
        "ETH": "Smart Contracts",
        "SOL": "Infrastructure",
        "ADA": "Layer-1",
    }
    name_map = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "ADA": "Cardano"}

    explorer_rows = []
    health_by_sym = {}

    for sym in base_assets:
        health_payload = compute_blockactivities_health_score(sym, db_path="crypto_data.db")
        h_score = health_payload["health_score"]
        pillars = health_payload.get("pillar_scores", {})
        health_by_sym[sym] = h_score

        econ_sub = page_data["economics"][page_data["economics"]["asset_symbol"] == sym]
        mcap = econ_sub["market_cap"].iloc[-1] if not econ_sub.empty else 0
        vol = econ_sub["volume_24h"].iloc[-1] if not econ_sub.empty else 0

        ecosystem_cat = category_map.get(sym, "DeFi")
        if selected_categories and ecosystem_cat not in selected_categories:
            continue
        if search_query and search_query.lower() not in sym.lower() and search_query.lower() not in name_map.get(sym, "").lower():
            continue
        if h_score < min_health:
            continue

        status = "Optimal" if h_score >= 70 else ("Watch" if h_score >= 50 else "Alert")
        explorer_rows.append({
            "Asset": sym,
            "Name": name_map.get(sym, sym),
            "Category": ecosystem_cat,
            "Market Cap": format_currency(mcap),
            "24h Volume": format_currency(vol),
            "Health": round(h_score, 1),
            "Status": status,
            "Source Code": round(pillars.get("sourcecode", 0), 1),
            "Network": round(pillars.get("network", 0), 1),
            "Economics": round(pillars.get("economics", 0), 1),
            "Sentiment": round(pillars.get("sentiment", 0), 1),
            "Accessibility": round(pillars.get("accessibility", 0), 1),
        })

    if explorer_rows:
        avg_h = sum(r["Health"] for r in explorer_rows) / len(explorer_rows)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Assets Shown", len(explorer_rows))
        with s2:
            st.metric("Avg Health", f"{avg_h:.1f}/100")
        with s3:
            top = max(explorer_rows, key=lambda r: r["Health"])
            st.metric("Top Asset", f"{top['Asset']} ({top['Health']})")

        df_exp = pd.DataFrame(explorer_rows).sort_values(by="Health", ascending=False)
        st.dataframe(df_exp, use_container_width=True, hide_index=True)

        st.markdown("#### Open Project Detail")
        btn_cols = st.columns(len(explorer_rows))
        for i, row in enumerate(explorer_rows):
            with btn_cols[i]:
                sym = row["Asset"]

                def _open_detail(s=sym):
                    push_nav_history()
                    st.session_state.asset_symbol = s
                    st.session_state.nav_category = "Analytics & Terminal"
                    st.session_state.nav_section = "Project Detail Page"

                st.button(
                    f"Open {sym}",
                    key=f"explorer_open_{sym}",
                    use_container_width=True,
                    on_click=_open_detail,
                )
    else:
        st.info("No assets match your search / filter criteria. Lower Min Health or clear filters.")

elif current_view == "Categories":
  st.subheader("Ecosystem Categories Breakdown")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Browse protocols grouped by functional utility. "
      "Select a category to see representative assets and their current health scores.</p>",
      unsafe_allow_html=True,
  )

  cat_data = {
      "Layer-1": [
          {"Asset": "BTC", "Focus": "Store of Value / Settlement", "Health": compute_blockactivities_health_score("BTC")["health_score"]},
          {"Asset": "ADA", "Focus": "Research-driven Smart Contracts", "Health": compute_blockactivities_health_score("ADA")["health_score"]},
      ],
      "Smart Contracts": [
          {"Asset": "ETH", "Focus": "General-purpose Compute Layer", "Health": compute_blockactivities_health_score("ETH")["health_score"]},
      ],
      "Infrastructure": [
          {"Asset": "SOL", "Focus": "High-throughput Execution", "Health": compute_blockactivities_health_score("SOL")["health_score"]},
      ],
      "DeFi": [
          {"Asset": "—", "Focus": "Coming soon – lending, DEX, and yield protocols", "Health": None},
      ],
  }

  selected_cat = st.selectbox("Select Category", list(cat_data.keys()))
  rows = []
  for item in cat_data[selected_cat]:
      rows.append({
          "Asset": item["Asset"],
          "Primary Focus": item["Focus"],
          "Health Score": f"{item['Health']:.1f} / 100" if item["Health"] is not None else "—",
      })
  st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
  st.caption("Health scores are composite multi-pillar ratings (Source Code, Network, Economics, Sentiment, Accessibility).")

elif current_view == "All Projects":
  st.subheader("Complete Directory of Tracked Protocols")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Exhaustive list of all indexed blockchain networks currently covered by BN Analytics.</p>",
      unsafe_allow_html=True,
  )

  all_rows = []
  category_map = {"BTC": "Layer-1", "ETH": "Smart Contracts", "SOL": "Infrastructure", "ADA": "Layer-1"}
  for sym in ["BTC", "ETH", "SOL", "ADA"]:
      h = compute_blockactivities_health_score(sym)["health_score"]
      econ_sub = page_data["economics"][page_data["economics"]["asset_symbol"] == sym]
      mcap = econ_sub["market_cap"].iloc[-1] if not econ_sub.empty else 0
      all_rows.append({
          "Symbol": sym,
          "Category": category_map.get(sym, "—"),
          "Market Cap": format_currency(mcap),
          "Health Score": f"{h:.1f} / 100",
          "Status": "Active Tracking",
      })
  st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)
  st.info("More protocols (Layer-2s, DeFi blue-chips, and infrastructure projects) will be added in upcoming releases.")

elif current_view == "Search":
  st.subheader("Advanced Global Terminal Search")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Search across assets, metrics, repositories, and governance signals.</p>",
      unsafe_allow_html=True,
  )
  search_box = st.text_input("Search query", placeholder="e.g. BTC health, SOL TPS, whale inflow...")
  if search_box:
      q = search_box.lower()
      matches = []
      for sym in ["BTC", "ETH", "SOL", "ADA"]:
          if q in sym.lower() or any(k in q for k in ["health", "score", "tps", "whale", "market"]):
              h = compute_blockactivities_health_score(sym)["health_score"]
              matches.append({"Result": sym, "Type": "Asset", "Health Score": f"{h:.1f}/100", "Action": "Open Project Detail"})
      if matches:
          st.success(f"Found {len(matches)} matching record(s) for '{search_box}'.")
          st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
      else:
          st.warning("No matching institutional records found. Try a different symbol or keyword.")

elif current_view == "Profile":
  st.subheader("Institutional User Profile")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Account details, role permissions, and API access credentials.</p>",
      unsafe_allow_html=True,
  )

  p_col1, p_col2 = st.columns(2)
  with p_col1:
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px;">
              <h4 style="color: #10b981; margin-top: 0;">Account</h4>
              <p style="color: #f3f4f6; margin-bottom: 6px;"><b>Username:</b> {st.session_state.username}</p>
              <p style="color: #f3f4f6; margin-bottom: 6px;"><b>Role:</b> {st.session_state.role}</p>
              <p style="color: #f3f4f6; margin-bottom: 0;"><b>Status:</b> <span style="color: #10b981;">Active</span></p>
          </div>
      """, unsafe_allow_html=True)
  with p_col2:
      st.markdown("""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 20px; border-radius: 10px;">
              <h4 style="color: #3b82f6; margin-top: 0;">API Access</h4>
              <p style="color: #f3f4f6; margin-bottom: 6px;"><b>API Key:</b> <code>bn_live_99f8a2c10b</code></p>
              <p style="color: #f3f4f6; margin-bottom: 6px;"><b>Tier:</b> Fund Manager (demo)</p>
              <p style="color: #f3f4f6; margin-bottom: 0;"><b>Rate Limit:</b> 300 req/min</p>
          </div>
      """, unsafe_allow_html=True)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Role Permissions")
  perm_data = [
      {"Capability": "View multi-pillar dashboards", "Admin": "✓", "Portfolio Manager": "✓", "Analyst": "✓"},
      {"Capability": "Configure alerts & webhooks", "Admin": "✓", "Portfolio Manager": "✓", "Analyst": "—"},
      {"Capability": "Export executive PDF / CSV", "Admin": "✓", "Portfolio Manager": "✓", "Analyst": "✓"},
      {"Capability": "Manage users & API keys", "Admin": "✓", "Portfolio Manager": "—", "Analyst": "—"},
      {"Capability": "Regenerate simulated data", "Admin": "✓", "Portfolio Manager": "—", "Analyst": "—"},
  ]
  st.dataframe(pd.DataFrame(perm_data), use_container_width=True, hide_index=True)

elif current_view == "Watchlist":
  st.subheader("Custom Asset Watchlist")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Tracked assets with live health scores and quick status indicators.</p>",
      unsafe_allow_html=True,
  )

  watch_rows = []
  for sym, label in [("BTC", "Bitcoin"), ("ETH", "Ethereum"), ("SOL", "Solana"), ("ADA", "Cardano")]:
      payload = compute_blockactivities_health_score(sym)
      h = payload["health_score"]
      pillars = payload.get("pillar_scores", {})
      status = "Optimal" if h >= 70 else ("Watch" if h >= 50 else "Alert")
      watch_rows.append({
          "Symbol": sym,
          "Name": label,
          "Health": round(h, 1),
          "Source Code": round(pillars.get("sourcecode", 0), 1),
          "Network": round(pillars.get("network", 0), 1),
          "Economics": round(pillars.get("economics", 0), 1),
          "Sentiment": round(pillars.get("sentiment", 0), 1),
          "Accessibility": round(pillars.get("accessibility", 0), 1),
          "Status": status,
      })
  st.dataframe(pd.DataFrame(watch_rows), use_container_width=True, hide_index=True)

  st.markdown("---")
  st.markdown("### Quick Actions")

  def _nav_to_asset_detail(symbol: str):
      push_nav_history()
      st.session_state.asset_symbol = symbol
      st.session_state.nav_category = "Analytics & Terminal"
      st.session_state.nav_section = "Project Detail Page"

  def _nav_to_overview():
      push_nav_history()
      st.session_state.nav_category = "Analytics & Terminal"
      st.session_state.nav_section = "Overview Dashboard"

  wcols = st.columns(5)
  for col, sym in zip(wcols[:4], ["BTC", "ETH", "SOL", "ADA"]):
      with col:
          st.button(
              f"Open {sym}",
              use_container_width=True,
              key=f"watch_{sym.lower()}_detail",
              on_click=_nav_to_asset_detail,
              args=(sym,),
          )
  with wcols[4]:
      st.button(
          "Overview",
          use_container_width=True,
          key="watch_overview",
          on_click=_nav_to_overview,
      )

# ===================== LEARN – TUTORIALS (EXPANDED) =====================
elif current_view == "Tutorials":
  st.subheader("Terminal Tutorials & Guides")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Step-by-step walkthroughs to help you get the most from the BN Analytics terminal.</p>",
      unsafe_allow_html=True,
  )

  tut_filter = st.selectbox("Filter by level", ["All Levels", "Beginner", "Intermediate", "Advanced"])

  tutorials = [
      {"Title": "Navigating the Multi-Pillar Dashboard", "Level": "Beginner", "Time": "5 min", "Summary": "Learn how Health Scores are composed and how to switch between Network, Economics, Sentiment, and Accessibility views."},
      {"Title": "Setting Up Automated Health Alerts", "Level": "Beginner", "Time": "4 min", "Summary": "Configure sidebar thresholds and connect Slack/Telegram webhooks for real-time notifications. Understand the audit log."},
      {"Title": "Using Project Explorer & Filters", "Level": "Intermediate", "Time": "6 min", "Summary": "Filter assets by category, search by symbol, and jump into the Project Detail experience."},
      {"Title": "Exporting Executive PDF & CSV Reports", "Level": "Intermediate", "Time": "3 min", "Summary": "Generate board-ready reports directly from the sidebar for any tracked asset."},
      {"Title": "Interpreting Whale Transactions & Order Flow", "Level": "Advanced", "Time": "8 min", "Summary": "Understand large transfer signals, net taker flow, and how they relate to accumulation or distribution phases."},
      {"Title": "Reading the 5-Pillar Health Score", "Level": "Beginner", "Time": "7 min", "Summary": "Deep explanation of weighting (Source Code 25%, Network 20%, Economics 20%, Sentiment 15%, Accessibility 20%) and how scores update."},
      {"Title": "Working with the API Playground", "Level": "Intermediate", "Time": "5 min", "Summary": "Authenticate, call /v1/metrics, /v1/economics and /v1/network, and interpret the JSON response."},
      {"Title": "Token Unlock Calendar Interpretation", "Level": "Advanced", "Time": "6 min", "Summary": "How to combine unlock schedules with health scores and net-flow data to anticipate supply shocks."},
      {"Title": "Strategy Grid-Search Basics", "Level": "Advanced", "Time": "10 min", "Summary": "Using the InstitutionalAnalyticsEngine to optimise MA crossover parameters and evaluate Sharpe / max drawdown."},
      {"Title": "Alert History & Compliance Trail", "Level": "Intermediate", "Time": "4 min", "Summary": "Where alert events are stored, how webhook delivery status is recorded, and how to review history in Settings."},
  ]

  shown = 0
  for t in tutorials:
      if tut_filter != "All Levels" and t["Level"] != tut_filter:
          continue
      shown += 1
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 16px 18px; border-radius: 8px; margin-bottom: 12px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                  <b style="color: #ffffff; font-size: 0.95rem;">{t['Title']}</b>
                  <span style="color: #9ca3af; font-size: 0.78rem;">{t['Level']} · {t['Time']}</span>
              </div>
              <p style="color: #9ca3af; font-size: 0.85rem; margin: 0;">{t['Summary']}</p>
          </div>
      """, unsafe_allow_html=True)
  if shown == 0:
      st.info("No tutorials match the selected level.")

# ===================== LEARN – GUIDES (EXPANDED) =====================
elif current_view == "Guides":
  st.subheader("Risk Management & Compliance Guides")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Best practices for institutional users operating the terminal in production environments.</p>",
      unsafe_allow_html=True,
  )

  guides = [
      {"Title": "Establishing Alert Thresholds", "Body": "Start with a conservative health-score floor (e.g. 45–55). Tighten thresholds once you understand normal volatility for each asset. Always keep the audit log enabled."},
      {"Title": "Webhook & Incident Response", "Body": "Route alerts to a dedicated Slack/Telegram channel. Pair automated notifications with a simple runbook for escalation. Test the webhook from Settings after any URL change."},
      {"Title": "Export & Audit Trail Hygiene", "Body": "Use PDF/CSV exports for investment committee packs. Alert audit logs in the institutional database provide a lightweight compliance trail that includes webhook delivery status."},
      {"Title": "Role-Based Access Hygiene", "Body": "Limit Regenerate Data and user-management actions to Admin. Analysts should focus on read + export workflows. Portfolio Managers may configure alerts but not manage users."},
      {"Title": "Data Refresh Cadence", "Body": "The simulated dataset can be regenerated from Settings (Admin only). In production this will be replaced by scheduled pipeline jobs. Always note the ‘Data refreshed’ timestamp in the sidebar."},
      {"Title": "Interpreting False Positives", "Body": "A single low health reading does not automatically mean risk. Cross-check pillar scores, recent whale flow, and unlock calendar before acting on an alert."},
      {"Title": "Webhook Security Notes", "Body": "Never hard-code production webhook secrets in the UI. Prefer environment variables or a secrets manager. Rotate URLs periodically and monitor delivery status in the alert history."},
      {"Title": "Preparing for Live Data Migration", "Body": "When moving from simulated to live CoinGecko / on-chain / GitHub sources, keep the same pillar interfaces. Caching and graceful fallbacks (already present) will minimise downtime."},
  ]
  for g in guides:
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 16px 18px; border-radius: 8px; margin-bottom: 12px;">
              <b style="color: #10b981; font-size: 0.95rem;">{g['Title']}</b>
              <p style="color: #f3f4f6; font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">{g['Body']}</p>
          </div>
      """, unsafe_allow_html=True)

# ===================== LEARN – GLOSSARY (EXPANDED) =====================
elif current_view == "Glossary":
  st.subheader("Blockchain & Finance Glossary")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>Quick reference for terms used across the terminal and research content.</p>",
      unsafe_allow_html=True,
  )

  glossary_items = {
      "Health Score": "Composite 0–100 rating built from five pillars: Source Code (25%), Network (20%), Economics (20%), Sentiment (15%), and Accessibility (20%).",
      "Net Taker Flow": "Difference between aggressive buy volume and aggressive sell volume. Positive values indicate net buying pressure.",
      "MVRV": "Market Value to Realized Value ratio — a valuation multiple comparing current market cap to the aggregate cost basis of holders.",
      "TPS": "Transactions per second — a core network throughput metric.",
      "Whale Transaction": "Large on-chain transfer (typically > $1M notional) that can signal institutional accumulation or distribution.",
      "TVL": "Total Value Locked — aggregate USD value of assets deposited in DeFi protocols.",
      "Funding Rate": "Periodic payment exchanged between long and short positions in perpetual futures markets.",
      "STH / LTH": "Short-Term Holders vs Long-Term Holders. Cohorts defined by the age of UTXOs or tokens since last movement.",
      "Open Interest (OI)": "Total number of outstanding derivative contracts that have not been settled.",
      "Delta Skew": "Difference in implied volatility between out-of-the-money calls and puts; used as a measure of market sentiment.",
      "Realized Cap": "Aggregate value of all tokens priced at the value when they last moved on-chain.",
      "Token Unlock": "Scheduled release of previously locked tokens into circulating supply, often creating sell pressure.",
      "Composite Pillar": "One of the five scored dimensions that feed the overall Health Score.",
      "Alert Audit Log": "Immutable record of every health-score breach and webhook delivery attempt stored in the institutional database.",
      "Webhook Dispatcher": "Component that posts alert payloads to Slack, Telegram or custom HTTP endpoints when thresholds are breached.",
      "RBAC": "Role-Based Access Control — Admin, Portfolio Manager, and Analyst roles with different privileges.",
      "Grid Search": "Systematic evaluation of parameter combinations (e.g. short/long MA windows) to optimise a trading strategy.",
      "Sharpe Ratio": "Risk-adjusted return metric: (mean excess return) / (standard deviation of returns), annualised.",
      "Max Drawdown": "Largest peak-to-trough decline in cumulative returns over a given period.",
      "Blob Gas": "Post-Dencun Ethereum fee market for data availability used by Layer-2 rollups.",
  }
  term_search = st.text_input("Filter glossary", placeholder="Type a term or keyword...")
  matched = 0
  for term, definition in glossary_items.items():
      if term_search and term_search.lower() not in term.lower() and term_search.lower() not in definition.lower():
          continue
      matched += 1
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
              <b style="color: #10b981;">{term}</b>
              <p style="color: #f3f4f6; font-size: 0.85rem; margin-top: 4px; margin-bottom: 0;">{definition}</p>
          </div>
      """, unsafe_allow_html=True)
  if matched == 0:
      st.info("No glossary entries match your search.")

# ===================== SETTINGS (HARDENED AUTOMATION + FEEDBACK) =====================
elif current_view == "Settings":
  st.subheader("Terminal Settings & Configurations")
  st.markdown("Manage data, API gateway access, notification endpoints, and UI preferences.")

  st.markdown("---")
  st.markdown("### Data Management")
  st.markdown(
      f"<p style='color: #9ca3af; font-size: 0.9rem;'>"
      f"<b>Last data refresh:</b> {data_last_refreshed}<br/>"
      "Regenerate the simulated institutional dataset (trades + multi-pillar metrics). "
      "This is useful when you want a fresh demo environment."
      "</p>",
      unsafe_allow_html=True,
  )

  col_reset1, col_reset2 = st.columns([1, 2])
  with col_reset1:
      if st.button("🔄 Regenerate All Data", type="primary", use_container_width=True):
          if st.session_state.role != "Admin":
              st.error("Only Admin users can regenerate the dataset.")
          else:
              with st.spinner("Regenerating trades and pillar metrics..."):
                  try:
                      if os.path.exists("crypto_data.db"):
                          os.remove("crypto_data.db")
                      generate_simulated_trades(num_records=5000, db_path="crypto_data.db")
                      generate_all_crypto_metrics(days=30, db_path="crypto_data.db")
                      st.cache_data.clear()
                      st.success("✅ Data regenerated successfully. Dashboard will reload with fresh metrics.")
                      st.rerun()
                  except Exception as e:
                      st.error(f"Regeneration failed: {e}")

  with col_reset2:
      st.info("This will delete and recreate `crypto_data.db`. Alert logs and user accounts are **not** affected.")

  st.markdown("---")
  st.markdown("### Notification Preferences & Automation Status")
  st.markdown(
      f"**Current Health Alert Threshold:** `{alert_health_min}`  \n"
      f"**Webhook configured:** `{'Yes' if webhook_url_input else 'No'}`  \n"
      f"**Current asset under watch:** `{asset_symbol}` (score: {current_check_score:.1f})"
  )
  st.caption("Adjust the threshold and webhook URL from the sidebar. Changes take effect immediately.")

  # Test webhook from Settings
  st.markdown("#### Test Webhook Delivery")
  test_url = st.text_input("Webhook URL to test", value=webhook_url_input or "", key="settings_test_webhook")
  if st.button("Send Test Payload"):
      if not test_url:
          st.warning("Enter a webhook URL first.")
      else:
          try:
              payload = {"text": f"BNAnalytics Settings Test – {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} – User: {st.session_state.username}"}
              res = requests.post(test_url, json=payload, timeout=5)
              if res.status_code in [200, 201]:
                  log_alert_event(asset_symbol, current_check_score, alert_health_min, "Settings Test", "Delivered")
                  st.success("✅ Test payload delivered successfully and logged.")
              else:
                  log_alert_event(asset_symbol, current_check_score, alert_health_min, "Settings Test", f"Failed ({res.status_code})")
                  st.warning(f"Webhook responded with status {res.status_code}")
          except Exception as e:
              log_alert_event(asset_symbol, current_check_score, alert_health_min, "Settings Test", f"Error: {str(e)[:40]}")
              st.error(f"Delivery failed: {e}")

  st.markdown("---")
  st.markdown("### Alert History & Audit Trail")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.85rem;'>"
      "Complete record of health-score breaches and webhook delivery attempts. "
      "Useful for compliance and post-incident review."
      "</p>",
      unsafe_allow_html=True,
  )

  alert_hist = get_alert_history(limit=100)
  if not alert_hist.empty:
      st.dataframe(alert_hist, use_container_width=True, hide_index=True)

      # Admin-only clear
      if st.session_state.role == "Admin":
          if st.button("🗑️ Clear All Alert Logs (Admin only)", type="secondary"):
              try:
                  conn = sqlite3.connect("bnanalytics_institutional.db")
                  conn.execute("DELETE FROM alert_audit_logs")
                  conn.commit()
                  conn.close()
                  st.success("Alert history cleared.")
                  st.rerun()
              except Exception as e:
                  st.error(f"Could not clear logs: {e}")
  else:
      st.caption("No alert events have been recorded yet.")

  st.markdown("---")
  st.markdown("### Account & Access")
  st.markdown(f"**Logged in as:** `{st.session_state.username}`")
  st.markdown(f"**Role:** `{st.session_state.role}`")
  st.markdown("**API Key (demo):** `bn_live_99f8a2c10b`")
  st.caption("Role permissions are enforced for data regeneration and log management.")