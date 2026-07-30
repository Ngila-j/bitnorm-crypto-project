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
    from exchange_adapter import (
        get_mode as exchange_mode,
        get_orderbook as exchange_get_orderbook,
        refresh_symbol as exchange_refresh_symbol,
    )
except ImportError:  # pragma: no cover
    exchange_mode = lambda: "unavailable"
    exchange_get_orderbook = None
    exchange_refresh_symbol = None
try:
    from prophet import Prophet
except ImportError:  # pragma: no cover - fallback for environments without Prophet
    Prophet = None
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
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
  st.session_state.nav_category = "🏠 Landing"
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


# --- Brand header ---
st.sidebar.markdown(
    """
    <div style="padding: 4px 0 8px 0;">
      <div style="font-size: 1.35rem; font-weight: 700; color: #f3f4f6; letter-spacing: 0.04em;">BN ANALYTICS</div>
      <div style="font-size: 0.75rem; color: #9ca3af; margin-top: 2px;">Institutional terminal</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if os.path.exists("logo.png"):
  st.sidebar.image("logo.png", width=40)

st.sidebar.markdown(
    f"""
    <div style="background:#111827; border:1px solid #374151; border-radius:8px; padding:10px 12px; margin: 6px 0 10px 0;">
      <div style="font-size:0.78rem; color:#9ca3af;">Signed in</div>
      <div style="font-size:0.88rem; color:#f3f4f6; font-weight:600;">{st.session_state.username}
        <span style="color:#10b981; font-weight:500;">· {st.session_state.role}</span></div>
      <div style="font-size:0.72rem; color:#6b7280; margin-top:4px;">Data · {data_last_refreshed}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.nav_history:
  st.sidebar.button(
      "← Go Back",
      use_container_width=True,
      key="sidebar_go_back",
      on_click=go_back,
  )

nav_categories = {
    "🏠 Landing": [
        "Home",
        "Features Overview",
        "Pricing",
        "Docs / API",
        "Blog / Resources",
    ],
    "📊 Insights": [
        "Research Reports",
        "Market Analysis",
        "News",
    ],
    "🖥 Terminal": [
        "Overview Dashboard",
        "Project Explorer",
        "Project Detail Page",
        "Settings",
    ],
    "🗂 Projects": [
        "Categories",
        "All Projects",
        "Search",
    ],
    "👤 Account": [
        "Profile",
        "Watchlist",
    ],
    "📖 Learn": [
        "Tutorials",
        "Guides",
        "Glossary",
    ],
}

# Map legacy category names → new keys (session may still hold old labels)
_nav_alias = {
    "Landing & Marketing": "🏠 Landing",
    "Insights": "📊 Insights",
    "Analytics & Terminal": "🖥 Terminal",
    "Projects & Explorer": "🗂 Projects",
    "Account & Watchlist": "👤 Account",
    "Learn & Resources": "📖 Learn",
}
if st.session_state.nav_category in _nav_alias:
  st.session_state.nav_category = _nav_alias[st.session_state.nav_category]

category_for_section = {
    page: category
    for category, pages in nav_categories.items()
    for page in pages
}
if st.session_state.nav_section not in category_for_section:
  st.session_state.nav_section = "Home"
if st.session_state.nav_category not in nav_categories:
  st.session_state.nav_category = category_for_section.get(
      st.session_state.nav_section, "🏠 Landing"
  )

available_pages = nav_categories[st.session_state.nav_category]
if st.session_state.nav_section not in available_pages:
  st.session_state.nav_section = available_pages[0]

st.sidebar.markdown(
    f"""
    <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin: 12px 0 6px 0;">
      Navigation
    </div>
    <div style="font-size:0.78rem; color:#9ca3af; margin-bottom:8px;">
      Now: <span style="color:#10b981; font-weight:600;">{st.session_state.nav_section}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

for category_name, pages in nav_categories.items():
  is_active_category = st.session_state.nav_category == category_name
  with st.sidebar.expander(
      category_name,
      expanded=is_active_category,
  ):
    for page_name in pages:
      is_current = page_name == st.session_state.nav_section
      label = f"● {page_name}" if is_current else page_name
      if st.sidebar.button(
          label,
          use_container_width=True,
          key=f"nav_{category_name}_{page_name}",
          type="primary" if is_current else "secondary",
      ):
        if page_name != st.session_state.nav_section or category_name != st.session_state.nav_category:
          push_nav_history()
        st.session_state.nav_category = category_name
        st.session_state.nav_section = page_name
        st.rerun()

# --- Controls: Asset + Alerts + Exports (collapsed by default for cleaner scroll) ---
st.sidebar.markdown(
    """
    <div style="font-size:0.72rem; color:#6b7280; text-transform:uppercase; letter-spacing:0.06em; margin: 14px 0 6px 0;">
      Workspace controls
    </div>
    """,
    unsafe_allow_html=True,
)

asset_symbol = st.sidebar.selectbox(
    "Target asset",
    ["BTC", "ETH", "SOL", "ADA"],
    key="asset_symbol",
)

with st.sidebar.expander("🔔 Alerts", expanded=False):
  alert_health_min = st.slider("Min health warning", 0, 100, 45, key="sidebar_alert_slider")
  webhook_url_input = st.text_input(
      "Webhook URL",
      placeholder="https://hooks.slack.com/...",
      key="sidebar_webhook_url",
  )
  auto_webhook = st.checkbox("Auto-dispatch on breach", value=False, key="sidebar_auto_webhook")

current_check_score = compute_blockactivities_health_score(
    asset_symbol, db_path="crypto_data.db"
)["health_score"]

# Session-level breach tracking to avoid logging every Streamlit rerun
if "alert_breach_keys" not in st.session_state:
  st.session_state.alert_breach_keys = set()

breach_key = f"{asset_symbol}:{alert_health_min}:{int(current_check_score)}"
is_breach = current_check_score < alert_health_min

# Compact status chip under asset selector
if is_breach:
  st.sidebar.warning(f"⚠ {asset_symbol} {current_check_score:.1f} < {alert_health_min}")
  if breach_key not in st.session_state.alert_breach_keys:
    try:
      conn_log = sqlite3.connect("bnanalytics_institutional.db")
      c_log = conn_log.cursor()
      c_log.execute(
          "INSERT INTO alert_audit_logs (asset_symbol, health_score, threshold, status) VALUES (?, ?, ?, ?)",
          (asset_symbol, current_check_score, alert_health_min, "Triggered - Warning"),
      )
      conn_log.commit()
      conn_log.close()
      st.session_state.alert_breach_keys.add(breach_key)
    except Exception:
      pass
  if auto_webhook and webhook_url_input and f"wh_{breach_key}" not in st.session_state.alert_breach_keys:
    try:
      payload = {
          "text": (
              f"BNAnalytics Auto-Dispatch: {asset_symbol} Health Score "
              f"dropped to {current_check_score:.1f} (Threshold: {alert_health_min})!"
          )
      }
      res = requests.post(webhook_url_input, json=payload, timeout=4)
      status_msg = "Webhook Dispatched" if res.status_code in [200, 201] else f"Webhook Failed ({res.status_code})"
      conn_log = sqlite3.connect("bnanalytics_institutional.db")
      c_log = conn_log.cursor()
      c_log.execute(
          "INSERT INTO alert_audit_logs (asset_symbol, health_score, threshold, status) VALUES (?, ?, ?, ?)",
          (asset_symbol, current_check_score, alert_health_min, status_msg),
      )
      conn_log.commit()
      conn_log.close()
      st.session_state.alert_breach_keys.add(f"wh_{breach_key}")
    except Exception as e:
      st.sidebar.error(f"Auto-webhook failed: {e}")
else:
  st.sidebar.caption(f"✓ {asset_symbol} health {current_check_score:.1f} ≥ {alert_health_min}")

# Broadcast lives inside alerts expander UX via second expander actions
if st.sidebar.button("Broadcast webhook now", use_container_width=True, key="sidebar_broadcast"):
  if not webhook_url_input:
    st.sidebar.error("Open Alerts and enter a webhook URL first.")
  else:
    try:
      payload = {
          "text": (
              f"BNAnalytics Manual Dispatch: {asset_symbol} Health Score "
              f"is {current_check_score:.1f} (Threshold: {alert_health_min})."
          )
      }
      res = requests.post(webhook_url_input, json=payload, timeout=4)
      if res.status_code in [200, 201]:
        st.sidebar.success("Webhook dispatched.")
        status_msg = "Manual Webhook OK"
      else:
        st.sidebar.warning(f"Webhook status {res.status_code}")
        status_msg = f"Manual Webhook Failed ({res.status_code})"
      conn_log = sqlite3.connect("bnanalytics_institutional.db")
      c_log = conn_log.cursor()
      c_log.execute(
          "INSERT INTO alert_audit_logs (asset_symbol, health_score, threshold, status) VALUES (?, ?, ?, ?)",
          (asset_symbol, current_check_score, alert_health_min, status_msg),
      )
      conn_log.commit()
      conn_log.close()
    except Exception as e:
      st.sidebar.error(f"Connection failed: {e}")

with st.sidebar.expander("⬇ Exports", expanded=False):
  export_frame = page_data["economics"][
      page_data["economics"]["asset_symbol"] == asset_symbol
  ]
  if not export_frame.empty:
    csv_data = export_frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"{asset_symbol} CSV",
        data=csv_data,
        file_name=f"{asset_symbol}_bnanalytics.csv",
        mime="text/csv",
        use_container_width=True,
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
  st.download_button(
      label=f"{asset_symbol} Executive PDF",
      data=pdf_buffer,
      file_name=f"{asset_symbol}_Report.pdf",
      mime="application/pdf",
      use_container_width=True,
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
          st.session_state.nav_category = "🖥 Terminal"
          st.session_state.nav_section = "Overview Dashboard"
          st.rerun()
  with hero_col2:
      if st.button("View Ecosystem Maps", key="hero_view_ecosystems"):
          push_nav_history()
          st.session_state.nav_category = "🖥 Terminal"
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

  # Alert audit summary
  st.markdown("### Recent Alert Activity")
  try:
      conn_alerts = sqlite3.connect("bnanalytics_institutional.db")
      alert_df = pd.read_sql(
          "SELECT timestamp, asset_symbol, health_score, threshold, status "
          "FROM alert_audit_logs ORDER BY log_id DESC LIMIT 8",
          conn_alerts,
      )
      conn_alerts.close()
      if not alert_df.empty:
          st.dataframe(alert_df, use_container_width=True, hide_index=True)
      else:
          st.caption("No alert events logged yet. Alerts appear when health falls below your sidebar threshold.")
  except Exception:
      st.caption("Alert log unavailable.")

elif current_view == "Features Overview":
    st.subheader("BN Analytics Feature Suite")
    st.markdown(
        "<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 20px;'>"
        "Enterprise modules for multi-pillar intelligence, risk automation, and institutional reporting."
        "</p>",
        unsafe_allow_html=True,
    )

    features = [
        {
            "title": "5-Pillar Health Engine",
            "color": "#10b981",
            "body": "Composite scores from Source Code, Network, Economics, Sentiment, and Accessibility — visible on Overview, Project Detail, Explorer, and Watchlist.",
            "cta": "Project Detail Page",
        },
        {
            "title": "Institutional Terminal",
            "color": "#3b82f6",
            "body": "Overview dashboards, order-flow depth, whale tracking, technical matrices, and Project Explorer with live filters.",
            "cta": "Overview Dashboard",
        },
        {
            "title": "Automated Alert Dispatcher",
            "color": "#8b5cf6",
            "body": "Health thresholds, Slack/Telegram webhooks, auto-dispatch, and an audit log with CSV export in Settings.",
            "cta": "Settings",
        },
        {
            "title": "Executive PDF & CSV",
            "color": "#f59e0b",
            "body": "Board-ready exports from the sidebar for any tracked asset, paired with alert audit trails for compliance packs.",
            "cta": "Overview Dashboard",
        },
        {
            "title": "Production API",
            "color": "#ec4899",
            "body": "FastAPI contracts at /v1/health, /v1/flow, and /v1/exchange/* for Next.js or external systems (Phase 4).",
            "cta": "Docs / API",
        },
        {
            "title": "Exchange Adapter",
            "color": "#14b8a6",
            "body": "Mock or live ticker and order-book feeds via exchange_adapter.py — Overview Order Flow already consumes them.",
            "cta": "Overview Dashboard",
        },
    ]
    cols = st.columns(2)
    for i, f in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
                <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 10px; margin-bottom: 16px; min-height: 160px;">
                    <h3 style="color: {f['color']}; margin-top: 0; font-size: 1.05rem;">{f['title']}</h3>
                    <p style="color: #9ca3af; font-size: 0.88rem; line-height: 1.5; margin-bottom: 0;">{f['body']}</p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Jump into the product")
    j1, j2, j3 = st.columns(3)
    with j1:
        if st.button("Open Terminal Overview", use_container_width=True, key="feat_to_ov"):
            push_nav_history()
            st.session_state.nav_category = "🖥 Terminal"
            st.session_state.nav_section = "Overview Dashboard"
            st.rerun()
    with j2:
        if st.button("Open Project Explorer", use_container_width=True, key="feat_to_exp"):
            push_nav_history()
            st.session_state.nav_category = "🖥 Terminal"
            st.session_state.nav_section = "Project Explorer"
            st.rerun()
    with j3:
        if st.button("Open Docs / API", use_container_width=True, key="feat_to_docs"):
            push_nav_history()
            st.session_state.nav_category = "🏠 Landing"
            st.session_state.nav_section = "Docs / API"
            st.rerun()

elif current_view == "Pricing":
  st.subheader("Institutional Subscription Tiers")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 25px;'>"
      "Scalable plans for developers, analysts, funds, and enterprise infrastructure teams."
      "</p>",
      unsafe_allow_html=True,
  )
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
                <li>Historical Metrics Access</li>
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
                <li>Custom Data / Exchange Pipelines</li>
                <li>99.9% Uptime SLA</li>
                <li>Regulatory Compliance & Auditing</li>
                <li>Priority Slack & Phone Support</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown("### Compare at a glance")
  compare = [
      {"Capability": "Terminal dashboards", "Developer": "Limited", "Analyst": "Yes", "Fund Manager": "Full", "Enterprise": "Full + custom"},
      {"Capability": "5-pillar health scores", "Developer": "API only", "Analyst": "Yes", "Fund Manager": "Yes", "Enterprise": "Yes + private assets"},
      {"Capability": "Webhook alerts + audit log", "Developer": "—", "Analyst": "Yes", "Fund Manager": "Yes", "Enterprise": "Yes + SSO"},
      {"Capability": "PDF / CSV exports", "Developer": "—", "Analyst": "CSV", "Fund Manager": "PDF + CSV", "Enterprise": "Branded packs"},
      {"Capability": "Exchange adapter feeds", "Developer": "Mock", "Analyst": "Mock", "Fund Manager": "Live optional", "Enterprise": "Dedicated"},
  ]
  st.dataframe(pd.DataFrame(compare), use_container_width=True, hide_index=True)
  st.info("Pricing is illustrative for the prototype. Contact BitNorm sales for production contracts.")

elif current_view == "Docs / API":
  st.subheader("Documentation & API Access")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.95rem; margin-bottom: 20px;'>"
      "Programmatic access to health scores, order flow, and exchange snapshots. "
      "Local Phase 4 API runs at <b>http://127.0.0.1:8000</b> — open <b>/docs</b> for Swagger."
      "</p>",
      unsafe_allow_html=True,
  )
  doc_tab1, doc_tab2, doc_tab3 = st.tabs(["Interactive Playground", "Endpoints & Reference", "Authentication & Errors"])
  with doc_tab1:
    st.markdown("### Live API Sandbox")
    st.markdown(
        "<p style='color: #9ca3af; font-size: 0.85rem;'>"
        "Runs against the local health engine (same data as the terminal). "
        "For the real HTTP API, start <code>uvicorn api:app --port 8000</code>."
        "</p>",
        unsafe_allow_html=True,
    )

    col_play1, col_play2 = st.columns(2)
    with col_play1:
      sandbox_symbol = st.selectbox("Select Asset Symbol", ["BTC", "ETH", "SOL", "ADA"], key="sandbox_sym")
      sandbox_endpoint = st.selectbox(
          "Select Endpoint",
          ["/v1/health/{symbol}", "/v1/flow", "/v1/exchange/ticker/{symbol}", "/v1/exchange/orderbook/{symbol}", "/health"],
          key="sandbox_endpoint",
      )
    with col_play2:
      sandbox_key = st.text_input("API Key (demo)", value="bn_live_99f8a2c10b", type="password", key="sandbox_key")
      try_local = st.checkbox("Try local FastAPI (127.0.0.1:8000)", value=False)

    if st.button("Execute API Request", key="run_sandbox_req"):
      with st.spinner("Fetching..."):
        if try_local:
          path = sandbox_endpoint.replace("{symbol}", sandbox_symbol)
          url = f"http://127.0.0.1:8000{path}"
          try:
            res = requests.get(url, timeout=4)
            st.success(f"{res.status_code} · {url}")
            try:
              st.json(res.json())
            except Exception:
              st.code(res.text)
          except Exception as e:
            st.error(f"Could not reach local API: {e}. Is uvicorn running?")
        else:
          t_mod.sleep(0.3)
          hs = compute_blockactivities_health_score(sandbox_symbol, db_path="crypto_data.db")
          sample_payload = {
              "status": "success",
              "code": 200,
              "timestamp": pd.Timestamp.now().isoformat(),
              "endpoint": sandbox_endpoint,
              "data": {
                  "symbol": sandbox_symbol,
                  "health_score": float(hs.get("health_score", 0)),
                  "pillar_scores": hs.get("pillar_scores", {}),
                  "note": "Simulated sandbox response from terminal engine",
              },
          }
          st.success("Request processed successfully (200 OK)")
          st.json(sample_payload)

  with doc_tab2:
    st.markdown("### Phase 4 API Endpoints (local FastAPI)")
    endpoint_data = [
        {"Endpoint": "/health", "Method": "GET", "Description": "Service health + exchange mode"},
        {"Endpoint": "/v1/assets", "Method": "GET", "Description": "List tracked assets"},
        {"Endpoint": "/v1/health/{symbol}", "Method": "GET", "Description": "Composite + pillar health scores"},
        {"Endpoint": "/v1/flow", "Method": "GET", "Description": "Net taker flow table"},
        {"Endpoint": "/v1/exchange/ticker/{symbol}", "Method": "GET", "Description": "Exchange last price / volume"},
        {"Endpoint": "/v1/exchange/orderbook/{symbol}", "Method": "GET", "Description": "Bid/ask depth snapshot"},
        {"Endpoint": "/v1/exchange/refresh", "Method": "POST", "Description": "Refresh mock/live exchange tables"},
    ]
    st.dataframe(pd.DataFrame(endpoint_data), use_container_width=True, hide_index=True)
    st.markdown("### Code Snippets (local)")
    lang_tab1, lang_tab2, lang_tab3 = st.tabs(["Python", "cURL", "Node.js"])
    with lang_tab1:
      st.code(
          "import requests\n\n"
          "r = requests.get('http://127.0.0.1:8000/v1/health/BTC')\n"
          "print(r.json())\n\n"
          "r2 = requests.get('http://127.0.0.1:8000/v1/exchange/orderbook/BTC', params={'refresh': True})\n"
          "print(r2.json())",
          language="python",
      )
    with lang_tab2:
      st.code(
          "curl http://127.0.0.1:8000/v1/health/BTC\n"
          "curl 'http://127.0.0.1:8000/v1/exchange/ticker/ETH?refresh=true'",
          language="bash",
      )
    with lang_tab3:
      st.code(
          "const r = await fetch('http://127.0.0.1:8000/v1/health/BTC');\n"
          "console.log(await r.json());",
          language="javascript",
      )
  with doc_tab3:
    st.markdown("### Authentication & Error Handling")
    st.markdown("""
- **Local Phase 4 API**: no auth required on `127.0.0.1` (development only).
- **Production**: send institutional key via `X-API-Key` header.
- **Rate limits**: target **300 req/min** on standard tiers; enterprise can be unthrottled.
- **Error codes**:
  - `400` — invalid symbol or parameters  
  - `401` — missing/invalid API key  
  - `404` — asset not tracked  
  - `429` — rate limit; use backoff  
  - `500` — upstream data or exchange adapter failure  
""")
    st.caption("Keep uvicorn running in a second terminal while testing the playground with “Try local FastAPI”.")

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
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Institutional briefs aligned to multi-pillar telemetry and market structure. "
      "Expand a report for abstract and recommended terminal views."
      "</p>",
      unsafe_allow_html=True,
  )
  reports = [
      {
          "Date": "2026-07-15",
          "Title": "Q3 Macro Liquidity Outlook & Rate Path",
          "Focus": "Economics / Macro",
          "Status": "Published",
          "Abstract": "Reviews global liquidity conditions, rate-path scenarios, and implications for large-cap crypto market caps and correlation regimes.",
          "Terminal Links": "Overview → Macro Overview · Market Analysis → AI Select",
      },
      {
          "Date": "2026-07-01",
          "Title": "Whale Accumulation Patterns Across L1s",
          "Focus": "Network / On-Chain",
          "Status": "Published",
          "Abstract": "Maps large-wallet accumulation and exchange outflow clusters across BTC, ETH, and SOL using transfer and netflow proxies.",
          "Terminal Links": "Overview → On-Chain Activity · Project Detail → Capital Flows",
      },
      {
          "Date": "2026-06-20",
          "Title": "Validator Security & Code Velocity Benchmarks",
          "Focus": "Source Code",
          "Status": "Published",
          "Abstract": "Benchmarks commit velocity, active developer cohorts, and security-related repository signals feeding the Source Code pillar.",
          "Terminal Links": "Project Detail → 5 Pillars · Home → Leaderboard",
      },
      {
          "Date": "2026-06-05",
          "Title": "Accessibility Index: Exchange & Wallet Coverage",
          "Focus": "Accessibility",
          "Status": "Draft",
          "Abstract": "Draft framework for scoring exchange listings and wallet support — designed to absorb proprietary exchange coverage when the feed is connected.",
          "Terminal Links": "Project Detail → 5 Pillars (Accessibility) · Settings",
      },
  ]
  focus_filter = st.multiselect(
      "Filter by focus",
      ["Economics / Macro", "Network / On-Chain", "Source Code", "Accessibility"],
      default=["Economics / Macro", "Network / On-Chain", "Source Code", "Accessibility"],
  )
  for r in reports:
      if r["Focus"] not in focus_filter:
          continue
      with st.expander(f"{r['Date']}  ·  {r['Title']}  ·  [{r['Status']}]", expanded=False):
          st.markdown(f"**Focus:** {r['Focus']}")
          st.markdown(r["Abstract"])
          st.caption(f"Suggested views: {r['Terminal Links']}")
  st.info("Full report PDFs can reuse the executive export pipeline used for asset reports.")

elif current_view == "Market Analysis":
  ma_health = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
  ma_pillars = ma_health.get("pillar_scores", {})
  ma_composite = ma_health.get("health_score", 0)

  m_tab_overview, m_tab_trading, m_tab_ai, m_tab_unlock = st.tabs([
      "Overview", "Trading Data", "AI Select", "Token Unlock"
  ])

  with m_tab_overview:
      st.markdown("### Market Overview & Tracked Assets")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Highlights plus live economics trajectory for the sidebar target asset."
          "</p>",
          unsafe_allow_html=True,
      )

      # Live ranking by health
      rank_rows = []
      for sym in ["BTC", "ETH", "SOL", "ADA"]:
          h = compute_blockactivities_health_score(sym)["health_score"]
          econ = page_data["economics"][page_data["economics"]["asset_symbol"] == sym]
          mcap = econ["market_cap"].iloc[-1] if not econ.empty else 0
          rank_rows.append({"Asset": sym, "Health": round(h, 1), "Market Cap": format_currency(mcap)})
      st.dataframe(pd.DataFrame(rank_rows).sort_values("Health", ascending=False), use_container_width=True, hide_index=True)

      st.markdown("### Market Cap Trajectory")
      econ_df = page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol]
      if not econ_df.empty:
          render_history_chart(
              econ_df,
              "market_cap",
              f"{asset_symbol} Market Capitalization Trajectory",
              "USD ($)",
              color="#8b5cf6",
          )

  with m_tab_trading:
      st.markdown("### Trading Data & Order Flow")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Ledger-derived taker flow plus derivatives context. Exchange-native books can plug in here later."
          "</p>",
          unsafe_allow_html=True,
      )
      try:
          flow_df = compute_net_taker_flow(db_path="crypto_data.db")
          st.dataframe(flow_df, use_container_width=True, hide_index=True)
      except Exception as e:
          st.warning(f"Taker flow unavailable: {e}")

      t_col1, t_col2 = st.columns(2)
      with t_col1:
          st.markdown("""
              <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px;">
                  <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Derivatives Snapshot</h4>
                  <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Total Open Interest:</b> $18.4B (+4.2%)</p>
                  <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Avg Funding Rate:</b> +0.0115%</p>
                  <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;"><b>Long / Short Ratio:</b> 1.62</p>
              </div>
          """, unsafe_allow_html=True)
      with t_col2:
          st.markdown("""
              <div style="background-color: #1f2937; border: 1px solid #374151; padding: 18px; border-radius: 8px;">
                  <h4 style="color: #ffffff; margin-top: 0; font-size: 0.95rem;">Exchange Integration</h4>
                  <p style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 6px;">Ready for proprietary CEX volume, depth, and user-flow feeds.</p>
                  <span style="color: #3b82f6; font-size: 0.78rem; font-weight: bold;">● Status: Contract defined · feed pending</span>
              </div>
          """, unsafe_allow_html=True)

  with m_tab_ai:
      st.markdown("### AI Select & Multi-Pillar Scoring")
      st.markdown(
          "<p style='color: #9ca3af; font-size: 0.85rem;'>"
          "Model outlook combined with live composite health for the selected asset."
          "</p>",
          unsafe_allow_html=True,
      )
      bias = "Hold / Accumulate" if ma_composite >= 55 else "Monitor / Reduce Risk"
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

      # Optional lightweight forecast chart from economics history
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
      ]
      st.dataframe(pd.DataFrame(unlock_data), use_container_width=True, hide_index=True)

elif current_view == "News":
  st.subheader("Crypto Industry News Feed")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Curated headlines tagged to pillar themes for institutional monitoring."
      "</p>",
      unsafe_allow_html=True,
  )
  news_items = [
      {"Time": "2 hours ago", "Headline": "SEC Approves New Multi-Chain ETF Baskets", "Tag": "Regulatory", "Summary": "Expanded product shelf may increase institutional accessibility for multi-asset exposure."},
      {"Time": "5 hours ago", "Headline": "Network Throughput Surges Across Layer-1 Ecosystems", "Tag": "Network", "Summary": "Elevated TPS and active addresses support stronger Network pillar readings on major L1s."},
      {"Time": "1 day ago", "Headline": "Whale Wallet Accumulation Reaches 6-Month High", "Tag": "On-Chain", "Summary": "Large-wallet inflows coincide with exchange outflows — monitor Overview whale tables."},
      {"Time": "2 days ago", "Headline": "Developer Commit Velocity Rises on Major L1 Repos", "Tag": "Source Code", "Summary": "Higher commit counts feed Source Code pillar inputs used in composite health."},
      {"Time": "3 days ago", "Headline": "Institutional Custody Providers Expand Staking Support", "Tag": "Accessibility", "Summary": "Broader custody and staking options improve institutional accessibility scores."},
      {"Time": "4 days ago", "Headline": "Perpetual Funding Rates Normalize After Crowded Longs", "Tag": "Economics", "Summary": "Funding compression reduces leverage stress — relevant to Derivatives panels."},
  ]
  tag_options = sorted({n["Tag"] for n in news_items})
  selected_tags = st.multiselect("Filter by tag", tag_options, default=tag_options)
  for n in news_items:
      if n["Tag"] not in selected_tags:
          continue
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 12px 15px; border-radius: 8px; margin-bottom: 10px;">
              <span style="color: #10b981; font-size: 0.75rem; font-weight: 600;">{n['Tag']}</span>
              <span style="color: #6b7280; font-size: 0.75rem;"> · {n['Time']}</span>
              <p style="color: #f3f4f6; font-size: 0.9rem; margin: 4px 0 2px 0;"><b>{n['Headline']}</b></p>
              <p style="color: #9ca3af; font-size: 0.8rem; margin: 0;">{n['Summary']}</p>
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

      # Quick jump to full Project Detail
      if st.button("Open full Project Detail →", key="ov_to_detail"):
          push_nav_history()
          st.session_state.nav_category = "🖥 Terminal"
          st.session_state.nav_section = "Project Detail Page"
          st.rerun()

  with ov_sub_tab2:
      st.markdown("### Order Book Depth & Market Microstructure")
      _ex_mode = exchange_mode() if callable(exchange_mode) else "mock"
      st.markdown(
          f"<p style='color: #9ca3af; font-size: 0.85rem;'>"
          f"Aggregate depth from the exchange adapter "
          f"(mode: <b>{_ex_mode}</b>). Set EXCHANGE_MODE=live + API keys for proprietary books."
          f"</p>",
          unsafe_allow_html=True,
      )

      depth_col1, depth_col2 = st.columns([2, 1])
      with depth_col1:
          book_bids, book_asks = [], []
          if exchange_refresh_symbol and exchange_get_orderbook:
              try:
                  exchange_refresh_symbol(asset_symbol)
                  book = exchange_get_orderbook(asset_symbol)
                  book_bids = book.get("bids") or []
                  book_asks = book.get("asks") or []
              except Exception:
                  book_bids, book_asks = [], []

          fig_depth = go.Figure()
          if book_bids and book_asks:
              bid_prices = [lv[0] for lv in book_bids]
              bid_sizes = list(np.cumsum([lv[1] for lv in book_bids]))
              ask_prices = [lv[0] for lv in book_asks]
              ask_sizes = list(np.cumsum([lv[1] for lv in book_asks]))
              fig_depth.add_trace(go.Scatter(
                  x=bid_prices, y=bid_sizes, fill="tozeroy", name="Bids", line=dict(color="#10b981")
              ))
              fig_depth.add_trace(go.Scatter(
                  x=ask_prices, y=ask_sizes, fill="tozeroy", name="Asks", line=dict(color="#ef4444")
              ))
              depth_title = f"{asset_symbol} Order Book ({_ex_mode})"
          else:
              price_base = 65000 if asset_symbol == "BTC" else (3200 if asset_symbol == "ETH" else (145 if asset_symbol == "SOL" else 0.48))
              prices_dummy = np.linspace(price_base * 0.95, price_base * 1.05, 50)
              bids_cum = np.cumsum(np.random.exponential(10, 50))[::-1]
              asks_cum = np.cumsum(np.random.exponential(10, 50))
              fig_depth.add_trace(go.Scatter(
                  x=prices_dummy[:25], y=bids_cum[:25], fill="tozeroy", name="Bids", line=dict(color="#10b981")
              ))
              fig_depth.add_trace(go.Scatter(
                  x=prices_dummy[25:], y=asks_cum[25:], fill="tozeroy", name="Asks", line=dict(color="#ef4444")
              ))
              depth_title = f"{asset_symbol} Aggregate Order Book Depth (fallback)"

          fig_depth.update_layout(
              title=depth_title,
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

      # Net taker flow from analytics when available
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

    # Live health + latest metrics for this asset
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

    # Top-line composite metrics
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

    # --- TAB: 5 PILLARS (core product answer for leadership) ---
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

        # Pillar score bars via simple metrics row
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
        # Historical charts for key pillars
        econ_hist = page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol]
        net_hist = page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol]
        chart_c1, chart_c2 = st.columns(2)
        with chart_c1:
            render_history_chart(econ_hist, "market_cap", f"{asset_symbol} Market Cap", "USD", color="#8b5cf6")
        with chart_c2:
            render_history_chart(net_hist, "tx_tps", f"{asset_symbol} Network TPS", "TPS", color="#3b82f6")

    # --- TAB: Derivatives ---
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

    # --- TAB: On-Chain Supply ---
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

        # Active addresses trend
        render_history_chart(
            page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol],
            "active_addresses",
            f"{asset_symbol} Active Addresses",
            "Addresses",
            color="#10b981",
        )

    # --- TAB: Capital Flows & Exchange (prep for acquired exchange) ---
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

    # --- TAB: Regulatory ---
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

    # Summary strip
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
                    st.session_state.nav_category = "🖥 Terminal"
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
      """Callback runs before widgets instantiate on the next run."""
      push_nav_history()
      st.session_state.asset_symbol = symbol
      st.session_state.nav_category = "🖥 Terminal"
      st.session_state.nav_section = "Project Detail Page"

  def _nav_to_overview():
      push_nav_history()
      st.session_state.nav_category = "🖥 Terminal"
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

elif current_view == "Tutorials":
  st.subheader("Terminal Tutorials")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Step-by-step walkthroughs mapped to live terminal features."
      "</p>",
      unsafe_allow_html=True,
  )

  tutorials = [
      {
          "Title": "Navigating the Multi-Pillar Dashboard",
          "Level": "Beginner",
          "Time": "5 min",
          "Summary": "Understand how Health Scores are composed from five pillars.",
          "Steps": [
              "Open Overview Dashboard or Project Detail for any asset.",
              "Review the composite Health Score (0–100) at the top.",
              "Open the 5 Pillars tab on Project Detail to see Source Code, Network, Economics, Sentiment, and Accessibility.",
              "Use the sidebar Target Asset selector to switch protocols without leaving the page family.",
          ],
      },
      {
          "Title": "Setting Up Automated Health Alerts",
          "Level": "Beginner",
          "Time": "4 min",
          "Summary": "Configure thresholds, webhooks, and review the audit trail.",
          "Steps": [
              "In the sidebar, set Min Health Score Warning (e.g. 45–55).",
              "Paste a Slack or Telegram webhook URL.",
              "Optionally enable Auto-dispatch webhook on breach.",
              "Use Broadcast Webhook Alert Now for a manual test.",
              "Open Settings → Alert Audit Log to confirm events were recorded.",
          ],
      },
      {
          "Title": "Using Project Explorer & Filters",
          "Level": "Intermediate",
          "Time": "6 min",
          "Summary": "Filter, rank, and open assets into Project Detail.",
          "Steps": [
              "Go to Analytics & Terminal → Project Explorer.",
              "Filter by category and Min Health; search by symbol or name.",
              "Sort by Health (highest first) and inspect pillar columns.",
              "Click Open BTC / ETH / SOL / ADA to jump into Project Detail.",
              "Use ← Go Back to return to the Explorer.",
          ],
      },
      {
          "Title": "Exporting Executive PDF & CSV Reports",
          "Level": "Intermediate",
          "Time": "3 min",
          "Summary": "Produce board-ready exports from the sidebar.",
          "Steps": [
              "Select the target asset in the sidebar.",
              "Under Executive Report Exports, download the CSV feed.",
              "Generate the PDF report for committee packs.",
              "Pair exports with the Alert Audit Log CSV from Settings when documenting risk events.",
          ],
      },
      {
          "Title": "Interpreting Whale Transactions & Order Flow",
          "Level": "Advanced",
          "Time": "8 min",
          "Summary": "Read large transfers and net taker pressure in context.",
          "Steps": [
              "Open Overview → On-Chain Activity for whale rows on the selected asset.",
              "Open Overview → Order Flow for net taker Buy/Sell/Net_Flow tables.",
              "Cross-check Capital Flows on Project Detail for CEX / DeFi / OTC narrative.",
              "Treat large outflows from exchanges as potentially constructive; large inflows as possible distribution risk.",
          ],
      },
      {
          "Title": "Reading Market Analysis AI Select",
          "Level": "Intermediate",
          "Time": "5 min",
          "Summary": "Combine model outlook with live pillar scores.",
          "Steps": [
              "Open Insights → Market Analysis → AI Select.",
              "Note composite health and recommendation (Hold / Accumulate vs Monitor).",
              "Review pillar inputs panel — weak pillars drive the cautious bias.",
              "Use the market-cap forecast chart as directional context, not a trading signal.",
          ],
      },
  ]
  for t in tutorials:
      with st.expander(f"{t['Title']}  ·  {t['Level']} · {t['Time']}", expanded=False):
          st.markdown(f"*{t['Summary']}*")
          for i, step in enumerate(t["Steps"], 1):
              st.markdown(f"{i}. {step}")

elif current_view == "Guides":
  st.subheader("Risk Management & Compliance Guides")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Operating standards for institutional use of the terminal."
      "</p>",
      unsafe_allow_html=True,
  )

  guides = [
      {
          "Title": "Establishing Alert Thresholds",
          "Body": "Start with a conservative floor (45–55). Review 30-day health volatility per asset before tightening. Different assets can justify different thresholds once you know their normal range.",
      },
      {
          "Title": "Webhook & Incident Response",
          "Body": "Route alerts to a dedicated Slack/Telegram channel. Enable auto-dispatch only after a successful manual Broadcast test. Pair notifications with a short runbook: who acknowledges, who escalates, and when to open Project Detail.",
      },
      {
          "Title": "Export & Audit Trail Hygiene",
          "Body": "Use PDF/CSV exports for investment committee packs. Download the Alert Audit Log from Settings when documenting breaches. Clear the log only after archival if your compliance policy requires it.",
      },
      {
          "Title": "Role-Based Access Hygiene",
          "Body": "Limit Regenerate Data and user-management actions to Admin. Portfolio Managers may configure alerts and export. Analysts should focus on read, Explore, and export workflows.",
      },
      {
          "Title": "Exchange Integration (Post Phase 3)",
          "Body": "When the acquired exchange feed is connected, prefer exchange last price and order-book depth over simulated charts. Keep a public-API fallback. Store API keys in environment variables only — never in source control.",
      },
      {
          "Title": "Data Regeneration Safety",
          "Body": "Regenerate All Data rebuilds crypto_data.db only. Alert logs and institutional users are preserved. Announce regeneration to anyone demoting live so they are not surprised by score changes.",
      },
  ]
  for g in guides:
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 16px 18px; border-radius: 8px; margin-bottom: 12px;">
              <b style="color: #10b981; font-size: 0.95rem;">{g['Title']}</b>
              <p style="color: #f3f4f6; font-size: 0.85rem; margin-top: 6px; margin-bottom: 0;">{g['Body']}</p>
          </div>
      """, unsafe_allow_html=True)

elif current_view == "Glossary":
  st.subheader("Blockchain & Finance Glossary")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.9rem;'>"
      "Terms used across the terminal, research, and alert workflows."
      "</p>",
      unsafe_allow_html=True,
  )

  glossary_items = {
      "Health Score": "Composite 0–100 rating from five pillars: Source Code (25%), Network (20%), Economics (20%), Sentiment (15%), Accessibility (20%).",
      "Source Code Pillar": "Developer activity signals — commits, active developers, and repository quality scores.",
      "Network Pillar": "On-chain throughput and usage — TPS, active addresses, gas fees.",
      "Economics Pillar": "Market structure — market cap, volume, and tokenomics quality.",
      "Sentiment Pillar": "User and flow sentiment — buy/sell pressure and sentiment index.",
      "Accessibility Pillar": "How easily institutions can hold and trade the asset — exchange coverage and wallet support.",
      "Net Taker Flow": "Aggressive buy volume minus aggressive sell volume. Positive = net buying pressure.",
      "Buy/Sell Ratio": "Buy volume divided by sell volume. Values above 1 favor buyers.",
      "MVRV": "Market Value to Realized Value — market cap relative to aggregate holder cost basis.",
      "TPS": "Transactions per second — core network throughput metric.",
      "Whale Transaction": "Large on-chain transfer (often > $1M notional) that may signal accumulation or distribution.",
      "TVL": "Total Value Locked — USD value of assets deposited in DeFi protocols.",
      "Funding Rate": "Periodic payment between longs and shorts in perpetual futures markets.",
      "Open Interest (OI)": "Total outstanding derivatives contracts — a measure of leveraged positioning.",
      "Order Book Depth": "Cumulative size available at successive price levels for bids and asks.",
      "Webhook": "HTTP callback URL (e.g. Slack/Telegram) that receives automated alert JSON payloads.",
      "Alert Audit Log": "Append-only style record of health breaches and webhook dispatch results in the institutional database.",
      "Composite Score": "Weighted blend of the five pillar scores into a single institutional health rating.",
  }
  term_search = st.text_input("Filter glossary", placeholder="Type a term...")
  for term, definition in glossary_items.items():
      if term_search and term_search.lower() not in term.lower() and term_search.lower() not in definition.lower():
          continue
      st.markdown(f"""
          <div style="background-color: #1f2937; border: 1px solid #374151; padding: 12px 15px; border-radius: 6px; margin-bottom: 10px;">
              <b style="color: #10b981;">{term}</b>
              <p style="color: #f3f4f6; font-size: 0.85rem; margin-top: 4px; margin-bottom: 0;">{definition}</p>
          </div>
      """, unsafe_allow_html=True)

elif current_view == "Settings":
  st.subheader("Terminal Settings & Configurations")
  st.markdown("Manage data, alerts, API access, and notification endpoints.")

  st.markdown("---")
  st.markdown("### Data Management")
  st.markdown(
      f"<p style='color: #9ca3af; font-size: 0.9rem;'>"
      f"<b>Last data refresh:</b> {data_last_refreshed}<br/>"
      "Regenerate the simulated institutional dataset (trades + multi-pillar metrics)."
      "</p>",
      unsafe_allow_html=True,
  )

  col_reset1, col_reset2 = st.columns([1, 2])
  with col_reset1:
      if st.button("🔄 Regenerate All Data", type="primary", use_container_width=True):
          with st.spinner("Regenerating trades and pillar metrics..."):
              try:
                  if os.path.exists("crypto_data.db"):
                      os.remove("crypto_data.db")
                  generate_simulated_trades(num_records=5000, db_path="crypto_data.db")
                  generate_all_crypto_metrics(days=30, db_path="crypto_data.db")
                  st.cache_data.clear()
                  st.session_state.alert_breach_keys = set()
                  st.success("Data regenerated successfully. Reloading dashboard...")
                  st.rerun()
              except Exception as e:
                  st.error(f"Regeneration failed: {e}")

  with col_reset2:
      st.info("This will delete and recreate `crypto_data.db`. Alert logs and user accounts are not affected.")

  st.markdown("---")
  st.markdown("### Notification Preferences")
  n1, n2, n3 = st.columns(3)
  with n1:
      st.metric("Health Threshold", f"{alert_health_min}")
  with n2:
      st.metric("Webhook", "Configured" if webhook_url_input else "Not set")
  with n3:
      st.metric("Auto-dispatch", "On" if auto_webhook else "Off")
  st.caption("Adjust threshold, webhook URL, and auto-dispatch from the sidebar.")

  st.markdown("---")
  st.markdown("### Alert Audit Log")
  st.markdown(
      "<p style='color: #9ca3af; font-size: 0.85rem;'>"
      "Immutable-style trail of health breaches and webhook dispatches for compliance review."
      "</p>",
      unsafe_allow_html=True,
  )
  try:
      conn_a = sqlite3.connect("bnanalytics_institutional.db")
      audit_df = pd.read_sql(
          "SELECT log_id, timestamp, asset_symbol, health_score, threshold, status "
          "FROM alert_audit_logs ORDER BY log_id DESC LIMIT 50",
          conn_a,
      )
      conn_a.close()
      if not audit_df.empty:
          st.dataframe(audit_df, use_container_width=True, hide_index=True)
          a1, a2 = st.columns(2)
          with a1:
              st.download_button(
                  "Download Alert Log CSV",
                  audit_df.to_csv(index=False).encode("utf-8"),
                  file_name="bnanalytics_alert_audit.csv",
                  mime="text/csv",
                  use_container_width=True,
              )
          with a2:
              if st.button("Clear Alert Log", use_container_width=True):
                  conn_c = sqlite3.connect("bnanalytics_institutional.db")
                  conn_c.execute("DELETE FROM alert_audit_logs")
                  conn_c.commit()
                  conn_c.close()
                  st.session_state.alert_breach_keys = set()
                  st.success("Alert log cleared.")
                  st.rerun()
      else:
          st.caption("No alert events recorded yet. Breach events appear when health falls below threshold.")
  except Exception as e:
      st.warning(f"Could not load alert audit log: {e}")

  st.markdown("---")
  st.markdown("### Account & Access")
  st.markdown(f"**Logged in as:** `{st.session_state.username}`")
  st.markdown(f"**Role:** `{st.session_state.role}`")
  st.markdown("**API Key (demo):** `bn_live_99f8a2c10b`")
  st.caption(f"Data last refreshed: {data_last_refreshed}")
