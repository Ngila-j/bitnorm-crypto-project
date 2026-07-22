import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from analytics import (
    compute_blockactivities_health_score,
    compute_net_taker_flow,
    compute_user_sentiment_index,
    fetch_latest_crypto_metrics,
)
from pipeline import generate_all_crypto_metrics, generate_simulated_trades

# Page Configuration with wide layout
st.set_page_config(
    page_title="BitNorm / BNAnalytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling (BitNorm Dark Theme & Clean Navigation)
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #f3f4f6;
    }
    .sidebar .sidebar-content {
        background-color: #111827;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border: 1px solid #374151;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .ticker-bar {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 8px;
        font-weight: 600;
        color: #10b981;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Clean Sidebar Button Styling for Navigation */
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
""", unsafe_allow_html=True)

@st.cache_data
def load_dashboard_data():
    """Loads dashboard data from SQLite and initializes simulated datasets if missing."""
    conn = sqlite3.connect("crypto_data.db")
    cursor = conn.cursor()

    required_tables = [
        "customer_trades",
        "sourcecode_metrics",
        "network_metrics",
        "economics_metrics",
        "sentiment_metrics",
        "accessibility_metrics",
    ]

    missing_tables = [t for t in required_tables if cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is None]

    if missing_tables:
        conn.close()
        generate_simulated_trades(num_records=5000, db_path="crypto_data.db")
        generate_all_crypto_metrics(days=30, db_path="crypto_data.db")
        conn = sqlite3.connect("crypto_data.db")
        cursor = conn.cursor()

    trades = pd.read_sql("SELECT * FROM customer_trades", conn)
    sourcecode = pd.read_sql("SELECT * FROM sourcecode_metrics", conn)
    network = pd.read_sql("SELECT * FROM network_metrics", conn)
    economics = pd.read_sql("SELECT * FROM economics_metrics", conn)
    sentiment = pd.read_sql("SELECT * FROM sentiment_metrics", conn)
    accessibility = pd.read_sql("SELECT * FROM accessibility_metrics", conn)
    conn.close()

    if "timestamp" in trades.columns:
        trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    for frame in [sourcecode, network, economics, sentiment, accessibility]:
        if "metric_date" in frame.columns:
            frame["metric_date"] = pd.to_datetime(frame["metric_date"])

    return {
        "trades": trades,
        "sourcecode": sourcecode,
        "network": network,
        "economics": economics,
        "sentiment": sentiment,
        "accessibility": accessibility,
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
    chart_data = frame[["metric_date", metric_name]].copy().sort_values("metric_date")
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
        title_font_size=18,
    )
    st.plotly_chart(fig, use_container_width=True)

# Load data
page_data = load_dashboard_data()

# --- TOP LIVE TICKER SIMULATION ---
st.markdown("""
    <div class="ticker-bar">
        ⚡ BITNORM LIVE FEED: BTC/USD $65,901 (+3.4%) | ETH/USD $1,927 (+1.8%) | SOL/USD $142.50 (+5.1%) | ADA/USD $0.48 (-0.6%)
    </div>
""", unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
if "nav_section" not in st.session_state:
    st.session_state.nav_section = "Home"

st.sidebar.image("logo.png", width=50)
st.sidebar.title("BITNORM")
st.sidebar.caption("BNAnalytics Intelligence Module")
st.sidebar.markdown("---")

nav_mapping = {
    "Home": "Home",
    "Code Intelligence": "Code Intelligence",
    "Ledger Metrics": "Ledger Metrics",
    "Market Economics": "Market Economics",
    "Social Sentiment": "Social Sentiment",
    "Ecosystem Liquidity": "Ecosystem Liquidity"
}

for label, target_section in nav_mapping.items():
    if st.sidebar.button(label, key=f"nav_{label}"):
        st.session_state.nav_section = target_section

section = st.session_state.nav_section

st.sidebar.markdown("---")
asset_symbol = st.sidebar.selectbox("Target Asset", ["BTC", "ETH", "SOL", "ADA"], index=0)

st.sidebar.markdown("---")
st.sidebar.info("💡 **BitNorm Tip:** Pivot seamlessly between macro overview and micro vector inspections.")

# --- VIEW ROUTING ---
if section == "Home":
    st.title("BNAnalytics Command Center")
    st.markdown("##### Real-time automated health tracking, multi-dimensional code verification, and on-chain sentiment analysis across global web3 markets.")

    latest_economics = page_data["economics"].sort_values("metric_date").groupby("asset_symbol").tail(1)
    latest_health = []
    for symbol in ["BTC", "ETH", "SOL", "ADA"]:
        latest_health.append((symbol, compute_blockactivities_health_score(symbol, db_path="crypto_data.db")["health_score"]))

    health_df = pd.DataFrame(latest_health, columns=["asset_symbol", "health_score"])
    latest_economics = latest_economics.merge(health_df, on="asset_symbol", how="left")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Global Market Cap", format_currency(latest_economics["market_cap"].sum()), "+4.2%")
    with col2:
        st.metric("Global 24h Volume", format_currency(latest_economics["volume_24h"].sum()), "+12.8%")
    with col3:
        st.metric("Avg. Health Score", f"{latest_economics['health_score'].mean():.1f}/100", "Optimized")
    with col4:
        st.metric("Tracked Assets", f"{len(latest_economics)} Core", "Active")

    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    with col_a:
        fig_health = px.bar(
            health_df.sort_values("health_score", ascending=False),
            x="asset_symbol",
            y="health_score",
            color="health_score",
            color_continuous_scale="Tealgrn",
            title="BitNorm Composite Health Scores",
        )
        fig_health.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
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
        fig_market.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f3f4f6")
        st.plotly_chart(fig_market, use_container_width=True)

    st.markdown("### Selected Asset Deep-Dive")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards([
        ("Asset Health Score", f"{health_score['health_score']:.1f}/100", "Top Tier"),
        ("Code Metric", f"{health_score['pillar_scores']['sourcecode']:.1f}", "Active Devs"),
        ("Markets Metric", f"{health_score['pillar_scores']['economics']:.1f}", "Liquidity Strong"),
    ])

    fig_radar = go.Figure()
    fig_radar.add_trace(
        go.Scatterpolar(
            r=[
                health_score["pillar_scores"]["sourcecode"],
                health_score["pillar_scores"]["network"],
                health_score["pillar_scores"]["economics"],
                health_score["pillar_scores"]["sentiment"],
                health_score["pillar_scores"]["accessibility"],
            ],
            theta=["Code", "Ledger", "Markets", "Social", "Liquidity"],
            fill="toself",
            name=asset_symbol,
            line_color="#10b981",
        )
    )
    fig_radar.update_layout(
        title=f"{asset_symbol} 5-Vector Intelligence Profile",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#374151"),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f3f4f6"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

elif section == "Code Intelligence":
    st.subheader("Code & Developer Activity")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["sourcecode"] or {}
    render_metric_cards([
        ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
        ("Commit Velocity", f"{latest.get('commits', 0):,}", "Weekly Rate"),
        ("Active Developers", f"{latest.get('active_devs', 0):,}", "Contributors"),
    ])
    render_history_chart(page_data["sourcecode"][page_data["sourcecode"]["asset_symbol"] == asset_symbol], "commits", f"{asset_symbol} Commit Trend Line", "Commits", color="#3b82f6")

elif section == "Ledger Metrics":
    st.subheader("Ledger Health & Throughput")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["network"] or {}
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards([
        ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
        ("Active Addresses", f"{latest.get('active_addresses', 0):,}", "On-Chain Users"),
        ("Transactions / Sec (TPS)", f"{latest.get('tx_tps', 0):.2f}", "Throughput"),
    ])
    render_history_chart(page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol], "tx_tps", f"{asset_symbol} Ledger TPS Performance", "TPS", color="#10b981")

elif section == "Market Economics":
    st.subheader("Tokenomics & Market Economics")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["economics"] or {}
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards([
        ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
        ("Market Cap", format_currency(latest.get("market_cap", 0)), "Valuation"),
        ("24h Volume", format_currency(latest.get("volume_24h", 0)), "Liquidity"),
    ])
    render_history_chart(page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol], "market_cap", f"{asset_symbol} Market Cap Valuation Trend", "Market Cap", color="#8b5cf6")

elif section == "Social Sentiment":
    st.subheader("User Sentiment & Market Mood")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["sentiment"] or {}
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards([
        ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
        ("Sentiment Index", f"{latest.get('user_sentiment_index', 0):.3f}", "Aggregated Index"),
        ("Buy / Sell Pressure", f"{latest.get('buy_sell_ratio', 0):.2f}", "Ratio"),
    ])
    render_history_chart(page_data["sentiment"][page_data["sentiment"]["asset_symbol"] == asset_symbol], "user_sentiment_index", f"{asset_symbol} User Sentiment Evolution", "Sentiment Index", color="#f59e0b")

elif section == "Ecosystem Liquidity":
    st.subheader("Accessibility & Ecosystem Liquidity")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["accessibility"] or {}
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards([
        ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
        ("Exchange Availability", f"{latest.get('exchange_count', 0):,}", "Listings"),
        ("Wallet Support Score", f"{latest.get('wallet_support_score', 0):.2f}", "Integration Score"),
    ])
    render_history_chart(page_data["accessibility"][page_data["accessibility"]["asset_symbol"] == asset_symbol], "wallet_support_score", f"{asset_symbol} Wallet Support Progress", "Support Score", color="#ec4899")

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #9ca3af;'>© 2013–2026 BitNorm.com — Professional Multi-Dimensional Crypto Intelligence & BNAnalytics Platform.</p>", unsafe_allow_html=True)