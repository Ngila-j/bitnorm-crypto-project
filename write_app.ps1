$path = Join-Path $PSScriptRoot 'app.py'
@'
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

ASSET_SYMBOLS = ["BTC", "ETH", "SOL", "ADA"]

st.set_page_config(
    page_title="BlockActivities.com / BN Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #0b0f19 0%, #111827 100%);
        color: #f9fafb;
    }
    .stSidebar {
        background-color: #111827;
    }
    .block-card {
        background: linear-gradient(135deg, rgba(31,41,55,0.95), rgba(17,24,39,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .ticker-bar {
        background: linear-gradient(90deg, #111827, #1f2937);
        color: #10b981;
        border-radius: 10px;
        padding: 10px 14px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .stSelectbox > div > div {
        background-color: #1f2937;
        color: #f9fafb;
        border: 1px solid #374151;
        border-radius: 8px;
    }
    h1, h2, h3, h4 {
        color: #f9fafb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dashboard_data():
    """Loads and normalizes multi-pillar crypto metric data from SQLite."""
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

    missing_tables = [
        table_name
        for table_name in required_tables
        if cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone() is None
    ]

    if missing_tables:
        conn.close()
        generate_simulated_trades(num_records=5000, db_path="crypto_data.db")
        generate_all_crypto_metrics(days=30, db_path="crypto_data.db")
        conn = sqlite3.connect("crypto_data.db")

    trades = pd.read_sql("SELECT * FROM customer_trades", conn)
    sourcecode = pd.read_sql("SELECT * FROM sourcecode_metrics", conn)
    network = pd.read_sql("SELECT * FROM network_metrics", conn)
    economics = pd.read_sql("SELECT * FROM economics_metrics", conn)
    sentiment = pd.read_sql("SELECT * FROM sentiment_metrics", conn)
    accessibility = pd.read_sql("SELECT * FROM accessibility_metrics", conn)
    conn.close()

    normalized_frames = {}
    for name, frame in {
        "trades": trades,
        "sourcecode": sourcecode,
        "network": network,
        "economics": economics,
        "sentiment": sentiment,
        "accessibility": accessibility,
    }.items():
        if frame.empty:
            normalized_frames[name] = frame
            continue

        normalized = frame.copy()
        if "asset_symbol" in normalized.columns:
            normalized["asset_symbol"] = normalized["asset_symbol"].astype(str).str.upper()
        if "metric_date" in normalized.columns:
            normalized["metric_date"] = pd.to_datetime(normalized["metric_date"])
        if "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])

        numeric_cols = [col for col in normalized.columns if col not in {"asset_symbol", "metric_date", "timestamp"}]
        for col in numeric_cols:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
            normalized[col] = normalized[col].ffill().bfill()

        if "asset_symbol" in normalized.columns and "metric_date" in normalized.columns:
            normalized = normalized.sort_values(["asset_symbol", "metric_date"]).reset_index(drop=True)
        normalized_frames[name] = normalized

    return normalized_frames


def format_currency(value):
    if value is None:
        return "—"
    if abs(value) >= 1e12:
        return f"${value / 1e12:,.2f}T"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f}M"
    return f"${value:,.2f}"


def render_metric_cards(metrics):
    cols = st.columns(len(metrics))
    for col, (title, value, delta) in zip(cols, metrics):
        with col:
            st.markdown("<div class='block-card'>", unsafe_allow_html=True)
            st.metric(label=title, value=value, delta=delta)
            st.markdown("</div>", unsafe_allow_html=True)


def render_history_chart(frame, metric_name, title, y_label, color="#10b981"):
    if frame.empty:
        st.info("No historical data is available yet for this asset.")
        return

    chart_data = frame[["metric_date", metric_name]].copy().sort_values("metric_date")
    fig = px.line(
        chart_data,
        x="metric_date",
        y=metric_name,
        markers=True,
        title=title,
        labels={"metric_date": "Date", metric_name: y_label},
    )
    fig.update_traces(line_color=color, line_width=3)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f9fafb",
        title_font_size=18,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dual_series(frame, first_metric, second_metric, title, first_label, second_label, first_color="#3b82f6", second_color="#10b981"):
    if frame.empty:
        st.info("No historical data is available yet for this asset.")
        return

    chart_data = frame[["metric_date", first_metric, second_metric]].copy().sort_values("metric_date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data["metric_date"], y=chart_data[first_metric], mode="lines+markers", name=first_label, line=dict(color=first_color, width=3)))
    fig.add_trace(go.Scatter(x=chart_data["metric_date"], y=chart_data[second_metric], mode="lines+markers", name=second_label, line=dict(color=second_color, width=3), yaxis="y2"))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=first_label,
        yaxis2=dict(title=second_label, overlaying="y", side="right"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f9fafb",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


page_data = load_dashboard_data()

st.markdown(
    """
    <div class="ticker-bar">
        ⚡ LIVE FEED: BTC/USD $65,901 (+3.4%) | ETH/USD $1,927 (+1.8%) | SOL/USD $142.50 (+5.1%) | ADA/USD $0.48 (-0.6%)
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("BlockActivities")
st.sidebar.caption("Institutional Crypto Intelligence")

section = st.sidebar.selectbox(
    "Navigation",
    [
        "Home / Command Center",
        "Code Intelligence",
        "Ledger Metrics",
        "Market Economics",
        "Social Sentiment",
        "Ecosystem Liquidity",
    ],
)
asset_symbol = st.sidebar.selectbox("Target Asset", ASSET_SYMBOLS, index=0)

st.sidebar.markdown("---")
st.sidebar.info("Use the sidebar to pivot between macro overview and deep-dive pillar analytics.")

if section == "Home / Command Center":
    st.title("BlockActivities / BN Analytics")
    st.markdown("##### A high-end intelligence command center covering code health, on-chain activity, market economics, sentiment, and ecosystem access.")

    latest_economics = page_data["economics"].sort_values("metric_date").groupby("asset_symbol").tail(1)
    latest_health = []
    for symbol in ASSET_SYMBOLS:
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
            title="Composite Health Score by Asset",
            labels={"asset_symbol": "Asset", "health_score": "Health Score"},
        )
        fig_health.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f9fafb")
        st.plotly_chart(fig_health, use_container_width=True)

    with col_b:
        fig_market = px.bar(
            latest_economics,
            x="asset_symbol",
            y="market_cap",
            color="market_cap",
            color_continuous_scale="Purples",
            title="Latest Market Cap Allocation",
            labels={"asset_symbol": "Asset", "market_cap": "Market Cap"},
        )
        fig_market.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#f9fafb")
        st.plotly_chart(fig_market, use_container_width=True)

    st.subheader(f"Selected Asset Snapshot: {asset_symbol}")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Sourcecode Pillar", f"{health_score['pillar_scores']['sourcecode']:.1f}", "Developer signal"),
            ("Economics Pillar", f"{health_score['pillar_scores']['economics']:.1f}", "Market strength"),
        ]
    )

    radar = go.Figure()
    radar.add_trace(
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
    radar.update_layout(
        title=f"{asset_symbol} Pillar Intelligence Radar",
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], gridcolor="#374151")),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f9fafb",
    )
    st.plotly_chart(radar, use_container_width=True)

    flow_df = compute_net_taker_flow(db_path="crypto_data.db")
    sentiment_df = compute_user_sentiment_index(db_path="crypto_data.db")
    col_c, col_d = st.columns(2)
    with col_c:
        buy_sell_fig = px.bar(
            flow_df,
            x="asset_symbol",
            y=["Buy", "Sell"],
            barmode="group",
            title="Customer Buy/Sell Flow by Asset",
            labels={"asset_symbol": "Asset", "value": "Volume"},
        )
        st.plotly_chart(buy_sell_fig, use_container_width=True)
    with col_d:
        sentiment_fig = px.bar(
            sentiment_df,
            x="asset_symbol",
            y="Sentiment_Index",
            color="Sentiment_Index",
            color_continuous_scale="RdYlGn",
            title="User Sentiment Index by Asset",
            labels={"asset_symbol": "Asset", "Sentiment_Index": "Sentiment Index"},
        )
        st.plotly_chart(sentiment_fig, use_container_width=True)

elif section == "Code Intelligence":
    st.subheader("Code Intelligence")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["sourcecode"] or {}
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Commits", f"{latest.get('commits', 0):,}", "Execution rate"),
            ("Active Developers", f"{latest.get('active_devs', 0):,}", "Contributor depth"),
        ]
    )
    source_frame = page_data["sourcecode"][page_data["sourcecode"]["asset_symbol"] == asset_symbol]
    render_dual_series(source_frame, "commits", "active_devs", f"{asset_symbol} Commit Velocity & Developer Growth", "Commits", "Active Developers", first_color="#3b82f6", second_color="#10b981")
    render_history_chart(source_frame, "repo_score", f"{asset_symbol} Repository Score Trend", "Repo Score", color="#8b5cf6")

elif section == "Ledger Metrics":
    st.subheader("Ledger Metrics")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["network"] or {}
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Active Addresses", f"{latest.get('active_addresses', 0):,}", "Growth"),
            ("TPS", f"{latest.get('tx_tps', 0):.2f}", "Throughput"),
        ]
    )
    network_frame = page_data["network"][page_data["network"]["asset_symbol"] == asset_symbol]
    render_dual_series(network_frame, "active_addresses", "tx_tps", f"{asset_symbol} Wallet Growth & Throughput", "Active Addresses", "TPS", first_color="#10b981", second_color="#f59e0b")
    render_history_chart(network_frame, "gas_fee_gwei", f"{asset_symbol} Gas Fee Trend", "Gas Fee (Gwei)", color="#ef4444")

elif section == "Market Economics":
    st.subheader("Market Economics")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["economics"] or {}
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Market Cap", format_currency(latest.get("market_cap", 0)), "Valuation"),
            ("24h Volume", format_currency(latest.get("volume_24h", 0)), "Liquidity"),
        ]
    )
    economics_frame = page_data["economics"][page_data["economics"]["asset_symbol"] == asset_symbol].copy()
    economics_frame["valuation_ratio"] = economics_frame["market_cap"] / (economics_frame["volume_24h"] + 1e-5)
    render_dual_series(economics_frame, "market_cap", "volume_24h", f"{asset_symbol} Market Cap & Volume", "Market Cap", "24h Volume", first_color="#8b5cf6", second_color="#3b82f6")
    render_history_chart(economics_frame, "valuation_ratio", f"{asset_symbol} Valuation Ratio Trend", "Market Cap / Volume", color="#f59e0b")

elif section == "Social Sentiment":
    st.subheader("Social Sentiment")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["sentiment"] or {}
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Sentiment Index", f"{latest.get('user_sentiment_index', 0):.3f}", "Mood"),
            ("Buy/Sell Pressure", f"{latest.get('buy_sell_ratio', 0):.2f}", "Pressure ratio"),
        ]
    )
    sentiment_frame = page_data["sentiment"][page_data["sentiment"]["asset_symbol"] == asset_symbol]
    render_dual_series(sentiment_frame, "user_sentiment_index", "buy_sell_ratio", f"{asset_symbol} Sentiment & Buy/Sell Pressure", "Sentiment Index", "Buy/Sell Ratio", first_color="#f59e0b", second_color="#10b981")
    render_history_chart(sentiment_frame, "user_sentiment_index", f"{asset_symbol} Community Mood Evolution", "Sentiment Index", color="#f59e0b")

elif section == "Ecosystem Liquidity":
    st.subheader("Ecosystem Liquidity")
    snapshot = fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db")
    health_score = compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db")
    latest = snapshot["accessibility"] or {}
    render_metric_cards(
        [
            ("Health Score", f"{health_score['health_score']:.1f}/100", "Composite"),
            ("Exchange Count", f"{latest.get('exchange_count', 0):,}", "Listings"),
            ("Wallet Support", f"{latest.get('wallet_support_score', 0):.2f}", "Integration score"),
        ]
    )
    accessibility_frame = page_data["accessibility"][page_data["accessibility"]["asset_symbol"] == asset_symbol]
    render_dual_series(accessibility_frame, "exchange_count", "wallet_support_score", f"{asset_symbol} Exchange Reach & Wallet Support", "Exchange Count", "Wallet Support Score", first_color="#ec4899", second_color="#3b82f6")
    render_history_chart(accessibility_frame, "wallet_support_score", f"{asset_symbol} Wallet Support Trajectory", "Support Score", color="#ec4899")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#9ca3af;'>© 2026 BlockActivities.com — Multi-dimensional crypto intelligence for the modern digital asset economy.</p>", unsafe_allow_html=True)
'@
Set-Content -Path $path -Encoding utf8
