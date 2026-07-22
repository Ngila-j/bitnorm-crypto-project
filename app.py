import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
from analytics import compute_net_taker_flow, compute_user_sentiment_index

st.set_page_config(page_title="Bitnorm CryptoPulse Dashboard", layout="wide")

# Title and Overview
st.title("🚀 Bitnorm CryptoPulse: Market Intelligence & Customer Trading Analytics")
st.markdown("An end-to-end platform bridging live crypto market telemetry, machine learning trend forecasting, and customer trading behavior.")

# Load database connections and analytics data
@st.cache_data
def load_data():
    conn = sqlite3.connect("crypto_data.db")
    df_trades = pd.read_sql("SELECT * FROM customer_trades", conn)
    conn.close()
    df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
    return df_trades

df_trades = load_data()
flow_df = compute_net_taker_flow()
sentiment_df = compute_user_sentiment_index()

# Sidebar Filters
st.sidebar.header("Filter Analytics")
selected_asset = st.sidebar.selectbox("Select Asset Symbol", options=["ALL"] + list(flow_df['asset_symbol']))

# Executive Metrics Row
col1, col2, col3, col4 = st.columns(4)
total_volume = df_trades['trade_amount_usd'].sum()
total_trades = len(df_trades)
unique_users = df_trades['user_id'].nunique()
avg_trade = df_trades['trade_amount_usd'].mean()

col1.metric("Total Platform Trading Volume", f"${total_volume:,.2f}")
col2.metric("Total Customer Trades", f"{total_trades:,}")
col3.metric("Active Traders", f"{unique_users:,}")
col4.metric("Average Trade Size", f"${avg_trade:,.2f}")

st.markdown("---")

# Section 1: Customer Trading Activity Visualizations
st.header("📊 Customer Trading Activity & Platform Sentiment")

# Filter data if specific asset is selected
if selected_asset != "ALL":
    filtered_trades = df_trades[df_trades['asset_symbol'] == selected_asset]
    filtered_flow = flow_df[flow_df['asset_symbol'] == selected_asset]
else:
    filtered_trades = df_trades
    filtered_flow = flow_df

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Trading Volume by Asset (Buy vs Sell)")
    fig_volume = px.bar(
        flow_df, 
        x='asset_symbol', 
        y=['Buy', 'Sell'], 
        barmode='group',
        labels={'value': 'Volume (USD)', 'asset_symbol': 'Asset'},
        title="Total Customer Buy/Sell Volume per Asset"
    )
    st.plotly_chart(fig_volume, use_container_width=True)

with col_b:
    st.subheader("Customer Sentiment Index")
    fig_sentiment = px.bar(
        sentiment_df, 
        x='asset_symbol', 
        y='Sentiment_Index',
        color='Sentiment_Index',
        color_continuous_scale='RdYlGn',
        labels={'Sentiment_Index': 'Sentiment Index (-1 to +1)', 'asset_symbol': 'Asset'},
        title="Net Buying Sentiment Index per Asset"
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)

# Section 2: Time-Series Trend & Order Distribution
col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Cumulative Trade Volume Over Time")
    # Resample daily volume
    daily_volume = filtered_trades.set_index('timestamp').resample('D')['trade_amount_usd'].sum().reset_index()
    fig_time = px.line(
        daily_volume, 
        x='timestamp', 
        y='trade_amount_usd',
        labels={'trade_amount_usd': 'Daily Volume (USD)', 'timestamp': 'Date'},
        title="Platform Daily Trading Volume Trend"
    )
    st.plotly_chart(fig_time, use_container_width=True)

with col_d:
    st.subheader("Customer Trade Distribution by Asset")
    asset_counts = filtered_trades['asset_symbol'].value_counts().reset_index()
    asset_counts.columns = ['asset_symbol', 'count']
    fig_pie = px.pie(
        asset_counts, 
        names='asset_symbol', 
        values='count',
        hole=0.4,
        title="Proportion of Trades per Cryptocurrency"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")
st.markdown("### 🔗 Project Resources & Deployment")
st.markdown("- **GitHub Repository:** [Ngila-j/bitnorm-crypto-project](https://github.com/Ngila-j/bitnorm-crypto-project)")
st.markdown("- **Live Dashboard:** Hosted live on Streamlit Cloud.")