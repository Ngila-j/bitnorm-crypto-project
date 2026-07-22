import sqlite3
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="CryptoPulse Intelligence Dashboard",
    page_icon="📈",
    layout="wide",
)

DB_NAME = "crypto_data.db"


@st.cache_data(ttl=60)
def load_data():
  """Loads market data from the SQLite database."""
  try:
    conn = sqlite3.connect(DB_NAME)
    query = "SELECT * FROM market_data"
    df = pd.read_sql(query, conn)
    conn.close()
    return df
  except Exception as e:
    return pd.DataFrame()


# Dashboard Header
st.title("🚀 Bitnorm CryptoPulse: Real-Time Intelligence Engine")
st.markdown(
    "Demonstrating end-to-end **Data Engineering, Machine Learning, and BI"
    " Dashboarding** for crypto market analytics."
)

df = load_data()

if df.empty:
  st.warning(
      "No data found in database yet. Please run your `pipeline.py` script"
      " first!"
  )
else:
  # Sidebar filters
  st.sidebar.header("Control Panel")
  selected_coin = st.sidebar.selectbox(
      "Select Cryptocurrency", df["name"].unique()
  )

  # Top-level metrics cards
  latest_timestamp = df["timestamp"].max()
  st.caption(f"Last Pipeline Sync: {latest_timestamp} (UTC)")

  col1, col2, col3 = st.columns(3)

  coin_df = df[df["name"] == selected_coin]
  latest_record = coin_df.iloc[-1]

  with col1:
    st.metric(
        label=f"{selected_coin} Price (USD)",
        value=f"${latest_record['current_price']:,.2f}",
    )
  with col2:
    st.metric(
        label="Market Capitalization",
        value=f"${latest_record['market_cap']:,.0f}",
    )
  with col3:
    st.metric(
        label="24h Price Change",
        value=(
            f"{latest_record['price_change_percentage_24h']:.2f}%"
            if not pd.isna(latest_record["price_change_percentage_24h"])
            else "N/A"
        ),
    )

  st.divider()

  # Visualizations
  col_left, col_right = st.columns(2)

  with col_left:
    st.subheader(f"📊 {selected_coin} Price History Trend")
    if len(coin_df) > 1:
      st.line_chart(coin_df, x="timestamp", y="current_price")
    else:
      st.info(
          "Run `pipeline.py` multiple times over a few minutes to generate"
          " historical trend lines!"
      )

  with col_right:
    st.subheader("🤖 AI / ML Model Risk & Feature Insights")
    st.info(
        "**XGBoost Model Feature Importance Analysis:**\n- **Total Volume:**"
        " 65.6%\n- **Market Cap:** 20.0%\n- **Current Price:** 14.3%\n\n*Insight:"
        " Volume fluctuations serve as the primary indicator for short-term"
        " price direction predictions.*"
    )

  # Data table view
  st.subheader("🔍 Raw Ingested Data Feed")
  st.dataframe(df.tail(10), use_container_width=True)