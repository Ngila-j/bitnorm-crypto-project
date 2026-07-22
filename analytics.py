import sqlite3
import pandas as pd

def load_customer_trades(db_path="crypto_data.db"):
    """Loads customer trades from SQLite into a Pandas DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM customer_trades", conn)
    conn.close()
    
    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def compute_net_taker_flow(db_path="crypto_data.db"):
    """
    Computes Net Taker Flow: Ratio of Buy Volume vs Sell Volume 
    aggregated by asset symbol.
    """
    df = load_customer_trades(db_path)
    
    # Group by asset and order type to sum trade amounts
    flow = df.groupby(['asset_symbol', 'order_type'])['trade_amount_usd'].sum().unstack(fill_value=0)
    
    if 'Buy' not in flow.columns:
        flow['Buy'] = 0.0
    if 'Sell' not in flow.columns:
        flow['Sell'] = 0.0
        
    # Calculate Net Flow and Ratio
    flow['Net_Flow'] = flow['Buy'] - flow['Sell']
    flow['Buy_Sell_Ratio'] = flow['Buy'] / (flow['Sell'] + 1e-5) # Prevent division by zero
    
    return flow.reset_index()

def compute_user_sentiment_index(db_path="crypto_data.db"):
    """
    Computes a User Sentiment Index (-1 to +1 scale) based on 
    net buying pressure relative to total trading volume per asset.
    """
    flow = compute_net_taker_flow(db_path)
    total_volume = flow['Buy'] + flow['Sell']
    
    # Sentiment Index: (Buy - Sell) / (Buy + Sell)
    flow['Sentiment_Index'] = (flow['Buy'] - flow['Sell']) / (total_volume + 1e-5)
    
    return flow[['asset_symbol', 'Buy', 'Sell', 'Sentiment_Index']]

if __name__ == "__main__":
    print("--- Net Taker Flow Summary ---")
    print(compute_net_taker_flow())
    print("\n--- User Sentiment Index ---")
    print(compute_user_sentiment_index())