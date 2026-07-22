import sqlite3
import random
from datetime import datetime, timedelta
import pandas as pd

def init_customer_trades_table(db_path="crypto_data.db"):
    """Creates the customer_trades table if it doesn't already exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_trades (
            trade_id TEXT PRIMARY KEY,
            timestamp TEXT,
            user_id TEXT,
            asset_symbol TEXT,
            order_type TEXT,
            trade_amount_usd REAL,
            execution_price REAL
        )
    """)
    
    conn.commit()
    conn.close()

def generate_simulated_trades(num_records=5000, db_path="crypto_data.db"):
    """Generates 5,000+ simulated customer trade logs and saves them to SQLite."""
    init_customer_trades_table(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if trades already exist to avoid duplicate inflation
    cursor.execute("SELECT COUNT(*) FROM customer_trades")
    count = cursor.fetchone()[0]
    if count >= num_records:
        print(f"Table already contains {count} records. Skipping generation.")
        conn.close()
        return

    assets = ["BTC", "ETH", "SOL", "BNB", "ADA"]
    order_types = ["Buy", "Sell"]
    base_prices = {"BTC": 65000.0, "ETH": 35000.0, "SOL": 180.0, "BNB": 580.0, "ADA": 0.45}
    
    start_time = datetime.now() - timedelta(days=30)
    trades_data = []
    
    print(f"Generating {num_records} simulated customer trades...")
    for i in range(1, num_records + 1):
        trade_id = f"TRD-{i:05d}"
        # Random timestamp within the last 30 days
        delta_minutes = random.randint(0, 30 * 24 * 60)
        timestamp = (start_time + timedelta(minutes=delta_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        
        user_id = f"USR-{random.randint(100, 999)}"
        asset = random.choice(assets)
        order_type = random.choices(order_types, weights=[0.55, 0.45])[0] # slight buy bias
        
        # Fluctuate execution price slightly around base price
        price_fluctuation = random.uniform(-0.03, 0.03)
        execution_price = base_prices[asset] * (1 + price_fluctuation)
        
        # Trade amount between $50 and $15,000
        trade_amount_usd = round(random.uniform(50.0, 15000.0), 2)
        
        trades_data.append((trade_id, timestamp, user_id, asset, order_type, trade_amount_usd, execution_price))
        
    cursor.executemany("""
        INSERT OR REPLACE INTO customer_trades 
        (trade_id, timestamp, user_id, asset_symbol, order_type, trade_amount_usd, execution_price)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, trades_data)
    
    conn.commit()
    conn.close()
    print("Successfully ingested 5,000+ customer trade records into crypto_data.db!")

if __name__ == "__main__":
    generate_simulated_trades()