import sqlite3
import time
from datetime import datetime
import pandas as pd
import requests

# SQLite database setup (creates a local file automatically)
DB_NAME = "crypto_data.db"


def init_db():
  """Initializes the SQLite database and creates the target table."""
  conn = sqlite3.connect(DB_NAME)
  cursor = conn.cursor()
  cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id TEXT,
            symbol TEXT,
            name TEXT,
            current_price REAL,
            market_cap REAL,
            total_volume REAL,
            price_change_percentage_24h REAL,
            timestamp TEXT
        )
    """)
  conn.commit()
  conn.close()
  print("Database initialized successfully.")


def fetch_crypto_data():
  """Fetches top cryptocurrency market data from the free CoinGecko API."""
  url = "https://api.coingecko.com/api/v3/coins/markets"
  params = {
      "vs_currency": "usd",
      "order": "market_cap_desc",
      "per_page": 20,  # Top 20 cryptocurrencies
      "page": 1,
      "sparkline": "false",
  }

  try:
    response = requests.get(url, params=params)
    if response.status_code == 200:
      data = response.json()
      return data
    else:
      print(f"Error fetching data: Status Code {response.status_code}")
      return []
  except Exception as e:
    print(f"Network error: {e}")
    return []


def process_and_store(data):
  """Cleans data using Pandas and loads it into SQLite."""
  if not data:
    print("No data to store.")
    return

  rows = []
  current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

  for coin in data:
    rows.append({
        "id": coin.get("id"),
        "symbol": coin.get("symbol"),
        "name": coin.get("name"),
        "current_price": coin.get("current_price"),
        "market_cap": coin.get("market_cap"),
        "total_volume": coin.get("total_volume"),
        "price_change_percentage_24h": coin.get("price_change_percentage_24h"),
        "timestamp": current_time,
    })

  df = pd.DataFrame(rows)

  # Data Cleaning / Engineering check: drop any rows with missing prices
  df = df.dropna(subset=["current_price"])

  # Save to SQLite database
  conn = sqlite3.connect(DB_NAME)
  df.to_sql("market_data", conn, if_exists="append", index=False)
  conn.close()

  print(
      f"Successfully ingested and stored records for {len(df)} coins at"
      f" {current_time}"
  )


if __name__ == "__main__":
  init_db()
  print("Starting data ingestion job...")
  raw_data = fetch_crypto_data()
  process_and_store(raw_data)