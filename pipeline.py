import sqlite3
import random
from datetime import datetime, timedelta


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

    cursor.execute("SELECT COUNT(*) FROM customer_trades")
    count = cursor.fetchone()[0]
    if count >= num_records:
        print(f"Table already contains {count} records. Skipping generation.")
        conn.close()
        return

    assets = ["BTC", "ETH", "SOL", "ADA"]
    order_types = ["Buy", "Sell"]
    base_prices = {"BTC": 65000.0, "ETH": 35000.0, "SOL": 180.0, "ADA": 0.45}

    start_time = datetime.now() - timedelta(days=30)
    trades_data = []

    print(f"Generating {num_records} simulated customer trades...")
    for i in range(1, num_records + 1):
        trade_id = f"TRD-{i:05d}"
        delta_minutes = random.randint(0, 30 * 24 * 60)
        timestamp = (start_time + timedelta(minutes=delta_minutes)).strftime("%Y-%m-%d %H:%M:%S")

        user_id = f"USR-{random.randint(100, 999)}"
        asset = random.choice(assets)
        order_type = random.choices(order_types, weights=[0.55, 0.45])[0]

        price_fluctuation = random.uniform(-0.03, 0.03)
        execution_price = base_prices[asset] * (1 + price_fluctuation)
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


def init_crypto_pillar_tables(db_path="crypto_data.db"):
    """Creates all crypto pillar tables if they do not already exist."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sourcecode_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            commits INTEGER,
            active_devs INTEGER,
            repo_score REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        );

        CREATE TABLE IF NOT EXISTS network_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            active_addresses INTEGER,
            tx_tps REAL,
            gas_fee_gwei REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        );

        CREATE TABLE IF NOT EXISTS economics_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            market_cap REAL,
            volume_24h REAL,
            tokenomics_score REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        );

        CREATE TABLE IF NOT EXISTS sentiment_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            user_sentiment_index REAL,
            buy_sell_ratio REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        );

        CREATE TABLE IF NOT EXISTS accessibility_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            exchange_count INTEGER,
            wallet_support_score REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        );
    """)

    conn.commit()
    conn.close()


def generate_sourcecode_metrics(days=30, db_path="crypto_data.db"):
    """Simulates source-code health signals for each crypto asset."""
    init_crypto_pillar_tables(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assets = {
        "BTC": {"base_commits": 9500, "dev_base": 700, "repo_base": 94.5, "commit_growth": 80},
        "ETH": {"base_commits": 8800, "dev_base": 640, "repo_base": 93.8, "commit_growth": 70},
        "SOL": {"base_commits": 7200, "dev_base": 520, "repo_base": 91.2, "commit_growth": 95},
        "ADA": {"base_commits": 6100, "dev_base": 410, "repo_base": 90.4, "commit_growth": 60},
    }

    start_date = datetime.now().date() - timedelta(days=days - 1)
    rows = []
    for asset_symbol, profile in assets.items():
        for idx in range(days):
            metric_date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            commits = max(3000, int(profile["base_commits"] + profile["commit_growth"] * idx + random.randint(-120, 140)))
            active_devs = max(180, int(profile["dev_base"] + (idx // 3) + random.randint(-20, 25)))
            repo_score = round(min(99.5, max(84.0, profile["repo_base"] + (idx * 0.01) + random.uniform(-0.25, 0.25))), 2)
            rows.append((asset_symbol, metric_date, commits, active_devs, repo_score))

    cursor.executemany("""
        INSERT OR REPLACE INTO sourcecode_metrics
        (asset_symbol, metric_date, commits, active_devs, repo_score)
        VALUES (?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return rows


def generate_network_metrics(days=30, db_path="crypto_data.db"):
    """Simulates blockchain network activity signals."""
    init_crypto_pillar_tables(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assets = {
        "BTC": {"addresses": 850000, "tps": 5.2, "gas": 18.5},
        "ETH": {"addresses": 620000, "tps": 24.0, "gas": 34.0},
        "SOL": {"addresses": 410000, "tps": 1850.0, "gas": 1.8},
        "ADA": {"addresses": 280000, "tps": 115.0, "gas": 0.6},
    }

    start_date = datetime.now().date() - timedelta(days=days - 1)
    rows = []
    for asset_symbol, profile in assets.items():
        for idx in range(days):
            metric_date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            active_addresses = int(profile["addresses"] + idx * 3500 + random.randint(-18000, 18000))
            tx_tps = round(max(1.0, profile["tps"] + idx * 0.15 + random.uniform(-0.4, 0.4)), 2)
            gas_fee_gwei = round(max(0.1, profile["gas"] + idx * 0.04 + random.uniform(-0.8, 0.8)), 2)
            rows.append((asset_symbol, metric_date, active_addresses, tx_tps, gas_fee_gwei))

    cursor.executemany("""
        INSERT OR REPLACE INTO network_metrics
        (asset_symbol, metric_date, active_addresses, tx_tps, gas_fee_gwei)
        VALUES (?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return rows


def generate_economics_metrics(days=30, db_path="crypto_data.db"):
    """Simulates macroeconomic and market activity signals."""
    init_crypto_pillar_tables(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assets = {
        "BTC": {"market_cap": 1.24e12, "volume": 3.2e10, "tokenomics": 95.0},
        "ETH": {"market_cap": 4.4e11, "volume": 1.6e10, "tokenomics": 93.2},
        "SOL": {"market_cap": 7.4e10, "volume": 2.8e9, "tokenomics": 90.8},
        "ADA": {"market_cap": 2.5e10, "volume": 7.8e8, "tokenomics": 88.5},
    }

    start_date = datetime.now().date() - timedelta(days=days - 1)
    rows = []
    for asset_symbol, profile in assets.items():
        for idx in range(days):
            metric_date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            market_cap = round(profile["market_cap"] * (1 + (idx * 0.002) + random.uniform(-0.03, 0.03)), 2)
            volume_24h = round(profile["volume"] * (1 + (idx * 0.001) + random.uniform(-0.05, 0.05)), 2)
            tokenomics_score = round(min(99.0, max(82.0, profile["tokenomics"] + random.uniform(-0.4, 0.4))), 2)
            rows.append((asset_symbol, metric_date, market_cap, volume_24h, tokenomics_score))

    cursor.executemany("""
        INSERT OR REPLACE INTO economics_metrics
        (asset_symbol, metric_date, market_cap, volume_24h, tokenomics_score)
        VALUES (?, ?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return rows


def generate_sentiment_metrics(days=30, db_path="crypto_data.db"):
    """Simulates market sentiment indicators for each asset."""
    init_crypto_pillar_tables(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assets = {
        "BTC": {"sentiment": 0.14, "ratio": 1.18},
        "ETH": {"sentiment": 0.08, "ratio": 1.11},
        "SOL": {"sentiment": 0.22, "ratio": 1.27},
        "ADA": {"sentiment": -0.05, "ratio": 0.95},
    }

    start_date = datetime.now().date() - timedelta(days=days - 1)
    rows = []
    for asset_symbol, profile in assets.items():
        for idx in range(days):
            metric_date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            user_sentiment_index = round(max(-0.9, min(0.9, profile["sentiment"] + (idx * 0.002) + random.uniform(-0.08, 0.08))), 4)
            buy_sell_ratio = round(max(0.7, min(1.6, profile["ratio"] + random.uniform(-0.08, 0.08))), 4)
            rows.append((asset_symbol, metric_date, user_sentiment_index, buy_sell_ratio))

    cursor.executemany("""
        INSERT OR REPLACE INTO sentiment_metrics
        (asset_symbol, metric_date, user_sentiment_index, buy_sell_ratio)
        VALUES (?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return rows


def generate_accessibility_metrics(days=30, db_path="crypto_data.db"):
    """Simulates exchange and wallet support availability."""
    init_crypto_pillar_tables(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    assets = {
        "BTC": {"exchanges": 240, "wallet": 0.96},
        "ETH": {"exchanges": 210, "wallet": 0.94},
        "SOL": {"exchanges": 180, "wallet": 0.91},
        "ADA": {"exchanges": 155, "wallet": 0.89},
    }

    start_date = datetime.now().date() - timedelta(days=days - 1)
    rows = []
    for asset_symbol, profile in assets.items():
        for idx in range(days):
            metric_date = (start_date + timedelta(days=idx)).strftime("%Y-%m-%d")
            exchange_count = int(profile["exchanges"] + idx + random.randint(-5, 6))
            wallet_support_score = round(max(0.75, min(0.99, profile["wallet"] + random.uniform(-0.02, 0.01))), 4)
            rows.append((asset_symbol, metric_date, exchange_count, wallet_support_score))

    cursor.executemany("""
        INSERT OR REPLACE INTO accessibility_metrics
        (asset_symbol, metric_date, exchange_count, wallet_support_score)
        VALUES (?, ?, ?, ?)
    """, rows)

    conn.commit()
    conn.close()
    return rows


def generate_all_crypto_metrics(days=30, db_path="crypto_data.db"):
    """Creates the pillar tables and populates them with simulated crypto metrics."""
    init_crypto_pillar_tables(db_path)
    generate_sourcecode_metrics(days=days, db_path=db_path)
    generate_network_metrics(days=days, db_path=db_path)
    generate_economics_metrics(days=days, db_path=db_path)
    generate_sentiment_metrics(days=days, db_path=db_path)
    generate_accessibility_metrics(days=days, db_path=db_path)
    print(f"Populated crypto pillar tables in {db_path} for {days} days of simulated data.")


if __name__ == "__main__":
    generate_simulated_trades()
    generate_all_crypto_metrics()