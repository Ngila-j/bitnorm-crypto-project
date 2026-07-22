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


def _normalize_score(value, lower, upper, invert=False):
    """Normalizes a value into a 0-100 score."""
    if upper <= lower:
        return 50.0
    raw = (value - lower) / (upper - lower) * 100.0
    if invert:
        raw = 100.0 - raw
    return round(max(0.0, min(100.0, raw)), 2)


def _get_latest_row(conn, table_name, asset_symbol):
    """Fetches the latest record for an asset from a given table."""
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT * FROM {table_name} WHERE asset_symbol = ? ORDER BY metric_date DESC LIMIT 1",
        (asset_symbol.upper(),),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [col[0] for col in cursor.description]
    return dict(zip(columns, row))


def _get_metric_range(conn, table_name, metric_name):
    """Returns the observed min/max range for a metric in a table."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT MIN({metric_name}), MAX({metric_name}) FROM {table_name}")
    minimum, maximum = cursor.fetchone()
    return minimum if minimum is not None else 0.0, maximum if maximum is not None else 1.0


def fetch_latest_crypto_metrics(asset_symbol, db_path="crypto_data.db"):
    """Fetches the latest pillar metrics for a given asset symbol from SQLite."""
    asset_symbol = asset_symbol.upper()
    conn = sqlite3.connect(db_path)
    try:
        snapshot = {
            "asset_symbol": asset_symbol,
            "sourcecode": _get_latest_row(conn, "sourcecode_metrics", asset_symbol),
            "network": _get_latest_row(conn, "network_metrics", asset_symbol),
            "economics": _get_latest_row(conn, "economics_metrics", asset_symbol),
            "sentiment": _get_latest_row(conn, "sentiment_metrics", asset_symbol),
            "accessibility": _get_latest_row(conn, "accessibility_metrics", asset_symbol),
        }
    finally:
        conn.close()
    return snapshot


def compute_blockactivities_health_score(asset_symbol, db_path="crypto_data.db"):
    """Computes normalized pillar scores and a weighted composite health score for an asset."""
    asset_symbol = asset_symbol.upper()
    conn = sqlite3.connect(db_path)
    try:
        source_row = _get_latest_row(conn, "sourcecode_metrics", asset_symbol) or {}
        network_row = _get_latest_row(conn, "network_metrics", asset_symbol) or {}
        economics_row = _get_latest_row(conn, "economics_metrics", asset_symbol) or {}
        sentiment_row = _get_latest_row(conn, "sentiment_metrics", asset_symbol) or {}
        accessibility_row = _get_latest_row(conn, "accessibility_metrics", asset_symbol) or {}

        source_commit_min, source_commit_max = _get_metric_range(conn, "sourcecode_metrics", "commits")
        source_dev_min, source_dev_max = _get_metric_range(conn, "sourcecode_metrics", "active_devs")
        source_repo_min, source_repo_max = _get_metric_range(conn, "sourcecode_metrics", "repo_score")

        network_addr_min, network_addr_max = _get_metric_range(conn, "network_metrics", "active_addresses")
        network_tps_min, network_tps_max = _get_metric_range(conn, "network_metrics", "tx_tps")
        network_gas_min, network_gas_max = _get_metric_range(conn, "network_metrics", "gas_fee_gwei")

        economics_mcap_min, economics_mcap_max = _get_metric_range(conn, "economics_metrics", "market_cap")
        economics_vol_min, economics_vol_max = _get_metric_range(conn, "economics_metrics", "volume_24h")
        economics_token_min, economics_token_max = _get_metric_range(conn, "economics_metrics", "tokenomics_score")

        sentiment_sent_min, sentiment_sent_max = _get_metric_range(conn, "sentiment_metrics", "user_sentiment_index")
        sentiment_ratio_min, sentiment_ratio_max = _get_metric_range(conn, "sentiment_metrics", "buy_sell_ratio")

        accessibility_ex_min, accessibility_ex_max = _get_metric_range(conn, "accessibility_metrics", "exchange_count")
        accessibility_wallet_min, accessibility_wallet_max = _get_metric_range(conn, "accessibility_metrics", "wallet_support_score")

        source_score = round(
            0.4 * _normalize_score(source_row.get("repo_score", 0), source_repo_min, source_repo_max)
            + 0.3 * _normalize_score(source_row.get("active_devs", 0), source_dev_min, source_dev_max)
            + 0.3 * _normalize_score(source_row.get("commits", 0), source_commit_min, source_commit_max),
            2,
        )

        network_score = round(
            0.4 * _normalize_score(network_row.get("active_addresses", 0), network_addr_min, network_addr_max)
            + 0.35 * _normalize_score(network_row.get("tx_tps", 0), network_tps_min, network_tps_max)
            + 0.25 * _normalize_score(network_row.get("gas_fee_gwei", 0), network_gas_min, network_gas_max, invert=True),
            2,
        )

        economics_score = round(
            0.4 * _normalize_score(economics_row.get("market_cap", 0), economics_mcap_min, economics_mcap_max)
            + 0.35 * _normalize_score(economics_row.get("volume_24h", 0), economics_vol_min, economics_vol_max)
            + 0.25 * _normalize_score(economics_row.get("tokenomics_score", 0), economics_token_min, economics_token_max),
            2,
        )

        sentiment_score = round(
            0.6 * _normalize_score(sentiment_row.get("user_sentiment_index", 0), sentiment_sent_min, sentiment_sent_max)
            + 0.4 * _normalize_score(sentiment_row.get("buy_sell_ratio", 0), sentiment_ratio_min, sentiment_ratio_max),
            2,
        )

        accessibility_score = round(
            0.55 * _normalize_score(accessibility_row.get("exchange_count", 0), accessibility_ex_min, accessibility_ex_max)
            + 0.45 * _normalize_score(accessibility_row.get("wallet_support_score", 0), accessibility_wallet_min, accessibility_wallet_max),
            2,
        )

        pillar_scores = {
            "sourcecode": source_score,
            "network": network_score,
            "economics": economics_score,
            "sentiment": sentiment_score,
            "accessibility": accessibility_score,
        }

        composite_score = round(
            0.25 * pillar_scores["sourcecode"]
            + 0.2 * pillar_scores["network"]
            + 0.2 * pillar_scores["economics"]
            + 0.15 * pillar_scores["sentiment"]
            + 0.2 * pillar_scores["accessibility"],
            2,
        )

        return {
            "asset_symbol": asset_symbol,
            "latest_metrics": {
                "sourcecode": source_row,
                "network": network_row,
                "economics": economics_row,
                "sentiment": sentiment_row,
                "accessibility": accessibility_row,
            },
            "pillar_scores": pillar_scores,
            "health_score": composite_score,
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print("--- Net Taker Flow Summary ---")
    print(compute_net_taker_flow())
    print("\n--- User Sentiment Index ---")
    print(compute_user_sentiment_index())
    print("\n--- BlockActivities Health Snapshot ---")
    for symbol in ["BTC", "ETH", "SOL", "ADA"]:
        print(symbol, compute_blockactivities_health_score(symbol))