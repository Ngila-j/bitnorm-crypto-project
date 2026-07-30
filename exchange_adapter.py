"""
BitNorm Exchange Adapter (Phase 4)

Normalizes exchange ticker / order-book data for the terminal.
Default mode is `mock` so the app works without real API keys.
Set EXCHANGE_MODE=live and provide credentials to hit a real REST API.
"""

from __future__ import annotations

import os
import time
import random
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def get_mode() -> str:
    mode = _env("EXCHANGE_MODE", "mock").lower()
    return mode if mode in ("mock", "live") else "mock"


def get_base_url() -> str:
    return _env("EXCHANGE_BASE_URL", "https://api.your-exchange.com").rstrip("/")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_exchange_tables(db_path: str = "crypto_data.db") -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_tickers (
            symbol TEXT PRIMARY KEY,
            last_price REAL,
            volume_24h REAL,
            bid REAL,
            ask REAL,
            source TEXT,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS exchange_orderbook_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            size REAL NOT NULL,
            snapshot_ts TEXT NOT NULL
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_ob_symbol_ts ON exchange_orderbook_levels(symbol, snapshot_ts)"
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

_MOCK_BASE = {
    "BTC": 65000.0,
    "ETH": 3200.0,
    "SOL": 145.0,
    "ADA": 0.48,
}


def fetch_ticker_mock(symbol: str) -> Dict[str, Any]:
    base = _MOCK_BASE.get(symbol.upper(), 100.0)
    last = round(base * (1 + random.uniform(-0.004, 0.004)), 4)
    spread = last * 0.0002
    return {
        "symbol": symbol.upper(),
        "last_price": last,
        "volume_24h": round(base * random.uniform(8000, 25000), 2),
        "bid": round(last - spread, 4),
        "ask": round(last + spread, 4),
        "source": "mock",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_orderbook_mock(symbol: str, depth: int = 15) -> Dict[str, List[List[float]]]:
    base = _MOCK_BASE.get(symbol.upper(), 100.0)
    mid = base * (1 + random.uniform(-0.002, 0.002))
    bids, asks = [], []
    for i in range(depth):
        bids.append([round(mid * (1 - 0.0005 * (i + 1)), 4), round(random.uniform(0.1, 5.0), 4)])
        asks.append([round(mid * (1 + 0.0005 * (i + 1)), 4), round(random.uniform(0.1, 5.0), 4)])
    return {"bids": bids, "asks": asks, "source": "mock"}


def fetch_ticker_live(symbol: str) -> Dict[str, Any]:
    """
    Live REST ticker. Adjust path/auth to your exchange API.
    Expected normalized return matches fetch_ticker_mock.
    """
    if requests is None:
        raise RuntimeError("requests package is required for live exchange mode")

    api_key = _env("EXCHANGE_API_KEY")
    base = get_base_url()
    # Placeholder path — replace with real endpoint
    url = f"{base}/api/v1/ticker"
    headers = {"X-API-KEY": api_key} if api_key else {}
    params = {"symbol": f"{symbol.upper()}-USDT"}

    res = requests.get(url, headers=headers, params=params, timeout=8)
    res.raise_for_status()
    data = res.json()

    # Map common field names; adjust when real schema is known
    last = float(data.get("last") or data.get("price") or data.get("lastPrice") or 0)
    vol = float(data.get("volume") or data.get("volume24h") or data.get("quoteVolume") or 0)
    bid = float(data.get("bid") or data.get("bestBid") or last)
    ask = float(data.get("ask") or data.get("bestAsk") or last)

    return {
        "symbol": symbol.upper(),
        "last_price": last,
        "volume_24h": vol,
        "bid": bid,
        "ask": ask,
        "source": "live",
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_orderbook_live(symbol: str, depth: int = 15) -> Dict[str, List[List[float]]]:
    if requests is None:
        raise RuntimeError("requests package is required for live exchange mode")

    api_key = _env("EXCHANGE_API_KEY")
    base = get_base_url()
    url = f"{base}/api/v1/orderbook"
    headers = {"X-API-KEY": api_key} if api_key else {}
    params = {"symbol": f"{symbol.upper()}-USDT", "depth": depth}

    res = requests.get(url, headers=headers, params=params, timeout=8)
    res.raise_for_status()
    data = res.json()

    bids = data.get("bids") or data.get("buys") or []
    asks = data.get("asks") or data.get("sells") or []
    # Ensure [[price, size], ...]
    def _norm(levels):
        out = []
        for lv in levels[:depth]:
            if isinstance(lv, (list, tuple)) and len(lv) >= 2:
                out.append([float(lv[0]), float(lv[1])])
            elif isinstance(lv, dict):
                out.append([float(lv.get("price", 0)), float(lv.get("size", lv.get("qty", 0)))])
        return out

    return {"bids": _norm(bids), "asks": _norm(asks), "source": "live"}


def fetch_ticker(symbol: str) -> Dict[str, Any]:
    if get_mode() == "live" and _env("EXCHANGE_API_KEY"):
        try:
            return fetch_ticker_live(symbol)
        except Exception:
            return fetch_ticker_mock(symbol)
    return fetch_ticker_mock(symbol)


def fetch_orderbook(symbol: str, depth: int = 15) -> Dict[str, List[List[float]]]:
    if get_mode() == "live" and _env("EXCHANGE_API_KEY"):
        try:
            return fetch_orderbook_live(symbol, depth=depth)
        except Exception:
            return fetch_orderbook_mock(symbol, depth=depth)
    return fetch_orderbook_mock(symbol, depth=depth)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_ticker(row: Dict[str, Any], db_path: str = "crypto_data.db") -> None:
    init_exchange_tables(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT OR REPLACE INTO exchange_tickers
        (symbol, last_price, volume_24h, bid, ask, source, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["symbol"],
            row["last_price"],
            row["volume_24h"],
            row.get("bid"),
            row.get("ask"),
            row.get("source", "mock"),
            row.get("updated_at"),
        ),
    )
    conn.commit()
    conn.close()


def save_orderbook(
    symbol: str,
    book: Dict[str, List[List[float]]],
    db_path: str = "crypto_data.db",
) -> str:
    init_exchange_tables(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Keep only latest snapshot per symbol to limit growth
    cur.execute("DELETE FROM exchange_orderbook_levels WHERE symbol = ?", (symbol.upper(),))
    rows = []
    for price, size in book.get("bids", []):
        rows.append((symbol.upper(), "bid", price, size, ts))
    for price, size in book.get("asks", []):
        rows.append((symbol.upper(), "ask", price, size, ts))
    cur.executemany(
        """
        INSERT INTO exchange_orderbook_levels (symbol, side, price, size, snapshot_ts)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    conn.close()
    return ts


def get_ticker(symbol: str, db_path: str = "crypto_data.db") -> Optional[Dict[str, Any]]:
    init_exchange_tables(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM exchange_tickers WHERE symbol = ?", (symbol.upper(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_orderbook(symbol: str, db_path: str = "crypto_data.db") -> Dict[str, List[List[float]]]:
    init_exchange_tables(db_path)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT side, price, size FROM exchange_orderbook_levels
        WHERE symbol = ?
        ORDER BY
          CASE WHEN side = 'bid' THEN price END DESC,
          CASE WHEN side = 'ask' THEN price END ASC
        """,
        (symbol.upper(),),
    ).fetchall()
    conn.close()
    bids, asks = [], []
    for side, price, size in rows:
        if side == "bid":
            bids.append([price, size])
        else:
            asks.append([price, size])
    return {"bids": bids, "asks": asks}


def refresh_symbol(symbol: str, db_path: str = "crypto_data.db", depth: int = 15) -> Dict[str, Any]:
    """Fetch ticker + book and persist. Returns combined snapshot."""
    ticker = fetch_ticker(symbol)
    book = fetch_orderbook(symbol, depth=depth)
    upsert_ticker(ticker, db_path=db_path)
    ts = save_orderbook(symbol, book, db_path=db_path)
    return {"ticker": ticker, "orderbook": book, "snapshot_ts": ts}


def refresh_all(symbols: Optional[List[str]] = None, db_path: str = "crypto_data.db") -> List[Dict[str, Any]]:
    symbols = symbols or ["BTC", "ETH", "SOL", "ADA"]
    return [refresh_symbol(s, db_path=db_path) for s in symbols]


if __name__ == "__main__":
    print(f"Exchange mode: {get_mode()}")
    results = refresh_all()
    for r in results:
        t = r["ticker"]
        print(f"{t['symbol']}: last={t['last_price']} vol={t['volume_24h']} source={t['source']}")
    print("Exchange tables updated in crypto_data.db")
