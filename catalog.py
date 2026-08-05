"""
BitNorm Project Catalog

Simulated announcement / project directory ready to be replaced by
BitcoinTalk Altcoin Announcements scraper output later.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

# Seed catalog — stands in for BitcoinTalk Altcoin Announcements
_SEED_PROJECTS = [
    {
        "project_name": "Bitcoin",
        "symbol": "BTC",
        "category": "Layer-1",
        "source": "core",
        "announcement_url": "https://bitcointalk.org/",
        "summary": "Peer-to-peer electronic cash and settlement network.",
        "status": "Tracked",
        "days_ago": 4000,
    },
    {
        "project_name": "Ethereum",
        "symbol": "ETH",
        "category": "Smart Contracts",
        "source": "core",
        "announcement_url": "https://bitcointalk.org/",
        "summary": "General-purpose smart contract and settlement layer.",
        "status": "Tracked",
        "days_ago": 3500,
    },
    {
        "project_name": "Solana",
        "symbol": "SOL",
        "category": "Infrastructure",
        "source": "core",
        "announcement_url": "https://bitcointalk.org/",
        "summary": "High-throughput L1 optimized for consumer applications.",
        "status": "Tracked",
        "days_ago": 1800,
    },
    {
        "project_name": "Cardano",
        "symbol": "ADA",
        "category": "Layer-1",
        "source": "core",
        "announcement_url": "https://bitcointalk.org/",
        "summary": "Research-driven proof-of-stake smart contract platform.",
        "status": "Tracked",
        "days_ago": 2800,
    },
    {
        "project_name": "NovaMesh Protocol",
        "symbol": "NMP",
        "category": "Infrastructure",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "Modular data-availability network for rollups (simulated announcement).",
        "status": "Catalog Only",
        "days_ago": 12,
    },
    {
        "project_name": "AetherLend",
        "symbol": "AETH",
        "category": "DeFi",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "Cross-chain money market with isolated risk pools (simulated).",
        "status": "Catalog Only",
        "days_ago": 8,
    },
    {
        "project_name": "CipherVault",
        "symbol": "CVT",
        "category": "Infrastructure",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "Institutional custody and policy-engine toolkit (simulated).",
        "status": "Catalog Only",
        "days_ago": 5,
    },
    {
        "project_name": "OrbitDEX",
        "symbol": "ORBX",
        "category": "DeFi",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "Intent-based DEX aggregator with MEV-aware routing (simulated).",
        "status": "Catalog Only",
        "days_ago": 3,
    },
    {
        "project_name": "Helios Identity",
        "symbol": "HLS",
        "category": "Identity",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "On-chain credentials and compliance attestations (simulated).",
        "status": "Catalog Only",
        "days_ago": 2,
    },
    {
        "project_name": "PulseOracle",
        "symbol": "PLS",
        "category": "Infrastructure",
        "source": "bitcointalk_sim",
        "announcement_url": "https://bitcointalk.org/index.php?board=159.0",
        "summary": "Low-latency price and event oracles for perps (simulated).",
        "status": "Catalog Only",
        "days_ago": 1,
    },
]


def init_catalog_table(db_path: str = "crypto_data.db") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS catalog_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_name TEXT NOT NULL,
            symbol TEXT,
            category TEXT,
            source TEXT,
            announcement_url TEXT,
            summary TEXT,
            status TEXT,
            announced_at TEXT,
            replies INTEGER DEFAULT 0,
            views INTEGER DEFAULT 0,
            github_repos TEXT,
            topic_type TEXT,
            UNIQUE(project_name, source)
        )
        """
    )
    # Migrate older DBs missing engagement / link columns
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(catalog_projects)")
    cols = {row[1] for row in cur.fetchall()}
    for col, decl in [
        ("replies", "INTEGER DEFAULT 0"),
        ("views", "INTEGER DEFAULT 0"),
        ("github_repos", "TEXT"),
        ("topic_type", "TEXT"),
    ]:
        if col not in cols:
            try:
                cur.execute(f"ALTER TABLE catalog_projects ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass
    conn.commit()
    conn.close()


def seed_catalog_projects(db_path: str = "crypto_data.db", force: bool = False) -> int:
    """Populate catalog with simulated projects. Returns row count."""
    init_catalog_table(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM catalog_projects")
    count = cur.fetchone()[0]
    if count > 0 and not force:
        conn.close()
        return count

    if force:
        cur.execute("DELETE FROM catalog_projects")

    rows = []
    for p in _SEED_PROJECTS:
        announced = (datetime.now() - timedelta(days=p["days_ago"])).strftime("%Y-%m-%d")
        rows.append(
            (
                p["project_name"],
                p["symbol"],
                p["category"],
                p["source"],
                p["announcement_url"],
                p["summary"],
                p["status"],
                announced,
            )
        )
    cur.executemany(
        """
        INSERT OR REPLACE INTO catalog_projects
        (project_name, symbol, category, source, announcement_url, summary, status, announced_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM catalog_projects").fetchone()[0]
    conn.close()
    return n


def load_catalog(db_path: str = "crypto_data.db") -> pd.DataFrame:
    init_catalog_table(db_path)
    seed_catalog_projects(db_path=db_path, force=False)
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        """
        SELECT id, project_name, symbol, category, source, announcement_url,
               summary, status, announced_at,
               COALESCE(replies, 0) AS replies,
               COALESCE(views, 0) AS views,
               github_repos,
               topic_type
        FROM catalog_projects
        ORDER BY announced_at DESC
        """,
        conn,
    )
    conn.close()
    if df.empty:
        return df
    df["replies"] = pd.to_numeric(df["replies"], errors="coerce").fillna(0).astype(int)
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    # Engagement score: views + 5 * replies (indexation-style attention proxy)
    df["engagement"] = df["views"] + 5 * df["replies"]
    tracked = {"BTC", "ETH", "SOL", "ADA"}
    df["tracked"] = df["symbol"].fillna("").astype(str).str.upper().isin(tracked)
    return df


def catalog_announcement_velocity(db_path: str = "crypto_data.db") -> pd.DataFrame:
    """Daily announcement counts for velocity chart."""
    df = load_catalog(db_path)
    if df.empty or "announced_at" not in df.columns:
        return pd.DataFrame(columns=["date", "count"])
    tmp = df.copy()
    tmp["date"] = tmp["announced_at"].astype(str).str[:10]
    out = tmp.groupby("date").size().reset_index(name="count").sort_values("date")
    return out


def search_catalog(query: str, db_path: str = "crypto_data.db") -> pd.DataFrame:
    df = load_catalog(db_path)
    if df.empty or not query:
        return df
    q = query.lower().strip()
    mask = (
        df["project_name"].str.lower().str.contains(q, na=False)
        | df["symbol"].str.lower().str.contains(q, na=False)
        | df["category"].str.lower().str.contains(q, na=False)
        | df["summary"].str.lower().str.contains(q, na=False)
        | df["source"].str.lower().str.contains(q, na=False)
    )
    return df[mask]


def catalog_categories(db_path: str = "crypto_data.db") -> List[str]:
    df = load_catalog(db_path)
    if df.empty:
        return []
    return sorted(df["category"].dropna().unique().tolist())


def catalog_as_records(db_path: str = "crypto_data.db") -> List[Dict[str, Any]]:
    df = load_catalog(db_path)
    return df.to_dict(orient="records")


if __name__ == "__main__":
    n = seed_catalog_projects(force=True)
    print(f"Seeded {n} catalog projects")
    print(load_catalog().head(10).to_string(index=False))
