"""
GitHub repository document → BNAnalytics sourcecode_metrics

Maps BitNorm indexation `GithubRepository` documents into the local
SQLite sourcecode_metrics table used by the Source Code pillar.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

DB_PATH_DEFAULT = "crypto_data.db"


def _parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        return text[:10] if len(text) >= 10 else None


def normalize_repo(raw: Dict[str, Any], asset_symbol: Optional[str] = None) -> Dict[str, Any]:
    owner = raw.get("owner") or ""
    name = raw.get("name") or ""
    repo_id = raw.get("_id") or f"{owner}/{name}".strip("/")

    contributors = raw.get("contributors") or []
    contributor_count = len(contributors) if isinstance(contributors, list) else int(contributors or 0)

    languages = raw.get("languages") or []
    lang_names = []
    if isinstance(languages, list):
        for lang in languages:
            if isinstance(lang, dict) and lang.get("name"):
                lang_names.append(lang["name"])
            elif isinstance(lang, str):
                lang_names.append(lang)

    symbol = (asset_symbol or raw.get("asset_symbol") or "").upper() or None
    if not symbol and name:
        symbol = "".join(ch for ch in name.upper() if ch.isalnum())[:6] or "UNK"

    metric_date = (
        _parse_date(raw.get("stat_updated_at"))
        or _parse_date(raw.get("pushed_at"))
        or _parse_date(raw.get("updated_at"))
        or datetime.now().strftime("%Y-%m-%d")
    )

    return {
        "asset_symbol": symbol,
        "metric_date": metric_date,
        "repo_id": str(repo_id),
        "owner": owner,
        "name": name,
        "url": raw.get("url") or (f"https://github.com/{owner}/{name}" if owner and name else ""),
        "commits_count": int(raw.get("commits_count") or 0),
        "contributor_count": contributor_count,
        "fork_count": int(raw.get("fork_count") or 0),
        "stargazers": int(raw.get("stargazers") or 0),
        "releases": int(raw.get("releases") or 0),
        "watchers": int(raw.get("watchers") or 0),
        "open_issues_count": int(raw.get("open_issues_count") or 0),
        "is_fork": bool(raw.get("is_fork") or False),
        "is_archived": bool(raw.get("is_archived") or False),
        "languages": lang_names,
        "pushed_at": _parse_date(raw.get("pushed_at")),
        "description": (raw.get("description") or "")[:300],
    }


def _repo_score(row: Dict[str, Any]) -> float:
    stars = float(row.get("stargazers") or 0)
    commits = float(row.get("commits_count") or 0)
    contribs = float(row.get("contributor_count") or 0)
    releases = float(row.get("releases") or 0)
    score = (
        25.0 * min(1.0, math.log10(stars + 1) / 5.0)
        + 25.0 * min(1.0, math.log10(commits + 1) / 5.0)
        + 25.0 * min(1.0, math.log10(contribs + 1) / 3.0)
        + 25.0 * min(1.0, math.log10(releases + 1) / 2.5)
    )
    if row.get("is_archived"):
        score *= 0.7
    if row.get("is_fork"):
        score *= 0.85
    return round(max(0.0, min(99.5, score)), 2)


def ensure_sourcecode_table(db_path: str = DB_PATH_DEFAULT) -> None:
    """sourcecode_metrics columns match pipeline.py."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sourcecode_metrics (
            asset_symbol TEXT NOT NULL,
            metric_date TEXT NOT NULL,
            commits INTEGER,
            active_devs INTEGER,
            repo_score REAL,
            PRIMARY KEY (asset_symbol, metric_date)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS github_repositories (
            repo_id TEXT PRIMARY KEY,
            asset_symbol TEXT,
            owner TEXT,
            name TEXT,
            url TEXT,
            commits_count INTEGER,
            contributor_count INTEGER,
            fork_count INTEGER,
            stargazers INTEGER,
            releases INTEGER,
            watchers INTEGER,
            open_issues_count INTEGER,
            is_fork INTEGER,
            is_archived INTEGER,
            languages TEXT,
            pushed_at TEXT,
            description TEXT,
            metric_date TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def upsert_github_repo(row: Dict[str, Any], db_path: str = DB_PATH_DEFAULT) -> None:
    ensure_sourcecode_table(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO github_repositories
        (repo_id, asset_symbol, owner, name, url, commits_count, contributor_count,
         fork_count, stargazers, releases, watchers, open_issues_count,
         is_fork, is_archived, languages, pushed_at, description, metric_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["repo_id"],
            row.get("asset_symbol"),
            row.get("owner"),
            row.get("name"),
            row.get("url"),
            row.get("commits_count"),
            row.get("contributor_count"),
            row.get("fork_count"),
            row.get("stargazers"),
            row.get("releases"),
            row.get("watchers"),
            row.get("open_issues_count"),
            1 if row.get("is_fork") else 0,
            1 if row.get("is_archived") else 0,
            json.dumps(row.get("languages") or []),
            row.get("pushed_at"),
            row.get("description"),
            row.get("metric_date"),
        ),
    )
    symbol = row.get("asset_symbol") or "UNK"
    metric_date = row.get("metric_date") or datetime.now().strftime("%Y-%m-%d")
    commits = int(row.get("commits_count") or 0)
    active_devs = int(row.get("contributor_count") or 0)
    repo_score = _repo_score(row)
    cur.execute(
        """
        INSERT OR REPLACE INTO sourcecode_metrics
        (asset_symbol, metric_date, commits, active_devs, repo_score)
        VALUES (?, ?, ?, ?, ?)
        """,
        (symbol, metric_date, commits, active_devs, repo_score),
    )
    conn.commit()
    conn.close()


def import_repos(
    repos: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH_DEFAULT,
    default_symbol: Optional[str] = None,
) -> int:
    count = 0
    for raw in repos:
        row = normalize_repo(raw, asset_symbol=raw.get("asset_symbol") or default_symbol)
        upsert_github_repo(row, db_path=db_path)
        count += 1
    return count


def import_repos_from_json_file(
    path: str,
    db_path: str = DB_PATH_DEFAULT,
    default_symbol: Optional[str] = None,
) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("repositories") or data.get("repos") or data.get("rows") or [data]
    return import_repos(data, db_path=db_path, default_symbol=default_symbol)


def _find_sample(filename: str) -> Optional[str]:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples", filename),
        os.path.join(os.getcwd(), "samples", filename),
        os.path.join(os.getcwd(), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def demo_import(db_path: str = DB_PATH_DEFAULT) -> int:
    sample_path = _find_sample("github_repositories.json")
    if sample_path:
        print(f"Loading samples from {sample_path}")
        return import_repos_from_json_file(sample_path, db_path=db_path)
    print("No samples/github_repositories.json found — skip import")
    return 0


if __name__ == "__main__":
    n = demo_import()
    print(f"Imported {n} GitHub repositories into sourcecode_metrics / github_repositories")
    conn = sqlite3.connect(DB_PATH_DEFAULT)
    cur = conn.cursor()
    cur.execute(
        "SELECT asset_symbol, metric_date, commits, active_devs, repo_score "
        "FROM sourcecode_metrics ORDER BY metric_date DESC LIMIT 10"
    )
    for r in cur.fetchall():
        print(r)
    conn.close()
