"""
BitcoinTalk topic → BNAnalytics catalog_projects

Maps BitNorm indexation `BitcointalkTopic` documents (and scraper AMQP
payloads) into the local SQLite catalog used by Project Explorer.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from catalog import init_catalog_table, load_catalog

DB_PATH_DEFAULT = "crypto_data.db"


def topic_url(topic_id: Any) -> str:
    return f"https://bitcointalk.org/index.php?topic={topic_id}.0"


def _parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    if not text:
        return None
    # ISO / Mongo-style
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(text.replace("Z", ""), fmt.replace("Z", "")).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except Exception:
        return text[:10] if len(text) >= 10 else None


def normalize_topic(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept scraper AMQP payload or Mongo BitcointalkTopic-shaped dict.
    """
    topic_id = raw.get("_id") or raw.get("id") or raw.get("topic_id")
    title = (raw.get("title") or "").strip() or f"Topic {topic_id}"
    topic_type = (raw.get("topicType") or raw.get("type") or "").lower()
    if topic_type not in ("ann", "ico"):
        # scraper uses type; indexation uses topicType
        t = (raw.get("type") or "").lower()
        topic_type = t if t in ("ann", "ico") else "ann"

    category = "ICO" if topic_type == "ico" else "Announcements"
    announced = _parse_date(raw.get("announcementDate") or raw.get("lastPostDate"))
    if not announced:
        announced = datetime.now().strftime("%Y-%m-%d")

    gh = raw.get("github_repositories") or raw.get("githubLinks") or []
    # scraper: list of [owner, name]; indexation: list of strings
    gh_norm: List[str] = []
    for item in gh:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            gh_norm.append(f"{item[0]}/{item[1]}")
        elif isinstance(item, str) and item.strip():
            gh_norm.append(item.strip())

    summary_parts = []
    if raw.get("username"):
        summary_parts.append(f"Posted by {raw['username']}")
    if raw.get("replies") is not None:
        summary_parts.append(f"{raw.get('replies', 0)} replies")
    if raw.get("views") is not None:
        summary_parts.append(f"{raw.get('views', 0)} views")
    if gh_norm:
        summary_parts.append("GitHub: " + ", ".join(gh_norm[:3]))
    summary = " · ".join(summary_parts) if summary_parts else title

    # Optional symbol guess from title tokens (best-effort)
    symbol = None
    for token in title.replace("[", " ").replace("]", " ").split():
        if token.isupper() and 2 <= len(token) <= 6 and token.isalpha():
            symbol = token
            break

    return {
        "project_name": title[:200],
        "symbol": symbol,
        "category": category,
        "source": "bitcointalk",
        "announcement_url": topic_url(topic_id) if topic_id is not None else raw.get("url") or "",
        "summary": summary[:500],
        "status": "Catalog Only",
        "announced_at": announced,
        "external_id": str(topic_id) if topic_id is not None else None,
        "github_repos": gh_norm,
        "topic_type": topic_type,
        "username": raw.get("username"),
        "replies": raw.get("replies"),
        "views": raw.get("views"),
    }


def upsert_catalog_row(row: Dict[str, Any], db_path: str = DB_PATH_DEFAULT) -> None:
    init_catalog_table(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Prefer match on announcement_url when present
    url = row.get("announcement_url") or ""
    name = row.get("project_name") or ""
    cur.execute(
        """
        SELECT id FROM catalog_projects
        WHERE announcement_url = ? OR (project_name = ? AND source = 'bitcointalk')
        LIMIT 1
        """,
        (url, name),
    )
    existing = cur.fetchone()
    if existing:
        cur.execute(
            """
            UPDATE catalog_projects
            SET project_name = ?, symbol = ?, category = ?, source = ?,
                announcement_url = ?, summary = ?, status = ?, announced_at = ?
            WHERE id = ?
            """,
            (
                row["project_name"],
                row.get("symbol"),
                row.get("category"),
                row.get("source", "bitcointalk"),
                url,
                row.get("summary"),
                row.get("status", "Catalog Only"),
                row.get("announced_at"),
                existing[0],
            ),
        )
    else:
        cur.execute(
            """
            INSERT INTO catalog_projects
            (project_name, symbol, category, source, announcement_url, summary, status, announced_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["project_name"],
                row.get("symbol"),
                row.get("category"),
                row.get("source", "bitcointalk"),
                url,
                row.get("summary"),
                row.get("status", "Catalog Only"),
                row.get("announced_at"),
            ),
        )
    conn.commit()
    conn.close()


def import_topics(
    topics: Iterable[Dict[str, Any]],
    db_path: str = DB_PATH_DEFAULT,
) -> int:
    count = 0
    for raw in topics:
        row = normalize_topic(raw)
        upsert_catalog_row(row, db_path=db_path)
        count += 1
    return count


def import_topics_from_json_file(path: str, db_path: str = DB_PATH_DEFAULT) -> int:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("topics") or data.get("rows") or [data]
    return import_topics(data, db_path=db_path)


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
    sample_path = _find_sample("bitcointalk_topics.json")
    if sample_path:
        print(f"Loading samples from {sample_path}")
        return import_topics_from_json_file(sample_path, db_path=db_path)
    print("No samples/bitcointalk_topics.json found — skip import")
    return 0


if __name__ == "__main__":
    n = demo_import()
    print(f"Imported {n} BitcoinTalk topics into catalog")
    df = load_catalog()
    print(df[df["source"] == "bitcointalk"].head(10).to_string(index=False))
