"""
BitNorm production API client (GraphQL).

Base URL: https://api.bitnorm.com/
Auth: Bearer token via BITNORM_API_TOKEN or session override.
Until a valid token is provided, calls return structured errors (not crashes).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import requests

DEFAULT_BASE_URL = os.environ.get("BITNORM_API_URL", "https://api.bitnorm.com/").rstrip("/") + "/"
DEFAULT_TIMEOUT = float(os.environ.get("BITNORM_API_TIMEOUT", "15"))


def get_token(explicit: Optional[str] = None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    return (os.environ.get("BITNORM_API_TOKEN") or os.environ.get("BN_API_TOKEN") or "").strip()


def graphql(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    token: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Execute a GraphQL request.

    Returns (ok, payload) where payload is either:
      {"data": ...} on success, or
      {"error": str, "status_code": int|None, "body": ...} on failure.
    """
    url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    tok = get_token(token)
    if tok:
        headers["Authorization"] = f"Bearer {tok}"

    body: Dict[str, Any] = {"query": query}
    if variables:
        body["variables"] = variables

    try:
        res = requests.post(url, json=body, headers=headers, timeout=timeout)
    except requests.exceptions.RequestException as e:
        return False, {"error": f"connection_failed: {e}", "status_code": None, "body": None}

    try:
        data = res.json()
    except Exception:
        data = {"raw": (res.text or "")[:1000]}

    if res.status_code == 401 or (
        isinstance(data, dict)
        and any("Unauthorized" in str(err.get("message", "")) for err in (data.get("errors") or []) if isinstance(err, dict))
    ):
        return False, {
            "error": "unauthorized",
            "status_code": res.status_code,
            "body": data,
            "hint": "Set BITNORM_API_TOKEN or paste a token in Settings. Confirm auth scheme with boss.",
        }

    if res.status_code >= 400:
        return False, {"error": f"http_{res.status_code}", "status_code": res.status_code, "body": data}

    if isinstance(data, dict) and data.get("errors") and not data.get("data"):
        return False, {"error": "graphql_errors", "status_code": res.status_code, "body": data}

    return True, data if isinstance(data, dict) else {"data": data}


def test_connection(token: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Lightweight probe: { __typename }."""
    ok, payload = graphql("{ __typename }", token=token, base_url=base_url)
    return {
        "ok": ok,
        "base_url": (base_url or DEFAULT_BASE_URL),
        "has_token": bool(get_token(token)),
        "result": payload,
    }


def introspect_query_fields(token: Optional[str] = None, base_url: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort Query field list (may be denied by server auth/ACL)."""
    q = """
    {
      __type(name: "Query") {
        name
        fields { name description }
      }
    }
    """
    ok, payload = graphql(q, token=token, base_url=base_url)
    return {"ok": ok, "result": payload}


# Placeholder helpers for when schema is known — safe no-ops until wired
def fetch_asset_health(symbol: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Reserved for production pillar query once schema is documented.
    Returns structured stub so UI can show 'not wired' cleanly.
    """
    return {
        "ok": False,
        "symbol": symbol,
        "error": "query_not_configured",
        "hint": "Await boss sample GraphQL query for pillar/asset health.",
    }
