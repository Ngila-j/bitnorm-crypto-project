"""
BitNorm Production API (Phase 4)

Lightweight FastAPI layer exposing health scores and exchange snapshots
for future Next.js / external consumers. Reads the same SQLite stores
as the Streamlit terminal.
"""

from __future__ import annotations

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from analytics import compute_blockactivities_health_score, compute_net_taker_flow
from exchange_adapter import (
    get_mode,
    get_orderbook,
    get_ticker,
    refresh_all,
    refresh_symbol,
)

app = FastAPI(
    title="BitNorm BN Analytics API",
    description="Production data contracts for health scores and exchange feeds",
    version="0.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("BN_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "env": os.environ.get("BN_ENV", "development"),
        "exchange_mode": get_mode(),
    }


@app.get("/v1/assets")
def list_assets():
    return {"assets": ["BTC", "ETH", "SOL", "ADA"]}


@app.get("/v1/health/{symbol}")
def asset_health(symbol: str):
    symbol = symbol.upper()
    if symbol not in ("BTC", "ETH", "SOL", "ADA"):
        raise HTTPException(status_code=404, detail="Asset not tracked")
    try:
        payload = compute_blockactivities_health_score(symbol, db_path="crypto_data.db")
        return {
            "symbol": symbol,
            "health_score": payload.get("health_score"),
            "pillar_scores": payload.get("pillar_scores"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/flow")
def net_taker_flow():
    try:
        df = compute_net_taker_flow(db_path="crypto_data.db")
        return {"rows": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/exchange/ticker/{symbol}")
def exchange_ticker(symbol: str, refresh: bool = Query(False)):
    symbol = symbol.upper()
    if refresh:
        refresh_symbol(symbol)
    row = get_ticker(symbol)
    if not row:
        refresh_symbol(symbol)
        row = get_ticker(symbol)
    if not row:
        raise HTTPException(status_code=404, detail="Ticker unavailable")
    return row


@app.get("/v1/exchange/orderbook/{symbol}")
def exchange_orderbook(symbol: str, refresh: bool = Query(False)):
    symbol = symbol.upper()
    if refresh:
        refresh_symbol(symbol)
    book = get_orderbook(symbol)
    if not book.get("bids") and not book.get("asks"):
        refresh_symbol(symbol)
        book = get_orderbook(symbol)
    return {"symbol": symbol, **book}


@app.post("/v1/exchange/refresh")
def exchange_refresh(symbols: Optional[List[str]] = None):
    results = refresh_all(symbols=symbols)
    return {
        "refreshed": [r["ticker"]["symbol"] for r in results],
        "mode": get_mode(),
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("BN_API_HOST", "0.0.0.0")
    port = int(os.environ.get("BN_API_PORT", "8000"))
    uvicorn.run("api:app", host=host, port=port, reload=True)
