# BitNorm / BN Analytics Terminal

Institutional-grade blockchain intelligence platform.

**Phases 1–3:** Streamlit terminal (multi-pillar health, alerts, explorer, learn).  
**Phase 4:** Production foundation — repo hygiene, CI, Docker, REST API, exchange adapter.

---

## Features

- Multi-pillar composite health scores (Source Code, Network, Economics, Sentiment, Accessibility)
- Live price ticker + exchange-adapter order books (mock or live)
- Overview, Project Detail, Explorer, Market Analysis, Watchlist
- Automated health alerts + webhook dispatcher + audit log
- Executive PDF / CSV export
- Production API (`api.py`) for future Next.js / external clients
- Docker Compose staging stack

---

## Project Structure

```
BITNORM-CRYPTO-PROJECT/
├── app.py                 # Streamlit terminal
├── analytics.py           # Health scores, net taker flow (keep on your machine)
├── pipeline.py            # Simulated data generation
├── exchange_adapter.py    # Exchange ticker/order book (mock + live hooks)
├── api.py                 # FastAPI production contracts
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
├── .github/workflows/ci.yml
└── README.md
```

---

## Quick Start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env          # optional
python pipeline.py            # seed demo DB
python exchange_adapter.py    # seed mock exchange tables

streamlit run app.py
```

### Production API

```bash
uvicorn api:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

Key routes:
- `GET /health`
- `GET /v1/health/{symbol}`
- `GET /v1/flow`
- `GET /v1/exchange/ticker/{symbol}?refresh=true`
- `GET /v1/exchange/orderbook/{symbol}?refresh=true`
- `POST /v1/exchange/refresh`

---

## Exchange integration

| Mode | How |
|------|-----|
| `EXCHANGE_MODE=mock` (default) | Simulated ticker + book — no keys needed |
| `EXCHANGE_MODE=live` | REST calls using `EXCHANGE_API_KEY` + `EXCHANGE_BASE_URL` |

1. Copy `.env.example` → `.env`
2. Fill exchange credentials when available
3. Adjust endpoint paths in `exchange_adapter.py` (`fetch_ticker_live` / `fetch_orderbook_live`) to match your exchange API
4. Overview → Order Flow uses the adapter automatically

---

## Docker (staging)

```bash
docker compose up --build
# Terminal: http://localhost:8501
# API:      http://localhost:8000/docs
```

---

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on push/PR:
- Install deps
- Syntax-check modules
- Seed pipeline data
- Mock exchange refresh

---

## Roadmap Status

- **Phase 1 — Prototype foundation:** Complete
- **Phase 2 — Core terminal polish:** Complete
- **Phase 3 — Insights, automation, Learn:** Complete
- **Phase 4 — Production foundation:** Complete (this release)
  - `.gitignore`, `.env.example`, CI workflow
  - `exchange_adapter.py` (mock + live hooks)
  - `api.py` FastAPI contracts
  - Docker + Compose staging stack
  - Streamlit order book wired to adapter

### Still ahead (post–Phase 4)

- Full **Next.js** marketing + app shell consuming `/v1/*`
- Real exchange endpoint mapping once API docs are finalized
- Managed staging/production hosts (Cloud Run, Fly, AWS, etc.)
- Hardened auth (replace demo auto-login)

---

## Notes

- Demo data is simulated unless exchange mode is `live` with valid keys.
- Never commit `.env` or database files with production secrets.
- Keep `analytics.py` alongside `app.py` in your working directory (imported by terminal + API).
