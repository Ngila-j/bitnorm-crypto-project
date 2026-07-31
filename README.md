# BitNorm / BN Analytics Terminal

Institutional-grade blockchain intelligence — the **BNAnalytics** module of the BitNorm ecosystem (BNCommunity · BNAnalytics · BNExchange · BNBusiness).

## Features

- 5-pillar composite health scores
- Overview, Project Detail, Explorer, Market Analysis, Watchlist
- Project **catalog** (ready for BitcoinTalk Altcoin Announcements)
- Automated health alerts + webhook audit log
- Exchange adapter (mock / live) for order books
- FastAPI `/v1/*` including `/v1/catalog`
- Docker Compose + CI workflow

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python pipeline.py
python catalog.py
streamlit run app.py
```

API:

```bash
uvicorn api:app --reload --port 8000
```

## Demo walkthrough

See [DEMO.md](DEMO.md) for a 3-minute boss walkthrough.

## Roadmap

- Phases 1–4: Complete (terminal, API, exchange adapter, catalog seed)
- Next (needs external access): live BNExchange API, BitcoinTalk scraper import
