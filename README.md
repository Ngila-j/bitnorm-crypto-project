# BitNorm / BN Analytics Terminal

Institutional-grade blockchain intelligence platform (Streamlit prototype).

This is the **Phase 1 reference implementation** of the BN Analytics terminal. It powers the multi-pillar health scores, project explorer, alerts, executive PDF reports, and the full navigation structure defined in the product sitemap.

---

## Features

- Multi-pillar composite health scores (Source Code, Network, Economics, Sentiment, Accessibility)
- Live price ticker (CoinGecko with simulated fallback)
- Overview Dashboard with order-book depth, technical matrix, and whale alerts
- Project Detail Page (derivatives, on-chain supply, capital flows, regulatory feed)
- Market Analysis (Overview, Trading Data, AI Select, Token Unlock)
- Automated health-score alerts + webhook dispatcher (Slack / Telegram)
- Executive PDF and CSV export
- Role-aware sidebar (Admin / Portfolio Manager / Analyst)
- Simulated institutional data layer (trades + 5 metric pillars)

---

## Project Structure

```
BITNORM-CRYPTO-PROJECT/
├── app.py                          # Main Streamlit application
├── analytics.py                    # Health scores, net taker flow, sentiment
├── pipeline.py                     # Simulated data generation
├── requirements.txt                # Clean dependency list
├── logo.png                        # Optional sidebar logo
├── crypto_data.db                  # Main metrics + trades database (auto-created)
├── bnanalytics_institutional.db    # Users, API keys, alert audit logs
└── README.md
```

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `prophet` is optional. The app includes a lightweight linear fallback if Prophet is not installed or fails to build.

### 3. (Optional) Pre-generate data

```bash
python pipeline.py
```

This creates `crypto_data.db` with 5,000 simulated trades and 30 days of pillar metrics for BTC, ETH, SOL, and ADA.  
The app will also auto-generate the data on first run if the tables are missing.

### 4. Launch the terminal

```bash
streamlit run app.py
```

The app will open in your browser (usually http://localhost:8501).

---

## Default Access

On first run the app auto-authenticates as:

- **Username:** `admin_lead`
- **Role:** `Admin`

(The institutional users table is seeded automatically.)

---

## Regenerating Data

If you want a fresh dataset:

1. Delete `crypto_data.db` (and optionally `bnanalytics_institutional.db`)
2. Run `python pipeline.py`
3. Restart the Streamlit app

---

## Key Modules

| Module | Responsibility |
|--------|----------------|
| `pipeline.py` | Creates tables and generates simulated trades + pillar metrics |
| `analytics.py` | Computes net taker flow, sentiment index, and composite health scores |
| `app.py` | Full Streamlit UI, navigation, charts, alerts, and PDF generation |

---

## Next Steps (Roadmap)

- **Phase 1 (current):** Stabilize prototype, clean dependencies, documentation
- **Phase 2:** Core terminal polish + production data contracts
- **Phase 3:** Insights, automation hardening, Learn section
- **Phase 4:** Production web app (Next.js + API) + launch

---

## Notes

- All data is **simulated** for demonstration purposes.
- CoinGecko live prices are used when the network is available; otherwise the ticker falls back to simulated values.
- The dark institutional theme is defined via custom CSS inside `app.py`.
