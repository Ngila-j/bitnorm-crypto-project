# BN Analytics — 3-minute demo script

Use this path when walking your boss through the live Streamlit app.

## 1. Home (30s)
- Point out the **BitNorm ecosystem** chips: BNCommunity · **BNAnalytics** · BNExchange
- Show **Live Asset Health** for BTC / ETH / SOL / ADA
- Click **Explore Terminal**

## 2. Overview Dashboard (45s)
- Select **BTC** in the sidebar Target asset
- Macro metrics + **Order Flow** (exchange adapter — mode shown as mock/live)
- Mention alerts: sidebar threshold + webhook

## 3. Project Detail → 5 Pillars (45s)
- Open **Project Detail Page**
- Open the **5 Pillars** tab: Source Code, Network, Economics, Sentiment, Accessibility
- Composite health score at the top

## 4. Project catalog (30s)
- **Projects → All Projects**: core tracked + simulated announcements
- **Categories**: filter by Layer-1 / DeFi / Infrastructure
- **Search**: try `DeFi` or `NovaMesh`

## 5. Alerts & Settings (30s)
- Sidebar: lower Min health warning until a warning appears (or show OK status)
- **Settings**: Exchange mode, Regenerate data, Alert audit log

## Optional
- **Docs / API**: playground + local `uvicorn` at `:8000`
- **Learn → Tutorials**: step-by-step terminal guides

## Talking points
- This is the **BNAnalytics** module of BitNorm (not the full community site)
- Exchange **live** mode waits on API keys from BNExchange
- Catalog is ready for **BitcoinTalk Altcoin Announcements** scraper output
