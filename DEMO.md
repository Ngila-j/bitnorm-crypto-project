# BN Analytics — 3-minute demo script

Use this path when walking your boss through the live Streamlit app.

## 0. Optional warm-up (15s)
- **Settings → Import both** (BitcoinTalk + GitHub samples) if catalog looks empty
- Confirm sidebar **Target asset** = BTC

## 1. Home (30s)
- Point out the **BitNorm ecosystem** chips: BNCommunity · **BNAnalytics** · BNExchange
- Show **Live Asset Health** for BTC / ETH / SOL / ADA
- Click **Explore Terminal**

## 2. Overview Dashboard (45s)
- Macro metrics + **Order Flow** (exchange adapter — mock/live in Settings)
- Mention alerts: sidebar health threshold + webhook

## 3. Project Detail → 5 Pillars (45s)
- Open **Project Detail Page**
- **5 Pillars** tab: Source Code, Network, Economics, Sentiment, Accessibility
- Scroll to **Linked GitHub repositories** (stars, commits, contributors from adapter)

## 4. Project catalog (30s)
- **Projects → All Projects**: filter source `bitcointalk` for ANN/ICO samples
- **Categories** / **Search**: try `NovaMesh` or `DeFi`

## 5. News (20s)
- **Insights → News**
- Filter tag **BitcoinTalk** / **Catalog** — announcements from the catalog

## 6. Alerts & Settings (20s)
- Sidebar health status chip
- **Settings**: Exchange mode, **Import samples**, Regenerate data, Alert audit log

## Optional
- **Docs / API**: playground includes `/v1/catalog` (local `uvicorn` on `:8000`)
- **Learn → Tutorials**

## Talking points
- This is the **BNAnalytics** module of BitNorm (not the full community site)
- Catalog + Source Code adapters match indexation schemas (`BitcointalkTopic`, `GithubRepository`)
- Exchange **live** mode waits on BNExchange API keys
- Live BitcoinTalk/Mongo feed can replace sample JSON without UI rewrites
