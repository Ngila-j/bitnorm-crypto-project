# BNAnalytics — status for leadership

**Date:** August 2026  
**Module:** BNAnalytics (BitNorm institutional terminal)  
**Repo:** bitnorm-crypto-project (Streamlit + local FastAPI)

---

## What is live today

| Capability | Status |
|------------|--------|
| 5-pillar health engine (Source Code, Network, Economics, Sentiment, Accessibility) | Working (local SQLite) |
| Custom pillar weights + radar charts | Working |
| Top navigation (category menus) + expanded workspace sidebar | Working |
| Project Explorer / Compare / Detail | Working |
| Catalog + News engagement (views, replies, ANN/ICO) | Working |
| GitHub repo risk badges (archived, fork, low contributors) | Working |
| BitcoinTalk + GitHub sample adapters (indexation-shaped) | Working |
| Alert rules (metric, threshold, channel, webhook) | Working |
| Catalog CSV export | Working |
| GraphQL client stub → `https://api.bitnorm.com/` | Ready; **Unauthorized without token** |
| Local FastAPI (`api.py`) | Working for local/dev |

Tracked demo assets: **BTC, ETH, SOL, ADA**.

---

## What we reviewed externally

- **BitNorm Studio** (blocksactivities-platform-1937): same 5-pillar product model; richer multi-asset chrome.
- **GitLab** bitcointalk-scraper + indexation models: adapters map topics/repos into our catalog and Source Code pillar.
- Studio’s documented REST host `api.bitnorm.io` does **not** resolve.
- Production GraphQL at **`https://api.bitnorm.com/`** responds; requires auth.

---

## Blocked on your side (please provide)

1. **GraphQL auth** for `api.bitnorm.com` — token + auth header scheme  
2. **One sample query** for asset / pillar health (or allow introspection)  
3. Optional: **Mongo/indexation export** of topics + GitHub repos for bulk history  
4. Optional: **BNExchange** API base URL + keys when ready for live order flow  

Until then we correctly keep scoring on local/simulated + imported sample data.

---

## Suggested next integration step

1. Confirm GraphQL token works in GraphiQL (`{ __typename }`).  
2. Share a minimal pillar/asset query.  
3. We map responses into Overview / Detail and expand beyond four assets **with real inputs**.

---

## Demo path (3 minutes)

See **DEMO.md** — Home radar → Overview 5-Pillar tab → Detail risk badges → All Projects CSV → Settings GraphQL test.
