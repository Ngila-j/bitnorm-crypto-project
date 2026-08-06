# BN Analytics — demo & status guide

Use this when walking someone through the live Streamlit app or reviewing the repo.

## 0. Optional warm-up (15s)

- **Settings → Import** BitcoinTalk + GitHub samples if catalog looks empty
- Sidebar **Target asset** = BTC (or ETH)

## 1. Home (45s)

- Ecosystem chips: BNCommunity · **BNAnalytics** · BNExchange
- **Protocol health leaderboard** (custom pillar weights)
- **Hot announcements** — top engagement topics from catalog
- CTAs: Live Analytics · Explorer · Compare · Detail

## 2. Overview Dashboard (60s)

Studio-style tabs:

| Tab | Show |
|-----|------|
| **Asset Overview** | Macro + vol context |
| **Market** | Order book / taker flow |
| **5-Pillar Health Engine** | Radar, pillar chips, weights |
| **On-Chain Activity** | Whales / network |

## 3. Project Detail (45s)

- **5 Pillars** tab: radar + composite (session weights)
- **Source Code history** + **Linked GitHub repositories**
- **Repo risk badges**: archived / fork / low contributors / no releases

## 4. Catalog & News (45s)

- **Projects → All Projects**: engagement, type, linked repos, **Tracked** symbols
- **Insights → News**: sort by engagement, ANN/ICO tags, velocity chart

## 5. Compare & Explorer (30s)

- **Compare Assets**: side-by-side pillars
- **Project Explorer**: Radar Score + DEV/NET/ECON/SENT/ACCESS

## 6. Settings (30s)

- Custom pillar weights (institutional vs equal 20%)
- **BitNorm GraphQL API** → `https://api.bitnorm.com/` (Test connection)
- Exchange mode, sample imports, regenerate data

## Architecture (one sentence)

Local SQLite + adapters (BitcoinTalk topics, GitHub repos) power the 5-pillar terminal; production GraphQL at `api.bitnorm.com` is stubbed until auth is provided.

## Talking points

| Ready now | Waiting on boss |
|-----------|-----------------|
| 5-pillar scores, radar, weights | GraphQL token + sample queries |
| Catalog engagement intelligence | Live multi-asset universe |
| GitHub risk signals | Mongo/indexation bulk export |
| Local FastAPI (`api.py`) | Exchange live keys (BNExchange) |

## Push checklist

```powershell
git add app.py catalog.py bitcointalk_adapter.py github_repo_adapter.py bitnorm_api.py DEMO.md .env.example
git status
git commit -m "Announcement intelligence, repo risk, GraphQL stub, Studio polish"
git push origin main
```

## Optional API probe

```powershell
# After BITNORM_API_TOKEN is set
curl.exe -X POST "https://api.bitnorm.com/" -H "Content-Type: application/json" -H "Authorization: Bearer YOUR_TOKEN" -d "{\"query\":\"{ __typename }\"}"
```
