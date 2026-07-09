# AI Inventory Optimizer: Predictive Stock Management Solution

A working implementation of the system described in your project report —
a Streamlit dashboard that forecasts product demand with **Facebook
Prophet** and turns that forecast into **reorder point / safety stock /
order quantity** recommendations.

## Project structure

```
ai-inventory-optimizer/
├── app.py                  # Streamlit dashboard (UI layer)
├── requirements.txt
├── src/
│   ├── simulation.py        # Synthetic weekly demand generator + product catalog
│   ├── scraper.py            # BeautifulSoup web-scraping module (+ safe fallback)
│   ├── forecasting.py        # Prophet forecasting wrapper
│   └── optimization.py       # Reorder point / safety stock / EOQ formulas
└── data/                     # (reserved for exported/cached CSVs)
```

This maps directly onto the report's chapters:

| Report section | Code |
|---|---|
| 3.2 Data Ingestion and Preprocessing | `src/simulation.py`, `src/scraper.py` |
| 3.3 ML and Forecasting Libraries | `src/forecasting.py` |
| 6.2 Optimization Logic | `src/optimization.py` |
| 6.1 Frontend (ProductSelector, ForecastChart, InventoryMetrics, ParameterControl, Export) | `app.py` |

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** installing `prophet` pulls in `cmdstanpy`, which downloads and
> compiles a small Stan binary on first install — this can take a few
> minutes the very first time, but only happens once.

## Run

```bash
streamlit run app.py
```

Then open the URL Streamlit
Live link for this project : https://infant88-ai-inventory-optimizer-predictive-stock-man-app-oe6uqj.streamlit.app/

## How it works

The dashboard is built to explain itself as it runs — no separate slide
needed to walk someone through it:

1. A **3-step banner** at the top ("Pick a product → AI predicts demand →
   Get a restock plan") frames the whole app before anyone touches it.
2. **Pick a product** in the sidebar (8 sample products across
   Electronics, Stationery, Furniture, Home, and Fitness are pre-loaded).
   The app generates **104 weeks of synthetic historical demand** for
   that product (trend + yearly seasonality + festive spikes + noise) —
   this stands in for real POS/e-commerce sales history.
3. **Prophet** fits that history and forecasts demand for the horizon you
   choose (4–26 weeks), with a 90% confidence interval. The chart is
   labeled in plain language (past sales vs. AI prediction vs. reorder
   line) with a caption explaining each color.
4. The **optimization module** aggregates the forecast over your lead
   time to get lead-time demand, computes **safety stock** using your
   chosen service level (Z-score), the **reorder point**, and an
   **EOQ-based suggested order quantity**. Each metric has a hover
   tooltip (ⓘ) explaining it in plain English, and a one-sentence plain
   summary ("once stock hits X, order Y units") ties it all together.
5. Every sidebar control has a plain-language question and a tooltip
   explaining its effect, so viewers can see cause and effect by moving
   a slider and watching the chart/numbers update live.
6. The full forecast table can be **downloaded as CSV**, and an
   "How this works" expander at the bottom gives a short recap for
   anyone who wants the underlying formulas.

## Using real data instead of simulated data

Swap `simulation.generate_demand_history()` in `app.py` for your own CSV
loader (any DataFrame with `ds` = date and `y` = units sold works with
`forecasting.run_forecast`). `src/scraper.py` includes a ready-to-adapt
BeautifulSoup scraper — point it at a product listing page and update the
CSS selectors in `SELECTORS` to match that site's HTML.

## Extending it (matches the report's "Future Scope" chapter)

- Swap the in-memory catalog for a real database (Postgres/SQLite).
- Add ARIMA / XGBoost as alternate forecasting backends (report section
  4.5 compares all three) and let the user pick a model.
- Add authentication + per-user session persistence for a multi-user
  deployment.
- Containerize with Docker and deploy to Render / Streamlit Community
  Cloud, as outlined in report section 3.6.
