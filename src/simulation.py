"""
Custom Simulation Engine
------------------------
Generates a synthetic product catalog and weekly demand history using
stochastic models (trend + seasonality + noise + occasional festive
spikes). This lets the forecasting engine be tested end-to-end without
needing real historical sales records, exactly as described in the
project report (Section 3.2 / 6.2 - Data Ingestion Module).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Static product catalog (stand-in for scraped e-commerce metadata)
# ---------------------------------------------------------------------
CATALOG = [
    {"product_id": "P001", "name": "Wireless Mouse",        "category": "Electronics", "price": 799,  "lead_time_days": 7,  "base_demand": 120, "trend": 0.15, "seasonality": 0.20},
    {"product_id": "P002", "name": "Bluetooth Headphones",  "category": "Electronics", "price": 1999, "lead_time_days": 10, "base_demand": 80,  "trend": 0.25, "seasonality": 0.35},
    {"product_id": "P003", "name": "Notebook Set (A5)",     "category": "Stationery",  "price": 149,  "lead_time_days": 5,  "base_demand": 200, "trend": 0.05, "seasonality": 0.10},
    {"product_id": "P004", "name": "Office Chair",          "category": "Furniture",   "price": 5499, "lead_time_days": 21, "base_demand": 25,  "trend": 0.10, "seasonality": 0.05},
    {"product_id": "P005", "name": "LED Desk Lamp",         "category": "Home",        "price": 899,  "lead_time_days": 12, "base_demand": 60,  "trend": 0.08, "seasonality": 0.30},
    {"product_id": "P006", "name": "Yoga Mat",              "category": "Fitness",     "price": 599,  "lead_time_days": 9,  "base_demand": 90,  "trend": 0.20, "seasonality": 0.40},
    {"product_id": "P007", "name": "Ceramic Coffee Mug",    "category": "Home",        "price": 299,  "lead_time_days": 6,  "base_demand": 150, "trend": 0.02, "seasonality": 0.15},
    {"product_id": "P008", "name": "Running Shoes",         "category": "Fitness",     "price": 2999, "lead_time_days": 14, "base_demand": 45,  "trend": 0.18, "seasonality": 0.45},
]


def get_catalog() -> pd.DataFrame:
    """Returns the product catalog as a DataFrame."""
    return pd.DataFrame(CATALOG)


def generate_demand_history(product_id: str, weeks: int = 104, seed: int | None = None) -> pd.DataFrame:
    """
    Generates synthetic weekly demand data for a given product using a
    trend + seasonal + festive-spike + noise stochastic model.

    Returns a DataFrame with columns: ds (week start date), y (units sold).
    """
    product = next((p for p in CATALOG if p["product_id"] == product_id), None)
    if product is None:
        raise ValueError(f"Unknown product_id: {product_id}")

    rng = np.random.default_rng(seed if seed is not None else abs(hash(product_id)) % (2**32))

    weeks_idx = np.arange(weeks)
    today = pd.Timestamp.today().normalize()
    last_monday = today - pd.Timedelta(days=today.weekday())
    dates = pd.date_range(end=last_monday, periods=weeks, freq="W-MON")

    base = product["base_demand"]
    trend = product["trend"] * weeks_idx  # gradual linear growth
    seasonal = base * product["seasonality"] * np.sin(2 * np.pi * weeks_idx / 52)
    noise = rng.normal(0, base * 0.08, size=weeks)

    # Festive spikes: boost demand around simulated "festive" weeks (e.g. every ~26 weeks)
    festive_spike = np.zeros(weeks)
    for spike_week in range(10, weeks, 26):
        window = slice(max(0, spike_week - 1), min(weeks, spike_week + 2))
        festive_spike[window] += base * 0.5

    demand = base + trend + seasonal + noise + festive_spike
    demand = np.clip(demand, a_min=0, a_max=None).round().astype(int)

    return pd.DataFrame({"ds": dates, "y": demand})


def generate_all_histories(weeks: int = 104) -> dict[str, pd.DataFrame]:
    """Generates demand history for every product in the catalog."""
    return {p["product_id"]: generate_demand_history(p["product_id"], weeks=weeks) for p in CATALOG}


if __name__ == "__main__":
    catalog = get_catalog()
    print(catalog)
    sample = generate_demand_history("P001", weeks=12)
    print(sample)
