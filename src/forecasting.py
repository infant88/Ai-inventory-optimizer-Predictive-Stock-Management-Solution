"""
Forecasting Engine (Prophet)
----------------------------
Wraps Facebook Prophet to turn weekly demand history into a future
demand forecast with confidence intervals, as described in the report
(Section 3.3 / 6.2).
"""

import logging

import pandas as pd
from prophet import Prophet

# Quiet down Prophet/cmdstanpy's chatty INFO logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def run_forecast(history: pd.DataFrame, periods: int = 12, freq: str = "W") -> pd.DataFrame:
    """
    Fits Prophet on `history` (columns: ds, y) and returns a DataFrame
    covering both history and the forecast horizon with columns:
    ds, yhat, yhat_lower, yhat_upper, is_forecast.
    """
    if history.empty or len(history) < 10:
        raise ValueError("Need at least 10 historical data points to forecast.")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.90,
    )
    model.fit(history[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result["yhat"] = result["yhat"].clip(lower=0)
    result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
    result["yhat_upper"] = result["yhat_upper"].clip(lower=0)
    result["is_forecast"] = result["ds"] > history["ds"].max()

    return result


def lead_time_demand_stats(forecast: pd.DataFrame, lead_time_weeks: float) -> tuple[float, float]:
    """
    Aggregates the forecasted future demand over the lead time window and
    returns (mean_demand, std_dev_demand) used by the optimization module.
    """
    future_only = forecast[forecast["is_forecast"]].copy()
    if future_only.empty:
        raise ValueError("Forecast contains no future periods.")

    n_weeks = max(1, round(lead_time_weeks))
    window = future_only.head(n_weeks)

    mean_weekly = window["yhat"].mean()
    # Approximate weekly std dev from the 90% confidence interval width
    approx_std_weekly = ((window["yhat_upper"] - window["yhat_lower"]) / (2 * 1.645)).mean()

    lead_time_demand = mean_weekly * n_weeks
    lead_time_std = approx_std_weekly * (n_weeks ** 0.5)  # variance scales with time
    return lead_time_demand, lead_time_std
