"""
AI Inventory Optimizer: Predictive Stock Management Solution
--------------------------------------------------------------
Streamlit dashboard tying together:
  - Data Ingestion (simulation.py / scraper.py)
  - Forecasting Engine (forecasting.py - Prophet)
  - Optimization Logic (optimization.py - reorder point, safety stock, EOQ)

Designed to be self-explanatory: a first-time viewer should be able to
watch it run and understand what's happening at each step without a
verbal walkthrough.

Run with:  streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent))
from src import forecasting, optimization, simulation

# ---------------------------------------------------------------------
# Page config & light styling
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI Inventory Optimizer",
    page_icon="\U0001F4E6",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .app-title { font-weight: 800; letter-spacing: -.01em; }
      .step-badge {
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 50%;
        background: #F2994A; color: #14181f; font-weight: 800; font-size: 0.85rem;
        margin-right: 8px; flex-shrink: 0;
      }
      .step-box {
        background: #14181f; border: 1px solid #262b36; border-radius: 10px;
        padding: 0.9rem 1rem; height: 100%;
      }
      .step-title { font-weight: 700; font-size: 0.95rem; color: #f2f4f7; display:flex; align-items:center; }
      .step-desc { color: #9aa4b2; font-size: 0.82rem; margin-top: 4px; margin-left: 34px; }
      .metric-sub { color: #6b7280; font-size: 0.78rem; }
      .legend-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    return simulation.get_catalog()


@st.cache_data(show_spinner=False)
def load_history(product_id: str, weeks: int = 104) -> pd.DataFrame:
    return simulation.generate_demand_history(product_id, weeks=weeks)


@st.cache_data(show_spinner=False)
def load_forecast(product_id: str, weeks: int, periods: int) -> pd.DataFrame:
    history = load_history(product_id, weeks=weeks)
    return forecasting.run_forecast(history, periods=periods, freq="W")


# ---------------------------------------------------------------------
# Header + plain-language intro (this IS the explanation people need)
# ---------------------------------------------------------------------
st.markdown("## \U0001F4E6 AI Inventory Optimizer")
st.caption("Predicts how much of a product you'll need — and tells you exactly when to reorder, and how much.")

s1, s2, s3 = st.columns(3)
with s1:
    st.markdown(
        """<div class="step-box"><div class="step-title"><span class="step-badge">1</span>Pick a product</div>
        <div class="step-desc">Choose any item from the sidebar. Its past sales history loads automatically.</div></div>""",
        unsafe_allow_html=True,
    )
with s2:
    st.markdown(
        """<div class="step-box"><div class="step-title"><span class="step-badge">2</span>AI predicts demand</div>
        <div class="step-desc">Prophet, an AI forecasting model, studies the trend and predicts future weekly demand.</div></div>""",
        unsafe_allow_html=True,
    )
with s3:
    st.markdown(
        """<div class="step-box"><div class="step-title"><span class="step-badge">3</span>Get a restock plan</div>
        <div class="step-desc">The app turns that prediction into: when to reorder, and how much to order.</div></div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------------------
# Sidebar - ProductSelector + ParameterControl components
# ---------------------------------------------------------------------
catalog = load_catalog()

with st.sidebar:
    st.markdown("### Controls")
    st.caption("Change anything below and the whole dashboard updates instantly.")
    st.divider()

    st.markdown("**① Product**")
    product_label_map = {f"{row.name} ({row.product_id})": row.product_id for row in catalog.itertuples()}
    selected_label = st.selectbox(
        "Which product are we planning for?",
        list(product_label_map.keys()),
        help="Each product has its own sales history, price, and delivery lead time.",
    )
    product_id = product_label_map[selected_label]
    product_row = catalog[catalog["product_id"] == product_id].iloc[0]

    st.divider()
    st.markdown("**② Forecast settings**")
    forecast_horizon = st.slider(
        "How many weeks ahead to predict?", 4, 26, 12,
        help="A longer horizon shows further into the future, but predictions get less certain the further out they go.",
    )
    service_level_pct = st.select_slider(
        "How sure do you want to be you won't run out?",
        options=[90, 95, 97.5, 99, 99.5], value=95, format_func=lambda v: f"{v}%",
        help="Higher = you keep more safety stock so stockouts are rarer, but you tie up more cash in inventory.",
    )
    lead_time_weeks = st.number_input(
        "How many weeks does a new order take to arrive?",
        min_value=0.5, max_value=20.0,
        value=round(product_row["lead_time_days"] / 7, 1), step=0.5,
        help="Supplier delivery time. Defaults to this product's catalog lead time — override if yours differs.",
    )

    st.divider()
    st.markdown("**③ Cost inputs** _(optional — fine-tunes order size)_")
    order_cost = st.number_input(
        "Cost to place one order (₹)", min_value=0.0, value=50.0, step=10.0,
        help="Admin/shipping cost incurred every time you place an order, regardless of quantity.",
    )
    holding_cost = st.number_input(
        "Cost to store one unit for a year (₹)", min_value=0.1,
        value=round(product_row["price"] * 0.15, 2), step=5.0,
        help="Warehousing, insurance, and capital cost of holding one unit in stock for a year.",
    )

    st.divider()
    st.info("Tip: try switching products — notice how a seasonal item (e.g. Yoga Mat) gets a very different-shaped forecast than a steady one (e.g. Notebook Set).", icon="\U0001F4A1")


# ---------------------------------------------------------------------
# Product header
# ---------------------------------------------------------------------
st.markdown(f"### <span class='app-title'>{product_row['name']}</span>", unsafe_allow_html=True)
st.caption(f"Category: {product_row['category']}  ·  Price: ₹{product_row['price']}  ·  Catalog lead time: {product_row['lead_time_days']} days")

history = load_history(product_id)
with st.spinner("Training the forecasting model on this product's sales history..."):
    try:
        forecast = load_forecast(product_id, weeks=104, periods=forecast_horizon)
    except ValueError as e:
        st.error(str(e))
        st.stop()

lead_time_demand, lead_time_std = forecasting.lead_time_demand_stats(forecast, lead_time_weeks)
annual_demand = history["y"].tail(52).sum()

recommendation = optimization.compute_recommendation(
    lead_time_demand=lead_time_demand,
    lead_time_demand_std=lead_time_std,
    service_level=service_level_pct / 100,
    annual_demand=annual_demand,
    order_cost=order_cost,
    holding_cost_per_unit=holding_cost,
)

# ---------------------------------------------------------------------
# ForecastChart Component
# ---------------------------------------------------------------------
st.markdown("#### \U0001F4C8 Step 2 — What the AI predicts")
st.caption(
    "**Blue** = actual past weekly sales. **Orange** = the AI's prediction for future weeks, "
    "with a shaded band showing its uncertainty. **Red dashed line** = the reorder point from Step 3 below."
)

hist_plot = history.tail(52)
fut_plot = forecast[forecast["is_forecast"]]

fig = go.Figure()
fig.add_trace(go.Scatter(x=hist_plot["ds"], y=hist_plot["y"], name="Actual past sales", mode="lines", line=dict(color="#5B8DEF", width=2)))
fig.add_trace(go.Scatter(x=fut_plot["ds"], y=fut_plot["yhat"], name="AI prediction", mode="lines", line=dict(color="#F2994A", width=2, dash="solid")))
fig.add_trace(go.Scatter(
    x=pd.concat([fut_plot["ds"], fut_plot["ds"][::-1]]),
    y=pd.concat([fut_plot["yhat_upper"], fut_plot["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(242,153,74,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="Prediction uncertainty range", hoverinfo="skip",
))
fig.add_hline(y=recommendation.reorder_point, line_dash="dot", line_color="#EB5757",
              annotation_text="Reorder here", annotation_position="top left")

fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=30, b=10),
    template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02),
    xaxis_title=None, yaxis_title="Units sold per week",
)
st.plotly_chart(fig, use_container_width=True)

st.write("")

# ---------------------------------------------------------------------
# InventoryMetrics Component
# ---------------------------------------------------------------------
st.markdown("#### \u2705 Step 3 — Your restock plan, in plain terms")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Demand while waiting for delivery", f"{recommendation.lead_time_demand:,.0f} units",
    help=f"Predicted units customers will buy during the {lead_time_weeks}-week wait for a new order to arrive.",
)
c2.metric(
    "Extra safety buffer", f"{recommendation.safety_stock:,.0f} units",
    help=f"Extra stock kept on hand in case demand is higher than expected, so you hit your {service_level_pct}% no-stockout target.",
)
c3.metric(
    "\U0001F6A9 Reorder when stock hits", f"{recommendation.reorder_point:,.0f} units",
    help="The moment your remaining stock drops to this number, place a new order — that's demand-while-waiting + safety buffer.",
)
c4.metric(
    "\U0001F4E6 How much to order", f"{recommendation.suggested_order_qty:,.0f} units",
    help="The cost-efficient order size (EOQ), balancing ordering cost against storage cost.",
)

st.success(
    f"**In one sentence:** once stock of **{product_row['name']}** falls to "
    f"**{recommendation.reorder_point:,.0f} units**, place a new order for "
    f"**{recommendation.suggested_order_qty:,.0f} units** — that keeps you {service_level_pct}% safe from running out "
    f"while the {lead_time_weeks}-week delivery is on its way.",
    icon="\U0001F3AF",
)

st.divider()

# ---------------------------------------------------------------------
# Export & Sharing Tools
# ---------------------------------------------------------------------
st.markdown("#### \U0001F4E4 Export")
export_df = forecast.rename(columns={"ds": "week", "yhat": "forecast", "yhat_lower": "lower_bound", "yhat_upper": "upper_bound"})
col_a, col_b = st.columns([1, 3])
with col_a:
    st.download_button(
        "\u2B07\uFE0F Download forecast (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{product_id}_forecast.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("\U0001F50D See the raw forecast numbers"):
    st.dataframe(export_df, use_container_width=True, hide_index=True)

with st.expander("\u2139\uFE0F How this works, for anyone new to the project"):
    st.markdown(
        """
        This dashboard demonstrates an AI-driven approach to a very common business problem:
        **"How much stock should I keep, and when should I reorder?"**

        1. **Historical data** — every product has a history of weekly sales (in a live deployment
           this would come from real POS/e-commerce records; here it's realistically simulated).
        2. **Forecasting (Prophet)** — an open-source AI model from Meta studies that history —
           trend, seasonality, spikes — and predicts future weekly demand, along with a confidence range.
        3. **Inventory math** — classic supply-chain formulas turn that prediction into concrete
           actions:
           - `Safety stock = Z-score × demand variability` — extra cushion for uncertainty
           - `Reorder point = demand during lead time + safety stock` — the trigger stock level
           - `Order quantity (EOQ)` — the most cost-efficient amount to reorder at once

        Change any control in the sidebar and every chart/number above recalculates live —
        that live recalculation *is* the AI + optimization pipeline running end-to-end.
        """
    )
