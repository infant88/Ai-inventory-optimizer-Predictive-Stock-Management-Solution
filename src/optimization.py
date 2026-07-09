"""
Optimization Logic
-------------------
Implements the inventory management formulas described in the report
(Section 6.2):

    Safety Stock  = Z * sigma  (sigma = std dev of lead-time demand)
    Reorder Point = Lead Time Demand + Safety Stock
    EOQ           = sqrt( (2 * D * S) / H )

Z is derived from the user-selected service level (e.g. 95% -> 1.645).
"""

from dataclasses import dataclass
from math import sqrt

# Common service-level -> Z-score lookup (one-sided normal distribution)
SERVICE_LEVEL_Z = {
    0.90: 1.2816,
    0.95: 1.6449,
    0.975: 1.9600,
    0.99: 2.3263,
    0.995: 2.5758,
}


def z_score_for_service_level(service_level: float) -> float:
    """Returns the closest Z-score for a given service level (0-1)."""
    closest = min(SERVICE_LEVEL_Z.keys(), key=lambda k: abs(k - service_level))
    return SERVICE_LEVEL_Z[closest]


@dataclass
class InventoryRecommendation:
    lead_time_demand: float
    safety_stock: float
    reorder_point: float
    suggested_order_qty: float
    service_level: float
    z_score: float


def compute_recommendation(
    lead_time_demand: float,
    lead_time_demand_std: float,
    service_level: float = 0.95,
    annual_demand: float | None = None,
    order_cost: float = 50.0,
    holding_cost_per_unit: float = 20.0,
    max_stock_capacity: float | None = None,
) -> InventoryRecommendation:
    """
    Computes safety stock, reorder point, and a suggested order quantity.

    If `annual_demand`, `order_cost` (S) and `holding_cost_per_unit` (H)
    are provided, the suggested order quantity uses the Economic Order
    Quantity (EOQ) formula; otherwise it defaults to the reorder point
    itself (order up to cover one lead-time cycle).
    """
    z = z_score_for_service_level(service_level)
    safety_stock = max(0.0, z * lead_time_demand_std)
    reorder_point = lead_time_demand + safety_stock

    if annual_demand and annual_demand > 0 and holding_cost_per_unit > 0:
        eoq = sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
        suggested_order_qty = eoq
    else:
        suggested_order_qty = reorder_point

    if max_stock_capacity is not None:
        suggested_order_qty = min(suggested_order_qty, max_stock_capacity)

    return InventoryRecommendation(
        lead_time_demand=round(lead_time_demand, 1),
        safety_stock=round(safety_stock, 1),
        reorder_point=round(reorder_point, 1),
        suggested_order_qty=round(suggested_order_qty, 1),
        service_level=service_level,
        z_score=z,
    )
