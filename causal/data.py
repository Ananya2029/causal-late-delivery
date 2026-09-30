"""
Build the analysis table from the Olist warehouse (built by the ecommerce-data-pipeline project).

Unit: one delivered order.  Treatment: delivered after the promised date (is_late).
Outcomes: review score, negative review (1-2 stars), and - for a customer's FIRST order -
whether they buy again within 6 months.
Confounders: only things known when the order was placed (before delivery could be late).
"""
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAREHOUSE = Path(os.environ.get(
    "WAREHOUSE_PATH", ROOT.parent / "ecommerce-data-pipeline" / "warehouse" / "olist.duckdb"))

QUERY = """
WITH main_item AS (          -- the most expensive item decides the order's category and seller
    SELECT order_id,
           arg_max(product_key, price) AS product_key,
           arg_max(seller_key, price)  AS seller_key
    FROM gold.fact_sales GROUP BY 1
),
firsts AS (                  -- each customer's first purchase date
    SELECT customer_key, min(purchased_at) AS first_at
    FROM gold.fact_orders WHERE order_status NOT IN ('canceled', 'unavailable') GROUP BY 1
),
repeat_6m AS (
    SELECT f.customer_key,
           max((o.purchased_at > f.first_at AND o.purchased_at <= f.first_at + INTERVAL 6 MONTH)::INT) AS bought_again_6m
    FROM firsts f JOIN gold.fact_orders o USING (customer_key)
    WHERE o.order_status NOT IN ('canceled', 'unavailable')
    GROUP BY 1
)
SELECT o.order_id,
       o.is_late::INT                          AS late,
       o.review_score,
       (o.review_score <= 2)::INT              AS negative_review,
       o.delivery_days - o.promised_days       AS days_vs_promise,   -- treatment intensity (>0 = late)
       (o.purchased_at = fr.first_at)::INT     AS is_first_order,
       r.bought_again_6m,
       o.purchased_at <= TIMESTAMP '2018-02-28' AS repeat_observable,   -- 6 months of follow-up exist
       -- confounders (all known at purchase time)
       o.items, o.items_value, o.freight_value, o.installments, o.promised_days,
       o.payment_type,
       month(o.purchased_at)                   AS purchase_month,
       dayofweek(o.purchased_at)               AS purchase_dow,
       p.category, p.weight_g, p.volume_cm3,
       c.state                                 AS customer_state,
       s.state                                 AS seller_state,
       c.lat AS c_lat, c.lng AS c_lng, s.lat AS s_lat, s.lng AS s_lng
FROM gold.fact_orders o
JOIN main_item mi USING (order_id)
JOIN gold.dim_product p ON p.product_key = mi.product_key
JOIN gold.dim_seller s ON s.seller_key = mi.seller_key
JOIN gold.dim_customer c USING (customer_key)
JOIN firsts fr USING (customer_key)
JOIN repeat_6m r USING (customer_key)
WHERE o.order_status = 'delivered' AND o.is_late IS NOT NULL AND o.review_score IS NOT NULL
"""


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def load(path=WAREHOUSE) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.sql(QUERY).df()
    df["distance_km"] = haversine_km(df.c_lat, df.c_lng, df.s_lat, df.s_lng)
    df = df.drop(columns=["c_lat", "c_lng", "s_lat", "s_lng"])
    num = ["items_value", "freight_value", "weight_g", "volume_cm3", "distance_km"]
    df = df.dropna(subset=num + ["installments", "promised_days"]).reset_index(drop=True)
    for c in ["items_value", "freight_value", "weight_g", "volume_cm3", "distance_km"]:
        df[f"log_{c}"] = np.log1p(df[c])
    df["same_state"] = (df.customer_state == df.seller_state).astype(int)
    return df


NUMERIC = ["items", "installments", "promised_days", "log_items_value", "log_freight_value",
           "log_weight_g", "log_volume_cm3", "log_distance_km", "same_state"]
CATEGORICAL = ["payment_type", "purchase_month", "purchase_dow", "category", "customer_state", "seller_state"]


def design_matrix(df: pd.DataFrame, top_categories: int = 30) -> pd.DataFrame:
    """Numeric confounders + one-hot categoricals (rare categories grouped as 'other')."""
    X = df[NUMERIC].astype(float).copy()
    cats = df[CATEGORICAL].copy()
    keep = cats["category"].value_counts().index[:top_categories]
    cats["category"] = cats["category"].where(cats["category"].isin(keep), "other")
    return pd.concat([X, pd.get_dummies(cats.astype(str), drop_first=True, dtype=float)], axis=1)
