"""Synthetic supply-chain order generator — the single source of training data.

WHY THIS EXISTS
---------------
Every other module in this project (EDA tools, the ML pipeline, the SQL warehouse,
the multi-agent demo) consumes ONE flat file: ``data/raw/supply_chain_orders.csv``.
Real client data is confidential, so we synthesize a dataset that *behaves* like the
public Kaggle "DataCo Smart Supply Chain" dataset — same canonical columns, same
business meaning — but is fully deterministic (``np.random.default_rng(42)``) so every
trainee sees identical numbers and every test is reproducible.

THE KEY TRICK: PLANTED SIGNAL
-----------------------------
A classifier can only "learn" patterns that exist. We generate the late-delivery
target from an explicit logistic model whose drivers mirror real logistics pain:

- **Tight schedules**: Same Day / First Class promise 1-2 days → most often late.
- **Hard geographies**: Africa and LATAM lanes have worse infrastructure → later.
- **Peak season**: Q4 (Oct-Dec) and the January hangover overload networks.
- **Big orders**: higher ``order_item_quantity`` → harder to fulfill on time.
- **Carrier quality**: SwiftShip Express is the worst performer, NordHaul the best.

The intercept is *calibrated by bisection* so the overall late rate lands at ~30%
(class imbalance is intentional — it drives the Week-3 imbalance lesson). A Gaussian
noise term keeps the problem realistic: a good model reaches ROC-AUC ≈ 0.75-0.82,
not 1.0.

LEAKAGE BY DESIGN
-----------------
``shipping_date`` and ``actual_shipping_days`` mathematically determine the target
(``late_delivery = actual > scheduled``). They are kept in the CSV *on purpose* so
trainees experience finding and dropping leakage columns in Lab 02.

Run: ``python data/generate_supply_chain_data.py`` (from the project root).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "raw" / "supply_chain_orders.csv"

RANDOM_STATE: int = 42
N_ORDERS: int = 12_000
N_CUSTOMERS: int = 2_500
N_DAYS: int = 730  # 2024-01-01 .. 2025-12-31
START_DATE: np.datetime64 = np.datetime64("2024-01-01")
TARGET_LATE_RATE: float = 0.30

# ----------------------------------------------------------------------------------
# Categorical vocabularies (§3.1 of ARCHITECTURE.md)
# ----------------------------------------------------------------------------------
MODES: list[str] = ["Same Day", "First Class", "Second Class", "Standard Class"]
MODE_PROBS: list[float] = [0.10, 0.18, 0.22, 0.50]
SCHEDULED_DAYS: dict[str, int] = {
    "Same Day": 1,
    "First Class": 2,
    "Second Class": 4,
    "Standard Class": 6,
}

MARKETS: list[str] = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
MARKET_PROBS: list[float] = [0.22, 0.26, 0.24, 0.18, 0.10]

# market → region → countries (plausible nesting, mirrors the DataCo hierarchy)
GEOGRAPHY: dict[str, dict[str, list[str]]] = {
    "LATAM": {
        "Central America": ["Mexico", "Guatemala", "Panama", "Costa Rica"],
        "South America": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
        "Caribbean": ["Dominican Republic", "Cuba", "Jamaica"],
    },
    "Europe": {
        "Western Europe": ["France", "Germany", "United Kingdom", "Netherlands", "Spain"],
        "Northern Europe": ["Sweden", "Norway", "Denmark", "Finland"],
        "Eastern Europe": ["Poland", "Czechia", "Romania", "Hungary"],
        "Southern Europe": ["Italy", "Portugal", "Greece"],
    },
    "Pacific Asia": {
        "Southeast Asia": ["Indonesia", "Vietnam", "Thailand", "Philippines", "Malaysia"],
        "Eastern Asia": ["China", "Japan", "South Korea"],
        "South Asia": ["India", "Bangladesh", "Pakistan"],
        "Oceania": ["Australia", "New Zealand"],
    },
    "USCA": {
        "East of USA": ["United States"],
        "West of USA": ["United States"],
        "US Center": ["United States"],
        "Canada": ["Canada"],
    },
    "Africa": {
        "West Africa": ["Nigeria", "Ghana", "Senegal"],
        "North Africa": ["Egypt", "Morocco", "Algeria"],
        "East Africa": ["Kenya", "Ethiopia", "Tanzania"],
        "Southern Africa": ["South Africa", "Botswana"],
    },
}

SEGMENTS: list[str] = ["Consumer", "Corporate", "Home Office"]
SEGMENT_PROBS: list[float] = [0.52, 0.30, 0.18]

# category → (base unit price, product names)
CATALOG: dict[str, tuple[float, list[str]]] = {
    "Electronics": (320.0, ["4K Action Camera", "Noise-Cancelling Headphones",
                            "Smart Fitness Watch", "Portable Bluetooth Speaker",
                            "Wireless Earbuds Pro"]),
    "Furniture": (450.0, ["Ergonomic Office Chair", "Standing Desk Pro",
                          "Walnut Bookshelf", "Modular Conference Table"]),
    "Office Supplies": (45.0, ["Laser Printer Paper 10-Ream", "Gel Pen 24-Pack",
                               "Heavy-Duty Stapler", "Whiteboard Marker Set"]),
    "Sporting Goods": (180.0, ["Carbon Trekking Poles", "Pro Composite Basketball",
                               "4-Person Camping Tent", "Insulated Hydration Pack"]),
    "Apparel": (75.0, ["Performance Rain Jacket", "Merino Base Layer",
                       "Trail Running Shorts", "Thermal Work Gloves"]),
    "Footwear": (110.0, ["Trail Running Shoes", "Steel-Toe Work Boots",
                         "Court Sneakers", "Waterproof Hiking Boots"]),
    "Garden & Outdoor": (95.0, ["Cordless Hedge Trimmer", "Solar Path Lights 8-Pack",
                                "Expandable Garden Hose", "Patio Heater"]),
    "Health & Beauty": (38.0, ["Electric Toothbrush", "Vitamin C Serum",
                               "Digital Body Scale", "Aromatherapy Diffuser"]),
    "Toys & Games": (55.0, ["1000-Piece Jigsaw Puzzle", "RC Off-Road Buggy",
                            "Strategy Board Game", "Building Blocks Mega Set"]),
    "Automotive Parts": (140.0, ["Ceramic Brake Pad Set", "LED Headlight Kit",
                                 "All-Weather Floor Mats", "OBD-II Diagnostic Scanner"]),
}

# ----------------------------------------------------------------------------------
# Carrier assignment (§3.3): correlated with shipping mode AND market.
# SwiftShip owns the premium Same Day / First Class lanes, Atlas the Standard lanes,
# NordHaul dominates Europe, Pacific Crest dominates Pacific Asia.
# ----------------------------------------------------------------------------------
CARRIERS: list[str] = ["SwiftShip Express", "Atlas Freight Co.",
                       "Pacific Crest Carriers", "NordHaul Logistics"]

# rows follow MODES order, columns follow CARRIERS order
MODE_CARRIER_WEIGHTS: np.ndarray = np.array([
    [0.70, 0.05, 0.10, 0.15],   # Same Day
    [0.60, 0.10, 0.15, 0.15],   # First Class
    [0.15, 0.40, 0.20, 0.25],   # Second Class
    [0.05, 0.55, 0.18, 0.22],   # Standard Class
])
# rows follow MARKETS order, columns follow CARRIERS order (multiplicative boost)
MARKET_CARRIER_MULT: np.ndarray = np.array([
    [1.00, 1.60, 0.40, 0.70],   # LATAM
    [0.90, 0.60, 0.20, 3.00],   # Europe → NordHaul
    [0.90, 0.50, 3.00, 0.40],   # Pacific Asia → Pacific Crest
    [1.20, 1.80, 0.50, 0.60],   # USCA → Atlas-leaning
    [1.00, 1.70, 0.50, 0.60],   # Africa
])

# ----------------------------------------------------------------------------------
# Planted signal: log-odds contributions to P(late). Tune here, not downstream.
# ----------------------------------------------------------------------------------
MODE_EFFECT: np.ndarray = np.array([1.90, 1.15, 0.10, -0.15])        # follows MODES
MARKET_EFFECT: np.ndarray = np.array([0.75, -0.15, 0.05, -0.30, 1.15])  # follows MARKETS
CARRIER_EFFECT: np.ndarray = np.array([0.85, 0.00, 0.10, -0.75])     # follows CARRIERS
MONTH_EFFECT: dict[int, float] = {1: 0.20, 10: 0.35, 11: 0.70, 12: 0.95}
QTY_EFFECT: float = 0.30   # per unit above 1
NOISE_SD: float = 0.25     # irreducible randomness → realistic (non-perfect) AUC


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_intercept(raw_logit: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept b0 such that mean(sigmoid(b0 + raw_logit)) == target.

    This decouples 'how strong is each driver' from 'what is the overall late rate':
    you can re-tune the planted effects above without breaking the ~30% class balance.
    """
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if float(np.mean(_sigmoid(mid + raw_logit))) > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


def _sample_geography(rng: np.random.Generator,
                      markets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample region then country, respecting the market → region → country nesting."""
    regions = np.empty(len(markets), dtype=object)
    countries = np.empty(len(markets), dtype=object)
    for market, region_map in GEOGRAPHY.items():
        mask = markets == market
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        chosen_regions = rng.choice(list(region_map), size=len(idx))
        regions[idx] = chosen_regions
        for region, country_list in region_map.items():
            sub = idx[chosen_regions == region]
            if len(sub):
                countries[sub] = rng.choice(country_list, size=len(sub))
    return regions, countries


def _sample_carriers(rng: np.random.Generator, mode_idx: np.ndarray,
                     market_idx: np.ndarray) -> np.ndarray:
    """Vectorized categorical sampling from per-row weights = mode base × market boost."""
    weights = MODE_CARRIER_WEIGHTS[mode_idx] * MARKET_CARRIER_MULT[market_idx]
    weights /= weights.sum(axis=1, keepdims=True)
    u = rng.random(len(mode_idx))
    picks = (np.cumsum(weights, axis=1) < u[:, None]).sum(axis=1)
    return np.minimum(picks, len(CARRIERS) - 1)


def generate_orders(n_orders: int = N_ORDERS, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Build the full canonical orders frame (§3.1 column order) deterministically."""
    rng = np.random.default_rng(seed)

    # --- customers (segment is fixed PER CUSTOMER → lossless DB normalization) ---
    pool_ids = np.arange(N_CUSTOMERS) + 1000
    pool_segments = rng.choice(SEGMENTS, size=N_CUSTOMERS, p=SEGMENT_PROBS)
    cust_pick = rng.integers(0, N_CUSTOMERS, size=n_orders)
    customer_ids = pool_ids[cust_pick]
    customer_segments = pool_segments[cust_pick]

    # --- dates / modes / geography / carriers ---
    order_dates = START_DATE + rng.integers(0, N_DAYS, size=n_orders).astype("timedelta64[D]")
    order_months = pd.DatetimeIndex(order_dates).month.to_numpy()

    mode_idx = rng.choice(len(MODES), size=n_orders, p=MODE_PROBS)
    modes = np.array(MODES, dtype=object)[mode_idx]
    scheduled = np.array([SCHEDULED_DAYS[m] for m in MODES])[mode_idx]

    market_idx = rng.choice(len(MARKETS), size=n_orders, p=MARKET_PROBS)
    markets = np.array(MARKETS, dtype=object)[market_idx]
    regions, countries = _sample_geography(rng, markets)

    carrier_idx = _sample_carriers(rng, mode_idx, market_idx)
    carriers = np.array(CARRIERS, dtype=object)[carrier_idx]

    # --- products & money ---
    cat_idx = rng.integers(0, len(CATALOG), size=n_orders)
    categories = np.array(list(CATALOG), dtype=object)[cat_idx]
    products = np.empty(n_orders, dtype=object)
    base_price = np.empty(n_orders)
    for i, (price, prods) in enumerate(CATALOG.values()):
        mask = cat_idx == i
        products[mask] = rng.choice(prods, size=int(mask.sum()))
        base_price[mask] = price

    quantity = rng.integers(1, 6, size=n_orders)
    sales = np.round(base_price * quantity * rng.uniform(0.85, 1.30, n_orders), 2)
    discount = np.round(rng.uniform(0.0, 0.25, n_orders), 2)
    profit = np.round(sales * rng.normal(0.12, 0.16, n_orders), 2)  # can go negative

    # --- target: calibrated logistic model over the planted drivers ---
    month_lookup = np.zeros(13)
    for month, effect in MONTH_EFFECT.items():
        month_lookup[month] = effect
    raw_logit = (
        MODE_EFFECT[mode_idx]
        + MARKET_EFFECT[market_idx]
        + CARRIER_EFFECT[carrier_idx]
        + month_lookup[order_months]
        + QTY_EFFECT * (quantity - 1)
        + rng.normal(0.0, NOISE_SD, n_orders)
    )
    intercept = _calibrate_intercept(raw_logit, TARGET_LATE_RATE)
    p_late = _sigmoid(intercept + raw_logit)
    late = rng.random(n_orders) < p_late

    # --- actual shipping days, CONSISTENT with the target by construction ---
    delay = rng.choice([1, 2, 3, 4], size=n_orders, p=[0.45, 0.30, 0.15, 0.10])
    early = rng.integers(0, 3, size=n_orders)
    actual = np.where(late, scheduled + delay, np.maximum(scheduled - early, 0))
    shipping_dates = order_dates + actual.astype("timedelta64[D]")

    df = pd.DataFrame({
        "order_id": np.arange(n_orders) + 10_000,
        "order_date": np.datetime_as_string(order_dates, unit="D"),
        "shipping_date": np.datetime_as_string(shipping_dates, unit="D"),
        "scheduled_shipping_days": scheduled.astype(int),
        "actual_shipping_days": actual.astype(int),
        "shipping_mode": modes,
        "carrier_name": carriers,
        "customer_id": customer_ids.astype(int),
        "customer_segment": customer_segments,
        "market": markets,
        "order_region": regions,
        "order_country": countries,
        "category_name": categories,
        "product_name": products,
        "order_item_quantity": quantity.astype(int),
        "sales": sales,
        "discount": discount,
        "profit": profit,
        "late_delivery": (actual > scheduled).astype(int),
    })
    assert (df["late_delivery"].to_numpy() == late.astype(int)).all(), "target drift"
    return df


def main() -> None:
    """Generate the canonical CSV and print a quick sanity summary."""
    df = generate_orders()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    late_rate = df["late_delivery"].mean()
    print(f"Wrote {len(df):,} rows -> {OUTPUT_PATH}")
    print(f"Overall late rate: {late_rate:.3f} (target ~{TARGET_LATE_RATE})")
    print("Late rate by shipping mode:")
    print(df.groupby("shipping_mode")["late_delivery"].mean().round(3).to_string())
    print("Late rate by carrier:")
    print(df.groupby("carrier_name")["late_delivery"].mean().round(3).to_string())


if __name__ == "__main__":
    main()
