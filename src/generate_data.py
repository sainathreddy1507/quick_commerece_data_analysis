from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    n_orders: int = 120_000
    seed: int = 42
    start_date: str = "2026-01-01"
    end_date: str = "2026-03-31"


def _weighted_choice(rng: np.random.Generator, values: list[str], probs: list[float], size: int) -> np.ndarray:
    return rng.choice(values, p=probs, size=size)


def build_synthetic_orders(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    zones = [
        "Koramangala",
        "HSR Layout",
        "Indiranagar",
        "Whitefield",
        "BTM Layout",
        "Jayanagar",
        "Marathahalli",
        "Electronic City",
    ]
    zone_weights = [0.16, 0.14, 0.13, 0.11, 0.12, 0.10, 0.12, 0.12]

    store_by_zone = {
        "Koramangala": ["KOR_01", "KOR_02"],
        "HSR Layout": ["HSR_01", "HSR_02"],
        "Indiranagar": ["IND_01", "IND_02"],
        "Whitefield": ["WHI_01", "WHI_02"],
        "BTM Layout": ["BTM_01", "BTM_02"],
        "Jayanagar": ["JAY_01", "JAY_02"],
        "Marathahalli": ["MAR_01", "MAR_02"],
        "Electronic City": ["ELC_01", "ELC_02"],
    }

    partners = [f"DP_{i:03d}" for i in range(1, 221)]
    partner_exp = rng.uniform(0.2, 1.0, size=len(partners))
    partner_efficiency = dict(zip(partners, partner_exp))

    start_ts = datetime.fromisoformat(cfg.start_date)
    end_ts = datetime.fromisoformat(cfg.end_date) + timedelta(days=1)
    total_seconds = int((end_ts - start_ts).total_seconds())

    order_offsets = rng.integers(0, total_seconds, size=cfg.n_orders)
    order_times = pd.to_datetime([start_ts + timedelta(seconds=int(x)) for x in order_offsets])

    hours = order_times.hour.values
    weekdays = order_times.dayofweek.values
    is_weekend = (weekdays >= 5).astype(int)

    hour_demand_boost = np.where((hours >= 18) & (hours <= 23), 1.7, 1.0)
    lunch_boost = np.where((hours >= 12) & (hours <= 14), 1.25, 1.0)
    late_night_penalty = np.where((hours <= 1) | (hours >= 23), 1.18, 1.0)
    weekend_boost = np.where(is_weekend == 1, 1.1, 1.0)

    weather = _weighted_choice(rng, ["clear", "rain", "storm"], [0.74, 0.22, 0.04], size=cfg.n_orders)
    weather_delay = np.where(weather == "rain", 1.30, np.where(weather == "storm", 1.75, 1.0))

    zone = _weighted_choice(rng, zones, zone_weights, size=cfg.n_orders)
    store_id = np.array([rng.choice(store_by_zone[z]) for z in zone])
    partner_id = rng.choice(partners, size=cfg.n_orders, replace=True)

    items_count = rng.poisson(3.2, size=cfg.n_orders) + 1
    basket_value = np.round(rng.gamma(shape=2.1, scale=180, size=cfg.n_orders), 2)
    distance_km = np.round(np.clip(rng.normal(2.7, 1.1, size=cfg.n_orders), 0.4, 8.0), 2)

    zone_congestion_index = pd.Series(zone).map(
        {
            "Koramangala": 1.30,
            "HSR Layout": 1.20,
            "Indiranagar": 1.25,
            "Whitefield": 1.15,
            "BTM Layout": 1.18,
            "Jayanagar": 1.05,
            "Marathahalli": 1.22,
            "Electronic City": 1.10,
        }
    ).values
    partner_factor = np.array([partner_efficiency[p] for p in partner_id])

    prep_base = 1.6 + 0.24 * items_count + rng.normal(0.0, 0.55, cfg.n_orders)
    prep_time_min = np.clip(prep_base * hour_demand_boost * lunch_boost * weekend_boost * np.where(zone_congestion_index > 1.2, 1.08, 1.0), 0.8, 12.0)

    pickup_wait_min = np.clip(
        rng.normal(1.1, 0.55, cfg.n_orders)
        * hour_demand_boost
        * weather_delay
        * np.where(zone_congestion_index > 1.2, 1.05, 1.0),
        0.2,
        8.0,
    )

    travel_base = (distance_km / 0.52) + rng.normal(0.35, 0.9, cfg.n_orders)
    traffic_multiplier = np.where((hours >= 8) & (hours <= 11), 1.25, 1.0) * np.where((hours >= 18) & (hours <= 21), 1.35, 1.0)
    delivery_travel_min = np.clip(
        travel_base * traffic_multiplier * weather_delay * late_night_penalty * zone_congestion_index / (0.85 + partner_factor),
        1.2,
        24.0,
    )

    order_to_delivery_min = prep_time_min + pickup_wait_min + delivery_travel_min
    sla_10_min = (order_to_delivery_min <= 10).astype(int)
    delay_min = np.clip(order_to_delivery_min - 10, 0, None)

    accepted_after_sec = np.clip((rng.normal(35, 12, cfg.n_orders) * (1.1 - partner_factor)).astype(int), 8, 130)
    packed_after_sec = np.clip((prep_time_min * 60 + rng.normal(20, 15, cfg.n_orders)).astype(int), 80, 1700)
    picked_after_sec = np.clip(((prep_time_min + pickup_wait_min) * 60 + rng.normal(30, 20, cfg.n_orders)).astype(int), 120, 2200)
    delivered_after_sec = np.clip((order_to_delivery_min * 60 + rng.normal(0, 25, cfg.n_orders)).astype(int), 180, 2600)

    df = pd.DataFrame(
        {
            "order_id": [f"O_{i:08d}" for i in range(1, cfg.n_orders + 1)],
            "order_ts": order_times,
            "zone": zone,
            "store_id": store_id,
            "partner_id": partner_id,
            "distance_km": distance_km,
            "items_count": items_count,
            "basket_value_inr": basket_value,
            "hour": hours,
            "weekday": weekdays,
            "is_weekend": is_weekend,
            "weather": weather,
            "zone_congestion_index": zone_congestion_index,
            "prep_time_min": np.round(prep_time_min, 2),
            "pickup_wait_min": np.round(pickup_wait_min, 2),
            "delivery_travel_min": np.round(delivery_travel_min, 2),
            "order_to_delivery_min": np.round(order_to_delivery_min, 2),
            "sla_10_min": sla_10_min,
            "delay_min": np.round(delay_min, 2),
            "accepted_after_sec": accepted_after_sec,
            "packed_after_sec": packed_after_sec,
            "picked_after_sec": picked_after_sec,
            "delivered_after_sec": delivered_after_sec,
        }
    ).sort_values("order_ts")

    return df


def main() -> None:
    cfg = Config()
    df = build_synthetic_orders(cfg)
    output_path = Path("data/quick_commerce_orders.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df):,} synthetic orders at {output_path}")


if __name__ == "__main__":
    main()
