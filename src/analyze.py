from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("data/quick_commerce_orders.csv")
OUTPUT_PATH = Path("outputs")


def _ensure_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Data file not found. Run `python src/generate_data.py` before analysis."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["order_ts"])
    return df


def _feature_importance(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    features = [
        "distance_km",
        "items_count",
        "hour",
        "is_weekend",
        "zone",
        "weather",
        "zone_congestion_index",
        "prep_time_min",
        "pickup_wait_min",
    ]
    target = "order_to_delivery_min"
    X = df[features]
    y = df[target]

    numeric_features = [
        "distance_km",
        "items_count",
        "hour",
        "is_weekend",
        "zone_congestion_index",
        "prep_time_min",
        "pickup_wait_min",
    ]
    categorical_features = ["zone", "weather"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=220,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))

    fitted_preprocessor = pipeline.named_steps["prep"]
    ohe = fitted_preprocessor.named_transformers_["cat"]
    encoded_names = list(ohe.get_feature_names_out(categorical_features))
    feature_names = numeric_features + encoded_names
    importances = pipeline.named_steps["model"].feature_importances_
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    top_imp = {k: round(float(v), 4) for k, v in imp_series.head(12).items()}

    metrics = {"model_mae_min": round(mae, 3), "model_r2": round(r2, 3)}
    return top_imp, metrics


def run_analysis() -> dict:
    df = _ensure_data()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    kpis = {
        "total_orders": int(len(df)),
        "avg_order_to_delivery_min": round(float(df["order_to_delivery_min"].mean()), 2),
        "p90_delivery_min": round(float(df["order_to_delivery_min"].quantile(0.90)), 2),
        "sla_10_min_rate_pct": round(float(df["sla_10_min"].mean() * 100), 2),
        "avg_delay_if_breached_min": round(
            float(df.loc[df["sla_10_min"] == 0, "delay_min"].mean()), 2
        ),
    }

    peak_hours = (
        df.groupby("hour", as_index=False)
        .agg(
            orders=("order_id", "count"),
            sla_pct=("sla_10_min", lambda x: round(float(np.mean(x) * 100), 2)),
            avg_delivery_min=("order_to_delivery_min", "mean"),
        )
        .sort_values("orders", ascending=False)
    )
    peak_hours["avg_delivery_min"] = peak_hours["avg_delivery_min"].round(2)

    location_perf = (
        df.groupby("zone", as_index=False)
        .agg(
            orders=("order_id", "count"),
            sla_pct=("sla_10_min", lambda x: round(float(np.mean(x) * 100), 2)),
            avg_delivery_min=("order_to_delivery_min", "mean"),
            avg_delay_min=("delay_min", "mean"),
        )
        .sort_values("orders", ascending=False)
    )
    location_perf["avg_delivery_min"] = location_perf["avg_delivery_min"].round(2)
    location_perf["avg_delay_min"] = location_perf["avg_delay_min"].round(2)

    partner_perf = (
        df.groupby("partner_id", as_index=False)
        .agg(
            orders=("order_id", "count"),
            sla_pct=("sla_10_min", lambda x: round(float(np.mean(x) * 100), 2)),
            avg_delivery_min=("order_to_delivery_min", "mean"),
            avg_distance_km=("distance_km", "mean"),
        )
        .query("orders >= 120")
        .sort_values(["sla_pct", "orders"], ascending=[False, False])
    )
    partner_perf["avg_delivery_min"] = partner_perf["avg_delivery_min"].round(2)
    partner_perf["avg_distance_km"] = partner_perf["avg_distance_km"].round(2)

    breach_df = df[df["sla_10_min"] == 0].copy()
    breach_root = {
        "prep_component_share_pct": round(
            float((breach_df["prep_time_min"].sum() / breach_df["order_to_delivery_min"].sum()) * 100),
            2,
        ),
        "pickup_component_share_pct": round(
            float((breach_df["pickup_wait_min"].sum() / breach_df["order_to_delivery_min"].sum()) * 100),
            2,
        ),
        "travel_component_share_pct": round(
            float((breach_df["delivery_travel_min"].sum() / breach_df["order_to_delivery_min"].sum()) * 100),
            2,
        ),
    }

    top_imp, model_metrics = _feature_importance(df)

    peak_hours.to_csv(OUTPUT_PATH / "peak_hours.csv", index=False)
    location_perf.to_csv(OUTPUT_PATH / "location_performance.csv", index=False)
    partner_perf.to_csv(OUTPUT_PATH / "partner_performance.csv", index=False)

    result = {
        "kpis": kpis,
        "model_metrics": model_metrics,
        "top_delay_drivers": top_imp,
        "breach_component_share": breach_root,
        "top_5_peak_hours": peak_hours.head(5).to_dict(orient="records"),
        "high_demand_locations": location_perf.head(5).to_dict(orient="records"),
    }
    (OUTPUT_PATH / "analysis_summary.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    output = run_analysis()
    print(json.dumps(output, indent=2))
