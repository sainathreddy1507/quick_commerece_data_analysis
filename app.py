from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Quick Commerce SLA Dashboard", layout="wide")

DATA_PATH = Path("data/quick_commerce_orders.csv")
SUMMARY_PATH = Path("outputs/analysis_summary.json")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_PATH, parse_dates=["order_ts"])


@st.cache_data
def load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text())


def main() -> None:
    st.title("Quick Commerce 10-Min Delivery SLA Dashboard")
    st.caption("Synthetic operational analytics for Zepto/Blinkit-style quick commerce")

    df = load_data()
    summary = load_summary()

    if df.empty or not summary:
        st.warning("Data/summary not found. Run `python src/generate_data.py` and `python src/analyze.py` first.")
        st.stop()

    k = summary["kpis"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Orders", f"{k['total_orders']:,}")
    c2.metric("Avg Delivery Time", f"{k['avg_order_to_delivery_min']} min")
    c3.metric("P90 Delivery Time", f"{k['p90_delivery_min']} min")
    c4.metric("SLA Within 10 Min", f"{k['sla_10_min_rate_pct']}%")
    c5.metric("Avg Delay (Breached)", f"{k['avg_delay_if_breached_min']} min")

    st.divider()
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 2])
    zones = sorted(df["zone"].unique().tolist())
    weathers = sorted(df["weather"].unique().tolist())
    selected_zones = filter_col1.multiselect("Zone", zones, default=zones)
    selected_weather = filter_col2.multiselect("Weather", weathers, default=weathers)
    hour_range = filter_col3.slider("Hour Range", min_value=0, max_value=23, value=(0, 23))

    filtered = df[
        (df["zone"].isin(selected_zones))
        & (df["weather"].isin(selected_weather))
        & (df["hour"].between(hour_range[0], hour_range[1]))
    ]

    st.subheader("SLA Performance by Hour")
    hour_agg = (
        filtered.groupby("hour", as_index=False)
        .agg(orders=("order_id", "count"), sla_pct=("sla_10_min", lambda x: x.mean() * 100))
        .sort_values("hour")
    )
    fig_hour = px.bar(hour_agg, x="hour", y="orders", title="Order Volume by Hour")
    st.plotly_chart(fig_hour, use_container_width=True)
    fig_hour_sla = px.line(hour_agg, x="hour", y="sla_pct", markers=True, title="SLA% by Hour")
    fig_hour_sla.update_yaxes(range=[0, 100])
    st.plotly_chart(fig_hour_sla, use_container_width=True)

    lcol, rcol = st.columns(2)
    with lcol:
        st.subheader("High-Demand Location Performance")
        zone_agg = (
            filtered.groupby("zone", as_index=False)
            .agg(
                orders=("order_id", "count"),
                sla_pct=("sla_10_min", lambda x: x.mean() * 100),
                avg_delivery=("order_to_delivery_min", "mean"),
            )
            .sort_values("orders", ascending=False)
        )
        zone_agg["avg_delivery"] = zone_agg["avg_delivery"].round(2)
        zone_agg["sla_pct"] = zone_agg["sla_pct"].round(2)
        st.dataframe(zone_agg, use_container_width=True)
        fig_zone = px.scatter(
            zone_agg,
            x="orders",
            y="sla_pct",
            size="avg_delivery",
            color="zone",
            title="Demand vs SLA by Zone",
        )
        fig_zone.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_zone, use_container_width=True)

    with rcol:
        st.subheader("Delivery Partner Efficiency")
        partner = (
            filtered.groupby("partner_id", as_index=False)
            .agg(
                orders=("order_id", "count"),
                sla_pct=("sla_10_min", lambda x: x.mean() * 100),
                avg_delivery=("order_to_delivery_min", "mean"),
                avg_distance=("distance_km", "mean"),
            )
            .query("orders >= 80")
            .sort_values("sla_pct", ascending=False)
        )
        partner["sla_pct"] = partner["sla_pct"].round(2)
        partner["avg_delivery"] = partner["avg_delivery"].round(2)
        partner["avg_distance"] = partner["avg_distance"].round(2)
        st.dataframe(partner.head(20), use_container_width=True)

        fig_partner = px.scatter(
            partner,
            x="avg_distance",
            y="avg_delivery",
            color="sla_pct",
            size="orders",
            title="Partner Avg Distance vs Delivery Time",
        )
        st.plotly_chart(fig_partner, use_container_width=True)

    st.subheader("Root Causes of Delay")
    root = summary.get("breach_component_share", {})
    root_df = pd.DataFrame(
        {
            "Component": ["Preparation", "Pickup Wait", "Travel"],
            "Share %": [
                root.get("prep_component_share_pct", 0),
                root.get("pickup_component_share_pct", 0),
                root.get("travel_component_share_pct", 0),
            ],
        }
    )
    fig_root = px.pie(root_df, names="Component", values="Share %", hole=0.45, title="Delay Contribution Mix")
    st.plotly_chart(fig_root, use_container_width=True)

    st.subheader("Model-Identified Delay Drivers")
    drv = summary.get("top_delay_drivers", {})
    drv_df = pd.DataFrame({"feature": list(drv.keys()), "importance": list(drv.values())}).sort_values(
        "importance", ascending=False
    )
    fig_drv = px.bar(drv_df.head(12), x="importance", y="feature", orientation="h", title="Top Delay Drivers")
    st.plotly_chart(fig_drv, use_container_width=True)

    st.caption("Model quality: MAE and R² are available in outputs/analysis_summary.json")


if __name__ == "__main__":
    main()
