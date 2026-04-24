from __future__ import annotations

import json
import csv
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


app = FastAPI(title="Quick Commerce SLA API")

SUMMARY_PATH = Path("outputs/analysis_summary.json")
LOCATION_PATH = Path("outputs/location_performance.csv")
PEAK_PATH = Path("outputs/peak_hours.csv")


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        return {}
    return json.loads(SUMMARY_PATH.read_text())


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.get("/api/summary")
def summary() -> JSONResponse:
    data = _load_summary()
    if not data:
        return JSONResponse(
            {"error": "summary_not_found", "message": "Run analysis locally and commit outputs."},
            status_code=404,
        )
    return JSONResponse(data)


@app.get("/api/location-performance")
def location_performance() -> JSONResponse:
    return JSONResponse(_load_csv(LOCATION_PATH))


@app.get("/api/peak-hours")
def peak_hours() -> JSONResponse:
    return JSONResponse(_load_csv(PEAK_PATH))


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    data = _load_summary()
    if not data:
        return HTMLResponse(
            """
            <html><body style="font-family:Arial;padding:24px">
            <h2>Quick Commerce SLA Dashboard</h2>
            <p>Data not found. Run analysis locally and push <code>outputs/analysis_summary.json</code>.</p>
            </body></html>
            """,
            status_code=200,
        )

    k = data.get("kpis", {})
    model = data.get("model_metrics", {})
    return HTMLResponse(
        f"""
        <html>
        <head><title>Quick Commerce SLA Dashboard</title></head>
        <body style="font-family:Arial, sans-serif; padding:24px; max-width:900px; margin:auto;">
          <h1>Quick Commerce 10-Min SLA Dashboard</h1>
          <p>Vercel deployment endpoint for project insights.</p>
          <h2>Core KPIs</h2>
          <ul>
            <li>Total Orders: <b>{k.get("total_orders", "NA")}</b></li>
            <li>Avg Delivery Time: <b>{k.get("avg_order_to_delivery_min", "NA")} min</b></li>
            <li>P90 Delivery Time: <b>{k.get("p90_delivery_min", "NA")} min</b></li>
            <li>SLA within 10 min: <b>{k.get("sla_10_min_rate_pct", "NA")}%</b></li>
            <li>Avg Delay (Breached): <b>{k.get("avg_delay_if_breached_min", "NA")} min</b></li>
          </ul>
          <h2>Model Quality</h2>
          <ul>
            <li>MAE: <b>{model.get("model_mae_min", "NA")} min</b></li>
            <li>R²: <b>{model.get("model_r2", "NA")}</b></li>
          </ul>
          <h2>API Endpoints</h2>
          <ul>
            <li><a href="/api/summary">/api/summary</a></li>
            <li><a href="/api/location-performance">/api/location-performance</a></li>
            <li><a href="/api/peak-hours">/api/peak-hours</a></li>
          </ul>
        </body>
        </html>
        """
    )
