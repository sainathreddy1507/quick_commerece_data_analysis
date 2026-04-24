# Quick Commerce (10-Min Delivery) Analysis

Professional end-to-end project for analyzing ultra-fast delivery performance (Zepto/Blinkit style), including:
- Realistic synthetic data generation
- SLA performance analysis
- Peak hour and high-demand location analysis
- Delivery partner efficiency benchmarking
- Root-cause decomposition for delays
- Interactive dashboard for presentation

## Project Structure

- `src/generate_data.py` - Builds synthetic order-level operations data
- `src/analyze.py` - Computes KPIs, root causes, and delay-driver model insights
- `app.py` - Streamlit dashboard
- `data/quick_commerce_orders.csv` - Generated dataset
- `outputs/analysis_summary.json` - KPI and model summary
- `outputs/*.csv` - Aggregated outputs for reporting

## Business Questions Covered

1. What is the current 10-minute SLA performance?
2. Which hours are peak demand, and how does SLA vary by hour?
3. Which locations drive most demand and delays?
4. Which delivery partners are most/least efficient?
5. What are the primary root causes of SLA breaches?

## Data Design (Synthetic But Realistic)

The synthetic generator simulates:
- Multi-zone operations (8 urban zones)
- Hourly demand variability (lunch + evening spikes)
- Weekday/weekend effects
- Weather impact (`clear`, `rain`, `storm`)
- Zone-level congestion index
- Delivery partner skill/efficiency variability
- Time components:
  - Preparation time
  - Pickup wait time
  - Travel time

SLA is defined as: `order_to_delivery_min <= 10`.

## Algorithms and Methods

- **Descriptive analytics**:
  - SLA %, P90 delivery time, average delay, demand distribution
- **Root cause analysis**:
  - Contribution share of prep, pickup, travel among breached orders
- **Predictive modeling**:
  - `RandomForestRegressor` to estimate delivery time
  - Feature importance used as explainable delay drivers
  - Evaluation metrics: MAE and R²

## Setup and Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate synthetic data:

```bash
python src/generate_data.py
```

3. Run analysis pipeline:

```bash
python src/analyze.py
```

4. Launch dashboard:

```bash
streamlit run app.py
```

## Presentation Guidance

For final presentation/demo:
- Start with KPI cards (SLA %, average delivery, P90)
- Show peak-hour bottlenecks and location hotspots
- Compare top and bottom partner cohorts
- Explain root-cause pie chart (prep vs pickup vs travel)
- Use model delay drivers to propose targeted interventions

## Suggested Operational Actions

- Increase partner supply in peak-hour/peak-zone windows
- Reduce pickup wait via better store readiness and batching
- Route optimization for longer distance during traffic spikes
- Targeted coaching/incentive plans for low-performing partners
- Weather-triggered dynamic SLA buffers and dispatch rules
