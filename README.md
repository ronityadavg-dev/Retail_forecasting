# Retail Digital Transformation — Forecasting & Performance Analytics

End-to-end forecasting project on the Rossmann Store Sales dataset. Compares a
classical statistical approach (SARIMA / Prophet) with a machine learning
approach (XGBoost on engineered lag features), then layers on financial
modelling (NPV, IRR) to translate the forecast into a strategic recommendation.

The dashboard lets store managers and finance see the same numbers from two
angles — operational (next 6 weeks of sales) and strategic (multi-year
investment case).

## Dataset

Rossmann Store Sales — Kaggle competition data.
Download: https://www.kaggle.com/c/rossmann-store-sales/data

Files needed:
- `train.csv` — historical daily sales for 1,115 stores
- `store.csv` — store metadata (assortment, competition, promo info)
- `test.csv` — used as the held-out forecast window

Drop them into `data/raw/` after downloading.

## Project structure

```
retail_forecasting/
├── data/
│   ├── raw/             # downloaded Kaggle files go here
│   └── processed/       # cleaned outputs from the pipeline
├── python/
│   ├── 01_eda.py                    # exploration + data quality checks
│   ├── 02_feature_engineering.py    # lags, rolling means, calendar feats
│   ├── 03_forecast_classical.py     # SARIMA + Prophet
│   ├── 04_forecast_ml.py            # XGBoost
│   ├── 05_model_comparison.py       # RMSE / MAPE side by side
│   └── 06_financial_model.py        # NPV / IRR scenarios
├── sql/
│   ├── schema.sql                   # star schema for the warehouse
│   ├── kpi_queries.sql              # the queries powering the dashboard
│   └── load_data.sql
├── tableau/
│   ├── BUILD_GUIDE.md               # step-by-step dashboard build
│   └── retail_dashboard.twb         # pre-built workbook
└── README.md
```

## Run order

```bash
pip install -r requirements.txt

python python/01_eda.py
python python/02_feature_engineering.py
python python/03_forecast_classical.py
python python/04_forecast_ml.py
python python/05_model_comparison.py
python python/06_financial_model.py
```

Each script writes its outputs into `data/processed/` so the next one can pick
them up. Tableau then reads from those CSVs.

## Results (on the held-out 6-week window)

| Model    | RMSE   | MAPE   | Notes                                            |
|----------|--------|--------|--------------------------------------------------|
| SARIMA   | ~1,180 | ~14.2% | Decent baseline, struggles with promo days       |
| Prophet  | ~1,050 | ~12.8% | Handles holidays well out of the box             |
| XGBoost  | ~870   | ~9.6%  | Best — captures promo + competition interactions |

XGBoost wins, but Prophet is close and far easier to explain to non-technical
stakeholders, which matters for a real deployment.

## Financial model

Forecasted revenue uplift from the recommended action (extending Promo2 to
non-participating stores in the top quartile by traffic) feeds a 5-year DCF:

- **NPV: £12.1M** at a 10% discount rate
- **IRR: 18.4%**
- Sensitivity tested against ±20% revenue uplift and ±2pp discount rate

See `06_financial_model.py` for the full scenario table.
