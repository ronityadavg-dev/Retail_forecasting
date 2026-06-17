"""
03_forecast_classical.py — SARIMA + Prophet.

Strategy: pick a representative store (one with full history, "average"
behaviour) and fit both models on it. In a real project you'd loop over all
1,115 stores or cluster them first — for this portfolio version one good
example is enough to demonstrate the technique and saves hours of compute.

Holding out the last 6 weeks as the test window, same as the Kaggle
competition setup.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# prophet's import is slow + chatty
warnings.filterwarnings("ignore")
from prophet import Prophet  # noqa

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

# pick a store — store 1 has clean history and isn't an outlier
TARGET_STORE = 1
HOLDOUT_DAYS = 42  # 6 weeks


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100)


def get_store_series(df, store_id):
    s = df[df["Store"] == store_id].copy()
    s = s.sort_values("Date").set_index("Date")
    # forward fill to a continuous daily index — store closed days got dropped earlier
    full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
    s = s.reindex(full_idx)
    s["Sales"] = s["Sales"].interpolate(method="linear").ffill().bfill()
    return s


def fit_sarima(train_y, test_y):
    # weekly seasonality is the obvious one for retail
    # order chosen from a small grid search done offline — hardcoding here
    model = SARIMAX(
        train_y,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.forecast(steps=len(test_y))
    return fc


def fit_prophet(train_df, test_df):
    # prophet wants columns named ds and y
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    # add German holiday support — Rossmann is a German chain
    m.add_country_holidays(country_name="DE")

    m.fit(train_df)

    future = m.make_future_dataframe(periods=len(test_df), freq="D")
    forecast = m.predict(future)
    yhat = forecast.set_index("ds").loc[test_df["ds"]]["yhat"].values
    return yhat, m, forecast


def main():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["Date"])
    series = get_store_series(df, TARGET_STORE)

    train_y = series.iloc[:-HOLDOUT_DAYS]["Sales"]
    test_y = series.iloc[-HOLDOUT_DAYS:]["Sales"]

    print(f"store {TARGET_STORE}: {len(train_y)} train days, {len(test_y)} test days")

    # ===== SARIMA =====
    print("\nfitting SARIMA(1,1,1)(1,1,1,7)...")
    sarima_pred = fit_sarima(train_y, test_y)
    print(f"  RMSE: {rmse(test_y, sarima_pred):.0f}")
    print(f"  MAPE: {mape(test_y, sarima_pred):.2f}%")

    # ===== Prophet =====
    print("\nfitting Prophet...")
    prophet_train = train_y.reset_index().rename(columns={"index": "ds", "Sales": "y"})
    prophet_test = test_y.reset_index().rename(columns={"index": "ds", "Sales": "y"})
    prophet_pred, m, full_fc = fit_prophet(prophet_train, prophet_test)
    print(f"  RMSE: {rmse(test_y, prophet_pred):.0f}")
    print(f"  MAPE: {mape(test_y, prophet_pred):.2f}%")

    # save predictions for the comparison + dashboard
    out = pd.DataFrame({
        "Date": test_y.index,
        "Store": TARGET_STORE,
        "actual": test_y.values,
        "sarima_pred": np.asarray(sarima_pred),
        "prophet_pred": prophet_pred,
    })
    out.to_csv(PROC / "forecast_classical.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_y.index, test_y.values, label="actual", color="black", linewidth=2)
    ax.plot(test_y.index, sarima_pred, label="SARIMA", linestyle="--")
    ax.plot(test_y.index, prophet_pred, label="Prophet", linestyle="--")
    ax.set_title(f"Store {TARGET_STORE} — 6-week forecast (classical models)")
    ax.set_ylabel("Sales (€)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROC / "forecast_classical.png", dpi=120)
    plt.close()

    print(f"\nsaved -> {PROC / 'forecast_classical.csv'}")


if __name__ == "__main__":
    main()
