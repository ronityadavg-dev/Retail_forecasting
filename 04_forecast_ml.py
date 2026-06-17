"""
04_forecast_ml.py — XGBoost on the engineered feature set.

This one trains across all stores at once (panel approach) — the lag features
+ store metadata give the model enough context to differentiate. Same 6-week
holdout as the classical models so the comparison is fair.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"

HOLDOUT_DAYS = 42

FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday",
    "year", "month", "day", "dow", "weekofyear",
    "is_weekend", "is_month_start", "is_month_end",
    "Assortment_enc", "StoreType_enc", "StateHoliday_enc",
    "CompetitionDistance",
    "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_28",
    "sales_roll_std_7", "sales_roll_std_28",
]
TARGET = "Sales"


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def mape(y, p):
    return float(mean_absolute_percentage_error(y, p) * 100)


def main():
    df = pd.read_csv(PROC / "features.csv", parse_dates=["Date"])
    df = df.sort_values("Date")

    cutoff = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df[df["Date"] <= cutoff].copy()
    test = df[df["Date"] > cutoff].copy()

    print(f"train: {len(train):,}  test: {len(test):,}")
    print(f"cutoff date: {cutoff.date()}")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # params from a small tuning round (early stopping decides n_estimators)
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        tree_method="hist",
        early_stopping_rounds=30,
    )

    print("\ntraining XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  best iter: {model.best_iteration}")

    preds = model.predict(X_test)
    print(f"\n  RMSE: {rmse(y_test, preds):.0f}")
    print(f"  MAPE: {mape(y_test, preds):.2f}%")

    # save
    out = test[["Date", "Store"]].copy()
    out["actual"] = y_test.values
    out["xgb_pred"] = preds
    out.to_csv(PROC / "forecast_ml.csv", index=False)

    # feature importance — useful for the dashboard / explanation
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    imp.to_csv(PROC / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    imp.head(15).plot(kind="barh", x="feature", y="importance", ax=ax, legend=False, color="seagreen")
    ax.invert_yaxis()
    ax.set_title("XGBoost feature importance — top 15")
    plt.tight_layout()
    plt.savefig(PROC / "feature_importance.png", dpi=120)
    plt.close()

    print(f"\nsaved -> {PROC / 'forecast_ml.csv'}")
    print(f"saved -> {PROC / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()
