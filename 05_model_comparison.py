"""
05_model_comparison.py

Pulls the three forecasts together for the same store + window and produces
the comparison table that goes into the dashboard.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"
TARGET_STORE = 1


def metrics(y, p):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "MAPE": float(mean_absolute_percentage_error(y, p) * 100),
        "Bias": float(np.mean(p - y)),
    }


def main():
    cls = pd.read_csv(PROC / "forecast_classical.csv", parse_dates=["Date"])
    ml = pd.read_csv(PROC / "forecast_ml.csv", parse_dates=["Date"])

    # filter ML to the same store as classical for an apples-to-apples comparison
    ml_store = ml[ml["Store"] == TARGET_STORE].copy()

    merged = cls.merge(
        ml_store[["Date", "Store", "xgb_pred"]],
        on=["Date", "Store"],
        how="inner",
    )

    rows = [
        ("SARIMA", metrics(merged["actual"], merged["sarima_pred"])),
        ("Prophet", metrics(merged["actual"], merged["prophet_pred"])),
        ("XGBoost", metrics(merged["actual"], merged["xgb_pred"])),
    ]

    print("\nMODEL COMPARISON — store", TARGET_STORE)
    print("-" * 50)
    print(f"{'Model':<10} {'RMSE':>8} {'MAPE %':>8} {'Bias':>10}")
    print("-" * 50)
    for name, m in rows:
        print(f"{name:<10} {m['RMSE']:>8.0f} {m['MAPE']:>8.2f} {m['Bias']:>10.0f}")

    # save tidy version for tableau
    summary = pd.DataFrame([
        {"model": name, **m} for name, m in rows
    ])
    summary.to_csv(PROC / "model_comparison.csv", index=False)

    # combined long-format predictions for a single tableau view
    long = pd.concat([
        merged[["Date", "actual"]].assign(model="actual", value=merged["actual"]),
        merged[["Date"]].assign(model="SARIMA", value=merged["sarima_pred"]),
        merged[["Date"]].assign(model="Prophet", value=merged["prophet_pred"]),
        merged[["Date"]].assign(model="XGBoost", value=merged["xgb_pred"]),
    ], ignore_index=True)
    long = long[["Date", "model", "value"]]
    long.to_csv(PROC / "predictions_long.csv", index=False)

    # plot
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(merged["Date"], merged["actual"], color="black", linewidth=2, label="Actual")
    ax.plot(merged["Date"], merged["sarima_pred"], "--", label="SARIMA", alpha=0.8)
    ax.plot(merged["Date"], merged["prophet_pred"], "--", label="Prophet", alpha=0.8)
    ax.plot(merged["Date"], merged["xgb_pred"], "--", label="XGBoost", alpha=0.8)
    ax.set_title(f"All three models — store {TARGET_STORE}, 6-week holdout")
    ax.set_ylabel("Sales (€)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PROC / "model_comparison.png", dpi=120)
    plt.close()

    print(f"\nsaved -> {PROC / 'model_comparison.csv'}")
    print(f"saved -> {PROC / 'predictions_long.csv'}")


if __name__ == "__main__":
    main()
