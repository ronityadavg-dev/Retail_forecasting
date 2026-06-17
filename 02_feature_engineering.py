"""
02_feature_engineering.py

Builds the feature set used by both the ML model and (partially) by Prophet
as regressors. Focus is on:
  - calendar features (day, month, week, isweekend)
  - lag features (sales 7, 14, 28 days ago)
  - rolling stats (7d / 28d mean & std)
  - store-level metadata joined in

I'm being a bit careful with the lag features to avoid leakage — they're
computed within each store group, sorted by date.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def load_and_merge():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(RAW / "store.csv")

    # drop closed days and the weird zero-sales-while-open rows
    train = train[train["Open"] == 1].copy()
    train = train[train["Sales"] > 0].copy()

    df = train.merge(store, on="Store", how="left")
    return df


def add_calendar_features(df):
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dow"] = df["Date"].dt.dayofweek
    df["weekofyear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    return df


def add_lag_features(df):
    # sort once, then shift within each store
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    for lag in [7, 14, 28]:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    # rolling stats — shift first so we don't include today's value
    for window in [7, 28]:
        df[f"sales_roll_mean_{window}"] = (
            df.groupby("Store")["Sales"].shift(1).rolling(window).mean().reset_index(0, drop=True)
        )
        df[f"sales_roll_std_{window}"] = (
            df.groupby("Store")["Sales"].shift(1).rolling(window).std().reset_index(0, drop=True)
        )

    return df


def encode_categoricals(df):
    # store metadata — keep it simple, ordinal where it makes sense
    assortment_map = {"a": 0, "b": 1, "c": 2}
    storetype_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    state_map = {"0": 0, "a": 1, "b": 2, "c": 3}

    df["Assortment_enc"] = df["Assortment"].map(assortment_map)
    df["StoreType_enc"] = df["StoreType"].map(storetype_map)
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df["StateHoliday_enc"] = df["StateHoliday"].map(state_map).fillna(0).astype(int)

    # competition distance — fill with median, missing usually means no nearby competitor
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    return df


def main():
    print("loading + merging...")
    df = load_and_merge()
    print(f"  rows after filter: {len(df):,}")

    print("adding calendar features...")
    df = add_calendar_features(df)

    print("adding lag features (this takes a sec)...")
    df = add_lag_features(df)

    print("encoding categoricals...")
    df = encode_categoricals(df)

    # drop early rows where lags are NaN
    before = len(df)
    df = df.dropna(subset=["sales_lag_28", "sales_roll_mean_28"])
    print(f"  dropped {before - len(df):,} rows with missing lags")

    out_path = OUT / "features.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}  ({len(df):,} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
