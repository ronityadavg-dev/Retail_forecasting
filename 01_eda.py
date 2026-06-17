"""
01_eda.py — first look at the Rossmann data.

Goal here is just to understand what we're working with before doing anything
clever. I want to know: how clean is it, what's the seasonality look like,
are there obvious outliers, and which stores are weird.

Outputs a few plots into data/processed/eda/ and prints a summary to stdout.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# paths — keeping it simple, no config file needed for a project this size
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "eda"
OUT.mkdir(parents=True, exist_ok=True)


def load():
    train = pd.read_csv(RAW / "train.csv", parse_dates=["Date"], low_memory=False)
    store = pd.read_csv(RAW / "store.csv")
    return train, store


def basic_checks(train, store):
    print("=" * 60)
    print("SHAPES")
    print("=" * 60)
    print(f"train: {train.shape}")
    print(f"store: {store.shape}")
    print(f"date range: {train['Date'].min().date()} -> {train['Date'].max().date()}")
    print(f"unique stores: {train['Store'].nunique()}")

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    # train is usually clean, store has gaps in CompetitionDistance and Promo2 fields
    print("train:")
    print(train.isna().sum()[train.isna().sum() > 0])
    print("\nstore:")
    print(store.isna().sum()[store.isna().sum() > 0])

    print("\n" + "=" * 60)
    print("ZERO SALES DAYS")
    print("=" * 60)
    # closed days = sales 0. Open=0 should match.
    closed = train[train["Open"] == 0]
    zero_sales_open = train[(train["Open"] == 1) & (train["Sales"] == 0)]
    print(f"closed days: {len(closed):,}")
    print(f"open but zero sales (suspicious): {len(zero_sales_open):,}")
    # these zero-sales-while-open rows are usually refurbishments — drop later


def seasonality_plots(train):
    # aggregate to daily total across all stores — gives a feel for overall seasonality
    daily = train.groupby("Date")["Sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily["Date"], daily["Sales"], linewidth=0.6)
    ax.set_title("Total daily sales across all stores")
    ax.set_ylabel("Sales (€)")
    plt.tight_layout()
    plt.savefig(OUT / "01_daily_sales_total.png", dpi=120)
    plt.close()

    # day of week — Sundays are mostly closed in DE retail
    train["dow"] = train["Date"].dt.dayofweek
    dow_avg = train[train["Open"] == 1].groupby("dow")["Sales"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    dow_avg.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], rotation=0)
    ax.set_title("Avg sales by day of week (open days only)")
    ax.set_ylabel("Avg sales (€)")
    plt.tight_layout()
    plt.savefig(OUT / "02_sales_by_dow.png", dpi=120)
    plt.close()

    # month-of-year seasonality
    train["month"] = train["Date"].dt.month
    monthly = train[train["Open"] == 1].groupby("month")["Sales"].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    monthly.plot(kind="bar", ax=ax, color="darkorange")
    ax.set_title("Avg sales by month — Dec spike from Christmas")
    ax.set_ylabel("Avg sales (€)")
    plt.tight_layout()
    plt.savefig(OUT / "03_sales_by_month.png", dpi=120)
    plt.close()


def promo_effect(train):
    # quick check on whether promotions actually move the needle
    open_days = train[train["Open"] == 1]
    promo_lift = open_days.groupby("Promo")["Sales"].mean()
    pct = (promo_lift[1] / promo_lift[0] - 1) * 100
    print("\n" + "=" * 60)
    print("PROMO LIFT")
    print("=" * 60)
    print(f"avg sales no promo: €{promo_lift[0]:.0f}")
    print(f"avg sales with promo: €{promo_lift[1]:.0f}")
    print(f"lift: +{pct:.1f}%")


def main():
    train, store = load()
    basic_checks(train, store)
    seasonality_plots(train)
    promo_effect(train)
    print(f"\nplots saved to: {OUT}")


if __name__ == "__main__":
    main()
