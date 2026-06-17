"""
06_financial_model.py

Translates the forecast into a strategic recommendation. The story:

  Top-quartile traffic stores not yet on Promo2 show ~6% lift in our XGBoost
  feature importance + counterfactual analysis. If we extend Promo2 to that
  group we project an annual revenue uplift, against a fixed setup cost +
  ongoing margin compression.

  Build a 5-year DCF, compute NPV and IRR, run sensitivity.

Numbers below are illustrative (assumed cost structures) — the *forecast*
inputs come from the model. This is the pattern stakeholders care about
seeing: forecast -> P&L -> investment case.
"""

import pandas as pd
import numpy as np
import numpy_financial as npf
from pathlib import Path

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# --- assumptions ----------------------------------------------------
N_STORES_ELIGIBLE = 280            # top quartile by traffic, not on Promo2
AVG_DAILY_SALES = 8_500            # € — rough avg from the data
DAYS_OPEN_PER_YEAR = 305
UPLIFT_PCT = 0.06                  # 6% from model counterfactual
MARGIN_HIT_PCT = 0.015             # 1.5% margin compression from discounting
GROSS_MARGIN = 0.28
SETUP_COST_PER_STORE = 18_000      # marketing + system + training
ANNUAL_RUN_COST_PER_STORE = 4_200
DISCOUNT_RATE = 0.10
HORIZON_YEARS = 5
EUR_TO_GBP = 0.86                  # rough fx
# --------------------------------------------------------------------


def project_cashflows(uplift_pct=UPLIFT_PCT,
                      margin_hit=MARGIN_HIT_PCT,
                      discount_rate=DISCOUNT_RATE):
    base_annual_revenue = AVG_DAILY_SALES * DAYS_OPEN_PER_YEAR * N_STORES_ELIGIBLE
    incremental_revenue = base_annual_revenue * uplift_pct

    # gross margin on the new revenue, less margin compression on the base
    incremental_gp = incremental_revenue * GROSS_MARGIN
    base_margin_drag = base_annual_revenue * margin_hit

    annual_run_cost = ANNUAL_RUN_COST_PER_STORE * N_STORES_ELIGIBLE
    annual_net = incremental_gp - base_margin_drag - annual_run_cost

    setup = SETUP_COST_PER_STORE * N_STORES_ELIGIBLE

    # year 0 = -setup, years 1..5 = annual_net (assume flat for simplicity,
    # could grow with a small growth rate but keeping conservative)
    cashflows_eur = [-setup] + [annual_net] * HORIZON_YEARS
    cashflows_gbp = [c * EUR_TO_GBP for c in cashflows_eur]

    npv = npf.npv(discount_rate, cashflows_gbp)
    irr = npf.irr(cashflows_gbp)

    return {
        "base_annual_rev_eur": base_annual_revenue,
        "incremental_rev_eur": incremental_revenue,
        "annual_net_eur": annual_net,
        "setup_eur": setup,
        "cashflows_gbp": cashflows_gbp,
        "npv_gbp": npv,
        "irr": irr,
    }


def sensitivity():
    rows = []
    for uplift in [0.04, 0.05, 0.06, 0.07, 0.08]:
        for dr in [0.08, 0.10, 0.12]:
            r = project_cashflows(uplift_pct=uplift, discount_rate=dr)
            rows.append({
                "uplift_pct": uplift,
                "discount_rate": dr,
                "npv_gbp_m": round(r["npv_gbp"] / 1e6, 2),
                "irr_pct": round(r["irr"] * 100, 1),
            })
    return pd.DataFrame(rows)


def main():
    base = project_cashflows()

    print("=" * 60)
    print("BASE CASE")
    print("=" * 60)
    print(f"eligible stores: {N_STORES_ELIGIBLE}")
    print(f"base annual revenue (€):  {base['base_annual_rev_eur']:>15,.0f}")
    print(f"incremental revenue (€):  {base['incremental_rev_eur']:>15,.0f}")
    print(f"annual net (€):           {base['annual_net_eur']:>15,.0f}")
    print(f"upfront setup (€):        {base['setup_eur']:>15,.0f}")
    print()
    print(f"NPV (£): {base['npv_gbp']:>14,.0f}  ({base['npv_gbp']/1e6:.2f}M)")
    print(f"IRR:     {base['irr']*100:>14.1f}%")

    # save cashflows for tableau
    cf = pd.DataFrame({
        "year": list(range(HORIZON_YEARS + 1)),
        "cashflow_gbp": base["cashflows_gbp"],
    })
    cf["cumulative_gbp"] = cf["cashflow_gbp"].cumsum()
    cf.to_csv(PROC / "financial_cashflows.csv", index=False)

    sens = sensitivity()
    sens.to_csv(PROC / "financial_sensitivity.csv", index=False)

    print("\nsensitivity table:")
    print(sens.to_string(index=False))

    print(f"\nsaved -> {PROC / 'financial_cashflows.csv'}")
    print(f"saved -> {PROC / 'financial_sensitivity.csv'}")


if __name__ == "__main__":
    main()
