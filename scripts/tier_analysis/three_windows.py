"""Deployed rule vs comparators over three windows: 26 yr, since VXN data (2007-11), recent real (2024-03).

Run: uv run python scripts/tier_analysis/three_windows.py
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, so the package imports work when run as a script

import dataclasses

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.allocation import intraday_engine as eng
from Strategy_Auto_Trader.core.trading_sessions import LSE, bar_end
from Strategy_Auto_Trader.synthetic_backtest_data.real_clean import clean_fund_bars

ROOT = str(pathlib.Path(__file__).resolve().parents[2]) + "/"
inp = eng.load_inputs()
eng.WINDOWS.update({"y26": (None, None), "vxn_era": ("2007-11-20", None), "recent": ("2024-03-25", None)})
WINDOWS = [("26 yr, incl. bridged 1999-2007", "y26"), ("since VXN data 2007-11-20", "vxn_era"), ("recent real 2024-03-25", "recent")]

# EQGB (live, GBP-hedged) leg, real hourly from 2017-10
raw = pd.read_csv(ROOT + "data/cache/ibkr_hourly_deep/EQGB.L.csv", index_col=0)
raw.index = pd.to_datetime(raw.index, utc=True)
close, _ = clean_fund_bars(raw["Close"].astype(float))
eqgb = pd.Series(close.to_numpy(), index=bar_end(close.index, LSE)).groupby(level=0).last()
logp = np.log(eqgb.reindex(inp.grid, method="ffill").to_numpy())
r_eqgb = np.concatenate([[0.0], np.diff(logp)])
start = inp.grid.searchsorted(eqgb.index[0])
hedged = dataclasses.replace(inp, log_ret=np.column_stack([np.where(np.arange(len(inp.grid)) >= start, r_eqgb, inp.log_ret[:, 0]), inp.log_ret[:, 1:]]))


def deployed(vxn_lo, vxn_hi, vix1=15.0, vix2=17.5):
    """As the daemon is configured: Nasdaq with a VXN deadband, else the VIX ladder (S&P, FTSE), else cash."""
    ladder = eng.tiers_vxn_vix(np.full(len(inp.vix), np.nan), inp.vix, 0.0, vix1, vix2)
    nasdaq = eng.tiers_vxn_deadband(inp.vxn, vxn_lo, vxn_hi) == 0
    return np.where(nasdaq, 0, ladder)


def stat_row(run, window):
    s = eng.window_stats(run, window)
    years = s["n_days"] / 252
    cagr = ((1 + s["ret_pct"] / 100) ** (1 / years) - 1) * 100
    return s, years, cagr


rules = {
    "DEPLOYED  VXN enter<=23 / hold<=24, VIX 15/17.5": deployed(23.0, 24.0),
    "two-state VXN enter<=23 / hold<=24 (no S&P/FTSE)": eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0),
    "plain VXN<=24 (no deadband)": eng.tiers_vxn_deadband(inp.vxn, 24.0, 24.0),
    "old thresholds VXN<=18, VIX 15/17.5 (plain)": eng.tiers_vxn_vix(inp.vxn, inp.vix, 18.0, 15.0, 17.5),
}

for label, window in WINDOWS:
    print(f"\n=== {label}")
    print(f"  {'rule':52s} | ret%     CAGR%  xSharpe rawSh  maxDD%  sw/yr | next-bar: ret% xSh")
    for name, tiers in rules.items():
        s, years, cagr = stat_row(eng.simulate(inp, tiers), window)
        n, _, _ = stat_row(eng.simulate(inp, tiers, "next_bar"), window)
        print(f"  {name:52s} | {s['ret_pct']:+7.0f} {cagr:+6.1f}  {s['xsharpe']:+.2f}  {s['sharpe']:+.2f} {s['max_dd_pct']:+7.1f} {s['sw_per_yr']:6.1f} |   {n['ret_pct']:+6.0f} {n['xsharpe']:+.2f}")
    for asset in ("NASDAQ", "SP500", "CASH"):
        s, years, cagr = stat_row(eng.buy_and_hold(inp, asset), window)
        print(f"  {'buy & hold ' + asset:52s} | {s['ret_pct']:+7.0f} {cagr:+6.1f}  {s['xsharpe']:+.2f}  {s['sharpe']:+.2f} {s['max_dd_pct']:+7.1f}     - |")
    print(f"  ({years:.1f} years, {s['n_days']} trading days)")

print("\n=== recent real window with the LIVE Nasdaq fund (EQGB, GBP-hedged, real hourly) instead of EQQQ")
tiers = rules["DEPLOYED  VXN enter<=23 / hold<=24, VIX 15/17.5"]
for label, source in (("EQQQ leg", inp), ("EQGB leg", hedged)):
    for fill in ("same_bar", "next_bar"):
        s, years, cagr = stat_row(eng.simulate(source, tiers, fill), "recent")
        print(f"  deployed, {label}, {fill:8s}: ret {s['ret_pct']:+.0f}%  CAGR {cagr:+.1f}%  xSharpe {s['xsharpe']:+.2f}  maxDD {s['max_dd_pct']:+.1f}%  sw/yr {s['sw_per_yr']:.1f}")
for label, source in (("EQQQ", inp), ("EQGB", hedged)):
    s, years, cagr = stat_row(eng.buy_and_hold(source, "NASDAQ"), "recent")
    print(f"  buy & hold {label}: ret {s['ret_pct']:+.0f}%  CAGR {cagr:+.1f}%  xSharpe {s['xsharpe']:+.2f}  maxDD {s['max_dd_pct']:+.1f}%")

tiers_dep = rules["DEPLOYED  VXN enter<=23 / hold<=24, VIX 15/17.5"]
share = np.bincount(tiers_dep, minlength=4) / len(tiers_dep)
print("\ntime in tier (deployed rule, all bars): " + "  ".join(f"{a} {s:.1%}" for a, s in zip(eng.ASSETS, share)))

print("\n=== annual returns %: deployed rule vs buy & hold (EQQQ leg)")
tab = pd.DataFrame({
    "deployed": eng.annual_returns(eng.simulate(inp, tiers_dep)),
    "Nasdaq": eng.annual_returns(eng.buy_and_hold(inp, "NASDAQ")),
    "cash": eng.annual_returns(eng.buy_and_hold(inp, "CASH")),
})
tab["data"] = ["bridged" if y < 2008 else "real" for y in tab.index]
tab.loc[2007, "data"] = "bridged/real"
print(tab.to_string())
