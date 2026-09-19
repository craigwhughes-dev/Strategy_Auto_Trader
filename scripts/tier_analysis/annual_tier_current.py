"""Year-by-year returns of the CURRENT tier rule (VXN enter<=23 / hold<=24, lower tiers off) vs Nasdaq and cash.

Run: uv run python scripts/tier_analysis/annual_current.py
Reads data_synthetic/hourly_spliced/ (build it first: see HANDOFF_26yr_rerun.md). Costs: flat 13 bps per switch.
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
tiers = eng.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)  # current config: Nasdaq or cash, lower tiers off
run = eng.simulate(inp, tiers)
nas = eng.buy_and_hold(inp, "NASDAQ")
cash = eng.buy_and_hold(inp, "CASH")

# EQGB (live, hedged) leg from its real start
raw = pd.read_csv(ROOT + "data/cache/ibkr_hourly_deep/EQGB.L.csv", index_col=0)
raw.index = pd.to_datetime(raw.index, utc=True)
close, _ = clean_fund_bars(raw["Close"].astype(float))
eqgb = pd.Series(close.to_numpy(), index=bar_end(close.index, LSE)).groupby(level=0).last()
logp = np.log(eqgb.reindex(inp.grid, method="ffill").to_numpy())
r_eqgb = np.concatenate([[0.0], np.diff(logp)])
start = inp.grid.searchsorted(eqgb.index[0])
hedged = dataclasses.replace(inp, log_ret=np.column_stack([np.where(np.arange(len(inp.grid)) >= start, r_eqgb, inp.log_ret[:, 0]), inp.log_ret[:, 1:]]))
run_h = eng.simulate(hedged, tiers)
nas_h = eng.buy_and_hold(hedged, "NASDAQ")

days = pd.DatetimeIndex(inp.days)
years = days.year
in_nas_bar = (tiers == 0)
share = pd.Series(in_nas_bar.astype(float)).groupby(years[inp.day_codes]).mean()
vxn_mean = pd.Series(inp.vxn).groupby(years[inp.day_codes]).mean()


def per_year(r):
    s = pd.Series(r.daily, index=days)
    ret = s.groupby(s.index.year).apply(lambda x: float(np.prod(1 + x) - 1) * 100)
    def maxdd(x):
        c = np.cumprod(1 + x.to_numpy())
        return float((c / np.maximum.accumulate(np.concatenate([[1.0], c]))[1:] - 1).min()) * 100
    dd = s.groupby(s.index.year).apply(maxdd)
    return ret, dd


ret, dd = per_year(run)
nret, ndd = per_year(nas)
cret, _ = per_year(cash)
hret, hdd = per_year(run_h)
nhret, _ = per_year(nas_h)
sw = pd.Series(run.switches, index=days).groupby(days.year).sum()

df = pd.DataFrame({
    "strategy": ret, "Nasdaq": nret, "cash": cret, "vs Nasdaq": ret - nret,
    "strat DD": dd, "Nasdaq DD": ndd, "% in Nasdaq": share * 100, "switches": sw, "avg VXN": vxn_mean,
})
df["data"] = ["bridged" if y < 2007 else ("bridged->real" if y == 2007 else "real") for y in df.index]
df["cum strat"] = (1 + df["strategy"] / 100).cumprod()
df["cum Nasdaq"] = (1 + df["Nasdaq"] / 100).cumprod()
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df.round(1).to_string())

print("\nEQGB (live hedged fund) leg, real hourly since 2017-10: strategy % vs EQGB buy&hold %")
h = pd.DataFrame({"strategy": hret, "EQGB": nhret, "strat DD": hdd}).loc[2018:]
print(h.round(1).to_string())

print("\nfull period:")
for label, r in (("strategy", run), ("Nasdaq", nas)):
    s = eng.window_stats(r, "all")
    yrs = s["n_days"] / 252
    print(f"  {label:9s} total {s['ret_pct']:+.0f}%  per yr {((1 + s['ret_pct'] / 100) ** (1 / yrs) - 1) * 100:+.1f}%  xSharpe {s['xsharpe']:+.2f}  maxDD {s['max_dd_pct']:+.1f}%  switches/yr {s['sw_per_yr']:.1f}  ({yrs:.1f} yrs)")
wins = int((df["vs Nasdaq"] > 0).sum())
print(f"  years beating Nasdaq: {wins} of {len(df)}; years positive: {int((df['strategy'] > 0).sum())} of {len(df)}; years negative: {int((df['strategy'] < 0).sum())}")
print(f"  worst year strategy {df['strategy'].min():+.1f}% ({df['strategy'].idxmin()}), best {df['strategy'].max():+.1f}% ({df['strategy'].idxmax()}); Nasdaq worst {df['Nasdaq'].min():+.1f}% ({df['Nasdaq'].idxmin()})")
