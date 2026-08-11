import pandas as pd
import numpy as np

results = []
for k in [20, 35, 50, 70, 100]:
    pos = f"data/journals/k_sweep_k{k}_pos.csv"
    df = pd.read_csv(pos)
    summ = df[df["date"] == "SUMMARY"].iloc[0]
    daily_df = df[df["date"] != "SUMMARY"].copy()
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["portfolio_value"] = pd.to_numeric(daily_df["portfolio_value"])
    daily = daily_df.sort_values("date").set_index("date")["portfolio_value"].resample("B").last().ffill()
    rets = daily.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
    sortino = (rets.mean() / rets[rets < 0].std()) * np.sqrt(252)
    results.append({
        "k": k,
        "final": float(summ["portfolio_value"]),
        "pnl": float(summ["realized_pnl_cum"]),
        "max_dd": float(summ["max_drawdown"]),
        "trades": int(float(summ["n_admitted"])),
        "sharpe": sharpe,
        "sortino": sortino,
    })

print(f"{'k':>5}  {'Final':>10}  {'P&L':>9}  {'Return':>7}  {'MaxDD':>6}  {'Trades':>6}  {'Sharpe':>7}  {'Sortino':>8}")
print("-" * 75)
for r in results:
    ret = (r["final"] / 10000 - 1) * 100
    print(f"{r['k']:>5}  {r['final']:>10,.0f}  {r['pnl']:>9,.0f}  {ret:>6.1f}%  {r['max_dd']:>6.1%}  {r['trades']:>6}  {r['sharpe']:>7.3f}  {r['sortino']:>8.3f}")
