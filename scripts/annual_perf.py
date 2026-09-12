import pandas as pd
import numpy as np
import sys

csv = sys.argv[1] if len(sys.argv) > 1 else "data_synthetic/journals/live_sim_synthetic_position_summary_20260912T092749.csv"

df = pd.read_csv(csv)
df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
df = df.sort_values("date")
df["year"] = df["date"].dt.year

annual = df.groupby("year").last()[["portfolio_value"]].copy()
annual["prev"] = annual["portfolio_value"].shift(1)
annual.loc[annual.index[0], "prev"] = 100000.0
annual["annual_return"] = (annual["portfolio_value"] / annual["prev"]) - 1

print("Annual P&L")
print(f"{'Year':<6} {'Port Value':>12} {'Annual Ret':>12}")
print("-" * 32)
for yr, row in annual.iterrows():
    print(f"{yr:<6} {row['portfolio_value']:>12,.0f} {row['annual_return']:>11.1%}")

r = annual["annual_return"].values
rfr = 0.04
excess = r - rfr
sharpe = excess.mean() / excess.std(ddof=1) if excess.std(ddof=1) > 0 else 0
downside = r[r < rfr] - rfr
sortino_denom = float(np.sqrt((downside**2).mean())) if len(downside) > 0 else 1.0
sortino = excess.mean() / sortino_denom if sortino_denom > 0 else 0

n_years = len(annual)
cagr = (annual["portfolio_value"].iloc[-1] / 100000.0) ** (1.0 / n_years) - 1

print(f"\nCAGR ({n_years}yr):          {cagr:.1%}")
print(f"Sharpe (ann, rfr=4%):  {sharpe:.2f}")
print(f"Sortino (ann, rfr=4%): {sortino:.2f}")
