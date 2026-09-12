import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

POS_SUM = "data_synthetic/journals/live_sim_synthetic_position_summary_20260912T092749.csv"
OUT = "reports/synth_26yr_fullrun_chart.png"
os.makedirs("reports", exist_ok=True)

df = pd.read_csv(POS_SUM)
df = df[df["date"] != "SUMMARY"].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").dropna(subset=["date"])
df["pnl"] = df["portfolio_value"] - 100_000.0

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Synthetic 26yr Backtest — optimised_new, £100k pot, 1999–2026", fontsize=13)

axes[0].fill_between(df["date"], df["deployed"], alpha=0.4, color="steelblue")
axes[0].plot(df["date"], df["deployed"], color="steelblue", linewidth=0.8)
axes[0].set_ylabel("Deployed £")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
axes[0].grid(True, alpha=0.3)

colors = np.where(df["pnl"] >= 0, "green", "red")
axes[1].fill_between(df["date"], df["pnl"], 0, where=df["pnl"] >= 0, alpha=0.35, color="green")
axes[1].fill_between(df["date"], df["pnl"], 0, where=df["pnl"] < 0,  alpha=0.35, color="red")
axes[1].plot(df["date"], df["pnl"], color="darkgreen", linewidth=0.9)
axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
axes[1].set_ylabel("Cumulative P&L £")
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:+,.0f}"))
axes[1].grid(True, alpha=0.3)

axes[2].step(df["date"], df["n_open"], where="post", color="darkorange", linewidth=0.9)
axes[2].fill_between(df["date"], df["n_open"], step="post", alpha=0.3, color="orange")
axes[2].set_ylabel("Open positions")
axes[2].set_xlabel("Date")
axes[2].grid(True, alpha=0.3)

# Annotate crash years
for yr, label in [(2000, "Dot-com"), (2008, "GFC"), (2020, "COVID")]:
    for ax in axes:
        ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="red", linewidth=0.7, linestyle=":", alpha=0.6)
axes[0].annotate("Dot-com", xy=(pd.Timestamp("2000-01-01"), axes[0].get_ylim()[1]),
                 xytext=(5, -15), textcoords="offset points", fontsize=7, color="red")
axes[0].annotate("GFC", xy=(pd.Timestamp("2008-01-01"), axes[0].get_ylim()[1]),
                 xytext=(5, -15), textcoords="offset points", fontsize=7, color="red")
axes[0].annotate("COVID", xy=(pd.Timestamp("2020-01-01"), axes[0].get_ylim()[1]),
                 xytext=(5, -15), textcoords="offset points", fontsize=7, color="red")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Chart saved: {OUT}")
