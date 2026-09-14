import pandas as pd

ps = pd.read_csv("data_synthetic/journals/live_sim_synthetic_position_summary_20260912T092749.csv")
ps["date"] = pd.to_datetime(ps["date"], errors="coerce")
ps = ps[ps["date"].notna()].sort_values("date")
ps["year"] = ps["date"].dt.year

# Year-end snapshot per year
annual = ps.groupby("year").last()[["portfolio_value", "realized_pnl_cum", "interest_cum"]].copy()

# Year-start values (prior year-end, or zero for first year)
annual["prev_pnl"]      = annual["realized_pnl_cum"].shift(1).fillna(0)
annual["prev_interest"] = annual["interest_cum"].shift(1).fillna(0)
annual["prev_port"]     = annual["portfolio_value"].shift(1).fillna(100_000.0)

annual["trade_pnl_yr"]    = annual["realized_pnl_cum"] - annual["prev_pnl"]
annual["interest_yr"]     = annual["interest_cum"]     - annual["prev_interest"]
annual["port_gain_yr"]    = annual["portfolio_value"]  - annual["prev_port"]
annual["port_ret"]        = annual["port_gain_yr"] / annual["prev_port"]

print(f"{'Year':<6} {'Port%':>7} {'TradePnL':>10} {'Interest':>10} {'Total gain':>11} {'Int%':>6}")
print("-" * 56)
for yr, r in annual.iterrows():
    int_pct = r["interest_yr"] / r["port_gain_yr"] * 100 if abs(r["port_gain_yr"]) > 1 else float("nan")
    int_s = f"{int_pct:>5.0f}%" if not (int_pct != int_pct) else "  n/a"
    print(f"{int(yr):<6} {r['port_ret']:>+7.1%} {r['trade_pnl_yr']:>+10,.0f}"
          f" {r['interest_yr']:>+10,.0f} {r['port_gain_yr']:>+11,.0f} {int_s:>6}")
