import pandas as pd

trades = pd.read_csv("data_synthetic/journals/synth_26yr.csv")
trades["date_closed"] = pd.to_datetime(trades["date_closed"], utc=True, errors="coerce")
trades["year"] = trades["date_closed"].dt.year
trades["group"] = trades["ticker"].apply(lambda t: "FTSE" if str(t).endswith(".L") else "SP500")

cnt      = trades.groupby("year").size().rename("n_trades")
ftse_cnt = trades[trades["group"] == "FTSE"].groupby("year").size().rename("n_ftse")
sp_cnt   = trades[trades["group"] == "SP500"].groupby("year").size().rename("n_sp")

ps = pd.read_csv("data_synthetic/journals/live_sim_synthetic_position_summary_20260912T092749.csv")
ps["date"] = pd.to_datetime(ps["date"], errors="coerce")
ps["year"] = ps["date"].dt.year
annual = ps.groupby("year").last()[["portfolio_value"]].copy()
annual["prev"] = annual["portfolio_value"].shift(1)
annual.loc[annual.index[0], "prev"] = 100_000.0
annual["port_ret"] = (annual["portfolio_value"] / annual["prev"]) - 1
annual = annual.join(cnt).join(ftse_cnt).join(sp_cnt).fillna(0)

print(f"{'Year':<6} {'Port £':>9} {'Port%':>7} {'Trades':>7} {'FTSE':>5} {'S&P':>5}")
print("-" * 44)
for yr, row in annual.iterrows():
    print(f"{int(yr):<6} {row['portfolio_value']:>9,.0f} {row['port_ret']:>+7.1%}"
          f" {int(row['n_trades']):>7} {int(row['n_ftse']):>5} {int(row['n_sp']):>5}")
total_t = int(annual["n_trades"].sum())
total_f = int(annual["n_ftse"].sum())
total_s = int(annual["n_sp"].sum())
print("-" * 44)
print(f"{'TOTAL':<6} {'':>9} {'':>7} {total_t:>7} {total_f:>5} {total_s:>5}")
