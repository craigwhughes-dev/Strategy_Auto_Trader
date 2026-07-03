"""Compare exit strategies across a selection of tickers using the consolidated engine.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.compare_exits
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

TEST_TICKERS = [
    "HSBA.L", "INTC", "T", "CSCO", "BA",
    "BATS.L", "F", "GOOGL", "GSK.L", "KO", "MU",
]

STRATEGIES = [
    {"name": "Base (RSI exit)",         "exit_on_rsi_reversal": True,  "max_hold_days": 0},
    {"name": "MACD+RSI exits",          "exit_on_rsi_reversal": True,  "exit_on_macd_cross": True, "max_hold_days": 0},
    {"name": "Consol exit",             "exit_on_rsi_reversal": True,  "exit_on_consolidation": True, "max_hold_days": 0},
    {"name": "Max-hold 240h",           "exit_on_rsi_reversal": True,  "max_hold_days": 240},
    {"name": "Max-hold 480h",           "exit_on_rsi_reversal": True,  "max_hold_days": 480},
    {"name": "No optional exits",       "exit_on_rsi_reversal": False, "max_hold_days": 0},
]


def _fetch(ticker: str) -> pd.DataFrame | None:
    from ..quant_hmm.quant_engine import fetch_hourly
    try:
        return fetch_hourly(ticker, period="730d")
    except Exception:
        return None


def main() -> int:
    from ..quant_hmm.consolidated_engine import consolidated_backtest

    print(f"Comparing {len(STRATEGIES)} exit strategies across {len(TEST_TICKERS)} tickers\n")

    price_data = {}
    for ticker in TEST_TICKERS:
        df = _fetch(ticker)
        if df is not None and len(df) > 300:
            price_data[ticker] = df
            print(f"  Fetched {ticker}: {len(df)} bars")
        else:
            print(f"  Skipped {ticker}: insufficient data")

    base_args = dict(
        min_train_bars=500, hmm_refit_bars=500,
        stop_loss_pct=0.05, take_profit_pct=0.15,
        vol_stop_mult=0.5, vol_stop_window=20,
        profit_stop_scale=0.5, min_stop_pct=0.05,
        initial_cash=20_000.0, trade_cost=10.0,
        use_kelly=True, regime_smooth=24, min_hold_bars=48,
        buy_threshold=3.0, sell_threshold=-3.0,
    )

    all_results = []
    t0 = time.time()
    total = len(price_data) * len(STRATEGIES)
    done = 0

    for ticker, df in price_data.items():
        close = df["Close"].dropna()
        for strat in STRATEGIES:
            done += 1
            args = {**base_args,
                    "exit_on_macd_cross": strat.get("exit_on_macd_cross", False),
                    "exit_on_rsi_reversal": strat.get("exit_on_rsi_reversal", False),
                    "exit_on_consolidation": strat.get("exit_on_consolidation", False),
                    "max_hold_days": strat.get("max_hold_days", 0)}
            try:
                bt = consolidated_backtest(df, **args)
                detail = bt["detail"]
                sells = detail[detail["trade_event"] == "SELL"]
                buys = detail[detail["trade_event"] == "BUY"]
                trade_pls = []
                buy_prices = buys["close"].tolist()
                sell_prices = sells["close"].tolist()
                sell_reasons = sells["sell_reason"].tolist() if "sell_reason" in sells else []
                for j in range(min(len(buy_prices), len(sell_prices))):
                    trade_pls.append((sell_prices[j] - buy_prices[j]) / buy_prices[j])
                wins = sum(1 for p in trade_pls if p > 0)
                losses = sum(1 for p in trade_pls if p < 0)
                avg_win = np.mean([p for p in trade_pls if p > 0]) * 100 if wins else 0
                avg_loss = np.mean([p for p in trade_pls if p < 0]) * 100 if losses else 0
                rr_tp = sum(1 for r in sell_reasons if "take_profit" in str(r))
                rr_sl = sum(1 for r in sell_reasons if "stop_loss" in str(r))

                all_results.append({
                    "ticker": ticker,
                    "strategy": strat["name"],
                    "sharpe": bt["sharpe_strategy"],
                    "total_return": bt["total_return_strategy"],
                    "max_dd": bt["max_drawdown_strategy"],
                    "pl": bt["total_pl"],
                    "bh_return": bt["total_return_bh"],
                    "n_trades": bt["n_buys"],
                    "wins": wins,
                    "losses": losses,
                    "win_rate": wins / (wins + losses) * 100 if (wins + losses) else 0,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "rr_tp": rr_tp,
                    "rr_sl": rr_sl,
                })
            except Exception as exc:
                all_results.append({
                    "ticker": ticker, "strategy": strat["name"],
                    "sharpe": float("nan"), "total_return": float("nan"),
                    "max_dd": float("nan"), "pl": float("nan"),
                    "bh_return": float("nan"), "n_trades": 0,
                    "wins": 0, "losses": 0, "win_rate": 0,
                    "avg_win": 0, "avg_loss": 0, "rr_tp": 0, "rr_sl": 0,
                })

        if done % len(STRATEGIES) == 0:
            elapsed = time.time() - t0
            print(f"  [{done}/{total}] {ticker} done ({elapsed:.0f}s)")

    df_res = pd.DataFrame(all_results)
    elapsed = time.time() - t0

    print(f"\n{'='*100}")
    print(f" Results by ticker ({elapsed:.0f}s)")
    print(f"{'='*100}")

    first_strat = STRATEGIES[0]["name"]
    for ticker in price_data:
        sub = df_res[df_res["ticker"] == ticker].copy()
        baseline = sub[sub["strategy"] == first_strat]
        bh_ret = baseline["bh_return"].iloc[0] * 100 if len(baseline) else 0

        print(f"\n  {ticker}  (B&H: {bh_ret:+.1f}%)")
        print(f"  {'Strategy':<22s} {'P&L':>10s} {'Return':>8s} {'Win%':>6s} {'W/L':>7s} {'AvgW':>6s} {'AvgL':>7s} {'TP':>4s} {'SL':>4s}")
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*4} {'-'*4}")

        for _, row in sub.iterrows():
            ret = row["total_return"] * 100 if np.isfinite(row["total_return"]) else float("nan")
            profitable = " $" if row["pl"] > 0 else ""
            print(f"  {row['strategy']:<22s} {row['pl']:>+9,.0f} {ret:>+7.1f}% {row['win_rate']:>5.0f}% "
                  f"{row['wins']:>3d}/{row['losses']:<3d} {row['avg_win']:>+5.1f}% {row['avg_loss']:>+6.1f}% "
                  f"{row['rr_tp']:>4d} {row['rr_sl']:>4d}{profitable}")

    print(f"\n{'='*100}")
    print(f" Aggregate (average across {len(price_data)} tickers)")
    print(f"{'='*100}")

    agg = df_res.groupby("strategy").agg({
        "total_return": "mean",
        "sharpe": "mean",
        "max_dd": "mean",
        "pl": "mean",
        "n_trades": "mean",
        "win_rate": "mean",
        "avg_win": "mean",
        "avg_loss": "mean",
    })
    strat_order = [s["name"] for s in STRATEGIES]
    agg = agg.loc[[s for s in strat_order if s in agg.index]]

    print(f"\n  {'Strategy':<22s} {'Avg P&L':>10s} {'Avg Return':>10s} {'Win%':>6s} {'AvgW':>7s} {'AvgL':>7s} {'Sharpe':>7s} {'Trades':>7s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for strat_name, row in agg.iterrows():
        print(f"  {strat_name:<22s} {row['pl']:>+9,.0f} {row['total_return']*100:>+9.1f}% "
              f"{row['win_rate']:>5.0f}% {row['avg_win']:>+6.1f}% {row['avg_loss']:>+6.1f}% "
              f"{row['sharpe']:>7.3f} {row['n_trades']:>6.0f}")

    print(f"\n  {'Strategy':<22s} {'Profitable':>11s} {'Profitable %':>13s}")
    print(f"  {'-'*22} {'-'*11} {'-'*13}")
    for strat in STRATEGIES:
        strat_pls = df_res[df_res["strategy"] == strat["name"]].set_index("ticker")["pl"]
        n_profit = sum(1 for pl in strat_pls if np.isfinite(pl) and pl > 0)
        n_total = sum(1 for pl in strat_pls if np.isfinite(pl))
        print(f"  {strat['name']:<22s} {n_profit:>7d}/{n_total:<3d} {n_profit/n_total*100 if n_total else 0:>12.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
