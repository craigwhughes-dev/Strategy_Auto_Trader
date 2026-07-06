"""Generate a trade-by-trade Excel report for a universe of tickers.

Usage:
    uv run python -m Strategy_Auto_Trader.markov_cli.trade_report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def _fetch(ticker: str) -> pd.DataFrame | None:
    from ..quant_hmm.quant_engine import fetch_hourly
    return fetch_hourly(ticker, period="730d")


def _get_company_info(ticker: str) -> tuple[str, str]:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or info.get("industry") or ""
        return name, sector
    except Exception:
        return ticker, ""


def main() -> int:
    parser = argparse.ArgumentParser(prog="trade-report")
    parser.add_argument("--tickers", type=Path, default=ROOT / "config" / "scan_tickers.json",
                        help="JSON file with ticker lists")
    parser.add_argument("--index", default="ftse100",
                        help="Key in tickers JSON (default: ftse100)")
    parser.add_argument("--start-date", default="2026-01-12",
                        help="Only include trades entered on or after this date")
    parser.add_argument(
        "--max-downside-vol", type=float, default=0.25,
        help="Max downside vol — skip stocks above this (default: 0.25 = 25%%)"
    )
    parser.add_argument("--buy-threshold", type=float, default=3.0,
                        help="Minimum composite score to trigger BUY (default: 3.0)")
    parser.add_argument("--lot-size", type=float, default=100.0,
                        help="Amount invested per trade in GBP (default: 100)")
    parser.add_argument("--trade-cost", type=float, default=1.0,
                        help="Cost per trade event in GBP (default: 1)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "trade_report.xlsx",
                        help="Output Excel file")
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date)

    with open(args.tickers, encoding="utf-8") as f:
        ticker_data = json.load(f)
    tickers = ticker_data.get(args.index, [])
    if not tickers:
        print(f"No tickers found under key '{args.index}'")
        return 1

    print(
        f"Trade report: {len(tickers)} {args.index} tickers, start={args.start_date}, "
        f"max_downside_vol={args.max_downside_vol:.2%}, "
        f"lot=GBP{args.lot_size:.0f}, cost=GBP{args.trade_cost:.0f}"
    )

    from ..quant_hmm.consolidated_engine import consolidated_backtest

    all_trades = []
    summary_rows = []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        if i % 10 == 0 or i == 1:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(tickers)}] {elapsed:.0f}s...")

        df = _fetch(ticker)
        if df is None or len(df) < 300:
            continue

        close_arr = df["Close"].dropna()
        returns = close_arr.pct_change().dropna()
        downside_vol = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)) * np.sqrt(252))
        if downside_vol > args.max_downside_vol:
            continue

        try:
            bt = consolidated_backtest(
                df, buy_threshold=args.buy_threshold,
                stop_loss_pct=0.05, take_profit_pct=0.15,
                vol_stop_mult=0.5, vol_stop_window=20,
                profit_stop_scale=0.5, min_stop_pct=0.05,
                initial_cash=args.lot_size, trade_cost=args.trade_cost,
                exit_on_rsi_reversal=True, max_hold_days=240,
                use_kelly=True, regime_smooth=24, min_hold_bars=48,
            )
        except Exception:
            continue

        detail = bt["detail"]
        if detail.empty:
            continue

        company_name, sector = _get_company_info(ticker)

        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]
        buy_dates = buys.index.tolist()
        sell_dates = sells.index.tolist()

        for j, buy_date in enumerate(buy_dates):
            if pd.Timestamp(buy_date).tz is not None:
                buy_ts = pd.Timestamp(buy_date).tz_localize(None)
            else:
                buy_ts = pd.Timestamp(buy_date)
            if buy_ts < start_date:
                continue

            buy_row = buys.loc[buy_date]
            entry_price = float(buy_row["close"])
            stop_level = float(buy_row["stop_level"]) if pd.notna(buy_row.get("stop_level")) else entry_price * 0.95
            target_level = float(buy_row["target_level"]) if pd.notna(buy_row.get("target_level")) else entry_price * 1.15
            shares = args.lot_size / entry_price

            matching_sells = [s for s in sell_dates if s > buy_date]

            if matching_sells:
                sell_date = matching_sells[0]
                sell_row = sells.loc[sell_date]
                exit_price = float(sell_row["close"])
                sell_reason = str(sell_row["sell_reason"])
                status = "CLOSED"
            else:
                sell_date = detail.index[-1]
                exit_price = float(detail["close"].iloc[-1])
                sell_reason = ""
                status = "OPEN"

            sell_ts = pd.Timestamp(sell_date)
            if sell_ts.tz is not None:
                sell_ts = sell_ts.tz_localize(None)
            hours_held = int((sell_ts - buy_ts).total_seconds() / 3600)
            days_held = hours_held // 24

            trade_slice = detail.loc[buy_date:sell_date, "strategy_return"]
            cumulative_return = (1 + trade_slice).prod() - 1
            gross_pl = args.lot_size * cumulative_return
            costs = args.trade_cost * (2 if status == "CLOSED" else 1)
            net_pl = gross_pl - costs
            net_pl_pct = net_pl / args.lot_size * 100

            all_trades.append({
                "Ticker": ticker,
                "Company": company_name,
                "Sector": sector,
                "Status": status,
                "Entry Date": str(buy_date)[:16],
                "Entry Price": round(entry_price, 4),
                "Stop Loss": round(stop_level, 4),
                "Take Profit": round(target_level, 4),
                "Shares": round(shares, 6),
                "Exit Date": str(sell_date)[:16],
                "Exit Price": round(exit_price, 4),
                "Exit Reason": sell_reason if sell_reason else ("open" if status == "OPEN" else ""),
                "Hours Held": hours_held,
                "Days Held": days_held,
                "Gross P&L": round(gross_pl, 2),
                "Costs": round(costs, 2),
                "Net P&L": round(net_pl, 2),
                "Net P&L %": round(net_pl_pct, 2),
                "Lot Size": args.lot_size,
            })

        ticker_trades = [t for t in all_trades if t["Ticker"] == ticker]
        if ticker_trades:
            total_pl = sum(t["Net P&L"] for t in ticker_trades)
            n_trades = len(ticker_trades)
            n_wins = sum(1 for t in ticker_trades if t["Net P&L"] > 0)
            n_open = sum(1 for t in ticker_trades if t["Status"] == "OPEN")
            summary_rows.append({
                "Ticker": ticker,
                "Company": company_name,
                "Sector": sector,
                "Trades": n_trades,
                "Wins": n_wins,
                "Losses": n_trades - n_wins - n_open,
                "Open": n_open,
                "Win Rate": round(n_wins / (n_trades - n_open) * 100, 1) if (n_trades - n_open) > 0 else 0,
                "Total Net P&L": round(total_pl, 2),
                "Avg P&L per Trade": round(total_pl / n_trades, 2),
            })

    elapsed = time.time() - t0
    print(f"\n  Done: {len(all_trades)} trades from {len(summary_rows)} tickers in {elapsed:.0f}s")

    if not all_trades:
        print("  No trades found.")
        return 1

    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(summary_rows).sort_values("Total Net P&L", ascending=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        trades_df.to_excel(writer, sheet_name="All Trades", index=False)

        winners = trades_df[trades_df["Net P&L"] > 0].sort_values("Net P&L", ascending=False)
        winners.to_excel(writer, sheet_name="Winners", index=False)

        losers = trades_df[trades_df["Net P&L"] < 0].sort_values("Net P&L")
        losers.to_excel(writer, sheet_name="Losers", index=False)

        open_pos = trades_df[trades_df["Status"] == "OPEN"].sort_values("Net P&L", ascending=False)
        if not open_pos.empty:
            open_pos.to_excel(writer, sheet_name="Open Positions", index=False)

        total_pl = trades_df["Net P&L"].sum()
        total_costs = trades_df["Costs"].sum()
        n_trades = len(trades_df)
        closed = trades_df[trades_df["Status"] == "CLOSED"]
        n_wins = (closed["Net P&L"] > 0).sum()
        n_losses = (closed["Net P&L"] < 0).sum()
        n_open = (trades_df["Status"] == "OPEN").sum()
        avg_win = closed[closed["Net P&L"] > 0]["Net P&L"].mean() if n_wins > 0 else 0
        avg_loss = closed[closed["Net P&L"] < 0]["Net P&L"].mean() if n_losses > 0 else 0
        avg_days = closed["Days Held"].mean() if len(closed) else 0
        capital_deployed = n_trades * args.lot_size

        stats = pd.DataFrame([
            {"Metric": "Start Date", "Value": args.start_date},
            {"Metric": "Lot Size", "Value": f"GBP{args.lot_size:.0f}"},
            {"Metric": "Trade Cost", "Value": f"GBP{args.trade_cost:.0f}"},
            {"Metric": "", "Value": ""},
            {"Metric": "Total Trades", "Value": n_trades},
            {"Metric": "Closed Trades", "Value": len(closed)},
            {"Metric": "Open Positions", "Value": int(n_open)},
            {"Metric": "Wins", "Value": int(n_wins)},
            {"Metric": "Losses", "Value": int(n_losses)},
            {"Metric": "Win Rate", "Value": f"{n_wins/(n_wins+n_losses)*100:.1f}%" if (n_wins+n_losses) else "N/A"},
            {"Metric": "", "Value": ""},
            {"Metric": "Total Net P&L", "Value": f"GBP{total_pl:,.2f}"},
            {"Metric": "Total Costs", "Value": f"GBP{total_costs:,.2f}"},
            {"Metric": "Avg Win", "Value": f"GBP{avg_win:,.2f}"},
            {"Metric": "Avg Loss", "Value": f"GBP{avg_loss:,.2f}"},
            {"Metric": "Avg Days Held", "Value": f"{avg_days:.1f}"},
            {"Metric": "Capital Deployed", "Value": f"GBP{capital_deployed:,.0f}"},
            {"Metric": "Return on Capital", "Value": f"{total_pl/capital_deployed*100:.2f}%" if capital_deployed else "N/A"},
            {"Metric": "", "Value": ""},
            {"Metric": "Tickers with Trades", "Value": len(summary_rows)},
            {"Metric": "Profitable Tickers", "Value": sum(1 for s in summary_rows if s["Total Net P&L"] > 0)},
        ])
        stats.to_excel(writer, sheet_name="Stats", index=False)

    print(f"\n  Report saved: {args.output}")

    total_pl = trades_df["Net P&L"].sum()
    n_wins = (trades_df[trades_df["Status"] == "CLOSED"]["Net P&L"] > 0).sum()
    n_losses = (trades_df[trades_df["Status"] == "CLOSED"]["Net P&L"] < 0).sum()
    n_open = (trades_df["Status"] == "OPEN").sum()
    print(f"\n  Total trades:  {len(trades_df)}")
    print(f"  Wins/Losses:   {n_wins}/{n_losses} (open: {n_open})")
    print(f"  Total Net P&L: GBP{total_pl:,.2f}")
    print(f"  Win rate:      {n_wins/(n_wins+n_losses)*100:.1f}%" if (n_wins+n_losses) else "")

    return 0


if __name__ == "__main__":
    sys.exit(main())
