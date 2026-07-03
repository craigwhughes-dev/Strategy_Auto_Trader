"""CLI for the HMM quant engine: HMM regime probabilities on hourly data.

Usage:
    uv run python -m Strategy_Auto_Trader.quant_hmm.quant_run --ticker HSBA.L
    uv run python -m Strategy_Auto_Trader.quant_hmm.quant_run --ticker CSCO --entry-prob 0.70
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .quant_engine import fetch_hourly, quant_backtest
from .sentiment import composite_sentiment, vix_regime

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-hmm")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--period", default="730d", help="yfinance period (default: 730d)")
    parser.add_argument("--entry-prob", type=float, default=0.65,
                        help="P(Bull) threshold to enter (default: 0.65)")
    parser.add_argument("--exit-prob", type=float, default=0.40,
                        help="P(Bull) threshold to exit (default: 0.40)")
    parser.add_argument("--stop-loss", type=float, default=0.05,
                        help="Hard stop-loss from entry (default: 0.05 = 5%%)")
    parser.add_argument("--take-profit", type=float, default=0.15,
                        help="Take-profit from entry (default: 0.15 = 15%%)")
    parser.add_argument("--volume-min", type=float, default=1.0,
                        help="Min volume ratio for entry (default: 1.0 = at least average)")
    parser.add_argument("--initial-cash", type=float, default=20_000.0)
    parser.add_argument("--trade-cost", type=float, default=10.0)
    parser.add_argument("--no-kelly", dest="kelly", action="store_false", default=True)
    parser.add_argument("--regime-smooth", type=int, default=24,
                        help="Rolling window (bars) for smoothing P(Bull) (default: 24 = 1 day)")
    parser.add_argument("--min-hold", type=int, default=48,
                        help="Min bars before regime exit can fire (default: 48 = 2 days)")
    parser.add_argument("--no-sentiment", dest="sentiment", action="store_false", default=True,
                        help="Disable sentiment signals (options IV, VIX, insider, short interest)")
    return parser


def _print_results(bt: dict, args, detail: pd.DataFrame) -> None:
    """Sharpe/return/drawdown, trade stats, P&L, and current-state console report."""
    def _pct(v): return f"{v*100:.2f}%" if np.isfinite(v) else "N/A"
    def _f(v): return f"{v:.3f}" if np.isfinite(v) else "N/A"

    print(f"\n  {'':30s}  {'Strategy':>10s}  {'Buy & Hold':>10s}")
    print(f"  {'Sharpe (annualised)':30s}  {_f(bt['sharpe_strategy']):>10s}  {_f(bt['sharpe_bh']):>10s}")
    print(f"  {'Total return':30s}  {_pct(bt['total_return_strategy']):>10s}  {_pct(bt['total_return_bh']):>10s}")
    print(f"  {'Max drawdown':30s}  {_pct(bt['max_drawdown_strategy']):>10s}  {_pct(bt['max_drawdown_bh']):>10s}")

    trades = bt["trade_results"]
    n_trades = len(trades)
    if n_trades:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        win_rate = len(wins) / n_trades * 100
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean(losses) * 100 if losses else 0
        print(f"\n  Trades: {n_trades}  |  Wins: {len(wins)}  Losses: {len(losses)}  "
              f"Open: {bt['n_buys'] - bt['n_sells']}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Avg win: {avg_win:+.1f}%  |  Avg loss: {avg_loss:+.1f}%")
        print(f"  Kelly fraction: {bt['final_kelly']*100:.1f}%")
    else:
        print(f"\n  No completed trades.")

    ic = bt["initial_cash"]
    fp = bt["final_portfolio"]
    pl = bt["total_pl"]
    sign = "+" if pl >= 0 else ""
    bh_fp = ic * (1 + bt["total_return_bh"])
    bh_pl = bh_fp - ic
    bh_sign = "+" if bh_pl >= 0 else ""
    print(f"\n  P&L (GBP{ic:,.0f} initial, GBP{args.trade_cost:.0f}/trade):")
    print(f"  {'Strategy':20s}  {sign}GBP{abs(pl):,.2f}  ({sign}{pl/ic*100:.1f}%)")
    print(f"  {'Buy & Hold':20s}  {bh_sign}GBP{abs(bh_pl):,.2f}  ({bh_sign}{bh_pl/ic*100:.1f}%)")

    last = detail.iloc[-1]
    print(f"\n  Current state:")
    print(f"  P(Bull): {last['p_bull']:.1%}  |  P(Bear): {last['p_bear']:.1%}")
    print(f"  Volume ratio: {last['volume_ratio']:.1f}x avg")
    in_pos = last["position"] > 0
    if in_pos:
        print(f"  IN POSITION  entry={last['entry_price']:.2f}  "
              f"stop={last['stop_level']:.2f}  target={last['target_level']:.2f}")
    else:
        print(f"  FLAT — waiting for P(Bull) > {args.entry_prob:.0%} + volume confirmation")


def _print_exit_breakdown(detail: pd.DataFrame, n_trades: int) -> None:
    if not n_trades:
        return
    sells = detail[detail["trade_event"] == "SELL"]
    print(f"\n  Exit breakdown:")
    for reason_prefix in ["stop_loss", "take_profit", "regime_exit"]:
        count = sells["sell_reason"].str.contains(reason_prefix, na=False).sum()
        if count:
            sub_trades = []
            for idx, row in sells[sells["sell_reason"].str.contains(reason_prefix, na=False)].iterrows():
                ep = row.get("entry_price")
                if ep and ep > 0:
                    sub_trades.append((row["close"] - ep) / ep)
            avg_pl = np.mean(sub_trades) * 100 if sub_trades else 0
            print(f"    {reason_prefix:<15s}: {count:>3d} trades, avg P&L {avg_pl:+.1f}%")


def _print_sentiment_detail(sent_data: dict) -> None:
    print(f"\n  Sentiment & Alternative Data:")
    print(f"  {'Composite':20s}  {sent_data['sentiment_label'].upper():>10s}  "
          f"(score: {sent_data['sentiment_score']:+.3f}, sources: {sent_data['confidence']}/4)")

    opts = sent_data.get("options", {})
    if opts.get("put_call_ratio") is not None:
        pc_label = {1: "Contrarian bullish", -1: "Bearish", 0: "Neutral"}
        print(f"  {'Put/Call ratio':20s}  {opts['put_call_ratio']:>10.3f}  "
              f"{pc_label.get(opts['put_call_signal'], '')}")
    if opts.get("iv_current") is not None:
        iv_label = {1: "Low IV (cheap entry)", -1: "High IV (risky)", 0: "Normal"}
        print(f"  {'Implied volatility':20s}  {opts['iv_current']:>10.1%}  "
              f"{iv_label.get(opts['iv_signal'], '')}")
        if opts.get("iv_rank") is not None:
            print(f"  {'IV rank':20s}  {opts['iv_rank']:>9.1f}%")
    if opts.get("skew") is not None:
        print(f"  {'Volatility skew':20s}  {opts['skew']:>10.4f}  "
              f"{'(fearful)' if opts['skew'] > 0.05 else '(calm)' if opts['skew'] < -0.02 else ''}")

    vix_d = sent_data.get("vix", {})
    if vix_d.get("vix_current") is not None:
        print(f"  {'VIX':20s}  {vix_d['vix_current']:>10.2f}  "
              f"regime={vix_d['vix_regime']}  "
              f"term={vix_d.get('vix_term_structure', 'N/A')}")

    ins = sent_data.get("insider", {})
    if ins.get("insider_net") != 0:
        print(f"  {'Insider net (90d)':20s}  {ins['insider_net']:>+10d}  "
              f"(buys={ins['insider_buys_90d']} sells={ins['insider_sells_90d']})")

    si = sent_data.get("short_interest", {})
    if si.get("short_pct_float") is not None:
        print(f"  {'Short % of float':20s}  {si['short_pct_float']:>9.2f}%  "
              f"days-to-cover={si.get('short_ratio', 'N/A')}")


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    print(f"\nQuant HMM Engine — {args.ticker}")
    print(f"  Hourly data, HMM regime probabilities, volume confirmation, Kelly sizing")
    print(f"  Entry: P(Bull)>{args.entry_prob}  Exit: P(Bull)<{args.exit_prob}")
    print(f"  Stop: {args.stop_loss*100:.0f}%  Target: {args.take_profit*100:.0f}%  Kelly: {'on' if args.kelly else 'off'}")

    print(f"\n  Fetching hourly data for {args.ticker}...")
    t0 = time.time()
    df = fetch_hourly(args.ticker, period=args.period)
    if df is None or df.empty:
        print(f"  No data for {args.ticker}")
        return 1

    print(f"  {len(df)} hourly bars | {df.index[0]} -> {df.index[-1]}")

    # Company info
    try:
        import yfinance as yf
        info = yf.Ticker(args.ticker).info
        name = info.get("longName") or info.get("shortName") or args.ticker
        sector = info.get("sector") or ""
        print(f"  {name}  [{sector}]" if sector else f"  {name}")
    except Exception:
        name = args.ticker
        sector = ""

    # Sentiment signals
    sent_score = 0.0
    vix_sig = 0
    sent_data = None
    if args.sentiment:
        print(f"\n  Fetching sentiment data...")
        sent_data = composite_sentiment(args.ticker)
        sent_score = sent_data.get("sentiment_score", 0.0)
        vix_info = sent_data.get("vix", {})
        vix_sig = vix_info.get("vix_signal", 0)
        print(f"  Sentiment: {sent_data['sentiment_label']} ({sent_score:+.3f})  "
              f"confidence: {sent_data['confidence']}/4")

    print(f"\n  Running backtest...")
    bt = quant_backtest(
        df,
        entry_prob=args.entry_prob,
        exit_prob=args.exit_prob,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        volume_min_ratio=args.volume_min,
        initial_cash=args.initial_cash,
        trade_cost=args.trade_cost,
        use_kelly=args.kelly,
        sentiment_score=sent_score,
        vix_signal=vix_sig,
        regime_smooth=args.regime_smooth,
        min_hold_bars=args.min_hold,
    )

    elapsed = time.time() - t0
    detail = bt["detail"]

    if detail.empty:
        print("  Insufficient data for backtest.")
        return 1

    _print_results(bt, args, detail)
    _print_exit_breakdown(detail, len(bt["trade_results"]))
    if sent_data:
        _print_sentiment_detail(sent_data)

    # Save
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = args.ticker.replace("/", "-").replace("\\", "-")
    run_dir = DATA_DIR / f"{safe}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(run_dir / "quant_backtest.csv")
    print(f"\n  Output: {run_dir}")
    print(f"  Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
