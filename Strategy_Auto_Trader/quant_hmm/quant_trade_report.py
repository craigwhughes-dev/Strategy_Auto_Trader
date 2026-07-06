"""Trade report using the HMM quant engine (HMM regime probabilities on hourly data).

Usage:
    uv run python -m Strategy_Auto_Trader.quant_hmm.quant_trade_report
    uv run python -m Strategy_Auto_Trader.quant_hmm.quant_trade_report --watchlist watchlist_ftse.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .quant_engine import fetch_hourly, quant_backtest
from .sentiment import composite_sentiment, vix_regime
from .vol_screen import screen_tickers

ROOT = Path(__file__).resolve().parent.parent.parent

warnings.filterwarnings("ignore", category=FutureWarning)


def _get_company_info(ticker: str) -> tuple[str, str]:
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or info.get("industry") or ""
        return name, sector
    except Exception:
        return ticker, ""


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quant-trade-report")
    parser.add_argument("--watchlist", type=Path, default=ROOT / "config" / "watchlist_ftse.json",
                        help="Watchlist JSON file")
    parser.add_argument("--start-date", default="2026-01-12",
                        help="Only include trades entered on or after this date")
    parser.add_argument("--period", default="730d",
                        help="yfinance period for hourly data (default: 730d)")
    parser.add_argument("--entry-prob", type=float, default=0.65)
    parser.add_argument("--exit-prob", type=float, default=0.40)
    parser.add_argument("--stop-loss", type=float, default=0.05)
    parser.add_argument("--take-profit", type=float, default=0.15)
    parser.add_argument("--volume-min", type=float, default=1.0)
    parser.add_argument("--lot-size", type=float, default=100.0,
                        help="Amount invested per trade in GBP (default: 100)")
    parser.add_argument("--trade-cost", type=float, default=1.0,
                        help="Cost per trade event in GBP (default: 1)")
    parser.add_argument("--no-kelly", dest="kelly", action="store_false", default=True)
    parser.add_argument("--regime-smooth", type=int, default=24,
                        help="Rolling window (bars) for smoothing P(Bull) (default: 24 = 1 day)")
    parser.add_argument("--min-hold", type=int, default=48,
                        help="Min bars before regime exit can fire (default: 48 = 2 days)")
    parser.add_argument("--no-sentiment", dest="sentiment", action="store_false", default=True,
                        help="Disable sentiment signals")
    parser.add_argument("--no-vol-screen", dest="vol_screen", action="store_false", default=True,
                        help="Disable pre-screening out choppy/mean-reverting tickers")
    parser.add_argument("--min-trend-quality", type=float, default=0.0,
                        help="Min trend-quality score to keep a ticker when vol-screening (default: 0.0)")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "quant_trade_report.xlsx")
    return parser


def _load_watchlist_tickers(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        wl = json.load(f)
    return [t["ticker"] if isinstance(t, dict) else t for t in wl.get("tickers", [])]


def _safe_round(val, decimals: int = 3):
    """Round val to decimals places; return None for missing or non-finite values."""
    try:
        v = float(val)
        return round(v, decimals) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _fetch_daily_indicators_table(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """Download daily OHLCV and compute every daily indicator as a time series.

    Covers RSI, SMA20/50/200, MACD, ATR, and Bollinger Band width — indicators
    that add useful context to each trade entry from the hourly engine.

    Returns a DataFrame indexed by normalised date (tz-naive), or None on failure.
    """
    import yfinance as yf
    from ..core.momentum import compute_rsi, compute_sma, compute_macd, compute_bollinger, compute_atr

    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    if len(close) < 50:
        return None
    high = df["High"] if "High" in df.columns else None
    low = df["Low"] if "Low" in df.columns else None

    rsi = compute_rsi(close)
    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)
    macd_line, macd_sig, macd_hist = compute_macd(close)
    bb_mid, bb_upper, bb_lower = compute_bollinger(close)
    bb_width = ((bb_upper - bb_lower) / bb_mid).where(bb_mid > 0)
    atr_series = compute_atr(close, high, low)

    out = pd.DataFrame({
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "above_sma20": (close > sma20),
        "above_sma50": (close > sma50),
        "above_sma200": (close > sma200).where(sma200.notna()),
        "pct_from_sma20": ((close - sma20) / sma20).where(sma20 > 0) * 100,
        "pct_from_sma50": ((close - sma50) / sma50).where(sma50 > 0) * 100,
        "pct_from_sma200": ((close - sma200) / sma200).where(sma200 > 0) * 100,
        "macd_line": macd_line,
        "macd_signal_line": macd_sig,
        "macd_histogram": macd_hist,
        "macd_trend": (macd_hist > 0).map({True: "bullish", False: "bearish"}),
        "atr": atr_series,
        "bb_width": bb_width,
    })
    out.index = pd.to_datetime(out.index).normalize()
    return out


def _lookup_daily_at(daily_df: pd.DataFrame | None, timestamp) -> dict:
    """Return daily indicator values for the trading day on or just before timestamp.

    Hourly timestamps are tz-aware; strips tz before comparing with the daily index.
    Returns an empty dict when the DataFrame is None or no prior daily row exists.
    """
    if daily_df is None:
        return {}
    try:
        ts = pd.Timestamp(timestamp)
        if ts.tz is not None:
            ts = ts.tz_localize(None)
        target = ts.normalize()
        prior = daily_df.index[daily_df.index <= target]
        if not len(prior):
            return {}
        row = daily_df.loc[prior[-1]]
        return {
            k: (None if (isinstance(v, float) and not np.isfinite(v)) else v)
            for k, v in row.items()
        }
    except Exception:
        return {}


def _build_trade_row(
    ticker: str, company_name: str, sector: str,
    buy_date, buy_row: pd.Series, sell_dates: list, sells: pd.DataFrame,
    detail: pd.DataFrame, start_date: pd.Timestamp, args,
    sent_data: dict | None, sent_score: float,
    *,
    daily_df: pd.DataFrame | None = None,
    vol_profile: dict | None = None,
) -> dict | None:
    """Build one trade row from a BUY event, matched to the next SELL (or still open).

    Returns None if the trade entered before start_date (caller should skip it).
    """
    buy_ts = pd.Timestamp(buy_date)
    if buy_ts.tz is not None:
        buy_ts = buy_ts.tz_localize(None)
    if buy_ts < start_date:
        return None

    entry_price = float(buy_row["close"])
    p_bull_entry = float(buy_row["p_bull"])
    vol_ratio_entry = float(buy_row["volume_ratio"])
    kelly_at_entry = float(buy_row["kelly_fraction"])
    stop_level = float(buy_row["stop_level"]) if pd.notna(buy_row.get("stop_level")) else entry_price * (1 - args.stop_loss)
    target_level = float(buy_row["target_level"]) if pd.notna(buy_row.get("target_level")) else entry_price * (1 + args.take_profit)
    shares = args.lot_size / entry_price

    matching_sells = [s for s in sell_dates if s > buy_date]

    if matching_sells:
        sell_date = matching_sells[0]
        sell_row = sells.loc[sell_date]
        exit_price = float(sell_row["close"])
        sell_reason = str(sell_row["sell_reason"])
        p_bull_exit = float(sell_row["p_bull"])
        status = "CLOSED"
        exit_row = sell_row
    else:
        sell_date = detail.index[-1]
        exit_price = float(detail["close"].iloc[-1])
        sell_reason = "open"
        p_bull_exit = float(detail["p_bull"].iloc[-1])
        status = "OPEN"
        exit_row = detail.iloc[-1]

    sell_ts = pd.Timestamp(sell_date)
    if sell_ts.tz is not None:
        sell_ts = sell_ts.tz_localize(None)
    days_held = (sell_ts - buy_ts).days
    hours_held = int((sell_ts - buy_ts).total_seconds() / 3600)

    # P&L from raw price change (full lot deployed per trade,
    # not scaled by Kelly position fraction)
    price_return = (exit_price - entry_price) / entry_price
    gross_pl = args.lot_size * price_return
    costs = args.trade_cost * (2 if status == "CLOSED" else 1)
    net_pl = gross_pl - costs
    net_pl_pct = net_pl / args.lot_size * 100

    # Simplify sell reason for display
    reason_short = sell_reason
    if "stop_loss" in sell_reason:
        reason_short = "Stop Loss"
    elif "take_profit" in sell_reason:
        reason_short = "Take Profit"
    elif "regime_exit" in sell_reason:
        reason_short = "Regime Exit"

    trade_row = {
        "Ticker": ticker,
        "Company": company_name,
        "Sector": sector,
        "Status": status,
        "Entry Date": str(buy_date)[:16],
        "Entry Price": round(entry_price, 4),
        "P(Bull) Entry": round(p_bull_entry, 3),
        "Vol Ratio Entry": round(vol_ratio_entry, 2),
        "Kelly %": round(kelly_at_entry * 100, 1),
        "Stop Loss": round(stop_level, 4),
        "Take Profit": round(target_level, 4),
        "Shares": round(shares, 6),
        "Exit Date": str(sell_date)[:16],
        "Exit Price": round(exit_price, 4),
        "P(Bull) Exit": round(p_bull_exit, 3),
        "Exit Reason": reason_short,
        "Hours Held": hours_held,
        "Days Held": days_held,
        "Gross P&L": round(gross_pl, 2),
        "Costs": round(costs, 2),
        "Net P&L": round(net_pl, 2),
        "Net P&L %": round(net_pl_pct, 2),
        "Lot Size": args.lot_size,
    }
    # ── Quant HMM extras (already in hourly detail, not previously reported) ─
    trade_row["P(Bull) Smooth Entry"] = _safe_round(buy_row.get("p_bull_smooth"), 3)
    trade_row["P(Bear) Entry"] = _safe_round(buy_row.get("p_bear"), 3)
    trade_row["P(Bull) Smooth Exit"] = _safe_round(exit_row.get("p_bull_smooth"), 3)
    trade_row["P(Bear) Exit"] = _safe_round(exit_row.get("p_bear"), 3)
    trade_row["Vol Ratio Exit"] = _safe_round(exit_row.get("volume_ratio"), 2)
    kf_exit = exit_row.get("kelly_fraction")
    trade_row["Kelly % Exit"] = (
        _safe_round(float(kf_exit) * 100, 1) if kf_exit is not None else None
    )

    # ── Daily indicators at entry (RSI, SMAs, MACD, Markov — not in quant engine) ─
    d = _lookup_daily_at(daily_df, buy_date)
    trade_row["RSI Entry"] = _safe_round(d.get("rsi"), 1)
    trade_row["Above SMA20"] = d.get("above_sma20")
    trade_row["% from SMA20"] = _safe_round(d.get("pct_from_sma20"), 1)
    trade_row["Above SMA50"] = d.get("above_sma50")
    trade_row["% from SMA50"] = _safe_round(d.get("pct_from_sma50"), 1)
    trade_row["Above SMA200"] = d.get("above_sma200")
    trade_row["% from SMA200"] = _safe_round(d.get("pct_from_sma200"), 1)
    trade_row["MACD Hist Entry"] = _safe_round(d.get("macd_histogram"), 4)
    trade_row["MACD Trend Entry"] = d.get("macd_trend")
    trade_row["ATR Entry"] = _safe_round(d.get("atr"), 4)
    trade_row["BB Width Entry"] = _safe_round(d.get("bb_width"), 4)
    # ── Sentiment (full detail including VIX, insider, IV signal) ────────────
    if sent_data:
        opts = sent_data.get("options", {})
        vix_d = sent_data.get("vix", {})
        ins = sent_data.get("insider", {})
        si = sent_data.get("short_interest", {})
        trade_row["Sentiment"] = sent_data.get("sentiment_label", "")
        trade_row["Sent Score"] = round(sent_score, 3)
        trade_row["Sent Confidence"] = sent_data.get("confidence")
        trade_row["P/C Ratio"] = opts.get("put_call_ratio")
        trade_row["IV Rank"] = opts.get("iv_rank")
        trade_row["IV Current"] = _safe_round(opts.get("iv_current"), 4)
        trade_row["IV Signal"] = {
            1: "Low (cheap)", -1: "High (risky)", 0: "Normal",
        }.get(opts.get("iv_signal", 0), "")
        trade_row["Skew"] = _safe_round(opts.get("skew"), 4)
        trade_row["VIX"] = _safe_round(vix_d.get("vix_current"), 2)
        trade_row["VIX Regime"] = vix_d.get("vix_regime")
        trade_row["VIX Signal"] = vix_d.get("vix_signal")
        trade_row["Insider Net"] = ins.get("insider_net")
        trade_row["Insider Signal"] = {
            1: "Buying", -1: "Selling", 0: "Neutral",
        }.get(ins.get("insider_signal", 0), "")
        trade_row["Short %"] = si.get("short_pct_float")
        trade_row["Short Ratio"] = _safe_round(si.get("short_ratio"), 1)

    # ── Volatility profile (constant for all trades of this ticker) ──────────
    if vol_profile:
        trade_row["Ann Vol"] = _safe_round(vol_profile.get("ann_vol"), 3)
        trade_row["Efficiency Ratio"] = _safe_round(vol_profile.get("efficiency_ratio"), 3)
        trade_row["Autocorr"] = _safe_round(vol_profile.get("autocorr"), 3)
        trade_row["Choppiness"] = _safe_round(vol_profile.get("choppiness_idx"), 1)
        trade_row["Sign Change Freq"] = _safe_round(vol_profile.get("sign_change_freq"), 3)
        trade_row["Trend Quality"] = _safe_round(vol_profile.get("trend_quality"), 3)

    return trade_row


def _build_ticker_summary_row(
    ticker: str, company_name: str, sector: str, ticker_trades: list[dict],
    bt: dict, sent_data: dict | None, sent_score: float,
    vol_profiles_by_ticker: dict,
) -> dict:
    total_pl = sum(t["Net P&L"] for t in ticker_trades)
    n_tt = len(ticker_trades)
    closed_tt = [t for t in ticker_trades if t["Status"] == "CLOSED"]
    n_wins = sum(1 for t in closed_tt if t["Net P&L"] > 0)
    n_losses = sum(1 for t in closed_tt if t["Net P&L"] < 0)
    n_open = sum(1 for t in ticker_trades if t["Status"] == "OPEN")
    avg_hours = np.mean([t["Hours Held"] for t in closed_tt]) if closed_tt else 0
    summary_row = {
        "Ticker": ticker,
        "Company": company_name,
        "Sector": sector,
        "Trades": n_tt,
        "Wins": n_wins,
        "Losses": n_losses,
        "Open": n_open,
        "Win Rate %": round(n_wins / (n_wins + n_losses) * 100, 1) if (n_wins + n_losses) > 0 else 0,
        "Total Net P&L": round(total_pl, 2),
        "Avg P&L/Trade": round(total_pl / n_tt, 2),
        "Avg Hours Held": round(avg_hours, 0),
        "Sharpe": round(bt["sharpe_strategy"], 3) if np.isfinite(bt["sharpe_strategy"]) else "N/A",
        "Sortino": round(bt["sortino_strategy"], 3) if np.isfinite(bt.get("sortino_strategy", float("nan"))) else "N/A",
        "Calmar": round(bt["calmar_strategy"], 3) if np.isfinite(bt.get("calmar_strategy", float("nan"))) else "N/A",
        "Max DD": f"{bt['max_drawdown_strategy']*100:.1f}%" if np.isfinite(bt["max_drawdown_strategy"]) else "N/A",
    }
    if sent_data:
        summary_row["Sentiment"] = sent_data.get("sentiment_label", "")
        summary_row["Sent Score"] = round(sent_score, 3)
    vp = vol_profiles_by_ticker.get(ticker)
    if vp:
        summary_row["Trend Quality"] = vp["trend_quality"]
        summary_row["Efficiency Ratio"] = vp["efficiency_ratio"]
    return summary_row


def _build_sentiment_row(ticker: str, company_name: str, sent_data: dict, sent_score: float) -> dict:
    opts = sent_data.get("options", {})
    ins = sent_data.get("insider", {})
    si = sent_data.get("short_interest", {})
    return {
        "Ticker": ticker,
        "Company": company_name,
        "Sentiment": sent_data.get("sentiment_label", ""),
        "Score": round(sent_score, 3),
        "Confidence": sent_data.get("confidence", 0),
        "P/C Ratio": opts.get("put_call_ratio"),
        "P/C Signal": {1: "Bullish", -1: "Bearish", 0: "Neutral"}.get(opts.get("put_call_signal", 0), ""),
        "IV Current": opts.get("iv_current"),
        "IV Rank": opts.get("iv_rank"),
        "IV Signal": {1: "Low (cheap)", -1: "High (risky)", 0: "Normal"}.get(opts.get("iv_signal", 0), ""),
        "Skew": opts.get("skew"),
        "Insider Net 90d": ins.get("insider_net", 0),
        "Insider Signal": {1: "Buying", -1: "Selling", 0: "Neutral"}.get(ins.get("insider_signal", 0), ""),
        "Short % Float": si.get("short_pct_float"),
        "Short Ratio": si.get("short_ratio"),
    }


def _build_exit_breakdown_rows(closed: pd.DataFrame) -> list[dict]:
    """Per-exit-reason win-rate / P&L breakdown over closed trades."""
    reason_stats = []
    for reason in closed["Exit Reason"].unique():
        subset = closed[closed["Exit Reason"] == reason]
        reason_stats.append({
            "Exit Reason": reason,
            "Count": len(subset),
            "Wins": (subset["Net P&L"] > 0).sum(),
            "Losses": (subset["Net P&L"] < 0).sum(),
            "Win Rate %": round((subset["Net P&L"] > 0).sum() / len(subset) * 100, 1),
            "Avg Net P&L": round(subset["Net P&L"].mean(), 2),
            "Total Net P&L": round(subset["Net P&L"].sum(), 2),
            "Avg Hours Held": round(subset["Hours Held"].mean(), 0),
        })
    return reason_stats


def _build_stats_rows(
    args, trades_df: pd.DataFrame, closed: pd.DataFrame,
    summary_rows: list[dict], skipped: list[str],
) -> list[dict]:
    """The flat metric/value rows written to the 'Stats' summary sheet."""
    total_pl = trades_df["Net P&L"].sum()
    total_costs = trades_df["Costs"].sum()
    n_trades = len(trades_df)
    n_wins = (closed["Net P&L"] > 0).sum() if not closed.empty else 0
    n_losses = (closed["Net P&L"] < 0).sum() if not closed.empty else 0
    n_open = (trades_df["Status"] == "OPEN").sum()
    avg_win = closed[closed["Net P&L"] > 0]["Net P&L"].mean() if n_wins > 0 else 0
    avg_loss = closed[closed["Net P&L"] < 0]["Net P&L"].mean() if n_losses > 0 else 0
    avg_hours = closed["Hours Held"].mean() if not closed.empty else 0
    capital_deployed = n_trades * args.lot_size
    best_trade = trades_df.loc[trades_df["Net P&L"].idxmax()] if len(trades_df) else None
    worst_trade = trades_df.loc[trades_df["Net P&L"].idxmin()] if len(trades_df) else None

    stats_data = [
        {"Metric": "Engine", "Value": "Quant HMM (HMM regime probabilities, hourly data)"},
        {"Metric": "Start Date", "Value": args.start_date},
        {"Metric": "Lot Size", "Value": f"GBP{args.lot_size:.0f}"},
        {"Metric": "Trade Cost", "Value": f"GBP{args.trade_cost:.0f}"},
        {"Metric": "Entry Threshold", "Value": f"P(Bull) > {args.entry_prob}"},
        {"Metric": "Exit Threshold", "Value": f"P(Bull) < {args.exit_prob}"},
        {"Metric": "Stop Loss", "Value": f"{args.stop_loss*100:.0f}%"},
        {"Metric": "Take Profit", "Value": f"{args.take_profit*100:.0f}%"},
        {"Metric": "Volume Min", "Value": f"{args.volume_min}x avg"},
        {"Metric": "Kelly Sizing", "Value": "On" if args.kelly else "Off"},
        {"Metric": "Vol Screen", "Value": f"On (min trend_quality={args.min_trend_quality})" if args.vol_screen else "Off"},
        {"Metric": "", "Value": ""},
        {"Metric": "Total Trades", "Value": n_trades},
        {"Metric": "Closed Trades", "Value": int(len(closed)) if not closed.empty else 0},
        {"Metric": "Open Positions", "Value": int(n_open)},
        {"Metric": "Wins", "Value": int(n_wins)},
        {"Metric": "Losses", "Value": int(n_losses)},
        {"Metric": "Win Rate", "Value": f"{n_wins/(n_wins+n_losses)*100:.1f}%" if (n_wins+n_losses) else "N/A"},
        {"Metric": "", "Value": ""},
        {"Metric": "Total Net P&L", "Value": f"GBP{total_pl:,.2f}"},
        {"Metric": "Total Costs", "Value": f"GBP{total_costs:,.2f}"},
        {"Metric": "Avg Win", "Value": f"GBP{avg_win:,.2f}"},
        {"Metric": "Avg Loss", "Value": f"GBP{avg_loss:,.2f}"},
        {"Metric": "Avg Hours Held", "Value": f"{avg_hours:.0f}"},
        {"Metric": "Capital Deployed", "Value": f"GBP{capital_deployed:,.0f}"},
        {"Metric": "Return on Capital", "Value": f"{total_pl/capital_deployed*100:.2f}%" if capital_deployed else "N/A"},
        {"Metric": "", "Value": ""},
        {"Metric": "Tickers with Trades", "Value": len(summary_rows)},
        {"Metric": "Profitable Tickers", "Value": sum(1 for s in summary_rows if s["Total Net P&L"] > 0)},
        {"Metric": "Tickers Skipped", "Value": len(skipped)},
    ]
    if best_trade is not None:
        stats_data.append({"Metric": "Best Trade", "Value": f"{best_trade['Ticker']} {best_trade['Entry Date']} GBP{best_trade['Net P&L']:+.2f}"})
    if worst_trade is not None:
        stats_data.append({"Metric": "Worst Trade", "Value": f"{worst_trade['Ticker']} {worst_trade['Entry Date']} GBP{worst_trade['Net P&L']:+.2f}"})
    return stats_data


def _write_excel_report(
    args, trades_df: pd.DataFrame, summary_df: pd.DataFrame,
    sentiment_rows: list[dict], vol_profiles_by_ticker: dict, skipped: list[str],
) -> None:
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

        if sentiment_rows:
            pd.DataFrame(sentiment_rows).to_excel(writer, sheet_name="Sentiment", index=False)

        if vol_profiles_by_ticker:
            vol_df = pd.DataFrame(vol_profiles_by_ticker.values()).rename(columns={
                "ticker": "Ticker", "ann_vol": "Ann Vol", "efficiency_ratio": "Efficiency Ratio",
                "autocorr": "Autocorr", "choppiness_idx": "Choppiness Idx",
                "sign_change_freq": "Sign Change Freq", "trend_quality": "Trend Quality",
            })
            vol_df["Kept"] = vol_df["Trend Quality"] >= args.min_trend_quality
            vol_df = vol_df.sort_values("Trend Quality", ascending=False)
            vol_df.to_excel(writer, sheet_name="Vol Screen", index=False)

        closed = trades_df[trades_df["Status"] == "CLOSED"]
        if not closed.empty:
            reason_stats = _build_exit_breakdown_rows(closed)
            pd.DataFrame(reason_stats).sort_values("Count", ascending=False).to_excel(
                writer, sheet_name="Exit Breakdown", index=False)

        summary_rows = summary_df.to_dict("records")
        stats_data = _build_stats_rows(args, trades_df, closed, summary_rows, skipped)
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="Stats", index=False)


def _print_final_summary(trades_df: pd.DataFrame) -> None:
    total_pl = trades_df["Net P&L"].sum()
    closed = trades_df[trades_df["Status"] == "CLOSED"]
    n_wins = (closed["Net P&L"] > 0).sum() if not closed.empty else 0
    n_losses = (closed["Net P&L"] < 0).sum() if not closed.empty else 0
    n_open = (trades_df["Status"] == "OPEN").sum()
    print(f"\n  Total trades:  {len(trades_df)}")
    print(f"  Wins/Losses:   {n_wins}/{n_losses} (open: {n_open})")
    print(f"  Total Net P&L: GBP{total_pl:,.2f}")
    if (n_wins + n_losses) > 0:
        print(f"  Win rate:      {n_wins/(n_wins+n_losses)*100:.1f}%")


def main() -> int:
    args = _build_arg_parser().parse_args()

    start_date = pd.Timestamp(args.start_date).tz_localize(None)

    tickers = _load_watchlist_tickers(args.watchlist)
    if not tickers:
        print("No tickers found in watchlist.")
        return 1

    print(f"Quant HMM Trade Report: {len(tickers)} tickers, start={args.start_date}")
    print(f"  Entry: P(Bull)>{args.entry_prob}  Exit: P(Bull)<{args.exit_prob}")
    print(f"  Stop: {args.stop_loss*100:.0f}%  Target: {args.take_profit*100:.0f}%")
    print(f"  Lot: GBP{args.lot_size:.0f}  Cost: GBP{args.trade_cost:.0f}  Kelly: {'on' if args.kelly else 'off'}")

    # Volatility-character pre-screen: drop tickers with choppy/mean-reverting
    # price action where the HMM regime strategy historically underperforms
    # (see vol_screen.py — efficiency ratio & autocorrelation are the strongest predictors)
    vol_profiles_by_ticker = {}
    if args.vol_screen:
        print(f"\n  Screening {len(tickers)} tickers for volatility character (trend_quality >= {args.min_trend_quality})...")
        kept_tickers, vol_profiles = screen_tickers(
            tickers, min_trend_quality=args.min_trend_quality, verbose=False)
        vol_profiles_by_ticker = {p["ticker"]: p for p in vol_profiles}
        excluded = [t for t in tickers if t not in kept_tickers]
        print(f"  Kept {len(kept_tickers)}/{len(tickers)} tickers; excluded as choppy: "
              f"{', '.join(excluded) if excluded else '(none)'}")
        tickers = kept_tickers

    # Fetch VIX regime once (shared across all tickers)
    vix_sig = 0
    if args.sentiment:
        print(f"  Fetching VIX regime...")
        vix_data = vix_regime()
        vix_sig = vix_data.get("vix_signal", 0)
        print(f"  VIX: {vix_data.get('vix_current', 'N/A')} "
              f"({vix_data.get('vix_regime', 'N/A')}) "
              f"term={vix_data.get('vix_term_structure', 'N/A')}")

    all_trades = []
    summary_rows = []
    sentiment_rows = []
    skipped = []
    t0 = time.time()

    for i, ticker in enumerate(tickers, 1):
        elapsed = time.time() - t0
        if i == 1 or i % 5 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker} ({elapsed:.0f}s)")

        df = fetch_hourly(ticker, period=args.period)
        if df is None or len(df) < 600:
            skipped.append(ticker)
            continue

        daily_df = _fetch_daily_indicators_table(ticker, period=args.period)

        # Per-ticker sentiment (options, insider, short interest)
        sent_score = 0.0
        sent_data = None
        if args.sentiment:
            try:
                sent_data = composite_sentiment(ticker, include_vix=False)
                sent_score = sent_data.get("sentiment_score", 0.0)
            except Exception:
                sent_data = None

        try:
            bt = quant_backtest(
                df,
                entry_prob=args.entry_prob,
                exit_prob=args.exit_prob,
                stop_loss_pct=args.stop_loss,
                take_profit_pct=args.take_profit,
                volume_min_ratio=args.volume_min,
                initial_cash=args.lot_size,
                trade_cost=args.trade_cost,
                use_kelly=args.kelly,
                sentiment_score=sent_score,
                vix_signal=vix_sig,
                regime_smooth=args.regime_smooth,
                min_hold_bars=args.min_hold,
            )
        except Exception as e:
            print(f"    {ticker}: backtest failed: {e}")
            skipped.append(ticker)
            continue

        detail = bt["detail"]
        if detail.empty:
            skipped.append(ticker)
            continue

        company_name, sector = _get_company_info(ticker)

        buys = detail[detail["trade_event"] == "BUY"]
        sells = detail[detail["trade_event"] == "SELL"]
        buy_dates = buys.index.tolist()
        sell_dates = sells.index.tolist()

        for buy_date in buy_dates:
            trade_row = _build_trade_row(
                ticker, company_name, sector, buy_date, buys.loc[buy_date],
                sell_dates, sells, detail, start_date, args, sent_data, sent_score,
                daily_df=daily_df,
                vol_profile=vol_profiles_by_ticker.get(ticker),
            )
            if trade_row is not None:
                all_trades.append(trade_row)

        # Per-ticker summary
        ticker_trades = [t for t in all_trades if t["Ticker"] == ticker]
        if ticker_trades:
            summary_rows.append(_build_ticker_summary_row(
                ticker, company_name, sector, ticker_trades, bt,
                sent_data, sent_score, vol_profiles_by_ticker,
            ))
            if sent_data:
                sentiment_rows.append(_build_sentiment_row(ticker, company_name, sent_data, sent_score))

    elapsed = time.time() - t0
    print(f"\n  Done: {len(all_trades)} trades from {len(summary_rows)} tickers in {elapsed:.0f}s")
    if skipped:
        print(f"  Skipped: {len(skipped)} tickers (no data / too short)")

    if not all_trades:
        print("  No trades found.")
        return 1

    trades_df = pd.DataFrame(all_trades)
    summary_df = pd.DataFrame(summary_rows).sort_values("Total Net P&L", ascending=False)

    _write_excel_report(args, trades_df, summary_df, sentiment_rows, vol_profiles_by_ticker, skipped)
    print(f"\n  Report saved: {args.output}")

    _print_final_summary(trades_df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
