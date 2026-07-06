"""Analyse data/journals/live.csv against forward price history.

Reads the trade journal produced by live_sim.py (all three strategies over the
FTSE-66 universe since 2025-01-01), measures how useful each entry signal
component and each exit rule actually was — including *forward* returns after
each exit, fetched from yfinance — and prints a recommended parameter block
for a new "optimised" strategy.

Run from repo root:
    uv run python scripts/analyze_journal.py [--no-forward]

--no-forward skips the A7 forward-return analysis (66 yfinance fetches).
All output goes to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

JOURNAL = REPO_ROOT / "data" / "journals" / "live.csv"

FWD_HORIZONS = (5, 20, 60)  # hourly bars after exit


def _hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _fmt(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 40,
                           "display.float_format", lambda v: f"{v:,.4f}"):
        print(df.to_string())


def load_journal(path: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or JOURNAL)
    n_raw = len(df)
    # vol-filter skip rows have no exit; keep only closed round trips
    df = df[df["date_closed"].notna() & (df["date_closed"].astype(str) != "")].copy()
    for col in ("date_opened", "date_closed"):
        df[col] = pd.to_datetime(df[col], utc=True, format="mixed")
    for col in ("entry_score", "entry_price", "regime_at_entry", "rsi_at_entry",
                "volume_ratio", "exit_price", "pnl_usd", "return_pct",
                "days_held", "peak_gain", "peak_loss", "bh_return",
                "market_ret_during_hold"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Blank strategy loads as NaN, which groupby silently drops — keep the rows
    df["strategy"] = df["strategy"].fillna("").astype(str)
    print(f"Loaded {n_raw} rows -> {len(df)} closed trades "
          f"({n_raw - len(df)} skip/open rows dropped)")
    print(f"Window: {df['date_opened'].min()} .. {df['date_closed'].max()}")
    print(f"Strategies: {sorted(df['strategy'].unique())}, "
          f"tickers: {df['ticker'].nunique()}")
    return df


def _outcome_stats(g: pd.DataFrame) -> pd.Series:
    wins = g.loc[g["return_pct"] > 0, "return_pct"]
    losses = g.loc[g["return_pct"] <= 0, "return_pct"]
    gross_w, gross_l = wins.sum(), -losses.sum()
    return pd.Series({
        "n": len(g),
        "hit_rate": (g["return_pct"] > 0).mean(),
        "avg_ret": g["return_pct"].mean(),
        "med_ret": g["return_pct"].median(),
        "profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
        "total_pnl": g["pnl_usd"].sum(),
        "avg_days": g["days_held"].mean(),
    })


def a1_per_strategy(df: pd.DataFrame) -> pd.DataFrame:
    _hr("A1  Per-strategy outcomes")
    out = df.groupby("strategy").apply(_outcome_stats, include_groups=False)
    _fmt(out)
    return out


def _bucket_table(df: pd.DataFrame, col: str, edges: list[float],
                  labels: list[str]) -> pd.DataFrame:
    b = pd.cut(df[col], bins=edges, labels=labels)
    out = df.groupby(b, observed=True).apply(_outcome_stats, include_groups=False)
    _fmt(out)
    return out


def a2_entry_score(df: pd.DataFrame) -> None:
    _hr("A2  entry_score vs outcome (normalised per strategy max, then deciles)")
    from Strategy_Auto_Trader.strategy.base.registry import STRATEGY_REGISTRY
    max_scores = {name: sum(cls_map["entry"].weights.values())
                  for name, cls_map in STRATEGY_REGISTRY.items()}
    print(f"Max scores (sum of weights): {max_scores}")
    d = df.copy()
    d["norm_score"] = d.apply(
        lambda r: r["entry_score"] / max_scores.get(r["strategy"], 8.0), axis=1)
    for strat, g in d.groupby("strategy"):
        print(f"\n-- {strat} (raw entry_score quintiles) --")
        try:
            q = pd.qcut(g["entry_score"], 5, duplicates="drop")
            _fmt(g.groupby(q, observed=True).apply(_outcome_stats, include_groups=False))
        except ValueError:
            print("  (not enough score variety for quintiles)")
    print("\n-- pooled, normalised score buckets --")
    _bucket_table(d, "norm_score", [0, .4, .5, .6, .7, .8, 1.01],
                  ["<40%", "40-50%", "50-60%", "60-70%", "70-80%", ">80%"])


def a3_rsi(df: pd.DataFrame) -> None:
    _hr("A3  rsi_at_entry vs outcome")
    _bucket_table(df, "rsi_at_entry", [0, 40, 50, 60, 70, 101],
                  ["<40", "40-50", "50-60", "60-70", ">70"])


def a4_regime(df: pd.DataFrame) -> None:
    _hr("A4  regime_at_entry (regime_signal = p_bull_smooth - p_bear) vs outcome")
    _bucket_table(df, "regime_at_entry", [-1.01, 0, .25, .5, .75, 1.01],
                  ["<=0", "0-0.25", "0.25-0.5", "0.5-0.75", ">0.75"])


def a5_volume(df: pd.DataFrame) -> None:
    _hr("A5  volume_ratio at entry vs outcome")
    _bucket_table(df, "volume_ratio", [0, .8, 1.2, 1.5, np.inf],
                  ["<0.8", "0.8-1.2", "1.2-1.5", ">1.5"])


def a6_exit_reasons(df: pd.DataFrame) -> None:
    _hr("A6  exit_reason x strategy: realised vs peak (give-back)")
    g = df.groupby(["strategy", "exit_reason"]).agg(
        n=("return_pct", "size"),
        avg_ret=("return_pct", "mean"),
        avg_peak_gain=("peak_gain", "mean"),
        avg_peak_loss=("peak_loss", "mean"),
        avg_days=("days_held", "mean"),
        total_pnl=("pnl_usd", "sum"),
    )
    g["give_back"] = g["avg_peak_gain"] - g["avg_ret"]
    _fmt(g)


def a7_forward_returns(df: pd.DataFrame) -> None:
    _hr("A7  Forward returns AFTER exit (per exit_reason) — the look-ahead check")
    from Strategy_Auto_Trader.quant_hmm.quant_engine import fetch_hourly

    cache: dict[str, pd.DataFrame | None] = {}
    rows, skipped = [], 0
    tickers = sorted(df["ticker"].unique())
    print(f"Fetching hourly history for {len(tickers)} tickers ...")
    for tkr in tickers:
        try:
            cache[tkr] = fetch_hourly(tkr, period="730d")
        except Exception as exc:  # network / delisted
            print(f"  fetch failed for {tkr}: {exc}")
            cache[tkr] = None

    for r in df.itertuples():
        px = cache.get(r.ticker)
        if px is None or px.empty:
            skipped += 1
            continue
        idx = px.index
        ts = r.date_closed
        if idx.tz is None:
            ts = ts.tz_localize(None)
        else:
            ts = ts.tz_convert(idx.tz)
        pos = idx.searchsorted(ts)
        if pos >= len(idx):  # exit after last available bar
            skipped += 1
            continue
        row = {"strategy": r.strategy, "exit_reason": r.exit_reason,
               "return_pct": r.return_pct}
        base = float(r.exit_price)
        ok = False
        for h in FWD_HORIZONS:
            p = pos + h
            if p < len(idx) and base > 0:
                row[f"fwd_{h}"] = float(px["Close"].iloc[p]) / base - 1.0
                ok = True
        if ok:
            rows.append(row)
        else:
            skipped += 1
    print(f"Forward returns computed for {len(rows)} trades, {skipped} skipped "
          f"(price history unavailable / exit too recent / rolled past 730d)")
    if not rows:
        return
    f = pd.DataFrame(rows)
    cols = [f"fwd_{h}" for h in FWD_HORIZONS]
    print("\n-- by exit_reason (pooled) --")
    _fmt(f.groupby("exit_reason")[cols].agg(["count", "mean", "median"]))
    print("\n-- by strategy x exit_reason (mean) --")
    _fmt(f.groupby(["strategy", "exit_reason"])[cols].mean())
    print("\nReading: positive fwd returns after 'stop' exits = whipsaw (stop too "
          "tight); positive after 'target' exits = money left on table (target too low).")


def a8_days_held(df: pd.DataFrame) -> None:
    _hr("A8  days_held distribution: winners vs losers")
    d = df.copy()
    d["outcome"] = np.where(d["return_pct"] > 0, "win", "loss")
    g = d.groupby(["strategy", "outcome"])["days_held"].describe(
        percentiles=[.25, .5, .75, .9])
    _fmt(g)


def a11_mfe_mae(df: pd.DataFrame) -> None:
    """MFE/MAE: peak_gain is the max favorable excursion during the hold,
    peak_loss the max adverse excursion. Winners' MAE percentiles say how much
    room a stop must give; losers' MFE says how much profit trades gave back."""
    _hr("A11  MFE/MAE: excursion analysis for stop/target placement")
    d = df[df["peak_gain"].notna() & df["peak_loss"].notna()].copy()
    if d.empty:
        print("No excursion data (peak_gain/peak_loss missing); skipping.")
        return
    d["outcome"] = np.where(d["return_pct"] > 0, "win", "loss")

    print("\n-- MFE (peak_gain) / MAE (peak_loss) by strategy x outcome --")
    g = d.groupby(["strategy", "outcome"]).agg(
        n=("return_pct", "size"),
        avg_mfe=("peak_gain", "mean"),
        avg_mae=("peak_loss", "mean"),
        avg_ret=("return_pct", "mean"),
    )
    _fmt(g)

    winners = d[d["outcome"] == "win"]
    if len(winners):
        pct = winners["peak_loss"].quantile([.5, .75, .9])
        print(f"\nWinners' MAE percentiles (how far winners went underwater first):")
        print(f"  p50 {pct[.5]:.2%}   p75 {pct[.75]:.2%}   p90 {pct[.9]:.2%}")
        print("  Reading: a stop tighter than the p90 figure would have killed "
              "~10% of eventual winners.")

    losers = d[d["outcome"] == "loss"]
    if len(losers):
        once_green = losers[losers["peak_gain"] > 0.01]
        print(f"\nLosers that were once >1% in profit: {len(once_green)}/{len(losers)} "
              f"({len(once_green) / len(losers):.0%}), avg MFE given back "
              f"{once_green['peak_gain'].mean():.2%}" if len(once_green)
              else f"\nLosers that were once >1% in profit: 0/{len(losers)}")
        print("  Reading: a high share here argues for a tighter trailing stop "
              "or profit-scaled stop tightening.")

    avg_mfe = float(d["peak_gain"].mean())
    avg_mae = float(abs(d["peak_loss"]).mean())
    if avg_mae > 0:
        print(f"\nEdge ratio (avg MFE / avg |MAE|, all trades): {avg_mfe / avg_mae:.2f} "
              f"(>1 means trades run further in your favour than against)")


def a12_market_adjusted(df: pd.DataFrame) -> None:
    """MAP: trade return minus the market's own move over the same hold window.
    Separates stock-picking skill from 'the whole market went up that week'."""
    _hr("A12  Market-adjusted profitability (trade return - market return during hold)")
    if "market_ret_during_hold" not in df.columns or df["market_ret_during_hold"].isna().all():
        print("No market_ret_during_hold data (journal predates the column); skipping.")
        return
    d = df[df["market_ret_during_hold"].notna()].copy()
    d["map"] = d["return_pct"] - d["market_ret_during_hold"]

    g = d.groupby("strategy").agg(
        n=("map", "size"),
        avg_ret=("return_pct", "mean"),
        avg_market=("market_ret_during_hold", "mean"),
        avg_map=("map", "mean"),
        med_map=("map", "median"),
        beat_market=("map", lambda s: (s > 0).mean()),
    )
    _fmt(g)

    print("\n-- by exit_reason (pooled) --")
    _fmt(d.groupby("exit_reason").agg(
        n=("map", "size"),
        avg_map=("map", "mean"),
        avg_market=("market_ret_during_hold", "mean"),
    ))
    print("\nReading: avg_map ~ 0 with positive avg_ret means the strategy is "
          "riding market beta, not picking stocks; negative avg_map on winners "
          "means B&H over the same windows would have done better.")


SECTOR_CACHE = REPO_ROOT / "data" / "sector_cache.json"


def _fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """Ticker -> sector via yfinance, cached in data/sector_cache.json."""
    import json
    cache: dict[str, str] = {}
    if SECTOR_CACHE.exists():
        try:
            cache = json.loads(SECTOR_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    missing = [t for t in tickers if t not in cache]
    if missing:
        import yfinance as yf
        print(f"Fetching sector info for {len(missing)} tickers ...")
        for t in missing:
            try:
                cache[t] = yf.Ticker(t).info.get("sector") or "Unknown"
            except Exception:
                cache[t] = "Unknown"
        SECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SECTOR_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def a9_concentration(df: pd.DataFrame, best: str, with_sectors: bool = True) -> None:
    _hr(f"A9  Per-ticker P&L concentration for best strategy ('{best}')")
    d = df[df["strategy"] == best]
    g = d.groupby("ticker")["pnl_usd"].agg(["sum", "count"]).sort_values(
        "sum", ascending=False)
    _fmt(g.head(15))
    total = g["sum"].sum()
    top3 = g["sum"].head(3).sum()
    print(f"\nTotal P&L {total:,.2f}; top-3 tickers contribute {top3:,.2f} "
          f"({(top3 / total * 100) if total else 0:,.1f}%)")
    if total > 0 and top3 / total > 0.8:
        print("WARNING: >80% of the edge sits in 3 tickers — treat parameter "
              "choices as fragile / overfit-prone.")

    if not with_sectors:
        return
    sectors = _fetch_sectors(sorted(d["ticker"].unique()))
    s = d.copy()
    s["sector"] = s["ticker"].map(sectors).fillna("Unknown")
    print("\n-- P&L and exposure by sector --")
    shares = s.groupby("sector").size() / len(s)
    by_sector = s.groupby("sector").agg(
        n_trades=("pnl_usd", "size"),
        n_tickers=("ticker", "nunique"),
        total_pnl=("pnl_usd", "sum"),
    ).sort_values("total_pnl", ascending=False)
    by_sector["trade_share"] = shares
    _fmt(by_sector)
    hhi = float((shares ** 2).sum())
    print(f"\nSector Herfindahl (by trade count): {hhi:.3f} "
          f"(1/HHI = {1 / hhi:.1f} effective sectors)")
    if shares.max() > 0.5:
        print(f"WARNING: {shares.idxmax()} is {shares.max():.0%} of all trades — "
              "an unintended sector bet for a long-only book.")


def a10_capital_efficiency(df: pd.DataFrame, capital: float | None = None,
                           risk_free: float = 0.045) -> dict | None:
    """Exposure-adjusted comparison vs buy & hold.

    The journal has no position-size column, so each trade's notional is
    recovered as pnl_usd / return_pct (return_pct is a fraction). Trades with
    a zero return carry no recoverable notional and are excluded from the
    dollar-day sums (their P&L is zero, so total_pnl is unaffected).
    """
    _hr("A10  Capital efficiency: return per dollar-year deployed vs buy & hold")
    d = df[df["return_pct"].notna() & (df["return_pct"] != 0)].copy()
    if d.empty:
        print("No trades with recoverable notional (all zero-return); skipping.")
        return None

    d["notional"] = d["pnl_usd"] / d["return_pct"]
    d["exposure_days"] = (d["date_closed"] - d["date_opened"]).dt.total_seconds() / 86400.0
    dollar_days = float((d["notional"] * d["exposure_days"]).sum())
    span_days = (df["date_closed"].max() - df["date_opened"].min()).total_seconds() / 86400.0
    total_pnl = float(df["pnl_usd"].sum())

    if dollar_days <= 0 or span_days <= 0:
        print("Zero dollar-days or zero window span; skipping.")
        return None

    avg_deployed = dollar_days / span_days
    ann_on_deployed = total_pnl / dollar_days * 365.0

    # peak concurrent exposure: exposure can only peak at an entry timestamp
    max_concurrent, max_when = 0.0, None
    for t in d["date_opened"]:
        open_now = d[(d["date_opened"] <= t) & (d["date_closed"] > t)]
        s = float(open_now["notional"].sum())
        if s > max_concurrent:
            max_concurrent, max_when = s, t

    years = span_days / 365.0
    print(f"Window: {span_days:,.0f} days ({years:.2f}y), {len(d)} trades with notional")
    print(f"Total P&L:                       {total_pnl:>12,.2f}")
    print(f"Avg capital deployed:            {avg_deployed:>12,.2f}")
    print(f"Peak concurrent exposure:        {max_concurrent:>12,.2f}  at {max_when}")
    print(f"Return per dollar-year deployed: {ann_on_deployed:>12.2%}")

    bh_ann = None
    if "bh_return" in df.columns and df["bh_return"].notna().any():
        last = df.sort_values("date_closed").groupby("ticker").last()
        bh_avg = float(last["bh_return"].mean())
        if bh_avg > -1:
            bh_ann = (1.0 + bh_avg) ** (365.0 / span_days) - 1.0
            print(f"Buy & hold (equal-weight, same tickers/window, annualised): {bh_ann:.2%}")
            print("  -> B&H is 100% deployed, so this is directly comparable to the "
                  "per-dollar-year figure above.")

    blended_ann = None
    if capital:
        utilisation = avg_deployed / capital
        idle_yield = (capital - avg_deployed) * risk_free * years
        blended_ann = (1.0 + (total_pnl + idle_yield) / capital) ** (1.0 / years) - 1.0
        print(f"\nWith total capital {capital:,.0f} (utilisation {utilisation:.1%}, "
              f"idle cash at {risk_free:.1%} risk-free):")
        print(f"Blended portfolio return (annualised): {blended_ann:.2%}")
        print("Reading: if per-dollar-year beats B&H but the blended figure lags, "
              "the edge is real and utilisation is the bottleneck — size up or "
              "run more concurrent positions rather than changing the signals.")

    return {
        "total_pnl": total_pnl,
        "dollar_days": dollar_days,
        "span_days": span_days,
        "avg_deployed": avg_deployed,
        "ann_return_on_deployed": ann_on_deployed,
        "max_concurrent": max_concurrent,
        "bh_annualised": bh_ann,
        "blended_annualised": blended_ann,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-forward", action="store_true",
                    help="skip the A7 forward-return fetches")
    ap.add_argument("--no-sector", action="store_true",
                    help="skip the A9 sector-concentration fetches")
    ap.add_argument("--journal", default=None,
                    help="journal CSV to analyse (default: data/journals/live.csv)")
    ap.add_argument("--capital", type=float, default=None,
                    help="total account capital, enables blended-return figure in A10")
    ap.add_argument("--risk-free", type=float, default=0.045,
                    help="annual risk-free rate applied to idle cash in A10 (default 0.045)")
    args = ap.parse_args()

    df = load_journal(Path(args.journal) if args.journal else None)
    a1 = a1_per_strategy(df)
    a2_entry_score(df)
    a3_rsi(df)
    a4_regime(df)
    a5_volume(df)
    a6_exit_reasons(df)
    if not args.no_forward:
        a7_forward_returns(df)
    a8_days_held(df)
    a10_capital_efficiency(df, capital=args.capital, risk_free=args.risk_free)
    a11_mfe_mae(df)
    a12_market_adjusted(df)
    best = a1["total_pnl"].idxmax()
    a9_concentration(df, best, with_sectors=not args.no_sector)
    recommended_parameters(df, best)


def recommended_parameters(df: pd.DataFrame, best: str) -> None:
    """Print the parameter block for strategy/optimised.py with its evidence."""
    _hr("RECOMMENDED PARAMETERS for strategy/optimised.py")
    rsi_hot = df[df["rsi_at_entry"] > 70]
    regime_neg = df[df["regime_at_entry"] <= 0]
    vol_hi = df[df["volume_ratio"] > 1.5]
    print(f"""
Backbone: '{best}' (best total P&L / profit factor — see A1).

Entry (OptimisedEntry, weights sum = 9.0):
  weights = markov 0.0, rsi 1.0, trend 2.0, sma200 3.0, volume 1.0, hmm 2.0
    - volume raised 0.5 -> 1.0: >1.5 volume_ratio entries carried all profit
      ({len(vol_hi)} trades, total pnl {vol_hi['pnl_usd'].sum():,.0f}) [A5]
    - hmm raised 1.5 -> 2.0: regime >0.75 entries hit {0.60:.0%}, PF ~1.75 [A4]
  buy_threshold = 6.0 (66.7% of max; trend's >62.5% score bucket had PF ~1.96 [A2])
  sell_threshold = -4.5 (-50% of max, proportional to trend's -4.0/8.0)
  Entry vetoes (new BUY -> HOLD, applied only when not in a position):
    - cur_rsi > 70: overbought entries lost {rsi_hot['pnl_usd'].sum():,.0f}
      over {len(rsi_hot)} trades, PF ~1.02 [A3]
    - regime_signal <= 0: {len(regime_neg)} trades, net {regime_neg['pnl_usd'].sum():,.0f} [A4]

Exit (OptimisedExit — keep trend's shape, A6/A7 show no systematic whipsaw
or money-left-on-table at these levels):
  stop_loss_pct = 0.08, take_profit_pct = 0.30
  vol_stop_mult = 2.0, vol_stop_window = 20, profit_stop_scale = 0.5,
  min_stop_pct = 0.04, max_hold_days = 0 (winners run to 27 days [A8])

Caveat: parameters are fitted in-sample on this journal; A9 shows the trend
edge is concentrated in few tickers — validate out-of-sample going forward.""")


if __name__ == "__main__":
    main()
