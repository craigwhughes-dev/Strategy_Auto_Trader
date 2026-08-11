"""
vol_threshold_sweep.py — Test max_downside_vol thresholds for optimised strategy.

Phase 1 (default): profiles top-K tickers, sweeps thresholds in memory.
Phase 2 (--live-sim): runs live_sim on top-K intersection for chosen thresholds.

Usage:
    # Phase 1 only (fast, ~15 min):
    uv run python scripts/vol_threshold_sweep.py

    # Phase 2: run live_sim for given thresholds (slow, ~45 min each):
    uv run python scripts/vol_threshold_sweep.py --live-sim --thresholds 0.35 0.40 None
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_top_k(path: Path) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    return data["tickers"]


def classify_tickers(tickers: list[str], ftse_wl: Path, sp500_wl: Path) -> dict[str, str]:
    """Returns {ticker: 'ftse'|'sp500'|'unknown'}."""
    def load_wl(p: Path) -> set[str]:
        with open(p) as f:
            wl = json.load(f)
        items = wl.get("tickers", [])
        return {t["ticker"] if isinstance(t, dict) else t for t in items}

    ftse_set = load_wl(ftse_wl)
    sp500_set = load_wl(sp500_wl)
    return {
        t: ("ftse" if t in ftse_set else "sp500" if t in sp500_set else "unknown")
        for t in tickers
    }


def fetch_profiles(tickers: list[str], period: str = "2y") -> list[dict]:
    """Download vol profiles for all tickers. Returns list of profile dicts."""
    sys.path.insert(0, str(ROOT))
    from Strategy_Auto_Trader.quant_hmm.vol_screen import volatility_profile

    profiles = []
    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        prof = volatility_profile(ticker, period=period)
        if prof is None:
            print("FAILED")
        else:
            print(f"downside_vol={prof['downside_vol']:.3f} trend_quality={prof['trend_quality']:.2f}")
            profiles.append(prof)
    return profiles


def sweep_thresholds(
    profiles: list[dict],
    market_by_ticker: dict[str, str],
    thresholds: list[float | None],
    min_trend_quality: float = 0.0,
) -> None:
    """Print threshold sweep table."""
    print("\n" + "=" * 72)
    print(f"{'Threshold':>12}  {'Total':>6}  {'FTSE':>6}  {'S&P500':>8}  Tickers in top-K")
    print("=" * 72)

    for thresh in thresholds:
        kept = [
            p for p in profiles
            if p["trend_quality"] >= min_trend_quality
            and (thresh is None or p["downside_vol"] <= thresh)
        ]
        kept_tickers = [p["ticker"] for p in kept]
        ftse_kept = [t for t in kept_tickers if market_by_ticker.get(t) == "ftse"]
        sp_kept = [t for t in kept_tickers if market_by_ticker.get(t) == "sp500"]
        label = "None (no cap)" if thresh is None else f"{thresh:.2f}"
        ticker_str = ", ".join(sorted(kept_tickers)) if kept_tickers else "(none)"
        print(f"{label:>12}  {len(kept_tickers):>6}  {len(ftse_kept):>6}  {len(sp_kept):>8}  {ticker_str}")

    print("=" * 72)


def print_profiles_table(profiles: list[dict], market_by_ticker: dict[str, str]) -> None:
    """Print full profile table sorted by downside_vol."""
    profiles_sorted = sorted(profiles, key=lambda p: p["downside_vol"])
    print("\nFull profile table (sorted by downside_vol asc):")
    print(f"{'Ticker':12} {'Market':6} {'DownVol':>8} {'AnnVol':>8} {'TrendQ':>7} {'EffRatio':>9} {'Autocorr':>9} {'SignChg':>8}")
    print("-" * 80)
    for p in profiles_sorted:
        m = market_by_ticker.get(p["ticker"], "?")
        print(
            f"{p['ticker']:12} {m:6} {p['downside_vol']:>8.3f} {p['ann_vol']:>8.3f}"
            f" {p['trend_quality']:>7.2f} {p['efficiency_ratio']:>9.3f}"
            f" {p['autocorr']:>9.3f} {p['sign_change_freq']:>8.3f}"
        )


def save_profiles_csv(profiles: list[dict], market_by_ticker: dict[str, str], out: Path) -> None:
    import csv
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ticker", "market", "downside_vol", "ann_vol", "trend_quality",
            "efficiency_ratio", "autocorr", "choppiness_idx", "sign_change_freq",
        ])
        writer.writeheader()
        for p in sorted(profiles, key=lambda x: x["downside_vol"]):
            writer.writerow({
                "ticker": p["ticker"],
                "market": market_by_ticker.get(p["ticker"], "unknown"),
                "downside_vol": p["downside_vol"],
                "ann_vol": p["ann_vol"],
                "trend_quality": p["trend_quality"],
                "efficiency_ratio": p["efficiency_ratio"],
                "autocorr": p["autocorr"],
                "choppiness_idx": p.get("choppiness_idx"),
                "sign_change_freq": p["sign_change_freq"],
            })
    print(f"\nProfiles saved: {out}")


def run_live_sim(tickers: list[str], threshold: float | None, out_dir: Path) -> None:
    label = f"{threshold:.2f}".replace(".", "p") if threshold is not None else "none"
    journal = out_dir / f"vol_sweep_{label}.csv"
    pos_summary = out_dir / f"vol_sweep_{label}_pos_summary.csv"
    log = out_dir / f"vol_sweep_{label}.log"

    cmd = [
        "uv", "run", "python", "-m", "Strategy_Auto_Trader.markov_cli.live_sim",
        "--tickers", *tickers,
        "--strategies", "optimised",
        "--initial-cash", "10000",
        "--start-date", "2000-01-01",
        "--max-trades-per-day", "0",
        "--cost-model", "ibkr_tiered_spread",
        "--journal", str(journal),
        "--position-summary", str(pos_summary),
    ]

    thresh_label = f"{threshold:.2f}" if threshold is not None else "None"
    print(f"\nRunning live_sim for max_downside_vol={thresh_label} ({len(tickers)} tickers)...")
    print(f"  log: {log}")

    with open(log, "w") as logf:
        result = subprocess.run(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) — check {log}")
    else:
        print(f"  Done. journal: {journal}")
        _summarise_position_summary(pos_summary, thresh_label)


def _summarise_position_summary(path: Path, label: str) -> None:
    if not path.exists():
        print(f"  No position summary at {path}")
        return
    import csv
    rows = []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("  Position summary empty")
        return
    # Grab last row per (strategy, pot_size) as final equity
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("strategy"), r.get("pot_size"))
        by_key[key] = r
    print(f"\n  threshold={label} results:")
    for (strategy, pot), r in sorted(by_key.items()):
        equity = r.get("equity", "?")
        sharpe = r.get("sharpe", "?")
        max_dd = r.get("max_drawdown", "?")
        print(f"    strategy={strategy} pot={pot} equity={equity} sharpe={sharpe} max_dd={max_dd}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds", nargs="+", default=["0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50", "None"],
        help="max_downside_vol values to sweep (use 'None' for no cap)",
    )
    parser.add_argument("--min-trend-quality", type=float, default=0.0)
    parser.add_argument("--period", default="2y")
    parser.add_argument(
        "--top-k-file", default=str(ROOT / "state" / "top_k_universe.json"),
        help="Path to top_k_universe.json",
    )
    parser.add_argument(
        "--live-sim", action="store_true",
        help="Phase 2: run live_sim for tickers passing each threshold",
    )
    parser.add_argument(
        "--profiles-csv", default=str(ROOT / "scripts" / "vol_threshold_profiles.csv"),
        help="Output path for profiles CSV",
    )
    args = parser.parse_args()

    # Parse thresholds
    thresholds: list[float | None] = []
    for t in args.thresholds:
        thresholds.append(None if t.lower() == "none" else float(t))

    top_k_path = Path(args.top_k_file)
    if not top_k_path.exists():
        print(f"ERROR: top_k_universe.json not found at {top_k_path}")
        return 1

    tickers = load_top_k(top_k_path)
    print(f"Top-K universe: {len(tickers)} tickers from {top_k_path}")

    market_by_ticker = classify_tickers(
        tickers,
        ROOT / "config" / "watchlist_ftse.json",
        ROOT / "config" / "watchlist_sp500.json",
    )

    print(f"\nFetching vol profiles (period={args.period})...")
    profiles = fetch_profiles(tickers, period=args.period)
    print(f"\nProfiled {len(profiles)}/{len(tickers)} tickers successfully.")

    print_profiles_table(profiles, market_by_ticker)
    sweep_thresholds(profiles, market_by_ticker, thresholds, args.min_trend_quality)
    save_profiles_csv(profiles, market_by_ticker, Path(args.profiles_csv))

    if args.live_sim:
        out_dir = ROOT / "scripts"
        for thresh in thresholds:
            kept = [
                p["ticker"] for p in profiles
                if p["trend_quality"] >= args.min_trend_quality
                and (thresh is None or p["downside_vol"] <= thresh)
            ]
            if not kept:
                label = f"{thresh:.2f}" if thresh is not None else "None"
                print(f"\nSkipping threshold={label}: 0 tickers pass.")
                continue
            run_live_sim(kept, thresh, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
