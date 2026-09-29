"""X0b driver: reconcile every scored episode's peak/trough closes in the event register.

Run:  uv run python scripts/tier_analysis/x0b_reconcile_register.py

Loads the register and each market's cached daily closes, runs both reconciliation tracks
(see Strategy_Auto_Trader/research/reconcile.py), prints a per-market summary plus every flagged
or dropped episode, and writes a full per-episode report to
data/cache/yahoo_index/x0b_reconciliation.csv.

Fetches FRED NASDAQCOM (^IXIC) and SP500 (^GSPC) once each; everything else is intrinsic-only.
Research only; no engine or production code is touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from Strategy_Auto_Trader.research import reconcile as R
from Strategy_Auto_Trader.research.index_history import CACHE_DIR, cache_path

REGISTER = CACHE_DIR / "event_register.csv"
REPORT = CACHE_DIR / "x0b_reconciliation.csv"


def _load_close(market: str) -> pd.Series:
    path = cache_path(market)
    df = pd.read_csv(path, index_col=0, comment="#")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).normalize()
    return pd.Series(df["close"].to_numpy(dtype=float), index=idx, name=market).sort_index()


def main() -> None:
    reg = pd.read_csv(REGISTER)
    markets = sorted(reg["market"].unique())

    closes = {m: _load_close(m) for m in markets}
    # Pre-scan each series once (the scans are O(n) and reused across that market's episodes).
    flags = {
        m: {
            "unit_shift": R.scan_unit_shift(closes[m]),
            "spike": R.scan_spike(closes[m]),
            "stale": R.scan_stale(closes[m]),
        }
        for m in markets
    }

    fred: dict[str, tuple[pd.Series, pd.Timestamp]] = {}
    for sym, (sid, from_str) in R.SECOND_SOURCE.items():
        if sym in markets:
            try:
                series = R.fetch_fred_daily(sid, start=from_str)
                fred[sym] = (series, pd.Timestamp(from_str, tz="UTC"))
                print(f"FRED {sid} for {sym}: {len(series)} rows, {series.index.min().date()}..{series.index.max().date()}")
            except Exception as e:  # noqa: BLE001 - a fetch failure just leaves that market intrinsic-only
                print(f"FRED {sid} for {sym}: FETCH FAILED ({type(e).__name__}: {e}); intrinsic-only")

    results: list[R.EpisodeReconResult] = []
    for _, row in reg.iterrows():
        m = row["market"]
        f_series, f_from = fred.get(m, (None, None))
        results.append(
            R.reconcile_episode(row, closes[m], f_series, f_from, precomputed_flags=flags[m])
        )

    out = pd.DataFrame(
        [
            {
                "market": r.market,
                "peak_date": r.peak_date.date(),
                "trough_date": r.trough_date.date(),
                "stored_depth_pct": r.stored_depth_pct,
                "recomputed_depth_pct": r.recomputed_depth_pct,
                "depth_ok": r.depth_ok,
                "confirmation": r.confirmation,
                "second_source": r.second_source,
                "peak_rel_err": None if r.peak_rel_err is None else round(r.peak_rel_err, 5),
                "trough_rel_err": None if r.trough_rel_err is None else round(r.trough_rel_err, 5),
                "defects": ";".join(r.defect_flags),
                "dropped": r.dropped,
            }
            for r in results
        ]
    )
    out.to_csv(REPORT, index=False)

    total = len(results)
    dropped = out["dropped"].sum()
    conf = out["confirmation"].value_counts().to_dict()

    print(f"\n{'='*70}\nX0b RECONCILIATION SUMMARY\n{'='*70}")
    print(f"episodes: {total} | dropped: {dropped} | kept: {total - dropped}")
    print(f"confirmation: {conf}")
    print("\nper market:")
    for m in markets:
        sub = out[out["market"] == m]
        sec = sub[sub["confirmation"] == "second-source"]
        errs = pd.concat([sec["peak_rel_err"], sec["trough_rel_err"]]).dropna()
        maxerr = f"max_rel_err={errs.max():.4f}" if len(errs) else "intrinsic-only"
        print(f"  {m:>8}  n={len(sub):>2}  depth_ok={sub['depth_ok'].sum():>2}/{len(sub):>2}  "
              f"dropped={sub['dropped'].sum()}  {maxerr}")

    flagged = out[(~out["depth_ok"]) | (out["defects"] != "") | (out["confirmation"] == "second-source-FAIL")]
    if len(flagged):
        print(f"\n{'='*70}\nFLAGGED / DROPPED EPISODES\n{'='*70}")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(flagged.to_string(index=False))
    else:
        print("\nNo episodes flagged: all depth-consistent, no intrinsic defects, all cross-checks within tolerance.")

    print(f"\nreport written: {REPORT}")


if __name__ == "__main__":
    main()
