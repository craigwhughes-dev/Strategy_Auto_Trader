"""X2 driver: out-of-sample test of the deployed exit across 27 shocks / 12 markets / 6 decades.

Run:  uv run python scripts/tier_analysis/x2_exit_generalisation.py

Applies the 21-day realized-vol deadband (X1's cleared window) to every market's own index under two
transfer rules (fixed_pctl PRIMARY, absolute as the reading-3 confound check), scores protection /
exit_lag / idle_rate per episode, aggregates by shock, and compares the pre-2001 holdout (14 shocks)
against the from-2001 design set (13 shocks).

Owner-ratified criteria (2026-09-29), fixed before running:
  * protection: holdout median within 0.15 of design AND >= 0.40
  * idle_rate:  holdout median within 0.10 of design
  * consistency: protection >= 0.40 in >= 10 of 14 holdout shocks
Plus raw distributions + design-vs-holdout gaps (ratified), and two significance checks: a label-
permutation equivalence test and a random-timing chance null.

Research only; reuses the deployed deadband, changes no engine/production/deployed-rule code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from Strategy_Auto_Trader.research import exit_protection as XP
from Strategy_Auto_Trader.research.index_history import CACHE_DIR, cache_path

REGISTER = CACHE_DIR / "event_register.csv"
RULES = ["fixed_pctl", "absolute"]


def _load_close(market: str) -> pd.Series:
    df = pd.read_csv(cache_path(market), index_col=0, comment="#")
    idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True)).normalize()
    return pd.Series(df["close"].to_numpy(dtype=float), index=idx, name=market).sort_index()


def _random_timing_null(scored_eps, closes, states_unused, n=2000, seed=0):
    """Median protection if the exit day were uniform-random in (peak, trough], per scored episode.

    Returns (real_median, null_median, p) where p is the fraction of null draws whose median >= real.
    """
    rng = np.random.default_rng(seed)
    real = np.array([e["protection_clipped"] for e in scored_eps], dtype=float)
    real_med = float(np.median(real))
    draws = np.empty(n)
    for d in range(n):
        vals = []
        for e in scored_eps:
            close = closes[e["market"]]
            days = close.index
            pk = int(days.searchsorted(e["peak_date"], side="left"))
            tr = int(days.searchsorted(e["trough_date"], side="left"))
            if tr <= pk + 1:
                vals.append(0.0)
                continue
            ex = rng.integers(pk + 1, tr + 1)
            denom = close.iloc[pk] - close.iloc[tr]
            raw = (close.iloc[ex] - close.iloc[tr]) / denom if denom > 0 else 0.0
            vals.append(float(np.clip(raw, 0.0, 1.0)))
        draws[d] = np.median(vals)
    p = (np.sum(draws >= real_med) + 1) / (n + 1)
    return real_med, float(np.median(draws)), float(p)


def run_rule(reg: pd.DataFrame, closes: dict[str, pd.Series], rule: str) -> None:
    print(f"\n{'#'*74}\n# TRANSFER RULE: {rule}\n{'#'*74}")

    states: dict[str, np.ndarray] = {}
    thresholds: dict[str, tuple[float, float]] = {}
    for m, close in closes.items():
        st, lo, hi = XP.build_state(close, rule)
        states[m], thresholds[m] = st, (lo, hi)

    episodes: list[XP.EpisodeProtection] = []
    scored_rows: list[dict] = []
    for _, row in reg.iterrows():
        m = row["market"]
        sid = int(row["shock_id"])
        peak = pd.Timestamp(row["peak_date"], tz="UTC").normalize()
        trough = pd.Timestamp(row["trough_date"], tz="UTC").normalize()
        cat, raw, clipped, lag = XP.episode_protection(closes[m], states[m], peak, trough)
        ep = XP.EpisodeProtection(m, sid, sid <= XP.HOLDOUT_MAX_SHOCK, peak, trough, cat, raw, clipped, lag)
        episodes.append(ep)
        if cat == "scored":
            scored_rows.append({"market": m, "peak_date": peak, "trough_date": trough,
                                "protection_clipped": clipped, "protection_raw": raw, "exit_lag": lag,
                                "is_holdout": ep.is_holdout})

    # idle_rate per market over its full history
    idle = {}
    for m, close in closes.items():
        wins = [(pd.Timestamp(r["peak_date"], tz="UTC").normalize(), pd.Timestamp(r["end_date"], tz="UTC").normalize())
                for _, r in reg[reg["market"] == m].iterrows()]
        idle[m] = XP.market_idle_rate(states[m], close.index, wins)

    shock_df = XP.aggregate_by_shock(episodes)
    # attach a per-shock idle_rate = mean idle over member markets
    shock_idle = {}
    for sid, grp in reg.groupby("shock_id"):
        mkts = grp["market"].unique()
        shock_idle[int(sid)] = float(np.nanmean([idle[m] for m in mkts]))
    shock_df["idle_rate"] = shock_df["shock_id"].map(shock_idle)

    design = shock_df[~shock_df["is_holdout"]]
    holdout = shock_df[shock_df["is_holdout"]]

    d_prot = design["protection"].dropna()
    h_prot = holdout["protection"].dropna()
    d_idle = design["idle_rate"].dropna()
    h_idle = holdout["idle_rate"].dropna()

    print(f"\ncategories: scored={sum(e.category=='scored' for e in episodes)} "
          f"already_out={sum(e.category=='already_out' for e in episodes)} "
          f"never_exited={sum(e.category=='never_exited' for e in episodes)}")

    print("\nper-shock protection (clipped mean) and idle_rate:")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(shock_df.round(3).to_string(index=False))

    d_med, h_med = d_prot.median(), h_prot.median()
    di_med, hi_med = d_idle.median(), h_idle.median()
    prot_gap = abs(h_med - d_med)
    idle_gap = abs(hi_med - di_med)
    consistency = int((h_prot >= 0.40).sum())

    print(f"\n--- protection ---")
    print(f"  design  median={d_med:.3f}  n={len(d_prot)}  raw quartiles={np.round(np.quantile(d_prot,[.25,.5,.75]),3) if len(d_prot) else 'NA'}")
    print(f"  holdout median={h_med:.3f}  n={len(h_prot)}  raw quartiles={np.round(np.quantile(h_prot,[.25,.5,.75]),3) if len(h_prot) else 'NA'}")
    print(f"  design-vs-holdout median gap = {prot_gap:.3f}")
    print(f"--- idle_rate ---")
    print(f"  design median={di_med:.3f}  holdout median={hi_med:.3f}  gap={idle_gap:.3f}")

    # criteria
    c_primary = (prot_gap <= 0.15) and (h_med >= 0.40)
    c_cost = idle_gap <= 0.10
    c_consistency = consistency >= 10
    print(f"\n--- CRITERIA (owner-ratified) ---")
    print(f"  primary   (holdout within 0.15 of design AND >=0.40): {c_primary}  (gap {prot_gap:.3f}, holdout {h_med:.3f})")
    print(f"  cost      (idle within 0.10):                         {c_cost}  (gap {idle_gap:.3f})")
    print(f"  consistency (>=10 of 14 holdout shocks >=0.40):       {c_consistency}  ({consistency}/{len(h_prot)})")
    print(f"  ==> GENERALISES: {c_primary and c_cost and c_consistency}")

    # significance checks
    p_equiv = XP.permutation_pvalue(d_prot.to_numpy(), h_prot.to_numpy())
    print(f"\n--- significance ---")
    print(f"  equivalence (label-permutation, HIGH p supports same-distribution): p={p_equiv:.3f}")
    if scored_rows:
        real_med, null_med, p_null = _random_timing_null(scored_rows, closes, states)
        print(f"  chance null (real vs random-timing exit): real median={real_med:.3f}, "
              f"random median={null_med:.3f}, p(random>=real)={p_null:.3f}")


def main() -> None:
    reg = pd.read_csv(REGISTER)
    reg["shock_id"] = reg["shock_id"].astype(int)
    markets = sorted(reg["market"].unique())
    closes = {m: _load_close(m) for m in markets}

    n_hold = reg[reg["shock_id"] <= XP.HOLDOUT_MAX_SHOCK]["shock_id"].nunique()
    n_design = reg[reg["shock_id"] > XP.HOLDOUT_MAX_SHOCK]["shock_id"].nunique()
    print(f"shocks: holdout(pre-2001)={n_hold}  design(from-2001)={n_design}  markets={len(markets)}")
    print("NOTE: fixed_pctl thresholds use full-history percentiles (mild threshold-only look-ahead, flagged).")

    for rule in RULES:
        run_rule(reg, closes, rule)


if __name__ == "__main__":
    main()
