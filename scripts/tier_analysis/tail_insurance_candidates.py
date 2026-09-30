"""Candidate tail-insurance gates on an already-held fund, judged against the OWNER'S risk numbers.

Reports worst 1/3/5-day compounded loss alongside max drawdown, per
[[feedback_loss_tolerance_is_not_max_drawdown]]: the owner's 15% is a tolerance on the worst
5-day loss while invested, and about -24% is the accepted max drawdown. Quoting max DD against
15% conflates two different quantities.

Run: uv run python scripts/tier_analysis/tail_insurance_candidates.py [HOLDING]
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from Strategy_Auto_Trader.allocation import basket_engine as be
from Strategy_Auto_Trader.allocation import intraday_engine as eng

HOLDING = sys.argv[1] if len(sys.argv) > 1 else "WORLD"
HORIZONS = (1, 3, 5)
WORST_5D_TOLERANCE = -15.0   # owner's stated tolerance, worst 5-day loss while invested
ACCEPTED_MAX_DD = -24.0      # owner's accepted max drawdown
CANDIDATES = {
    "no gate (today)": None,
    "VXN 42/46": (42.0, 46.0),
    "VXN 40/44": (40.0, 44.0),
    "VXN 38/42": (38.0, 42.0),
    "VXN 34/38": (34.0, 38.0),
    "VXN 30/34": (30.0, 34.0),
    "VXN 26/30": (26.0, 30.0),
    "VXN 22/24": (22.0, 24.0),
    "VXN 20/24": (20.0, 24.0),
}

inp0 = eng.load_inputs()
inp, NAMES = be.extend_with_world(inp0)
ALWAYS_IN = np.zeros(len(inp.grid), dtype=int)


def rolling_worst(daily: np.ndarray, k: int) -> float:
    """Worst k-day compounded return (%). Same definition as t4_gap_stress.rolling_worst."""
    cs = np.concatenate([[0.0], np.cumsum(np.log1p(daily))])
    return float(np.expm1(cs[k:] - cs[:-k]).min() * 100)


def in_market_pct(tiers: np.ndarray, window: str) -> float:
    sl = eng._bounds(inp.days, window)
    lo, hi = inp.days[sl][0], inp.days[sl][-1]
    london = inp.grid.tz_convert("Europe/London").tz_localize(None).normalize()
    mask = (london >= lo) & (london <= hi)
    return float((tiers[mask] == 0).mean() * 100)


for window in ("real", "test"):
    sl = eng._bounds(inp.days, window)
    days = inp.days[sl]
    print(f"\n{'=' * 118}")
    print(f"== {HOLDING}: tail-insurance candidates  [{window}: {days[0].date()} .. {days[-1].date()}, {len(days)} days]")
    print(f"   tolerance: worst 5-day loss >= {WORST_5D_TOLERANCE:g}%; accepted max DD ~{ACCEPTED_MAX_DD:g}%")
    print(f"{'=' * 118}")
    print(f"  {'gate':16s} {'in-mkt%':>7} {'sw/yr':>6} {'CAGR%':>6} {'give-up':>8} "
          + " ".join(f"{'worst ' + str(k) + 'd%':>9}" for k in HORIZONS)
          + f" {'maxDD%':>7} {'verdict':>22}")
    base_cagr = None
    for label, gate in CANDIDATES.items():
        tiers = ALWAYS_IN if gate is None else eng.tiers_vxn_deadband(inp.vxn, *gate)
        run = be.basket_run(inp, NAMES, tiers, {HOLDING: 1.0})
        s = eng.window_stats(run, window)
        cagr = ((1 + s["ret_pct"] / 100) ** (252 / s["n_days"]) - 1) * 100
        if base_cagr is None:
            base_cagr = cagr
        worst = {k: rolling_worst(run.daily[sl], k) for k in HORIZONS}
        ok_5d = worst[5] >= WORST_5D_TOLERANCE
        ok_dd = s["max_dd_pct"] >= ACCEPTED_MAX_DD
        verdict = "within both" if (ok_5d and ok_dd) else (
            "5d ok, maxDD beyond" if ok_5d else ("maxDD ok, 5d beyond" if ok_dd else "BEYOND both"))
        print(f"  {label:16s} {in_market_pct(tiers, window):7.1f} {s['sw_per_yr']:6.1f} {cagr:6.1f} "
              f"{base_cagr - cagr:8.1f} " + " ".join(f"{worst[k]:9.1f}" for k in HORIZONS)
              + f" {s['max_dd_pct']:7.1f} {verdict:>22}")
