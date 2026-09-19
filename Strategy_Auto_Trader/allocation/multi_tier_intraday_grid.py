"""Threshold grid on the intraday-faithful engine, with an era holdout.

Two signal variants (same 4-tier ladder, different drivers):
  vxn_vix   Nasdaq if VXN<=a; else S&P if VIX<=b; FTSE if VIX<=c; else cash   (current design)
  vix_only  Nasdaq if VIX<=a; S&P if VIX<=b; FTSE if VIX<=c; else cash         (no VXN)
  vxn_deadband  Nasdaq|cash only: enter when VXN<=lo, exit only when VXN>hi (hi>=lo); width 0 = plain rule

Thresholds are chosen on TRAIN (2007-11 to 2019) and scored once on TEST (2020 to now). The
1999-2007 bridged years are reported but never used to choose. The old 15/17.5/18 thresholds
came from flawed timing models, so they appear only as one labelled reference row.

Selection metric is Sharpe of returns OVER CASH ("xSharpe"): raw Sharpe counts idle cash as
income with no volatility and so flatters strategies that sit in cash. Raw Sharpe is shown too.
A pick is also judged on its +-1 neighbours' average — a sharp peak its neighbours don't share
is noise, not a threshold.

Run:
    uv run python -m Strategy_Auto_Trader.allocation.multi_tier_intraday_grid
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import intraday_engine as eng

_OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "intraday_backtest"
_VIX_RANGE = range(12, 31)
_VXN_RANGE = range(14, 31)
_REPORT_WINDOWS = ("train", "test", "bridged", "real")
_STAT_KEYS = ("sharpe", "xsharpe", "ret_pct", "max_dd_pct", "sw_per_yr")
_COST_SWEEP = (0.0, 6.5, 13.0, 26.0)
_REFERENCE = {"vxn_vix": (18.0, 15.0, 17.5)}  # as deployed before this work — reference only
_SELECT = "xsharpe"
_VARIANTS = ("vxn_vix", "vix_only", "vxn_deadband")
_MAX_BAND = 10


def param_grid(variant: str) -> list[tuple[float, float, float]]:
    if variant == "vxn_deadband":  # (enter_at, exit_above, unused)
        return [(float(lo), float(lo + w), 0.0) for lo in _VXN_RANGE for w in range(_MAX_BAND + 1)]
    if variant == "vix_only":
        return [(float(a), float(b), float(c)) for a, b, c in itertools.combinations(_VIX_RANGE, 3)]
    return [(float(a), float(b), float(c)) for a in _VXN_RANGE for b, c in itertools.combinations(_VIX_RANGE, 2)]


def tiers_for(inp: eng.Inputs, variant: str, params: tuple[float, float, float]) -> np.ndarray:
    if variant == "vxn_deadband":
        return eng.tiers_vxn_deadband(inp.vxn, params[0], params[1])
    if variant == "vix_only":
        return eng.tiers_vix_only(inp.vix, *params)
    return eng.tiers_vxn_vix(inp.vxn, inp.vix, *params)


def evaluate(inp: eng.Inputs, variant: str, params, fill: str, cost_bps: float) -> dict:
    run = eng.simulate(inp, tiers_for(inp, variant, params), fill, cost_bps)
    row = {"variant": variant, "fill": fill, "cost_bps": cost_bps, "p1": params[0], "p2": params[1], "p3": params[2]}
    for window in _REPORT_WINDOWS:
        stats = eng.window_stats(run, window)
        row.update({f"{window}_{k}": stats[k] for k in _STAT_KEYS})
    return row


def evaluate_grid(inp: eng.Inputs, variant: str, fill: str, cost_bps: float = eng.DEFAULT_COST_BPS) -> pd.DataFrame:
    return pd.DataFrame([evaluate(inp, variant, p, fill, cost_bps) for p in param_grid(variant)])


def add_neighbour_mean(df: pd.DataFrame, column: str = f"train_{_SELECT}") -> pd.DataFrame:
    """Mean of `column` over each cell and its existing +-1 neighbours in every threshold dimension."""
    lookup = {(r.p1, r.p2, r.p3): getattr(r, column) for r in df.itertuples()}
    offsets = list(itertools.product((-1.0, 0.0, 1.0), repeat=3))
    smooth = []
    for r in df.itertuples():
        vals = [lookup[k] for d in offsets if (k := (r.p1 + d[0], r.p2 + d[1], r.p3 + d[2])) in lookup]
        smooth.append(float(np.nanmean(vals)))
    return df.assign(**{f"{column}_nbr": smooth})


def selection_diagnostics(df: pd.DataFrame) -> dict:
    """How much does train rank tell us about test? Spearman over the whole grid plus a top-decile check."""
    train, test = df[f"train_{_SELECT}"], df[f"test_{_SELECT}"]
    return {
        "cells": len(df),
        "spearman_train_vs_test": float(train.corr(test, method="spearman")),
        "grid_median_test": float(test.median()),
        "top_decile_train_median_test": float(df[train >= train.quantile(0.9)][f"test_{_SELECT}"].median()),
        "share_test_positive": float((test > 0).mean()),
        "best_test_anywhere": float(test.max()),
    }


def _fmt_row(r: pd.Series | object, label: str = "") -> str:
    def g(key: str) -> float:
        return getattr(r, key)

    return (f"{label}{g('p1'):5.1f} {g('p2'):5.1f} {g('p3'):5.1f} | train x{g('train_xsharpe'):+.2f} raw{g('train_sharpe'):+.2f} ({g('train_ret_pct'):+5.0f}%) "
            f"| test x{g('test_xsharpe'):+.2f} raw{g('test_sharpe'):+.2f} ({g('test_ret_pct'):+4.0f}%, dd {g('test_max_dd_pct'):+5.1f}%, {g('test_sw_per_yr'):4.1f} sw/yr) "
            f"| bridged x{g('bridged_xsharpe'):+.2f}")


def _top_block(df: pd.DataFrame, by: str, n: int, title: str) -> list[str]:
    lines = [f"  {title}", "     p1    p2    p3 | train (x = excess over cash) | test | bridged"]
    lines += [_fmt_row(r, "   ") for r in df.sort_values(by, ascending=False).head(n).itertuples()]
    return lines


def cost_sweep(inp: eng.Inputs, variant: str, params, fill: str) -> list[str]:
    lines = []
    for cost in _COST_SWEEP:
        r = evaluate(inp, variant, params, fill, cost)
        lines.append(f"    cost {cost:5.1f} bps -> train x{r['train_xsharpe']:+.2f}  test x{r['test_xsharpe']:+.2f} ({r['test_ret_pct']:+.0f}%)")
    return lines


def ladder_vs_two_state(inp: eng.Inputs, grid: pd.DataFrame) -> list[str]:
    """Do the S&P / FTSE tiers add anything? Compare, per VXN cut, the two-state (Nasdaq or cash) rule with the ladder."""
    lines = ["  Do tiers 2/3 add value? per VXN cut: two-state (Nasdaq|cash) vs ladder (median over VIX cuts)",
             "    cut | two-state train/test xSharpe | ladder median train/test | ladder best-by-train -> its test"]
    for cut in range(16, 31, 2):
        tiers = np.full(len(inp.vix), 3)
        tiers[inp.vxn <= cut] = 0
        run = eng.simulate(inp, tiers)
        two = (eng.window_stats(run, "train")[_SELECT], eng.window_stats(run, "test")[_SELECT])
        sub = grid[grid.p1 == cut]
        best = sub.sort_values(f"train_{_SELECT}", ascending=False).iloc[0]
        lines.append(f"    {cut:3d} | {two[0]:+.2f} / {two[1]:+.2f}               | {sub[f'train_{_SELECT}'].median():+.2f} / {sub[f'test_{_SELECT}'].median():+.2f}"
                     f"          | {best[f'train_{_SELECT}']:+.2f} -> {best[f'test_{_SELECT}']:+.2f}")
    return lines


def band_width_table(grid: pd.DataFrame) -> list[str]:
    """Effect of band width alone: median over every enter threshold, so no single threshold drives it."""
    g = grid.assign(width=grid.p2 - grid.p1)
    by = g.groupby("width").agg(train=(f"train_{_SELECT}", "median"), test=(f"test_{_SELECT}", "median"),
                                test_ret=("test_ret_pct", "median"), test_dd=("test_max_dd_pct", "median"),
                                sw=("test_sw_per_yr", "median"))
    lines = ["  Effect of band width (median across all enter thresholds)",
             "    width | train x | test x | test ret% | test dd% | test sw/yr"]
    lines += [f"    {int(w):5d} | {r.train:+.2f}   | {r.test:+.2f}  | {r.test_ret:+7.0f}   | {r.test_dd:+6.1f}   | {r.sw:5.1f}" for w, r in by.iterrows()]
    return lines


def deadband_vs_plain(grid: pd.DataFrame) -> list[str]:
    """Per enter threshold: the plain rule (width 0) against the band that scored best on TRAIN."""
    lines = ["  Per enter threshold: plain rule vs best band chosen on train",
             "    enter | plain train/test x, sw/yr    | best band (exit_above) train/test x, sw/yr"]
    for lo in range(16, 31, 2):
        sub = grid[grid.p1 == lo]
        plain = sub[sub.p2 == lo].iloc[0]
        best = sub.sort_values(f"train_{_SELECT}", ascending=False).iloc[0]
        lines.append(f"    {lo:5d} | {plain.train_xsharpe:+.2f} / {plain.test_xsharpe:+.2f}, {plain.test_sw_per_yr:5.1f}      "
                     f"| ({best.p2:4.0f}) {best.train_xsharpe:+.2f} / {best.test_xsharpe:+.2f}, {best.test_sw_per_yr:5.1f}")
    return lines


def tier_occupancy(inp: eng.Inputs, variant: str, params) -> str:
    tiers = tiers_for(inp, variant, params)
    share = np.bincount(tiers, minlength=4) / len(tiers)
    return "  time in tier (all bars): " + "  ".join(f"{a} {s:.0%}" for a, s in zip(eng.ASSETS, share))


def variant_report(inp: eng.Inputs, variant: str, same: pd.DataFrame, nxt: pd.DataFrame) -> list[str]:
    same = add_neighbour_mean(same)
    diag = selection_diagnostics(same)
    lines = [f"=== {variant} ({diag['cells']} cells, same-bar fill, 13 bps, xSharpe = Sharpe over cash) ===",
             f"  Spearman(train, test) over grid: {diag['spearman_train_vs_test']:+.2f}",
             f"  test xSharpe: grid median {diag['grid_median_test']:+.2f} | top-decile-by-train median {diag['top_decile_train_median_test']:+.2f} "
             f"| share positive {diag['share_test_positive']:.0%} | best anywhere {diag['best_test_anywhere']:+.2f}"]
    lines += _top_block(same, f"train_{_SELECT}", 8, "Top by raw train xSharpe")
    lines += _top_block(same, f"train_{_SELECT}_nbr", 8, "Top by neighbour-averaged train xSharpe (robust picks)")
    pick = same.sort_values(f"train_{_SELECT}_nbr", ascending=False).iloc[0]
    params = (float(pick.p1), float(pick.p2), float(pick.p3))
    nxt_row = nxt[(nxt.p1 == pick.p1) & (nxt.p2 == pick.p2) & (nxt.p3 == pick.p3)].iloc[0]
    lines += [f"  Robust pick {params}", tier_occupancy(inp, variant, params),
              f"  next-bar fill -> train x{nxt_row.train_xsharpe:+.2f}, test x{nxt_row.test_xsharpe:+.2f} ({nxt_row.test_ret_pct:+.0f}%)",
              "  cost sensitivity (same-bar):"] + cost_sweep(inp, variant, params, "same_bar")
    if variant == "vxn_vix":
        lines += ladder_vs_two_state(inp, same)
    if variant == "vxn_deadband":
        lines += band_width_table(same) + deadband_vs_plain(same)
    return lines


def context_block(inp: eng.Inputs) -> list[str]:
    lines = ["=== Context: buy and hold, xSharpe / return / max drawdown ===",
             "  asset    | train                   | test                    | bridged"]
    for asset in eng.ASSETS[:3]:
        run = eng.buy_and_hold(inp, asset)
        cells = [eng.window_stats(run, w) for w in ("train", "test", "bridged")]
        lines.append(f"  {asset:8s} | " + " | ".join(f"{c['xsharpe']:+.2f} / {c['ret_pct']:+5.0f}% / {c['max_dd_pct']:+5.1f}%" for c in cells))
    return lines


def reference_block(inp: eng.Inputs) -> list[str]:
    lines = ["=== Reference only: pre-existing thresholds VXN 18 / VIX 15 / 17.5 ==="]
    for fill in ("same_bar", "next_bar"):
        row = pd.Series(evaluate(inp, "vxn_vix", _REFERENCE["vxn_vix"], fill, eng.DEFAULT_COST_BPS))
        lines.append(_fmt_row(row, f"  {fill:8s} "))
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--variants", nargs="+", choices=_VARIANTS, default=list(_VARIANTS))
    args = p.parse_args()
    inp = eng.load_inputs()
    out_dir = args.out or _OUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, report = [], [f"Intraday-faithful tier backtest  {inp.days[0].date()} -> {inp.days[-1].date()}",
                          "Eras: train 2007-11..2019 | test 2020..now | bridged 1999..2007-11 (assumed intraday, never used to choose)", ""]
    report += context_block(inp) + [""] + reference_block(inp) + [""]
    for variant in args.variants:
        same, nxt = evaluate_grid(inp, variant, "same_bar"), evaluate_grid(inp, variant, "next_bar")
        frames += [same, nxt]
        report += variant_report(inp, variant, same, nxt) + [""]
    pd.concat(frames).to_csv(out_dir / "grid_results.csv", index=False)
    text = "\n".join(report)
    (out_dir / "report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
