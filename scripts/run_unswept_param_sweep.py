"""Isolated sweep of optimised_new parameters that have never been tested.

Covers six parameter groups, each isolating ONE change against the full
current-default baseline so effects don't tangle:

  1. regime_signal <= 0 veto (binary: on/off)
  2. buy_threshold (6.0 → 5.0 / 5.5 / 6.5 / 7.0)
  3. SMA200 weight (3.0 → 1.5 / 2.0 / 2.5 / 4.0)
  4. RSI weight   (1.0 → 0.5 / 1.5 / 2.0)
  5. HMM weight   (2.0 → 1.0 / 1.5 / 2.5 / 3.0)
  6. RSI > 70 veto threshold (70 → off / 60 / 65 / 75 / 80)

Uses per-ticker consolidated_backtest (same engine as live), NOT the
capital-arbitrated live_sim pipeline. Results are directional — they show
signal quality on individual tickers, not portfolio-level P&L or drawdown.
A follow-up full live_sim run is warranted for any parameter that shows
meaningful improvement here.

Usage:
    uv run python scripts/run_unswept_param_sweep.py
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, ".")  # run from repo root

from Strategy_Auto_Trader.core.cli_logging import setup_cli_logger
from Strategy_Auto_Trader.markov_cli.compare_exits import _fetch
from Strategy_Auto_Trader.plugins.types import EntryDecision, RegimeState
from Strategy_Auto_Trader.strategy.optimised_new import OptimisedNewEntry, OptimisedNewExit

logger = logging.getLogger(__name__)

# Same 11-ticker basket used by compare_admission_gates / compare_exits.
# Large enough for direction; small enough for a sub-5-minute full sweep.
TEST_TICKERS = [
    "HSBA.L", "INTC", "T", "CSCO", "BA",
    "BATS.L", "F", "GOOGL", "GSK.L", "KO", "MU",
]

_HOURLY_DEFAULTS = dict(
    min_train_bars=500, hmm_refit_bars=500,
    regime_smooth=24, min_hold_bars=48,
)

_RSI_DEFAULT = 70.0
_MIN_REGIME_SIGNAL = 0.0


# ---------------------------------------------------------------------------
# Variant entry classes for parameters that live inside evaluate()
# ---------------------------------------------------------------------------

class _NoRegimeVetoEntry(OptimisedNewEntry):
    """Identical to OptimisedNewEntry but the regime_signal <= 0 veto removed."""

    def evaluate(self, regime: RegimeState, mom: dict, _volume_ratio: float,
                 currently_in: bool = False) -> EntryDecision:
        from Strategy_Auto_Trader.core.momentum import composite_signal
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate

        if not self._vol_filter_ok:
            return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0,
                                 reason="vol_filter: unsuitable")
        raw = composite_signal(markov_signal=0.0, mom=mom, hmm_state=regime.hmm_vote,
                               buy_threshold=self._buy_t, sell_threshold=self._sell_t,
                               weights=self._weights)
        if self._quality_gate_enabled:
            gated = _apply_quality_gate(raw, mom, regime.regime_signal,
                                        currently_in=currently_in,
                                        gate_sensitivity=self._gate_sensitivity)
        else:
            gated = dict(raw, reason="", gate_fired=False)
        decision = EntryDecision(flag=gated["flag"], raw_flag=raw["flag"],
                                 score=float(raw.get("score", 0.0)),
                                 reason=gated.get("reason", ""),
                                 gate_fired=gated.get("gate_fired", False))
        if currently_in or decision.flag != "BUY":
            return decision
        if float(mom.get("cur_rsi", 50.0)) > _RSI_DEFAULT:
            return EntryDecision(flag="HOLD", raw_flag=decision.raw_flag,
                                 score=decision.score,
                                 reason=f"veto: RSI > {_RSI_DEFAULT:.0f}")
        if self._min_entry_score is not None and decision.score < self._min_entry_score:
            return EntryDecision(flag="HOLD", raw_flag=decision.raw_flag,
                                 score=decision.score,
                                 reason=f"veto: score < {self._min_entry_score:.1f}")
        # regime_signal veto REMOVED — this is the only difference
        return decision


class _CustomRSIVetoEntry(OptimisedNewEntry):
    """OptimisedNewEntry with a configurable RSI overbought threshold."""

    def __init__(self, *args, rsi_veto: float | None = 70.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._rsi_veto = rsi_veto  # None = veto disabled

    def evaluate(self, regime: RegimeState, mom: dict, _volume_ratio: float,
                 currently_in: bool = False) -> EntryDecision:
        from Strategy_Auto_Trader.core.momentum import composite_signal
        from Strategy_Auto_Trader.core.quality_gate import _apply_quality_gate

        if not self._vol_filter_ok:
            return EntryDecision(flag="HOLD", raw_flag="HOLD", score=0.0,
                                 reason="vol_filter: unsuitable")
        raw = composite_signal(markov_signal=0.0, mom=mom, hmm_state=regime.hmm_vote,
                               buy_threshold=self._buy_t, sell_threshold=self._sell_t,
                               weights=self._weights)
        if self._quality_gate_enabled:
            gated = _apply_quality_gate(raw, mom, regime.regime_signal,
                                        currently_in=currently_in,
                                        gate_sensitivity=self._gate_sensitivity)
        else:
            gated = dict(raw, reason="", gate_fired=False)
        decision = EntryDecision(flag=gated["flag"], raw_flag=raw["flag"],
                                 score=float(raw.get("score", 0.0)),
                                 reason=gated.get("reason", ""),
                                 gate_fired=gated.get("gate_fired", False))
        if currently_in or decision.flag != "BUY":
            return decision
        if self._rsi_veto is not None and float(mom.get("cur_rsi", 50.0)) > self._rsi_veto:
            return EntryDecision(flag="HOLD", raw_flag=decision.raw_flag,
                                 score=decision.score,
                                 reason=f"veto: RSI > {self._rsi_veto:.0f}")
        if regime.regime_signal is not None and regime.regime_signal <= _MIN_REGIME_SIGNAL:
            return EntryDecision(flag="HOLD", raw_flag=decision.raw_flag,
                                 score=decision.score,
                                 reason="veto: regime_signal <= 0")
        if self._min_entry_score is not None and decision.score < self._min_entry_score:
            return EntryDecision(flag="HOLD", raw_flag=decision.raw_flag,
                                 score=decision.score,
                                 reason=f"veto: score < {self._min_entry_score:.1f}")
        return decision


# ---------------------------------------------------------------------------
# Sweep group definitions
# ---------------------------------------------------------------------------

@dataclass
class Variant:
    name: str
    build: Callable[[], tuple]  # () -> (entry, exit_)
    group: str
    is_baseline: bool = False


def _base_entry(vol_filter_ok: bool = True, **kwargs):
    return OptimisedNewEntry(vol_filter_ok=vol_filter_ok, **kwargs)


def _base_exit():
    return OptimisedNewExit()


def build_variants() -> list[Variant]:
    variants: list[Variant] = []

    # Shared baseline — appears once per group for clear comparison
    def _baseline():
        return _base_entry(), _base_exit()

    # ── Group 1: regime_signal veto ─────────────────────────────────────
    variants.append(Variant("regime_veto ON (baseline)", _baseline,
                             "1_regime_veto", is_baseline=True))
    variants.append(Variant("regime_veto OFF", lambda: (_NoRegimeVetoEntry(), _base_exit()),
                             "1_regime_veto"))

    # ── Group 2: buy_threshold ───────────────────────────────────────────
    variants.append(Variant("buy_threshold=6.0 (baseline)", _baseline,
                             "2_buy_threshold", is_baseline=True))
    for bt in (5.0, 5.5, 6.5, 7.0):
        t = bt  # capture
        variants.append(Variant(
            f"buy_threshold={t:.1f}",
            lambda t=t: (_base_entry(buy_threshold=t, sell_threshold=-t), _base_exit()),
            "2_buy_threshold",
        ))

    # ── Group 3: SMA200 weight ───────────────────────────────────────────
    variants.append(Variant("sma200_w=3.0 (baseline)", _baseline,
                             "3_sma200_weight", is_baseline=True))
    for w in (1.5, 2.0, 2.5, 4.0):
        ww = w
        variants.append(Variant(
            f"sma200_w={ww:.1f}",
            lambda ww=ww: (_base_entry(weights={"sma200": ww}), _base_exit()),
            "3_sma200_weight",
        ))

    # ── Group 4: RSI weight ──────────────────────────────────────────────
    variants.append(Variant("rsi_w=1.0 (baseline)", _baseline,
                             "4_rsi_weight", is_baseline=True))
    for w in (0.5, 1.5, 2.0):
        ww = w
        variants.append(Variant(
            f"rsi_w={ww:.1f}",
            lambda ww=ww: (_base_entry(weights={"rsi": ww}), _base_exit()),
            "4_rsi_weight",
        ))

    # ── Group 5: HMM weight ──────────────────────────────────────────────
    variants.append(Variant("hmm_w=2.0 (baseline)", _baseline,
                             "5_hmm_weight", is_baseline=True))
    for w in (1.0, 1.5, 2.5, 3.0):
        ww = w
        variants.append(Variant(
            f"hmm_w={ww:.1f}",
            lambda ww=ww: (_base_entry(weights={"hmm": ww}), _base_exit()),
            "5_hmm_weight",
        ))

    # ── Group 6: RSI > threshold veto ───────────────────────────────────
    variants.append(Variant("rsi_veto=70 (baseline)",
                             lambda: (_CustomRSIVetoEntry(rsi_veto=70.0), _base_exit()),
                             "6_rsi_veto_threshold", is_baseline=True))
    variants.append(Variant("rsi_veto=OFF",
                             lambda: (_CustomRSIVetoEntry(rsi_veto=None), _base_exit()),
                             "6_rsi_veto_threshold"))
    for thresh in (60, 65, 75, 80):
        t = float(thresh)
        variants.append(Variant(
            f"rsi_veto={thresh}",
            lambda t=t: (_CustomRSIVetoEntry(rsi_veto=t), _base_exit()),
            "6_rsi_veto_threshold",
        ))

    return variants


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_variant(df: pd.DataFrame, variant: Variant) -> dict:
    from Strategy_Auto_Trader.quant_hmm.consolidated_engine import consolidated_backtest
    entry, exit_ = variant.build()
    try:
        bt = consolidated_backtest(df, entry_strategy=entry, exit_strategy=exit_,
                                   **_HOURLY_DEFAULTS)
        detail = bt["detail"]
        buys  = detail[detail["trade_event"] == "BUY"]["close"].tolist()
        sells = detail[detail["trade_event"] == "SELL"]["close"].tolist()
        pls = [(sells[i] - buys[i]) / buys[i]
               for i in range(min(len(buys), len(sells)))]
        wins = sum(1 for p in pls if p > 0)
        losses = sum(1 for p in pls if p < 0)
        return {
            "sharpe":       bt["sharpe_strategy"],
            "sortino":      bt.get("sortino_strategy", float("nan")),
            "total_return": bt["total_return_strategy"],
            "max_dd":       bt["max_drawdown_strategy"],
            "pl":           bt["total_pl"],
            "n_trades":     bt["n_buys"],
            "win_rate":     wins / (wins + losses) * 100 if (wins + losses) else 0,
            "wins":         wins,
            "losses":       losses,
        }
    except Exception as exc:
        logger.warning(f"    variant '{variant.name}' failed: {exc}")
        return {k: float("nan") for k in
                ("sharpe", "sortino", "total_return", "max_dd", "pl",
                 "n_trades", "win_rate", "wins", "losses")}


def main() -> int:
    setup_cli_logger("unswept_param_sweep")

    variants = build_variants()

    logger.info(f"Loading data for {len(TEST_TICKERS)} tickers...")
    price_data: dict[str, pd.DataFrame] = {}
    for ticker in TEST_TICKERS:
        df = _fetch(ticker)
        if df is not None and len(df) > 500:
            price_data[ticker] = df
            logger.info(f"  {ticker}: {len(df)} bars")
        else:
            logger.info(f"  {ticker}: skipped (insufficient data)")

    logger.info(f"\nRunning {len(variants)} variants across {len(price_data)} tickers...\n")

    rows = []
    t0 = time.time()
    for i, variant in enumerate(variants):
        ticker_results = []
        for ticker, df in price_data.items():
            r = _run_variant(df, variant)
            r.update({"ticker": ticker, "variant": variant.name, "group": variant.group})
            rows.append(r)
            ticker_results.append(r)
        elapsed = time.time() - t0
        avg_sharpe = np.nanmean([r["sharpe"] for r in ticker_results])
        avg_ret    = np.nanmean([r["total_return"] for r in ticker_results]) * 100
        avg_trades = np.nanmean([r["n_trades"] for r in ticker_results])
        logger.info(f"  [{i+1:2d}/{len(variants)}] {variant.name:<32s}  "
                    f"Sharpe={avg_sharpe:+.3f}  Ret={avg_ret:+.1f}%  "
                    f"Trades={avg_trades:.0f}  ({elapsed:.0f}s)")

    df_all = pd.DataFrame(rows)

    # ── Per-group aggregate summary ──────────────────────────────────────
    groups = df_all["group"].unique()
    logger.info(f"\n{'='*100}")
    logger.info(" AGGREGATE RESULTS BY GROUP (mean across tickers)")
    logger.info(f"{'='*100}")

    for group in sorted(groups):
        g = df_all[df_all["group"] == group]
        agg = g.groupby("variant").agg(
            sharpe=("sharpe", "mean"),
            sortino=("sortino", "mean"),
            ret=("total_return", lambda x: x.mean() * 100),
            dd=("max_dd", lambda x: x.mean() * 100),
            trades=("n_trades", "mean"),
            win_rate=("win_rate", "mean"),
        )

        # preserve insertion order (baseline first)
        order = df_all[df_all["group"] == group]["variant"].unique()
        agg = agg.reindex([v for v in order if v in agg.index])

        logger.info(f"\n  Group: {group}")
        logger.info(f"  {'Variant':<32s} {'Sharpe':>7s} {'Sortino':>8s} {'Return':>8s} {'MaxDD':>7s} {'Win%':>6s} {'Trades':>7s}")
        logger.info(f"  {'-'*32} {'-'*7} {'-'*8} {'-'*8} {'-'*7} {'-'*6} {'-'*7}")
        for vname, row in agg.iterrows():
            flag = " <-- baseline" if "(baseline)" in str(vname) else ""
            logger.info(f"  {vname:<32s} {row['sharpe']:>+7.3f} {row['sortino']:>+8.3f} "
                        f"{row['ret']:>+7.1f}% {row['dd']:>+6.1f}% {row['win_rate']:>5.0f}% "
                        f"{row['trades']:>6.0f}{flag}")

    elapsed_total = time.time() - t0
    logger.info(f"\nTotal elapsed: {elapsed_total:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
