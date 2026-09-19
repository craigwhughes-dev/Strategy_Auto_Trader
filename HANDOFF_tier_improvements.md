# Handoff — Tier strategy improvement experiments

Written 2026-09-19 (late night). Read this file first, then `HANDOFF_26yr_rerun.md` for broader context and `BACKTEST_LOG.md` top four entries for the numbers behind the decisions below.

## 0. What was just established (this session)

- **Stamp-duty fix (D4) does not affect tier backtests.** `allocation/*.py` use flat bps only.
- **Comparators on intraday engine (xSharpe):** current rule wins on 26yr and since-2007 over static blend, VT5%, VT10%, SMA200+VT5%. The [SUSPECT] figures (VT5% raw Sharpe 1.11 > deployed 0.79) were an artefact of raw Sharpe rewarding cash carry. xSharpe is the correct metric.
- **D2 (asymmetric re-entry 8d) rejected.** Tested on intraday engine. xSharpe drops on all windows including recent (0.37 → -0.01). Recovery-year drag (2009, 2010, 2024) outweighs whipsaw saving.
- **D1 decision: keep the current VXN 23/24 deadband rule.** No switch to VT5%.
- **Remaining known weakness:** recent 2024+ window xSh 0.37 vs comparators ~0.75, driven by VXN range-bound near 22-25 (18 sw/yr recently vs 7.4 average). This is a genuine open question but the 2.5yr window is too short to choose new parameters on.

Current rule summary (same-bar fill, 13 bps/switch, `data_synthetic/hourly_spliced/`):

| Window | xSharpe | sw/yr | maxDD% | ret% |
|---|---|---|---|---|
| 26yr (1999-2026) | +0.76 | 7.4 | -21.5 | +2,242 |
| since VXN data (2007-11-20+) | +0.96 | 9.3 | -14.9 | +1,001 |
| recent real (2024-03-25+) | +0.37 | 18.4 | -13.6 | +24 |

Sanity check: `uv run python scripts/tier_analysis/annual_tier_current.py` → 26yr +2,242%, xSh +0.76, DD -21.5%, 7.4 sw/yr. Run this first. If it doesn't match, stop.

## 1. Repo state

Branch: `main`, `e280f9b`. Tests: 1814 passing (`uv run python -m pytest tests -q`).

Relevant code:
- `allocation/intraday_engine.py` — `load_inputs`, `simulate`, `tiers_vxn_deadband`, `window_stats`, `annual_returns`, `WINDOWS`
- `allocation/intraday_comparators.py` — `static_blend`, `vol_target`, `sma_vol_target`, `_daily_simple_ret`, `_cash_daily`, `_run_blend` (new this session)
- `allocation/tier_cost_helper.py` — `switch_cost_bps`, `partial_rebalance_cost_bps` (new this session)
- `scripts/tier_analysis/annual_tier_current.py`, `three_windows.py` — reference scripts

**Do NOT read:** `data/`, `logs/`, `reports/`, `state/`, `.venv/`, `uv.lock`.
**Do NOT touch the daemon or `state/`.**
**User commits and pushes; do not commit unless asked.**

## 2. The three experiments

### Experiment A: Wider exit deadband sweep

**Goal:** the 1-point deadband (enter ≤23, exit >24) causes 18 sw/yr recently when VXN oscillates near the threshold. A wider exit reduces switches with a causal mechanism (require a larger VXN move to trigger exit). Test whether switching less improves xSharpe or drawdown on any window without sacrificing the long-run advantage.

**What to build:** script `scripts/tier_analysis/deadband_sweep.py` (not in allocation module — it's a research script). No new module needed; just uses `intraday_engine`.

**Algorithm:**
```python
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.setdefault("y26", (None, None))
eng.WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
eng.WINDOWS.setdefault("recent", ("2024-03-25", None))
eng.WINDOWS.setdefault("train", ("2007-11-20", "2019-12-31"))
eng.WINDOWS.setdefault("test", ("2020-01-01", None))

ENTER_AT = 23.0
COST_BPS = 13.0
exit_range = [24.0, 25.0, 26.0, 27.0, 28.0, 30.0, 33.0]

rows = []
for exit_above in exit_range:
    tiers = eng.tiers_vxn_deadband(inp.vxn, ENTER_AT, exit_above)
    run = eng.simulate(inp, tiers, cost_bps=COST_BPS)
    for wname, wkey in [("26yr", "y26"), ("since-2007", "vxn_era"), ("recent", "recent"), ("train", "train"), ("test", "test")]:
        s = eng.window_stats(run, wkey)
        rows.append({"exit_above": exit_above, "window": wname, **s})
```

**Output table:** for each exit_above value, show: xSharpe, maxDD%, sw/yr, ret% across all five windows. The train/test split (train=2007-2019, test=2020+) is essential — don't pick a winner purely on train.

**What to look for:**
- Does wider exit reduce sw/yr on the recent window without hurting 26yr/since-2007 xSharpe?
- Is there an exit_above where recent xSh > 0.37 without sacrificing 26yr xSh > 0.70?
- Is the Pareto frontier monotone (wider exit = fewer switches + lower xSharpe) or does it show a sweet spot?

**Caution:** all windows except `test` (2020+) are in-sample for the VXN 23/24 choice. A winning exit_above on train data is not validated. Only `test` and `recent` are genuinely out-of-sample.

---

### Experiment B: VXN-scaled position size

**Goal:** instead of binary 100%/0% Nasdaq allocation, scale the Nasdaq weight continuously by how far VXN is below the entry threshold. Preserves the VXN timing signal but reduces peak exposure (and thus drawdown) when VXN is just under 23. Analogous to vol-target but driven by implied vol level rather than realized vol.

**Weight formula:**
```
w(vxn) = clamp((enter_at - vxn) / (enter_at - full_weight_at), 0, 1)
```
Where:
- `enter_at = 23` (threshold where weight first becomes positive)
- `full_weight_at = 15` (VXN level where weight reaches 100%)
- E.g. VXN=20 → w = (23-20)/(23-15) = 3/8 = 37.5%
- VXN=15 → w = 100%
- VXN≥23 → w = 0% (all cash)

**What to build:** new function `vxn_scaled_blend` in `allocation/intraday_comparators.py`. Reuse `_run_blend` already there.

```python
def vxn_scaled_blend(
    inp: Inputs,
    enter_at: float = 23.0,
    full_weight_at: float = 15.0,
    cadence_days: int = 5,
    deadband: float = 0.02,
    pot_gbp: float = POT_GBP,
) -> Run:
    """Nasdaq weight scales linearly from 0 at VXN=enter_at to 1 at VXN=full_weight_at."""
    dr = _daily_simple_ret(inp)
    r_nasdaq = dr[:, _NASDAQ]
    r_cash = dr[:, _CASH]

    # Daily VXN: last non-NaN reading per day
    n_days = len(inp.days)
    daily_vxn = np.full(n_days, np.nan)
    for d in range(n_days):
        mask = inp.day_codes == d
        valid = inp.vxn[mask]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            daily_vxn[d] = valid[-1]

    # Target weight per day (lagged 1 day to avoid look-ahead)
    raw_w = (enter_at - daily_vxn) / (enter_at - full_weight_at)
    target_w_series = pd.Series(raw_w).clip(0.0, 1.0).shift(1).fillna(0.0)
    target_w = target_w_series.to_numpy()

    cost_fn = lambda dw: partial_rebalance_cost_bps(dw, pot_gbp)
    return _run_blend(r_nasdaq, r_cash, target_w, cadence_days, deadband, cost_fn, inp)
```

**Grid to run:** sweep `full_weight_at` in [12, 14, 15, 17, 20] — this controls how steep the ramp is. Also sweep `enter_at` in [22, 23, 24] to see if a slightly different threshold helps. Keep `cadence_days=5, deadband=0.02`.

**Output:** for each (enter_at, full_weight_at) pair, report xSharpe, maxDD%, average Nasdaq weight (in-market-time analog), sw/yr, ret% across the three windows. Compare against binary current rule.

**Tests to add:** `tests/allocation/test_intraday_comparators.py` — add `TestVxnScaledBlend` class:
- At VXN < full_weight_at: weight = 1.0 (fully in Nasdaq)
- At VXN = enter_at: weight = 0.0 (fully in cash)  
- At VXN between: weight is between 0 and 1
- Returns Run object with correct shape

**What to look for:** does VXN-scaled achieve the drawdown improvement of VT5% (−10%) without sacrificing the xSharpe of the current binary rule (+0.76)?

---

### Experiment C: Threshold grid search with train/test split

**Goal:** the 23/24 entry/exit pair was eyeballed. A systematic grid search on the TRAIN window (2007-2019) with reporting on the TEST window (2020+) may find better thresholds. This is the only experiment that could genuinely find a better threshold without overfitting (provided the test window is held out during selection).

**Grid:**
- `enter_at` in [19, 20, 21, 22, 23, 24, 25]
- `exit_above` in [enter_at, enter_at+1, ..., enter_at+10] (deadband 0..10)
- Total: ~55 combinations

**What to build:** script `scripts/tier_analysis/threshold_grid.py`.

```python
from itertools import product
from Strategy_Auto_Trader.allocation import intraday_engine as eng

inp = eng.load_inputs()
eng.WINDOWS.setdefault("y26", (None, None))
eng.WINDOWS.setdefault("vxn_era", ("2007-11-20", None))
eng.WINDOWS.setdefault("train", ("2007-11-20", "2019-12-31"))
eng.WINDOWS.setdefault("test", ("2020-01-01", None))
eng.WINDOWS.setdefault("recent", ("2024-03-25", None))

COST_BPS = 13.0
rows = []

for enter_at in [19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]:
    for deadband in range(0, 11):  # exit_above = enter_at + deadband
        exit_above = enter_at + deadband
        tiers = eng.tiers_vxn_deadband(inp.vxn, enter_at, exit_above)
        run = eng.simulate(inp, tiers, cost_bps=COST_BPS)
        row = {"enter_at": enter_at, "exit_above": exit_above, "deadband": deadband}
        for wname, wkey in [("train", "train"), ("test", "test"), ("recent", "recent"), ("y26", "y26")]:
            s = eng.window_stats(run, wkey)
            row[f"{wname}_xsh"] = s["xsharpe"]
            row[f"{wname}_maxdd"] = s["max_dd_pct"]
            row[f"{wname}_sw"] = s["sw_per_yr"]
            row[f"{wname}_ret"] = s["ret_pct"]
        rows.append(row)
```

**Output:**
1. Sort by `test_xsh` descending — top 10 rows (best test-window xSharpe)
2. Sort by `test_maxdd` ascending — top 10 rows (best test-window drawdown)
3. Highlight current 23/24 row as reference
4. Pareto frontier: combinations that are non-dominated on (test_xsh, test_maxdd)

**Selection rule:** a winner must beat the current rule on BOTH train_xsh and test_xsh. Don't pick a test winner that destroys train xSharpe — that's overfitting to the post-2020 period.

**What to look for:**
- Is there a threshold pair with test_xsh > 0.50 (current 0.37 on test, ~0.88 on the full test window) that also has train_xsh ≥ 0.90?
- Does a wider deadband (exit_above >> enter_at) dominate across windows?
- Is the 23/24 pair actually near the optimum, or is there a clear better option?

Note: the `test` window here is 2020-01-01 onwards (not the `recent` 2024+ window). 2020 includes COVID — in-sample for the threshold sweep table in BACKTEST_LOG but not for the original threshold eyeballing. `recent` 2024+ is fully out-of-sample for everything.

---

## 3. Order of execution

Run in this order — each builds on the prior:

**A first:** quickest, no new code, just a script. Immediately tells whether wider deadband helps and sets context for B and C.

**B second:** new function + tests. The VXN-scaled result determines whether a hybrid approach is worth pursuing before the grid search.

**C last:** most exhaustive, most in-sample risk. Use results from A and B to narrow the grid if needed.

After each experiment, update BACKTEST_LOG with exact commands, data range, result table, conclusion, and caveats. Follow the existing log format.

## 4. Cost model reminder

At £20k pot (EQGB.L ↔ CSH2.L):
- Full switch: 10.0 bps commission-only, 16.0 bps with spread. Use 13 bps (DEFAULT_COST_BPS) as flat assumption — it's bracketed and validated.
- Partial rebalance |Δw|: dominated by £1/side minimum → ~1 bps of NAV per rebalance at |Δw| ≤ 10%.

Helper: `from Strategy_Auto_Trader.allocation.tier_cost_helper import switch_cost_bps, partial_rebalance_cost_bps`

## 5. What to NOT do

- Do not change the live daemon, `state/`, or `config/overnight_strategy.json`.
- Do not read `data/`, `logs/`, `reports/`, `.venv/`, `uv.lock`.
- Do not run `live_sim.py`, `live_daemon.py`, or any IBKR-connecting code.
- Do not commit without being asked.
- Do not present in-sample winners as validated results — always show train and test windows separately.
- Do not use raw Sharpe as the primary metric; use xSharpe (excess-over-cash). `window_stats` returns both — use `xsharpe` key.
