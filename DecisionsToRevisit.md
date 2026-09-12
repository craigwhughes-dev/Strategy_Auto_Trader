---

### DONE, flagged for revisit: rolling-Sharpe volatility diagnosis + same-day cap (2026-09-02, real IBKR data — separate from the synthetic-data rebuild above)

**Note up front:** everything below used `live_sim.py`'s real `--source ibkr` path, not `--synthetic-data-dir` — the sigma/gamma fix and HMM re-warm above don't directly touch any number here. Documented anyway per explicit instruction, since the same fetch chain (`ibkr_data.py`/`data_cache.py`/`quant_engine.py`) was modified this session and is shared infrastructure — worth a sanity re-check once the current rebuild settles, in case cache/pacing behavior interacts.

**Context:** rolling-30d Sharpe on the `optimised_new` £100k full-universe backtest (2024-11-21 to 2026-08-27) swung wildly (+8.79 to -5.60). Two mechanisms found:
- Dec 2024 dips: backtest ramp-up noise (~5 real trades in that window) — not a real signal.
- Mar/Apr 2025 dips: real, driven by correlated same-day trade clustering — e.g. 22 trades closed same day on 2025-03-11 during the worst Sharpe stretch. corr(max same-day trade-event count in trailing 30d, rolling Sharpe) = -0.37 (real, moderate, not dominant) — the portfolio isn't diversified across ~70 tickers so much as making one correlated market-timing bet dressed as many, since entries/exits are all driven by the same shared HMM regime signal.

**Tested and rejected:** a strategy-owned `same_day_deployment_cap_pct` gate in `arbitrate()` (resolved off the strategy's Entry class via `resolve_same_day_deployment_cap()`, not a CLI flag — see `.claude/rules/strategy.md`'s Strategy-Owned Admission Attributes section), capping total new capital admitted per calendar day. Validated with `scripts/run_same_day_cap_sweep.ps1` + `scripts/analyze_same_day_cap_sweep.py` across cap_pct ∈ {0.10, 0.20, 0.35, 0.50}. **Result: negative** — no value reduced rolling-Sharpe volatility. 0.35/0.50 never bind (worst clustering day deploys under a third of the pot); 0.10/0.20 bind but make things worse (0.10: Sharpe std +9.8%, realized P&L -40.7%; 0.20: Sharpe std +2.1%). `same_day_deployment_cap_pct` is back to `None` on `OptimisedNewEntry` permanently — finding recorded in that file's own comment and in `project_same_day_cap_validation_20260902` memory. Don't re-run this exact sweep — it's settled, unless the revisit above turns up something that specifically changes it.

**Side fixes made along the way (committed/pushed, `2fb9c03`, `96d8dc4`, `990dc76`):**
- `arbitrate()` was mutating a shared `Candidate.record` across a `--pot-sizes` sweep, silently corrupting the trade journal for every pot size but the last.
- `fetch_hourly`'s incremental IBKR cache always opened a live reqHistoricalData call per ticker regardless of whether the caller needed today's newest bar — a full-universe backtest sweep collided with the live daemon's own polling for IBKR's account-wide pacing limit, timing out on nearly every ticker overnight. Added `historical_only` (threaded `IBKRDataClient.fetch_hourly` → `quant_engine.fetch_hourly` → `data_cache.fetch_hourly_cached` → `ticker_ranking.py`'s fetch functions) so a cache-served backtest skips the live gap-fill entirely; `live_sim.py` always passes `historical_only=True` (it never feeds the live daemon), `run.py`/the live daemon's own fetch path is untouched.

**Open next step, not started: correlation-aware admission gating.** The $-deployment cap failed because it throttles correlated winners and losers proportionally without changing the correlation *structure* driving the volatility — same-shaped bet, just smaller. A different mechanism would gate admission by how correlated a candidate's ticker/sector/regime-driver is with what's *already* queued to enter that day, not by aggregate $ amount — e.g. cap the number of highly-correlated (same-sector, or same-regime-flip-day) entries admitted together, or require some diversity-of-driver score among same-day admissions before the Kelly/cash gates even run. Would need: (1) a correlation or co-movement measure computable cheaply per candidate at arbitration time (sector bucket is the simplest starting point — actual return correlation would need a rolling covariance matrix, more expensive), (2) the same strategy-owned-attribute pattern as the rejected cap, (3) the same sweep-and-validate discipline against this exact £100k/top-70/2024-11-21 window before trusting a result. Not scoped in detail yet — worth a fresh design pass, not a variant of the rejected cap.

---

### BLOCKED on above sigma-fix rebuild: 26yr regime-split re-validation (2026-09-07, separate session)

**Context:** User raised regime-conditioned risk sizing (more risk in calm markets, less in volatile) after the 26yr synthetic backtest came back unprofitable. Before designing that, wanted to root-cause which regime is actually dragging the 26yr number down. This ran concurrently with / just before the sigma-fix work above — same root problem, discovered independently.

**Recommendation given (not yet acted on):** Don't switch strategies by regime — `live_sim.py` is one-pot-per-strategy, no clean multi-strategy capital-sharing (`.claude/rules/cli.md`), and strategies are standalone/no-inheritance by convention ([[feedback_strategies_standalone_no_inheritance]]). Instead scale `kelly_fraction` continuously inside one strategy's Entry class off a signal already computed (HMM P(bull), or the existing VIX/`trend_quality` signal) — cheaper to test, no engine changes, strategy-owned per existing convention. Not implemented — was gathering evidence first.

**Root-cause attempt found the existing regime-split table (this file's old "26yr synth results with Plan B=6" section, sourced from `data_synthetic/journals/synth_26yr.csv`) was contaminated:** journal was additive across two config generations (453 pre-Plan-B + 567 post-Plan-B trades = 1020 mixed, per the file's own contamination note) and predates `min_entry_score=7.0`/`vol_stop_mult=1.5` (adopted later the same day, `758d7ff`). Old contaminated files backed up to `scratch_backup_20260907/` (`synth_26yr.csv`, `synth_26yr_equity.csv`) rather than deleted.

**Attempted a clean re-run** (`scripts/run_synth_26yr.ps1`) to regenerate the journal cleanly on the *old, pre-sigma-fix* synthetic data. Hit process-management issues mid-session (a stray `Start-Job` plus a separate Bash background call both targeting the same output/log paths) — cleaned up, killed the strays, restarted once as a single tracked background run. That run was killed by session/process teardown before finishing: current `data_synthetic/journals/synth_26yr.csv` holds only 481 trades reaching to 2025-08-21 of the full 2000-2026 span, no equity file. **Not usable for regime-split analysis — incomplete.**

**Superseded regardless by the sigma fix above** — every number in the old contaminated table, and this incomplete rerun, used the over-dispersed (~2.6x too much intrabar variance) synthetic bars. Over-dispersion hits volatile-regime trades hardest (fatter intrabar swings → premature stop-outs), so the old table's "volatile regimes have much worse per-trade edge" finding is a specific candidate for reversal, not just a number that shifts slightly.

**Next steps once Night 4 HMM re-warm completes (2026-09-11):**

1. **26yr baseline + regime split** (primary validation — does sigma fix close the real-vs-synthetic gap?):
   ```powershell
   # journal/equity paths already clear — nothing to back up first
   powershell -File scripts\run_synth_26yr.ps1
   uv run python -m scripts.analyse_regime_split --journal data_synthetic/journals/synth_26yr.csv --equity data_synthetic/journals/synth_26yr_equity.csv
   ```
   Old result: real +22.1% vs synthetic -12.8% (35pp gap). Target: gap < 15pp. Old volatile regime: 16.3% WR / -2.88% mean_ret — expect material improvement since over-dispersed sigma was hitting stops on noise. Current live regime 2023-26 was already +0.70% / 61.9% WR — should stay positive.

2. **Plan A/B sweep** (verify Plan A still negative, Plan B=6 still positive with corrected sigma):
   ```powershell
   uv run python -m scripts.sweep_exit_params
   ```
   Old: Plan A negative (17.5% WR); Plan B=6 positive (+0.12pp). If Plan A's trade count no longer doubles after sigma fix (stops were firing on synthetic noise), mechanism may be worth a second look — but don't enable without a new clean sweep first.

3. **score_rr_by_vsmult.py** (score-stratified R:R — now baseline is score gate=7.0, vol_stop_mult=1.5):
   ```powershell
   uv run python -m scripts.score_rr_by_vsmult
   ```
   Old result (pre-score-gate, vol_stop_mult=0.5): Score-6 avg_loss -4.64%, WR 44.7%. With score gate, expect Score-6 population is higher quality (trades that would have been score 6 but below 7.0 are now excluded). Also check if vol_stop_mult=1.5 now aligns with real IBKR on synthetic.

4. **Score gate on synthetic** (adopted from real IBKR data only; first synthetic validation):
   Score gate=7.0 gave +0.63pp mean_ret on real (100→57 trades, 57.4%→62.7% WR). Run a sweep on synthetic with gate at 6.0/7.0/7.5/8.0 to check the synthetic optimum matches real — if they diverge, that's evidence of residual synthetic/real mismatch.
   ```powershell
   uv run python -m scripts.sweep_exit_params  # includes score gate sweep if added to _SWEEPS
   ```
   Add `"min_entry_score": [6.0, 7.0, 7.5, 8.0]` to `_SWEEPS` in `sweep_exit_params.py` before running.

5. **VIX kelly segmentation** (check if quintile non-monotonicity survives corrected sigma):
   ```powershell
   uv run python -m scripts.vix_kelly_segmentation --strategy optimised_new
   ```
   Old result: win_rate NOT monotone-degrading; trough at VIX 17–19, not extremes; f(vix) scaling NOT warranted. If non-monotonicity persists, verdict stands. If it becomes monotone after sigma fix, reconsider.

6. **Exit-parameter audit crash-window re-check** (see "Needs revisiting once sigma-fix rebuild completes" section below for specific scripts):
   Lower priority — the two currently-live parameters (`sell_threshold=-6.0`, `trend=1.0`) have strong real-window support. Crash-window confirmation is a belt-and-suspenders check, not a prerequisite.

7. Only after steps 1-3 confirm improved synthetic/real alignment: revisit regime-conditioned Kelly-scaling idea — with corrected numbers, not over-dispersed ones.

---

### IN PROGRESS: avg_loss fixes — breakeven trailing stop and regime-forced exit (2026-09-06/07)

**Context:** Exit parameter sweep + R:R analysis found the structural problem — not entry quality, not exit timing on winners. Losers fall straight to the -8% hard stop because the current trailing stop only fires when price > entry (profitable). Two targeted fixes to test:

**Why the current trailing stop doesn't help losers:**
In `Strategy_Auto_Trader/core/exits.py:93-101` (`_check_exit_conditions`):
```python
if peak_price_since_entry > entry_price and cur_close >= entry_price:
    drop_from_peak = ...
```
Both guards require the trade to have been profitable first. A loser that never rises above entry goes straight to the -8% hard stop. `peak_price_since_entry` is initialized to `cur_close` (= entry_price) at entry — see `consolidated_engine.py:625`.

**Success target:** Score-6 avg_loss shrinks from -7.24% to ≤ -4.63% (the break-even level at current 39% win rate and +3.54% avg_win). Or win rate rises above break-even 67% threshold.

---

#### Option A: Breakeven trailing stop (trail from entry, not from peak)

**Diagnosis:** Trailing stop needs to fire from entry_price as the reference, not from peak. This exits losers at `-vol_stop_pct%` from entry (≈ -3 to -4.5% at vol_stop_mult=0.5) instead of waiting for the -8% hard stop.

**Files to change:**

1. `Strategy_Auto_Trader/core/exits.py` — `_check_exit_conditions()`:
   - Add `breakeven_trailing: bool = False` parameter (before `use_sar_stop`)
   - Replace the trailing stop block (lines 93-101) with:
   ```python
   if effective_stop > 0.0 and not trailing_stop_hit:
       peak_price_since_entry = max(peak_price_since_entry, cur_close)
       if breakeven_trailing:
           # trail from entry_price minimum — exits losers before hard stop
           peak_ref = max(peak_price_since_entry, entry_price)
           drop_from_ref = (peak_ref - cur_close) / peak_ref
           if drop_from_ref >= effective_stop:
               trailing_stop_hit = True
               gain_pct = (cur_close - entry_price) / entry_price * 100
               sell_reason = (f"trailing_stop_be({drop_from_ref*100:.1f}% from ref, "
                             f"{gain_pct:+.1f}% from entry)")
       else:
           # existing behavior: only fires when in profit
           if peak_price_since_entry > entry_price and cur_close >= entry_price:
               drop_from_peak = (peak_price_since_entry - cur_close) / peak_price_since_entry
               if drop_from_peak >= effective_stop:
                   trailing_stop_hit = True
                   gain_pct = (cur_close - entry_price) / entry_price * 100
                   sell_reason = (f"trailing_stop({drop_from_peak*100:.1f}% from peak, "
                                 f"still +{gain_pct:.1f}% from entry)")
   ```

2. `Strategy_Auto_Trader/plugins/exit_rules.py` — `StandardExitRules`:
   - Add `breakeven_trailing: bool = False` to `__init__`, store as `self._breakeven_trailing`
   - Pass to `_check_exit_conditions(...)` call in `check()`

3. `Strategy_Auto_Trader/strategy/optimised_new.py` — `OptimisedNewExit.__init__`:
   - Add `breakeven_trailing: bool | None = None` param (pattern matches existing overrides)
   - Pass to `build_standard_exit_rules(defaults={..., "breakeven_trailing": False}, breakeven_trailing=breakeven_trailing)`
   - To test: set default to `True` in the defaults dict and sweep

**To sweep:** Add `"breakeven_trailing": [False, True]` to `_SWEEPS` in `scripts/sweep_exit_params.py`. The existing sweep infrastructure will run it. Also run `scripts/score_rr_by_vsmult.py` (already set up) with the flag toggled to see avg_loss change.

**Expected result:** avg_loss for score-6 drops from -7.24% toward -3 to -4%. Break-even win rate drops from 67% toward 50%. Mean_ret for score-6 may approach breakeven.

**Key risk:** avg_win may also shrink if the breakeven stop catches early partial winners before they run. Check peak_capture metric.

---

#### Option B: Regime-forced exit ignoring min_hold_bars

**Diagnosis:** When HMM flips to bear (p_bull < exit_prob), current code only exits if `bars_held >= min_hold_bars` (48 bars = ~2 days). Losers in an HMM bear regime sit in the trade for 2+ days accumulating losses before the signal exit is allowed.

**File to change — only `consolidated_engine.py`:**

In the per-bar loop after the exit-rules check (around line 591-598), add before the signal SELL check:
```python
# Regime-forced exit: HMM confirmed bear, bypass min_hold_bars
# (12-bar minimum = ~1.7 hours, avoids immediate-reversal whipsaws)
_min_hold_regime = getattr(_exit, "min_hold_bars_regime_exit", 12)
if (not exit_hit
        and regime_state.regime_signal is not None
        and regime_state.regime_signal <= 0
        and bars_held >= _min_hold_regime):
    trade_event = "SELL"
    sell_reason = f"regime_forced(signal={regime_state.regime_signal:.2f})"
    exit_hit = True
```

`regime_state.regime_signal` is already computed at each bar (it's the smoothed p_bull scalar — see `plugins/hmm_regime.py`). No new fields needed on `BarData` or `TradeState`. Strategy-owned via `min_hold_bars_regime_exit` attribute on the exit class (default 12 if not declared — following the strategy-owned pattern from `strategy.md`).

**To sweep:** Add a quick before/after run in `scripts/sweep_exit_params.py` by passing a patched exit class with `min_hold_bars_regime_exit = 6/12/24/48`.

---

#### Implementation status (2026-09-07)

Both options implemented. 1541 tests pass. Sweep running.

**Option A** — coded across 4 files:
- `core/exits.py`: `_check_exit_conditions` got `breakeven_trailing: bool = False` param; trailing block restructured with `peak_ref = max(peak, entry_price)` branch
- `plugins/exit_rules.py`: `StandardExitRules.__init__` stores `_breakeven_trailing`; passes to `_check_exit_conditions`
- `strategy/base/exit_overrides.py`: `_STANDARD_EXIT_RULES_PARAMS` now includes `"breakeven_trailing"`
- `strategy/optimised_new.py`: `OptimisedNewExit.__init__` takes `breakeven_trailing: bool | None = None`; defaults dict sets `True` for testing

**Option B** — engine-only in `quant_hmm/consolidated_engine.py`:
- Regime-forced exit block added inside `else:` branch, before signal SELL check
- `_min_hold_regime = getattr(_exit, "min_hold_bars_regime_exit", 12)` — strategy-owned, `None` = off
- Guards: `regime_state.regime_signal is not None and <= 0 and bars_held >= _min_hold_regime`

**Sweep** — `scripts/sweep_exit_params.py` updated:
- Baseline `vol_stop_mult` corrected 1.0 → 0.5; `breakeven_trailing=True` added to baseline
- `_SWEEPS` includes `"breakeven_trailing": [False, True]`
- New `_REGIME_FORCED_SWEEP` section: `[None, 6, 12, 24, 48]` (None=off)

#### Sweep results (2026-09-07, bg7233q55 — with breakeven_trailing=True baseline, now reverted)

**Plan A: NEGATIVE — do not enable.**
- `True`: 10564 trades, 17.5% win rate, -2.07% mean_ret (Sharpe -29.5)
- `False`: 5230 trades, 52.6% win rate, -1.82% mean_ret (Sharpe -15.1)
- Root cause: vol_stop_mult=0.5 trail from entry fires within 1-3 bars on normal price dips, triggering rapid stop-out cycling. Trade count doubles (recycled capital = more entries), win rate collapses.
- `breakeven_trailing` stays `False` in `optimised_new.py` defaults and sweep baseline. Feature code kept in `exits.py`/`exit_rules.py` for future testing with looser vol_stop_mult if needed.

**Plan B: marginal at 6 bars, but contaminated by bad Plan A baseline. Re-running sweep clean.**
- With breakeven=True as baseline: None=-2.07%, 6bars=-1.92%, 12bars=same as off, 24/48 worse
- Need clean re-run (breakeven=False) to see true Plan B effect

#### Clean sweep results (2026-09-07, bvg9pb1p2 — breakeven=False baseline, correct)

**Plan B: POSITIVE at 6 bars.**
```
None (off)   5230   52.6%   -1.82%   35.7%   Sharpe -15.109
6 bars       5604   51.2%   -1.70%   37.3%   Sharpe -14.783  ← adopted
12 bars      5230   52.6%   -1.82%   (identical to off)
24 bars      4843   54.4%   -2.00%   (worse)
48 bars      4587   57.1%   -2.11%   (worse)
```
- `min_hold_bars_regime_exit = 6` declared on `OptimisedNewExit`. Live daemon picks up automatically.
- 12-bar identical to off: all regime flips that matter happen in first 6 bars or not at all.

**Other sweep observations (not acted on):**
- `stop_loss_pct=0.10`: -1.66% mean_ret vs -1.82% — worth sweeping on real data before touching live
- `min_hold_bars=168`: 54.0% WR vs 52.6%, -1.79% vs -1.82% — marginal; leave

#### 26yr synth results with Plan B=6 (2026-09-07, b0qqy2zvp)

567 trades logged (up from 453 in old journal — Plan B recycling capital faster, enabling more entries).

**Regime split — NOTE: journal is additive; synth_26yr.csv now has 1020 trades (453 pre-PlanB + 567 new). Period counts are mixed. Clear the file before next clean run.**

```
Regime                     Type         Trades  WinRate  MeanRet   TradePnL
dot-com crash 2000-02      volatile         89    32.6%   -2.35%    -18,804
bull 2003-07               calm            623    57.9%   -0.59%    -29,469
GFC 2008-09                volatile     (no trades — VIX gate 20.0 blocked all entries)
recovery 2010-11           intermediate     21    52.4%   -0.76%     -1,486
bull 2012-19               calm            249    53.4%   -0.78%    -37,776
COVID/recovery 2020-21     intermediate     16    62.5%   +0.43%       -388
rate-hike 2022             volatile          1     0.0%   -3.41%       -665
recovery 2023-26           calm             21    61.9%   +0.70%     +1,591  ← current live regime
```

By type: calm 57.8% WR / -0.22% mean_ret; volatile 16.3% / -2.88%.
Equity: trade_pnl=-86,998 total; interest_est=+202,844; total_growth=+115,846.
Trading itself loses money across 26yr; portfolio grows from synthetic interest on cash balance.

**Key validation:** 2023-26 (current live regime) is POSITIVE (+0.70%, 61.9% WR). VIX gate prevents all GFC entries.

**score_rr_by_vsmult.py results (bpny1zzxr) — both variants now include Plan B=6:**

```
vol_stop_mult=1.0 + Plan B=6:
  Score 6:  3346  WR 37.9%  MeanRet -1.72%  AvgWin +2.81%  AvgLoss -4.49%  BE 61.5%  HS 16.6%
  Score 7:   389  WR 52.4%  MeanRet -1.99%  AvgWin +2.91%  AvgLoss -7.39%  BE 71.7%  HS 33.2%
  Score 8:   995  WR 41.0%  MeanRet -3.29%  AvgWin +3.09%  AvgLoss -7.73%  BE 71.4%  HS 43.0%
  ALL:      4730  WR 39.8%  MeanRet -2.07%

vol_stop_mult=0.5 + Plan B=6 (CURRENT LIVE):
  Score 6:  3701  WR 44.7%  MeanRet -1.62%  AvgWin +2.13%  AvgLoss -4.64%  BE 68.5%  HS 16.0%
  Score 7:   560  WR 65.7%  MeanRet -1.44%  AvgWin +1.92%  AvgLoss -7.90%  BE 80.4%  HS 25.5%
  Score 8:  1343  WR 56.1%  MeanRet -2.45%  AvgWin +1.88%  AvgLoss -8.00%  BE 81.0%  HS 33.9%
  ALL:      5604  WR 49.5%  MeanRet -1.80%
```

**vs HANDOFF success target:**
- Score-6 avg_loss: improved from -7.24% → -4.64% (2.6pp). Target ≤ -4.63% — effectively met.
- Score-6 win rate: 44.7% vs 68.5% breakeven — still losing money per-trade. Mean_ret -1.62%.

**Structural insight:**
- Score-7/8 avg_loss near hard stop (-7.90%, -8.00%) — Plan B=6 not cutting those (losers hit hard stop before 6 bars)
- vol_stop_mult=0.5 trades win rate (+44.7% vs 37.9%) for smaller avg_win (+2.13% vs +2.81%) — breakeven threshold rises proportionally, no free lunch
- All mean_rets still negative; trading loses money, portfolio grows from synthetic interest income

**Plan A/B summary:**
- Plan A (breakeven trailing): NEGATIVE — rejected
- Plan B=6 (regime-forced exit): POSITIVE — adopted; `min_hold_bars_regime_exit = 6` on OptimisedNewExit
- Success target (avg_loss ≤ -4.63%) EFFECTIVELY MET but win-rate gap (44.7% vs 68.5% needed) means strategy still structurally unprofitable on 26yr synthetic

#### Journal contamination note
`data_synthetic/journals/synth_26yr.csv` now has 1020 trades (453 pre-PlanB + 567 new). **Clear before next run:**
```powershell
Remove-Item data_synthetic\journals\synth_26yr.csv
```

#### Next ideas if win-rate gap persists
- Tighter hard stop: sweep 0.05/0.06 on real data (synthetic says 0.10 better, risky on live)
- ~~Score gating: only admit score >= 7 entries~~ — DONE: `min_entry_score=7.0` adopted in `758d7ff`; re-validate on corrected synthetic post-rebuild
- Entry filter: require regime_signal > 0.3 (not just > 0) at entry to demand stronger bull confirmation

**Reference files for context:**
- `scripts/sweep_exit_params.py` — sweep harness, 20 tickers, synthetic data
- `scripts/score_rr_by_vsmult.py` — score-stratified R:R comparison, use this to measure avg_loss change
- `data_synthetic/journals/synth_26yr.csv` — 453-trade vol-filtered journal (correct input for regime analysis)
- `scripts/analyse_regime_split.py` — regime-split P&L by calm/volatile/intermediate

---

### COMMITTED (2026-09-07, 758d7ff): score gate + vol_stop_mult 1.5 + min_hold_bars 0

Full sequence of `optimised_new.py` changes committed in `758d7ff` (built on top of 2026-09-03/04 audit that was committed in prior sessions):

**2026-09-03/04 audit (committed earlier):**
- `vol_stop_mult` 2.0 → 1.0
- `trend` weight 2.0 → 1.0, `sell_threshold` -4.5 → -6.0 (swept together; `_RSI_OVERBOUGHT=60` validated positive alone but not adopted — it overrides sell_threshold contribution once combined, confirmed via 2-way isolation)

**758d7ff additional changes (current live config):**
- `min_entry_score = 7.0` added to `OptimisedNewEntry` — score gate blocks entries below threshold. Sweep on real IBKR (20 tickers, 2.9yr): +0.63pp mean_ret (57.4% → 62.7% WR, 100 → 57 trades). Strongest single improvement of the session.
- `vol_stop_mult` 0.5 → **1.5** — re-swept with score gate active on real IBKR; 1.5 best Sharpe (+35.3 vs +12.2 at 0.5). Both real sweeps confirm looser > tighter. Note: synthetic diverged from real on this knob (old over-dispersed sigma was firing stops too eagerly) — real data overrules for live decisions.
- `min_hold_bars` 48 → **0** — inert across all values 0-168 on gated real sweep; score gate + Plan B=6 cover early exits, composite-signal SELL never fires with score gate active.
- `min_hold_bars_regime_exit = 6` (Plan B) — adopted, see above section.
- `breakeven_trailing = False` (Plan A) — feature code present in `exits.py`/`exit_rules.py` but disabled, see above section.

**Synthetic re-run needed after HMM rebuild (2026-09-11+):**
Run `scripts/sweep_exit_params.py` and `scripts/score_rr_by_vsmult.py` with current config as baseline to check:
- Does synthetic now agree with real on vol_stop_mult 1.5 > 0.5? (if yes, confirms no residual sigma artifact)
- Does score gate=7.0 improve synthetic mean_ret similarly to real (+0.63pp)?
- Does min_hold_bars inertness still hold? (expected yes — mechanism is unchanged)

Full history of the exit-parameter audit lives in `BACKTEST_LOG.md`'s 2026-09-03/04 entries and `optimised_new.py`'s own class-attribute comments.

### SUPERSEDED by the sigma-fix rebuild above: 26-year synthetic out-of-sample validation (2000-2026)

Original idea (2026-09-04): every parameter in the 2026-09-03/04 exit-parameter audit was tuned against the same two windows (2008 synthetic crash, prev-2yr real) — multiple-comparisons/overfitting-to-the-test-set risk. Planned to warm the synthetic HMM cache for 2000-2026 and re-validate. Never started — overtaken by the sigma-fix discovery (2026-09-07, see top of file), which found the synthetic bridge itself was generating ~2.6x too much intrabar variance. A 26yr re-run on the *old* data would have baked the same bug in at larger scale; the sigma-fix rebuild + re-warm already in progress supersedes this item outright — no separate 2000-2026 warm needed once that finishes, since the fix presumably gets folded into whatever range gets rebuilt.

### Needs revisiting once the sigma-fix rebuild completes: 2026-09-03/04 exit-parameter audit's synthetic-crash-window results

Every crash-window number in the 2026-09-03/04 audit and its follow-ups (this session) used the *old*, over-dispersed synthetic Jan2008-Jul2009 data — same root cause as the sigma-fix section at the top of this file (premature vol/trailing-stop fires from inflated intrabar noise). Real-window (prev-2yr, live IBKR data) numbers from the same sessions are **not** affected — only what touched `data_synthetic/`.

**Affected (re-run crash-window portion once Night 4 HMM re-warm + rebuild completes):**
- VIX gate threshold selection (2026-09-02/03) — the crash-protection case for `vix_entry_gate_threshold=20.0` (crash return -39.1% vs baseline -90.7%) leaned heavily on the old synthetic crash data, though the *final* full-history validation that confirmed 20.0 used real IBKR data only (2023-01-01+) — that part likely still holds.
- Exit-parameter audit #1 `min_hold_bars` — "keep 48" decision used old crash data. Since superseded anyway by the separate `min_hold_bars_regime_exit=6` mechanism from the 2026-09-06/07 avg_loss work (a different, additive knob — see above).
- Exit-parameter audit #2 `vol_stop_mult=1.0` adoption — crash-window comparison used old data. Since superseded anyway by the 2026-09-06 real-data sweep that further changed it to 0.5.
- Exit-parameter audit #3 `sell_threshold=-6.0` — **currently live**, not superseded by anything later in this file. Crash-window contribution to this decision (crash -8.6% alone, -7.5% combined with weights) needs re-checking; the real-window case (+25.2%/+22.5%) is unaffected and was already the dominant reason for adoption.
- Exit-parameter audit #4 `trend=1.0` weight + its dedicated crash-window follow-up (t1s3 vs t2s3, confirmed via 2 extra crash-only runs) — **currently live**, same caveat as #3: crash number needs re-checking, real number (+20.2% vs +12.2%) unaffected.
- Exit-parameter audit #5 `profit_stop_scale`/`min_stop_pct` validation — "keep current" decision partly used old crash data (crash -10.2% vs -13.2% off). Real-window part of the validation (+12.2% vs +7.6%) unaffected.
- Combined-winners check (sell_threshold+trend+RSI all together, and the sell_threshold+weights / sell_threshold+RSI 2-way isolation) — the crash-window numbers used to argue `sell_threshold+weights` has the best crash/real balance (-7.5% crash) all used old data. This was part of *why* `sell_threshold+weights` (not `sell_threshold` alone) got adopted — worth re-confirming that reasoning once rebuilt, since real-window numbers alone (+22.5% vs +25.2%) don't unambiguously favor the pairing over `sell_threshold` alone.

**Not affected (real-data-only, no re-check needed):**
- Correlation-admission-gate sweep (negative result, `2024-11-21` real window only).
- Exit-parameter audit #6 `_RSI_OVERBOUGHT` sweep — real-window only, no crash data used.
- VIX gate's full-history validation (`2023-01-01`+ real IBKR data) that confirmed threshold=20.0.

**Net effect:** the two parameters currently live from this session (`sell_threshold=-6.0`, `trend=1.0` weight) are probably still fine — their strongest evidence (real-window returns) doesn't depend on synthetic data at all — but the *specific pairing choice* (weights, not RSI, alongside sell_threshold) was partly a crash-window call and deserves a re-run once the rebuild + re-warm finishes, using the same scripts already built this session (`scripts/run_2way_combo_check.ps1`, `scripts/run_combined_winners_check.ps1`, `scripts/run_weight_grid_crash_followup.ps1` — all still in the repo, just point them at the new data).

---

### Investigation: VIX-conditional kelly scaling (2026-09-07/08)

**Hypothesis:** Multiply `kelly_fraction` by f(vix_level) at admission time to scale position sizes down in high-volatility regimes. Graduated risk instead of binary gate.

**Objections noted before running (important context):**
1. `kelly_fraction` on each candidate is computed unconditionally from all historical trades — multiplying by f(vix) applies a regime-conditional scale to a regime-unconditional estimate. Coherent only if win_rate AND payoff degrade monotonically with VIX; otherwise f(vix) shrinks a misspecified number.
2. Architecture constraint: `kelly_fraction` is immutable on the candidate (CLI contract). Entry.evaluate() can't reach it. Scaling requires touching `KellySizer`, `arbitrate()`, or the candidate — all engine-level, not strategy-owned.
3. f(vix) has strictly more free parameters than the binary gate. Same-day cap was killed on ablation; this surface is richer.

**Test performed:** Quintile segmentation of `data/journals/backtest.csv` (16,361 optimised_new trades) by VIX close at `date_opened`. Script: `scripts/vix_kelly_segmentation.py`.

**Results (equal-count quintiles):**
```
Q1  VIX 11.9–14.2   n=3256  wr=0.538  payoff=1.314  kelly=0.186
Q2  VIX 14.2–15.7   n=3233  wr=0.543  payoff=1.239  kelly=0.174  ← highest win_rate
Q3  VIX 15.8–17.1   n=3304  wr=0.507  payoff=1.188  kelly=0.092
Q4  VIX 17.1–19.1   n=3277  wr=0.484  payoff=1.277  kelly=0.080  ← trough
Q5  VIX 19.1–52.3   n=3291  wr=0.515  payoff=1.231  kelly=0.121  ← rebounds
```

Q5 split at live gate (VIX=25):
```
VIX 19.1–25.0  n=2596  wr=0.518  kelly=0.123
VIX 25.0–52.3  n=695   wr=0.502  kelly=0.113
```

**Key findings:**
1. Win_rate NOT monotone-degrading (`False` by code check). Q2 > Q1; Q5 rebounds above Q4.
2. Trough is at VIX 17–19 (moderate anxiety), not at extremes. Any linear f(vix) would shrink wrong direction — over-scales in safe Q1-Q2, misses the actual worst zone at Q3-Q4.
3. Splitting Q5 at 25: both sub-bins (19–25, 25+) behave identically (kelly ~0.113–0.123, similar to Q3). VIX gate at 25 is not protecting against a structurally worse kelly regime.
4. Average payoff barely moves across quintiles (avg_win 0.055–0.057%, avg_loss 0.042–0.046%). Win_rate drives all kelly variation — not payoff scaling.

**Verdict: Conditional kelly scaling NOT warranted.** Non-monotonic pattern means no simple f(vix) fits. The existing binary gate at 25 is a cleaner intervention. Do not pursue f(vix) kelly scaling without a materially different mechanism.

**Repeat after synthetic data updated:** Re-run `uv run python -m scripts.vix_kelly_segmentation --strategy optimised_new` on the updated backtest journal to check if synthetic-data correction changes the quintile distribution. Verdict may change if the sigma fix substantially alters trade outcomes (particularly avg_loss) across VIX regimes — though the non-monotonicity finding is structural and likely to survive.
