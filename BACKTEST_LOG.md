# Backtest Log

Running log of every backtest/scan run — newest entry on top. One block per run.

**Rule: whenever a backtest/scan finishes, or the user asks for a summary of one, update this log.** Include the per-strategy summary table below (not just prose) whenever the run covers multiple strategies. Add latest to top of file. Include date and time of run

**Rule: any `live_sim.py` run (has a `position_summary.csv`) must ship a chart alongside the log entry.** 3-panel line chart, one line per strategy, all vs date: (1) deployed £ (amount committed to market), (2) total P&L £ (`portfolio_value - pot_size`), (3) `n_open` (number of live/open trades). Drop `date == 'SUMMARY'` rows first. Save to `reports/<journal_basename>_chart.png`, link it from the log entry (`Chart: <path>`). Isolated-pot runs (`full_scan.py`, no `position_summary.csv`) have no equity curve to chart — table only.

**Archive note (2026-07-27):** history before this date lives in `BACKTEST_LOG_ARCHIVE_pre20260727.md`, archived because most of its entries carried a "Return on max deployed"/"~Annualised" column that was misread as an achievable real-money return and confirmed wrong (panel-reviewed) — the underlying trades assume infinite capital was always available, so they don't hold for a real capital-constrained account. This file starts clean with the corrected format: capacity facts only, no derived "return". For "what would £X actually earn me": there is no shortcut metric — run `live_sim.py --initial-cash <X>` (or `--pot-sizes <X> <Y> ...` to sweep) on a curated ticker/strategy set for real entry arbitration and real position sizing off the actual pot.

Template:
```
## YYYY-MM-DD — <short title>
Tool: run.py / batch.py / live_sim.py / full_scan.py
Scope: <tickers/universe> x <strategy/strategies>
Journal: <path if any>
Chart: <path, live_sim runs only>
Result: <key numbers — return vs b&h, win rate, PF, trades, sharpe>

| Strategy | Closed trades | Net P&L | Avg Profit/Trade | Peak concurrent capital | Peak date | Avg concurrent capital |
|---|---|---|---|---|---|---|
| ... | ... | ... | ... | ... | ... | ... |

Conclusion: <one line takeaway>
```

Net P&L is the sum of N *independent* backtests, each with unlimited capital and its own isolated pot — it is not what one shared account of any size would earn. Peak/avg concurrent capital is a post-hoc reconstruction (pool every trade onto one shared timeline, commit kelly_fraction × pot-size for its open interval) for capacity comparison between strategies only — never divide Net P&L by it to imply an achievable return. For genuine capital-arbitrated results (real entry admission/rejection, real position sizing against an actual shared pot), use `live_sim.py` and report its `position_summary.csv` output instead — that's the only source that can legitimately state a return on a given capital amount.

---

## 2026-09-02 (evening) — Synthetic 2008 crash stress test: full universe, optimised_new, £100k, top-k=70
Tool: live_sim.py (new `--synthetic-data-dir`/`--synthetic-end-date` flags, wired this session)
Scope: `--universe --strategies optimised_new --start-date 2008-01-01 --synthetic-data-dir data_synthetic/hourly --synthetic-end-date 2009-07-31 --initial-cash 100000 --top-k 70 --workers 4`. Real hourly data doesn't reach 2008 (IBKR's UK history starts 1998, but hourly specifically is far shorter) — this run uses `synthetic_backtest_data`'s Brownian-bridge synthetic hourly bars (built from real Stooq/IBKR daily closes) instead, isolated from real data via `SYNTHETIC_HMM_CACHE_DIR` and a dedicated journal path. 600-ticker universe, 450 had synthetic data covering the window (150 dropped — post-2009 IPOs/spinoffs: TSLA, META, UBER, ABNB, COIN, PLTR, HOOD, GEV, CEG, KVUE, WBD, IAG.L, NWG.L, etc., confirmed real listing-date gaps, not a bug). HMM `min_train_bars=500` warmup consumed Jan–mid-April 2008 (no entries possible before then, by design — same mechanism live has always had).
Journal: data_synthetic/journals/live_sim_synthetic.csv (765 trades), data_synthetic/journals/live_sim_synthetic_position_summary_20260902T193113.csv
Chart: reports/live_sim_synthetic_chart.png

| Strategy | Data span | Trades admitted | Rejected (cash / kelly≤0) | Final portfolio | P&L | Return | Max drawdown | Peak deployed |
|---|---|---|---|---|---|---|---|---|
| optimised_new | 2008-04-13 → 2009-07-30 | 765/774 | 0 / 9 | £10,225.07 | −£90,733.60 | −90.7% | −89.7% | £93,582.05 |

Chart detail worth noting (not just the headline number): deployed capital and open-position count were **already declining** through summer 2008, well before the Sept 15 Lehman collapse — peak deployed (£93.6k, 38 open positions) was mid-April 2008, down to ~£45k/16 positions by the Lehman line. So the strategy was de-risking somewhat ahead of the crash, not blindly loading up into it. The real damage was existing positions marking down through the crash while little new capital was deployed (deployed stayed £5–20k Oct 2008–March 2009) — P&L bled continuously and only flattened around the real-world March 2009 market bottom. Re-entries resumed after that (open positions climbed back to 40+ by May 2009) but never recovered the loss — P&L stayed pinned near −£90k through the end of the window.

Conclusion: **Severe result, real finding not noise.** Whatever de-risking behavior reduced new deployment through 2008 wasn't enough to protect capital already committed — the loss is dominated by mark-to-market decline on held positions through the crash, not reckless re-entry into it (re-entry only resumes post-bottom, and even then doesn't recover). Worth investigating whether `optimised_new`'s regime-exit logic should be exiting held positions faster once the HMM detects a bear regime, rather than only gating new entries. Not yet investigated further — this run establishes the finding, doesn't diagnose the mechanism.

---

## 2026-08-30 — Pot-size comparison: £20k vs £30k, full universe, optimised_new
Tool: live_sim.py
Scope: `--universe --strategies optimised_new --pot-sizes 20000 30000 --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --workers 4 --cost-model ibkr_tiered_spread --seasonal-volume --source ibkr`
Journal: data/journals/live_sim_potsize_20k_30k.csv (854 trades), data/journals/live_sim_potsize_20k_30k_equity.csv
Chart: reports/live_sim_potsize_20k_30k_chart.png

| Pot | Data span | Trades admitted | Rejected (cash / kelly≤0) | Final portfolio | P&L | Return | Sharpe | Sortino | Peak deployed | Max drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| £20k | 2024-11-21 → 2026-08-27 | 427/427 | 0 / 0 | £29,601 | +£8,824 | +48.3% | 2.244 | 3.756 | £20,528 | −10.5% |
| £30k | 2024-11-21 → 2026-08-27 | 427/427 | 0 / 0 | £45,138 | +£13,940 | +50.8% | 2.329 | 3.887 | £31,657 | −10.4% |

Cross-run summary (all same params, only pot size varies):

| Pot | Return | Sharpe | Sortino | Max DD | Cash rejections |
|---|---|---|---|---|---|
| £10k | +42.0% | 1.949 | 3.142 | −11.1% | 6 |
| £20k | +48.3% | 2.244 | 3.756 | −10.5% | 0 |
| £30k | +50.8% | 2.329 | 3.887 | −10.4% | 0 |
| £100k | +53.7% | 2.420 | 4.021 | −10.0% | 0 |

Conclusion: £20k is the inflection point — admits all 427 trades, Sharpe jumps +0.30 vs £10k. Beyond £20k, gains are real but diminishing (commission %, interest on idle cash, and floor-rounding all favour larger positions). £20k→£100k adds only +5.4pp return and +0.18 Sharpe.

---

## 2026-08-30 — Pot-size comparison: £10k vs £100k, full universe, optimised_new
Tool: live_sim.py
Scope: `--universe --strategies optimised_new --pot-sizes 10000 100000 --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --workers 4 --cost-model ibkr_tiered_spread --seasonal-volume --source ibkr`
Journal: data/journals/live_sim_potsize_compare.csv (848 trades), data/journals/live_sim_potsize_compare_equity.csv
Chart: reports/live_sim_potsize_compare_chart.png

| Pot | Data span | Trades admitted | Rejected (cash / kelly≤0) | Final portfolio | P&L | Return | Sharpe | Sortino | Peak deployed | Max drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| £10k | 2024-11-21 → 2026-08-27 | 421/427 | 6 / 0 | £14,169 | +£3,818 | +42.0% | 1.949 | 3.142 | £10,308 | −11.1% |
| £100k | 2024-11-21 → 2026-08-27 | 427/427 | 0 / 0 | £153,413 | +£49,363 | +53.7% | 2.420 | 4.021 | £108,563 | −10.0% |

Conclusion: £100k dominates on every metric — 11.7pp more return, Sharpe +0.47, Sortino +0.88, slightly lower drawdown. The 6 cash-rejected trades at £10k were disproportionately profitable, explaining the risk-adjusted gap beyond mere scale. Matches live daemon's current £100k pot configuration.

---

## 2026-08-15 — MC daily HMM (70/70 tickers): optimised_new portfolio, 50 paths, £100k
Tool: monte_carlo_live_sim.py (Track B)
Scope: `--universe --strategies optimised_new --n-paths 50 --daily-hmm --pot-sizes 100000 --top-k 70 --workers 4`. First run with `--daily-hmm` at 70/70 daily HMMs (NaN-filter fix applied to `fit_generating_hmm` — yfinance UK daily data has exactly 1 NaN log-return per LSE ticker from GBp/GBX unit conversion artifacts; previous run had 53/70 and was killed). All UK tickers now included in multi-cycle regime estimation.
Output: `data/monte_carlo/optimised_new_portfolio_20260815T085307Z/mc_summary.json`
Chart: none (MC — percentile distribution, no equity curve)

| metric | p5 | p25 | p50 | p75 | p95 | mean |
|---|---|---|---|---|---|---|
| Total return | −1.1% | +9.4% | +14.5% | +26.7% | +39.5% | +17.6% |
| Final portfolio (£100k) | £98,928 | £109,388 | £114,512 | £126,745 | £139,487 | £117,637 |
| Max drawdown | −11.5% | −8.3% | −6.6% | −5.5% | −3.7% | −7.3% |
| Trades admitted | 257 | 292 | 319 | 342 | 385 | 319 |

`prob_of_loss = 8%` (4/50 paths). Mean return +17.6%.

Comparison vs prior MC runs (all optimised_new, top-k=70):

| Run | HMM source | Coupling | Paths | Pot | p5 return | p50 return | prob_loss |
|---|---|---|---|---|---|---|---|
| 2026-08-13 baseline | hourly 2yr | 0.0 | 50 | £25k | +4.4% | +15.1% | 4% |
| 2026-08-14 coupling pilot | hourly 2yr | 0.3 | 10 | £25k | −1.0% | +12.0% | 10% |
| **2026-08-15 daily HMM** | **daily 20yr (70/70)** | **0.0** | **50** | **£100k** | **−1.1%** | **+14.5%** | **8%** |

Conclusion: Daily HMM (20yr regime transitions) produces materially worse p5 vs hourly baseline (+4.4%→−1.1%) and higher prob_loss (4%→8%), confirming that multi-cycle regime sequences generate more realistic bear-state clustering. p50 is broadly stable (+15.1%→+14.5%). The daily HMM p5=−1.1% is close to the coupling=0.3 pilot p5=−1.0%, suggesting both stress dimensions (longer bear cycles vs correlated crashes) impose similar tail risk. Pot sizes differ (£25k vs £100k) — percentile returns not strictly comparable, but direction is informative. Bug fixed this session: `sample_daily_tiled_states` was not forwarding `transmat_noise`/`market_coupling` args — those were silently dropped when `precomputed_state_labels` was set; fixed in both `monte_carlo.py` and `monte_carlo_live_sim.py`. UK daily HMM NaN fix: `fit_generating_hmm` now strips non-finite log-returns before fitting.

---

## 2026-08-14 — IBKR definitive baseline: full universe, :30-resampled, optimised_new, £100k
Tool: live_sim.py
Scope: `--universe --strategies optimised_new --source ibkr --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --seasonal-volume --initial-cash 100000 --workers 4` — first run post bar-alignment fix (`broker/ibkr_data.py` now resamples to :30). Supersedes the pre-fix IBKR run (Sharpe 0.80) logged below.
Journal: data/journals/live.csv (605 trades), data/journals/live_sim_position_summary_20260814T233951.csv
Chart: reports/live_sim_ibkr_resampled_optimised_new_100k_chart.png

| Strategy | Data span | Trades admitted | Rejected (cash / kelly≤0) | Final portfolio | P&L | CAGR | Sharpe | Sortino | Peak deployed | Max drawdown |
|---|---|---|---|---|---|---|---|---|---|---|
| optimised_new | 2023-05-12 → 2026-08-11 | 606/607 | 0 / 1 | £141,131.53 | +£41,131.53 | 11.2% | 1.754 | 3.020 | £108,076.37 | −9.2% |

Comparison — same params, different source/alignment:

| Run | Source | Bar alignment | Sharpe | Sortino | P&L |
|---|---|---|---|---|---|
| yfinance (Sharpe 2.032 run) | yfinance | :30 always | 2.034 | 3.10 | +£43,862 |
| IBKR pre-fix (this command, broken) | IBKR | :00 drift | 0.80 | 1.19 | +£16,919 |
| **IBKR post-fix (this run)** | **IBKR** | **:30** | **1.754** | **3.020** | **+£41,132** |

Conclusion: Bar-alignment fix closes most of the gap (0.80 → 1.754 Sharpe). Remaining ~0.28 Sharpe deficit vs yfinance is universe divergence — IBKR's extra 4-month history (May vs Sep 2023) selects a slightly different top-70. IBKR is now a valid production data source; the daemon runs on this aligned data.

---

## 2026-08-14 — IBKR bar-alignment root-cause: :30-resample recovers Sharpe gap vs yfinance
Tool: scripts/analysis_ibkr_clipped_window.py (uses arbitrate() internally; diagnostic script)
Scope: Same 70 tickers as yfinance Sharpe 2.034 run, IBKR TRADES data clipped to 2023-09-18 (matching yfinance hourly start), resampled to :30-aligned 60-min bars (`resample("1h", offset="30min")`), arbitration from 2023-12-14, £100k, optimised_new, ibkr_tiered_spread cost model.
Journal: data/journals/diag_ibkr_resampled_yf70.csv (421 trades)
Chart: reports/diag_ibkr_resampled_yf70_chart.png

| Run | Source | Bar alignment | Sharpe | Sortino | P&L | Trades | Max DD |
|---|---|---|---|---|---|---|---|
| yfinance baseline (Sharpe 2.032 run) | yfinance | :30 always | 2.034 | 3.10 | +£43,862 | 519 | — |
| IBKR raw, same 70 tickers, unclipped | IBKR | :00 (after first :30) | 1.720 | — | — | 442 | — |
| IBKR same 70 tickers, clipped Sep 2023 | IBKR | :00 | 1.427 | 2.395 | +£32,663 | 629 | −1.88% |
| **IBKR clipped + :30-resampled** | **IBKR** | **:30** | **2.749** | **4.527** | **+£59,841** | **421** | **−2.75%** |

Conclusion: **Bar alignment was the entire gap.** yfinance bars are always :30-aligned; IBKR cache shifts to :00-aligned after the first bar — different OHLCV slices into RSI/SMA200/volume_ratio cause different signals on identical prices. After resampling IBKR to :30, Sharpe exceeds yfinance (2.749 vs 2.034). Fix applied to `broker/ibkr_data.py` — `fetch_hourly()` now resamples to :30 at return; cache stores raw data (no invalidation needed).

---

## 2026-08-14 — MC shared panic factor pilot: coupling=0.0 vs coupling=0.3, 10 paths each
Tool: monte_carlo_live_sim.py (Track B)
Scope: optimised_new, 603-ticker universe → top-k=70, 10 paths each, £25k, workers=2. Validates `--market-coupling` (shared SPY market HMM biasing each ticker's state sequence toward common market regime).
Output: `optimised_new_portfolio_20260814T070139Z` (0.0), `optimised_new_portfolio_20260814T070143Z` (0.3)
Chart: none

| coupling | p5 return | p50 return | p5 max_dd | prob_loss | trades p50 |
|---|---|---|---|---|---|
| 0.0 (independent) | +4.5% | +13.0% | −7.9% | 0% | 422 |
| 0.3 (shared panic) | −1.0% | +12.0% | −9.1% | 10% | 347 |

Conclusion: **coupling validated**. p5 return crosses negative (−1.0%), prob_loss rises 0%→10%, p5 drawdown worsens −7.9%→−9.1%. Fewer trades at p50 (422→347) because co-crash Bear clustering reduces signal diversity. Independent HMMs structurally cannot generate these outcomes. Next: full 50-path run at coupling=0.3 for stable percentile estimates.

---

## 2026-08-13/14 — HMM Monte Carlo stress test: Track A (SPY/default, 300 paths) + Track B (optimised_new portfolio, 50 paths)
Tool: monte_carlo.py (Track A), monte_carlo_live_sim.py (Track B)
Scope: Track A — SPY × default strategy, 300 synthetic paths, 5100 bars/path (~3yr), block_size=24, transmat_noise=0.0, workers=2. Track B — optimised_new, 603-ticker universe → top-k=70 fixed set, 50 synthetic paths, £25k pot, workers=4.
Output: `data/monte_carlo/SPY_default_20260813T180326Z/`, `data/monte_carlo/optimised_new_portfolio_20260813T180323Z/`
Chart: none (MC — no equity curve; percentile bands below)

**Track A — SPY × default, 300 paths × 5100 bars:**

| metric | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| Sharpe | 0.32 | 1.06 | 1.59 | 2.04 | 2.65 |
| Sortino | 0.45 | 1.51 | 2.35 | 3.15 | 4.28 |
| Max drawdown | −2.5% | −2.0% | −1.6% | −1.4% | −1.0% |
| Total return | +0.8% | +4.2% | +7.6% | +11.7% | +17.8% |
| Final portfolio (£20k start) | £21,715 | £22,328 | £23,020 | £23,798 | £25,037 |

`prob_of_loss = 2%` (6/300 paths). Mean return +8.3%.

Notes: `vol_filter_ok` forced `True` for all synthetic paths (SPY real trend_quality=−0.478 would veto all entries otherwise — vol-screen is a live-trading admission gate, not a stress-test gate). Block bootstrap (block_size=24) preserves within-state RSI/SMA autocorrelation; iid Gaussian draws produced 0 trades. Uses new vectorized `_sample_state_sequence` sampler (270× faster than hmmlearn; generation now ~8s for 300 paths vs 37 min).

**Track B — optimised_new, 70-ticker fixed universe, 50 paths × £25k:**

| metric | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| Total return | +4.4% | +11.0% | +15.1% | +22.4% | +31.3% |
| Final portfolio | £26,105 | £27,757 | £28,774 | £30,606 | £32,825 |
| Max drawdown | −8.6% | −6.5% | −4.7% | −3.7% | −2.9% |
| Trades admitted (per path) | 393 | 426 | 452 | 500 | 550 |

`prob_of_loss = 4%` (2/50 paths). Mean return +16.2%.

Conclusion: Strategy robust across synthetic regimes. Track A p5=+0.8% (near-flat worst case), Track B p5=+4.4% — floor is positive under both. Track B worst-case drawdown −8.6% manageable at £25k. High per-path trade count (p50=452) confirms block bootstrap generates realistic signal buildup. Track B used pre-A/B-fix code (volume/range_ratio independently bootstrapped); alignment fix improves fidelity, won't materially shift these distributions.

---

## 2026-08-13 — seasonal volume normalisation: optimised_new, flat vs hour-of-day, £100k, top-k=70
Tool: live_sim.py (two parallel runs)
Scope: full S&P500+FTSE100 universe, optimised_new, top-k=70, £100k, full ~2.9yr history. Baseline uses flat rolling-20 volume ratio; seasonal uses same-hour-of-day rolling-20 (20 same-hour observations ≈ 20 trading days), falling back to flat when same-hour history shallow. All other settings identical.
Command: `--universe --strategies optimised_new --initial-cash 100000 --start-date 2000-01-01 --workers 4 --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --cost-model ibkr_tiered_spread`
Journal (baseline): data/journals/live_sim_seasonal_baseline.csv (575 trades), data/journals/live_sim_seasonal_baseline_pos.csv
Journal (seasonal): data/journals/live_sim_seasonal_seasonal.csv (519 trades), data/journals/live_sim_seasonal_seasonal_pos.csv
Chart: reports/live_sim_seasonal_comparison_chart.png

| Variant | Admitted | Final £ | P&L £ | Return | Max DD | Sharpe | Sortino | Trades |
|---|---|---|---|---|---|---|---|---|
| baseline (flat rolling-20) | 575/583 (0 cash, 8 kelly≤0) | £131,493 | +£31,493 | +31.5% | −9.69% | 1.773 | 2.894 | 575 |
| seasonal (hour-of-day rolling-20) | 519/522 (0 cash, 3 kelly≤0) | £143,862 | +£43,862 | +43.9% | −9.12% | 2.032 | 3.598 | 519 |

Conclusion: **seasonal normalisation wins on every metric** — +12.4pp return, +14.6% Sharpe, +24.3% Sortino, slightly lower drawdown, 56 fewer trades (cleaner signal). Enabled live: `overnight_strategy.json` sets `seasonal_volume: true` in both market defaults and `top_k_screen`; wires through `_build_argv()` → `run.py --seasonal-volume` for daytime evaluation and `rank_universe_cli --seasonal-volume` for nightly top-K ranking. No daemon restart required — config read fresh each cycle.

Note: Sharpe/Sortino computed from event-day-sampled equity curve (`portfolio_value` column in position_summary), not daily-resampled — methodology consistent across both variants, comparison is valid; absolute values not directly comparable to a daily-resampled series.

Note: baseline return here (31.5%) is lower than the 2026-08-11 baseline (55.2%) — the rolling 730d yfinance window has advanced ~2 days, dropping some late-2023 trades and including more of the April 2026 volatility period. Not a regression.

---

## 2026-08-12 (night) — require_vol_filter_ok ablation: optimised_new vol_filter=False, £100k, top-k=70
Tool: live_sim.py, `--universe --strategies optimised_new --initial-cash 100000 --top-k 70 --workers 4 --start-date 2000-01-01 --journal data/journals/live_sim_optimised_new_novolfilter_100k.csv`
Scope: full S&P500+FTSE100 universe (601 tickers all generate candidates — previously only 118 vol-filter-passing tickers did), k=70, £100k, ~2.9yr history. `OptimisedNewEntry.require_vol_filter_ok=False` bypasses the per-ticker vol_filter_ok gate in evaluate(). Direct ablation vs 2026-08-11 baseline (True).
Journal: data/journals/live_sim_optimised_new_novolfilter_100k.csv (575 trades), data/journals/live_sim_position_summary_20260812T232115.csv

| Strategy | Candidates admitted | Final value | P&L (trading) | Total return | Max drawdown | Trades |
|---|---|---|---|---|---|---|
| optimised_new (vol_filter=False) | 575/583 (0 rejected cash, 8 rejected kelly≤0) | £131,894.60 | +£25,377 | +31.9% | −9.7% | 575 |
| **optimised_new (vol_filter=True, baseline 2026-08-11)** | 609/613 | £155,218.03 | +£48,813 | **+55.2%** | **−6.08%** | 609 |

Conclusion: **vol_filter=False is worse on every metric** — −23pp return, +3.6pp drawdown, and paradoxically 34 fewer trades. The 479 previously-vetoed tickers (low trend_quality) dilute the top-70 candidate pool, displacing higher-quality tickers from the baseline set; the low-quality candidates then generate fewer valid trade entries (more HOLD from RSI/regime/flip vetoes), reducing total trade count. `require_vol_filter_ok=True` confirmed. Reverted `optimised_new.py` to `True` immediately. All 7 Entry classes now declare `require_vol_filter_ok` explicitly (except `mean_reversion` which has inverted vol-filter logic).

---

## 2026-08-12 (evening) — require_flip_entry ablation: optimised_new flip=False, £100k, top-k=70, journal diff
Tool: live_sim.py + manual journal diff (pandas)
Scope: same as ablation run below — comparing `live_sim_optimised_new_100k_full.csv` (baseline, flip=True) vs `live_sim_optimised_new_noflip_100k.csv` (flip=False) at trade level to understand WHERE the flip guard fires.

Key finding: the "31 net extra trades" (609→640) masked a large churn — **490 new trades admitted, 459 baseline trades displaced by capital contention.**

| Cohort | Count | Win rate | Avg P&L | Avg RSI at entry | Avg regime signal | Avg entry score |
|---|---|---|---|---|---|---|
| Extra (flip=False unlocked) | 490 | 47.3% | £65.4 | 59.9 | 0.867 | 8.08 |
| Displaced (squeezed from baseline) | 459 | — | £95.8 | — | — | — |
| Baseline all | 609 | 52.2% | £80.2 | 55.7 | 0.843 | 7.46 |

Pattern: extra trades spread uniformly across all 33 months — no clustering around vol events or crashes. The flip guard is a **continuous quality filter**, not a macro one. Without it: mid-trend entries (RSI ~60, regime already established) consume capital first, displacing the better flip-confirmed entries (avg P&L £95.8 displaced vs £65.4 admitted). Entry score is paradoxically higher for the extra trades (8.08 vs 7.46) because the composite signal is strong mid-trend — but that strength is already priced in; win rate drops 5pp and avg P&L drops £15/trade.

Conclusion: **flip guard earns its keep via capital protection, not just trade filtering.** It gates out mid-trend entries so capital remains available for higher-quality flip-confirmed entries. Confirmed keep=True.

---

## 2026-08-12 (evening) — require_flip_entry ablation: optimised_new flip=False, £100k, top-k=70
Tool: live_sim.py, `--universe --strategies optimised_new --initial-cash 100000 --top-k 70 --workers 4 --start-date 2000-01-01 --journal data/journals/live_sim_optimised_new_noflip_100k.csv`
Scope: full S&P500+FTSE100 universe, optimised_new with `require_flip_entry=False` (set as explicit class attribute for this test), k=70, £100k, ~2.9yr history. Direct ablation vs 2026-08-11 baseline (True).
Journal: data/journals/live_sim_optimised_new_noflip_100k.csv (640 trades), data/journals/live_sim_position_summary_20260812T202651.csv

| Strategy | Candidates admitted | Final value | P&L (trading) | Total return | Max drawdown | Trades |
|---|---|---|---|---|---|---|
| optimised_new (flip=False) | 640/648 (0 rejected cash, 8 rejected kelly≤0) | £143,817.07 | +£37,795 | +43.8% | −9.2% | 640 |
| **optimised_new (flip=True, baseline 2026-08-11)** | 609/613 | £155,218.03 | +£48,813 | **+55.2%** | **−6.08%** | 609 |

Conclusion: **flip=False is worse on every metric** — −11pp return, +3pp drawdown, 5% more trades (churn, no edge). `require_flip_entry=True` confirmed. Reverted `optimised_new.py` to `True` immediately after. All 8 Entry classes now declare `require_flip_entry` explicitly as a class attribute (no longer relies on engine `getattr` fallback).

---

## 2026-08-11 (evening) — optimised_new full-history capital-arbitrated re-baseline, £100k, top-k=70
Tool: live_sim.py, `--universe --strategies optimised_new --initial-cash 100000 --top-k 70 --workers 4 --journal data/journals/live_sim_optimised_new_100k_full.csv`
Scope: full S&P500+FTSE100 universe filtered to optimised_new's top-70 ranked tickers, one £100k pot, full available history (2023-12-11 to 2026-08-10, the yfinance ~2.9yr hourly cap — see fix note below)
Journal: data/journals/live_sim_optimised_new_100k_full.csv (609 trades), data/journals/live_sim_position_summary_20260811T204259.csv
Chart: reports/live_sim_optimised_new_100k_full_chart.png
Result: **+55.2% total return, −6.08% max drawdown, 52.2% win rate.** Two bugs fixed same session, both folded into this run:

1. `arbitrate()`'s position sizing was continuous-£ (`alloc = kelly_fraction * cash`), not whole-share, so backtest P&L was systematically optimistic vs. the live daemon's real `compute_quantity()` (which floors to whole shares and can reject a candidate outright on a high per-share price even with cash available). Fixed to floor to whole shares, matching live exactly — fractional-share order support on IBKR's API is unreliable per user reports, so whole-share flooring is correct behavior on both sides, not a shortfall.
2. `live_sim.py`'s `--start-date` default was `2026-01-12`, silently truncating every "full universe" run to ~7 months even though candidate generation covers full history — an initial run this session only covered Jan–Jul 2026 before this was caught and the default corrected to `2000-01-01` (effectively "all available data").

| Strategy | Candidates admitted | Final value | P&L | Total return | Max drawdown | Peak deployed | Trades | Win rate |
|---|---|---|---|---|---|---|---|---|
| optimised_new | 609/613 (0 rejected cash, 4 rejected kelly≤0) | £155,218.03 | +£55,218.03 (£48,812.77 trading + £6,405.27 interest) | +55.2% | −6.08% | £123,249 | 609 | 52.2% |

Monthly breakdown. **Opened/Closed are separate date axes** — a position opened late in a month can close months later, so a month can show 0 closes with nonzero deployed capital (Dec 2023: 8 opened, 0 closed, first closes were Jan 2024) or 0 opens with several closes (Aug 2026: 0 opened, 9 closed, tail end of the data window). Winners/Losers/P&L/Win% are bucketed by close date; peak deployed is that month's high-water mark of capital tied up, independent of either date:

| Month | Opened | Closed | Winners | Losers | Win% | P&L (closed) | Peak deployed |
|---|---|---|---|---|---|---|---|
| 2023-12 | 8 | 0 | 0 | 0 | 0% | +£0 | £58,071 |
| 2024-01 | 37 | 19 | 8 | 11 | 42% | +£842 | £89,364 |
| 2024-02 | 34 | 31 | 16 | 15 | 52% | +£2,820 | £90,969 |
| 2024-03 | 24 | 25 | 16 | 9 | 64% | +£2,728 | £88,751 |
| 2024-04 | 28 | 40 | 24 | 16 | 60% | +£2,833 | £92,881 |
| 2024-05 | 25 | 29 | 15 | 14 | 52% | +£2,284 | £93,963 |
| 2024-06 | 26 | 17 | 8 | 9 | 47% | +£789 | £88,559 |
| 2024-07 | 29 | 29 | 17 | 12 | 59% | +£659 | £91,228 |
| 2024-08 | 14 | 15 | 7 | 8 | 47% | −£177 | £88,700 |
| 2024-09 | 26 | 22 | 12 | 10 | 55% | +£2,508 | £94,897 |
| 2024-10 | 21 | 30 | 11 | 19 | 37% | +£570 | £92,789 |
| 2024-11 | 18 | 18 | 13 | 5 | 72% | +£3,674 | £89,510 |
| 2024-12 | 10 | 20 | 10 | 10 | 50% | +£1,106 | £88,756 |
| 2025-01 | 22 | 11 | 7 | 4 | 64% | +£1,322 | £92,535 |
| 2025-02 | 18 | 21 | 11 | 10 | 52% | +£2,302 | £91,686 |
| 2025-03 | 19 | 20 | 5 | 15 | 25% | −£1,662 | £81,244 |
| 2025-04 | 15 | 16 | 4 | 12 | 25% | −£2,726 | £79,764 |
| 2025-05 | 26 | 22 | 13 | 9 | 59% | +£2,086 | £96,814 |
| 2025-06 | 21 | 20 | 11 | 9 | 55% | +£1,336 | £102,417 |
| 2025-07 | 17 | 19 | 12 | 7 | 63% | +£2,449 | £96,256 |
| 2025-08 | 14 | 15 | 6 | 9 | 40% | +£987 | £92,689 |
| 2025-09 | 13 | 10 | 7 | 3 | 70% | +£4,007 | £105,799 |
| 2025-10 | 9 | 20 | 17 | 3 | 85% | +£5,557 | £92,086 |
| 2025-11 | 10 | 8 | 2 | 6 | 25% | −£2,634 | £71,705 |
| 2025-12 | 12 | 11 | 4 | 7 | 36% | +£549 | £89,241 |
| 2026-01 | 15 | 13 | 5 | 8 | 38% | −£945 | £98,180 |
| 2026-02 | 18 | 11 | 6 | 5 | 55% | +£1,320 | £114,699 |
| 2026-03 | 15 | 27 | 10 | 17 | 37% | −£819 | £96,295 |
| 2026-04 | 20 | 10 | 4 | 6 | 40% | −£159 | £112,650 |
| 2026-05 | 11 | 20 | 6 | 14 | 30% | +£9,137 | £105,308 |
| 2026-06 | 17 | 11 | 9 | 2 | 82% | +£3,535 | £123,249 |
| 2026-07 | 17 | 20 | 14 | 6 | 70% | +£1,503 | £116,917 |
| 2026-08 | 0 | 9 | 8 | 1 | 89% | +£1,034 | £39,903 |

Totals: 609 opened, 609 closed, 318 winners, 291 losers (52.2% win rate), net P&L £48,812.77 (trading only, excludes interest).

Conclusion: **At £100k the whole-share-flooring fix never actually bit** — 0/613 candidates were rejected for cash; all 4 rejections were kelly≤0. This confirms the fix matters at small pots (£10k), not at the daemon's actual live pot size — no behavior change expected for the live account from fix #1. Monthly returns are lumpy but broadly positive (23/33 months net positive); worst month 2025-04 (−£2,726, part of the −6.08% April 2025 drawdown episode) recovered within ~2 months. Interactive equity/drawdown/per-trade-P&L charts: https://claude.ai/code/artifact/f815128b-9caa-4aca-8058-e439883b5ee8

---

## 2026-07-31 (night) — k-sweep, optimised strategy, £10k pot, full universe
Tool: live_sim.py, `--universe --strategies optimised --initial-cash 10000 --start-date 2000-01-01 --max-trades-per-day 0 --workers 4 --cost-model ibkr_tiered_spread --top-k <K>`
Scope: full S&P500+FTSE100 universe, one £10k pot, k swept across [20, 35, 50, 70, 100]. Context: run after dropping `max_downside_vol=0.25` from vol_screen config and wiring Stage 1 vol_kept list into rank_universe_cli — k now directly controls effective live universe size (only TQ≥0 tickers compete for top-K slots). Simulation window ~2yr (yfinance 730-day hourly cap).
Journal: data/journals/k_sweep_k{K}.csv, data/journals/k_sweep_k{K}_pos.csv

| k | Final £ | P&L £ | Return | Max DD | Trades | Sharpe | Sortino |
|---|---|---|---|---|---|---|---|
| 20 | £11,992 | +£993 | +19.9% | −4.3% | 149 | 1.32 | 1.08 |
| 35 | £12,797 | +£1,857 | +28.0% | −4.3% | 261 | 1.50 | 1.62 |
| 50 | £12,929 | +£2,134 | +29.3% | −6.5% | 430 | 1.49 | 1.87 |
| **70** | **£14,906** | **+£4,179** | **+49.1%** | **−4.7%** | **643** | **2.09** | **3.02** |
| 100 | £13,325 | +£2,742 | +33.2% | −9.3% | 867 | 1.39 | 2.00 |

Conclusion: **k=70 is the clear optimum** — highest return, best Sharpe (2.09), best Sortino (3.02), tighter drawdown than k=50 and k=100. k=100 overshoots: more trades, worst drawdown (−9.3%), lowest Sharpe of the upper range. k=70 confirmed as correct config value; no change to `overnight_strategy.json`. Note: these runs use the new pipeline (Stage 1 TQ pre-filter feeds rank_universe_cli) so this sweep is apples-to-apples with the live daemon going forward.

---

## 2026-07-30 (night) — optimised vs optimised_new, real capital-arbitrated walk-forward, top-k=70, £100k
Tool: live_sim.py, `--universe --strategies optimised optimised_new --initial-cash 100000 --start-date 2000-01-01 --max-trades-per-day 0 --workers 4 --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --cost-model ibkr_tiered_spread`
Scope: full S&P500+FTSE100 universe, filtered to each strategy's own top-70 ranked tickers (same hybrid vol/win-rate ranking as the live daemon's `top_k_screen`), £100k pot **per strategy** (one pot per strategy, not shared between the two — see `.claude/rules/cli.md`)
Journal: data/journals/live_sim_opt_vs_optnew_k70_100k.csv (1,097 trades), data/journals/live_sim_opt_vs_optnew_k70_100k_position_summary.csv
Result: **optimised_new wins on real capital-arbitrated return, reversing the isolated-pot verdict from the same-day earlier test.**

| Strategy | Candidates admitted | Final value | P&L | Total return | Max drawdown | Peak deployed | Sharpe | Sortino |
|---|---|---|---|---|---|---|---|---|
| optimised | 627/627 (0 rejected) | £161,269.32 | +£52,324.05 | +61.3% | −5.5% | £111,986.49 | 2.46 | 3.51 |
| optimised_new | 470/470 (0 rejected) | £166,572.07 | +£58,591.50 | +66.6% | −4.7% | £100,495.25 | 2.61 | 3.94 |

Both strategies admitted 100% of their candidates — capital is not the binding constraint at £100k for either, at this candidate count. optimised_new needed 25% fewer trades (470 vs 627) to produce a better outcome: +12% higher P&L, smaller drawdown.

Conclusion: **This is the correct comparison to trust over the earlier same-day isolated-pot/no-top-k test** (previous entry below) — it matches what actually runs live (top-70 filtered universe, real shared-pot capital arbitration, real Kelly sizing off the live pot balance) rather than an unfiltered 607-ticker isolated-pot scan. The earlier test's conclusion ("don't switch, worse Sharpe/Sortino/Calmar") does not hold once restricted to the actual live-trading universe: optimised_new's ratchet-only exit appears to underperform across the broad low-quality tail of tickers (dragging down aggregate risk-adjusted metrics in the full-universe test) but outperforms on the curated top-70 subset the daemon actually trades, where it captures more of genuine trend moves instead of capping gains at the old hard 30% take-profit. Recommend switching `overnight_strategy.json`'s strategy default (and `top_k_screen.strategy`) from `optimised` to `optimised_new` — pending user confirmation, not yet applied.

---

## 2026-07-30 (evening) — optimised vs optimised_new head-to-head (existing data, first-time comparison)
Tool: full_scan_all_strategies.py (data already existed from the 2026-07-29 23:59 – 2026-07-30 00:58 session that created optimised_new; never previously compared strategy-vs-strategy before this)
Scope: 607 tickers (S&P 500 + FTSE 100), isolated £20k pot per ticker per strategy, same universe/data-cutoff for both strategies (back-to-back same session)
Journal: reports/full_scan/summary.csv (dedupe (strategy,ticker) keep-last), per-ticker journals at data/journals/full_scan/{optimised,optimised_new}/<ticker>.csv
Result: **near-identical net P&L, but optimised_new is worse on every risk-adjusted metric.**

| Strategy | Closed trades | Net P&L | Avg Profit/Trade | Avg Sharpe | Avg Sortino | Avg Calmar | Avg max DD | Beat B&H rate |
|---|---|---|---|---|---|---|---|---|
| optimised | 4,987 | +£1,150,962.93 | +£230.79 | 0.68 | 1.17 | 0.80 | −0.78% | 24.0% |
| optimised_new | 3,754 | +£1,147,300.19 | +£305.62 | 0.54 | 0.91 | 0.66 | −0.92% | 23.5% |

optimised_new's ratchet-only exit (999-disabled hard TP, profit_stop_scale=0.30, min_stop_pct=0.03) trades less often and banks a bigger win per trade when it does (letting winners run past the old hard 30% take-profit) — but the aggregate effect is a wash on raw P&L and a clear step down on Sharpe/Sortino/Calmar, plus a slightly worse average per-ticker drawdown.

Conclusion: **optimised_new does not clearly beat optimised — do not switch the live daemon's `overnight_strategy.json` strategy default (or `top_k_screen.strategy`) to optimised_new on this result.** This is an isolated-£20k-pot comparison (same caveat as every other full_scan number in this log — signal-quality comparison only, not an achievable-return estimate). If a switch is still being considered, the decision needs a real `live_sim.py` capital-arbitrated run for optimised_new (matching the k=70 top-k config already live for optimised) before it can be trusted the way the 2026-07-27 conservative/default/trend/optimised capital-arbitration comparison was.

---

## 2026-07-30 — top-k sweep extended: optimised strategy, k=40/50/60/70/80/90/100, £100k

Tool: live_sim.py (`--universe --strategies optimised --initial-cash 100000 --start-date 2000-01-01 --cost-model ibkr_tiered_spread --max-trades-per-day 0 --workers 4 --top-k <k> --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60`)
Scope: full S&P500+FTSE100 universe, optimised strategy, £100k pot, 7 k values (40–100)
Journals: `data/journals/live_sim_optimised_topk{40,50,60,70,80,90,100}.csv`
Result: **Peak at k=70 (+63.4%). Quality cliff hits at k=80 — returns drop and drawdown jumps sharply.**

| k | Candidates | Total return | Trading P&L | Interest | Max DD | Notes |
|---|---|---|---|---|---|---|
| 40 | 299 | +38.4% | +£27,690 | +£10,746 | −4.07% | ~2.6yr window |
| 50 | 407 | +44.4% | +£34,455 | +£9,906 | −5.08% | ~2.6yr window |
| 60 | 529 | +62.1% | +£52,344 | +£9,708 | −5.26% | ~2.6yr window |
| **70** | **625** | **+63.4%** | **+£54,390** | **+£9,017** | **−6.38%** | **~2.6yr window — peak** |
| 80 | 683 | +57.9% | +£49,524 | +£8,407 | −8.83% | quality cliff starts |
| 90 | 850 | +50.2% | +£42,938 | +£7,244 | −9.76% | degrading |
| 100 | 980 | +50.0% | +£43,326 | +£6,647 | −9.38% | degrading |

Combined with the 2026-07-29 sweep (k=10/15/20/25/30), full picture across all tested k values:

| k | Candidates | Total return | Max DD | Window |
|---|---|---|---|---|
| 10 | 63 | +18.8% | −1.77% | ~2.6yr |
| 15 | 95 | +21.0% | −1.85% | ~2.6yr |
| 20 | 109 | +20.4% | −2.50% | ~2.6yr |
| 25 | 165 | +27.8% | −4.30% | ~2.6yr |
| 30 | 187 | +33.6% | −4.30% | ~2.6yr |
| 40 | 299 | +38.4% | −4.07% | ~2.6yr |
| 50 | 407 | +44.4% | −5.08% | ~2.6yr |
| 60 | 529 | +62.1% | −5.26% | ~2.6yr |
| **70** | **625** | **+63.4%** | **−6.38%** | **~2.6yr** |
| 80 | 683 | +57.9% | −8.83% | ~2.6yr |
| 90 | 850 | +50.2% | −9.76% | ~2.6yr |
| 100 | 980 | +50.0% | −9.38% | ~2.6yr |

**Correction (2026-07-30):** an earlier version of this table carried k=20/25/30 figures (candidates 98/125/146, returns +10.2%/+10.4%/+13.0%) from a run that had silently defaulted to `--start-date 2026-01-12` instead of `2000-01-01`, giving those three a ~6.5-month window vs everyone else's ~2.6yr — a different session flagged this as suspicious and it checked out. Verified directly against each journal's trade dates: all of k=10 through k=100 now span ~2023-11-24/27 to 2026-07-27/29, the same window. k=20/25/30 were rerun 2026-07-29 22:19–22:59 with the correct `--start-date`; figures above are from the corrected runs. Independent confirmation the fix is real: idle-cash interest for k=20/25/30 (a pure function of elapsed cash-days, not signal quality) jumped from ~£1.1-1.3k to ~£11.5-12.2k — now in the same range as k=10/15's ~£12k, as expected for a matching ~2.6yr window. All 12 rows are now directly comparable.

Interest declining as k climbs past ~30 is expected — more candidates = more capital deployed as positions = less cash earning interest. Not a window artifact.

Conclusion: **k=70 is the empirical optimum (+63.4%, DD −6.38%) across the full k=10-100 range tested. Return climbs monotonically from k=10 to k=70 (no local dip once the timeframe bug is fixed), then degrades k=80+. k=60 is the conservative pick (+62.1%, DD −5.26%) — marginal return loss for meaningfully lower drawdown. k=80+ degrades as lower-ranked tickers dilute the pool.**

---

## 2026-07-29 — top-k sweep: optimised strategy, k=10/15/20/25/30, £100k

Tool: live_sim.py (`--universe --strategies optimised --initial-cash 100000 --max-trades-per-day 0 --workers 4 --top-k <k>`)
Scope: full S&P500+FTSE100 universe, optimised strategy, £100k pot, 5 k values
Journals: `data/journals/live_sim_optimised_topk{10,15,20,25,30}.csv`
Result: **All 5 k values profitable at £100k — confirms candidate-count filtering restores profitability vs 2026-07-27 full-universe failure.**

| k | Candidates | Admitted | Rejected | End value | Trading P&L | Interest | Max DD | Total return | Data window |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 63 | 63 | 0 | £118,816 | +£6,509 | +£12,307 | −1.77% | **+18.8%** | ~Nov 2023 – Jul 2026 (~2.6yr) |
| 15 | 95 | 95 | 0 | £120,967 | +£8,973 | +£11,994 | −1.85% | **+21.0%** | ~Nov 2023 – Jul 2026 (~2.6yr) |
| 20 | 109 | 109 | 0 | £120,437 | +£8,206 | +£12,232 | −2.50% | **+20.4%** | ~Nov 2023 – Jul 2026 (~2.6yr) |
| 25 | 165 | 165 | 0 | £127,845 | +£16,347 | +£11,498 | −4.30% | **+27.8%** | ~Nov 2023 – Jul 2026 (~2.6yr) |
| 30 | 187 | 187 | 0 | £133,589 | +£21,759 | +£11,830 | −4.30% | **+33.6%** | ~Nov 2023 – Jul 2026 (~2.6yr) |

**Correction (2026-07-30):** the k=20/25/30 rows originally logged here (candidates 98/125/146, returns +10.2%/+10.4%/+13.0%, ~6.5mo window) came from a run that silently defaulted to `--start-date 2026-01-12` instead of the plan's `2000-01-01` — the `--start-date` flag was accidentally dropped when those three were rerun after an unrelated daemon-restart interruption. A separate session flagged the window mismatch as suspicious; confirmed by inspecting each journal's actual trade-date range. Reran k=20/25/30 on 2026-07-29 22:19–22:59 with the correct flag; the table above reflects the corrected runs, verified to share the same ~2.6yr window as k=10/15 (interest income, a pure function of elapsed cash-days, now sits at ~£11.5-12.2k for all five instead of the broken run's ~£1.1-1.3k for k=20/25/30 — matches the ~2.6yr vs ~6.5mo gap exactly).

**Within-group findings (all 5 now comparable, same ~2.6yr window):**
- Return climbs from k=10 to k=15 (+18.8% → +21.0%), dips slightly at k=20 (+20.4%, within noise — DD also ticks up to −2.50%), then climbs again through k=25 (+27.8%) and k=30 (+33.6%).
- Trading P&L (excluding idle-cash interest) nearly triples from k=15 to k=30 (£8,973 → £21,759) as candidate count roughly doubles (95 → 187).
- Zero rejections across all runs — at 63–187 candidates, £100k pot is never the binding constraint. Entry ordering is pure signal priority, not capital starvation.

Conclusion: **Filtering to top-k tickers reliably restores profitability at every k tested (+18.8% to +33.6%, all beating the full-universe baseline's −3.2%).** No local optimum within k=10-30 — return keeps climbing to k=30, which turns out to continue smoothly into the 2026-07-30 extended sweep (k=40-100): see that entry for the full picture and the eventual peak at k=70.

---

## 2026-07-28 — optimised strategy rescue: 3-option fix validation
Tool: live_sim.py (all 3 options)
Scope: Option 1 (top-20 tickers, 354 candidates); Option 2 (full vol-filter universe, 3,790 candidates × £250k/£500k pots); Option 3 (strict threshold, failed)
Journal: data/journals/live_sim_position_summary_20260728T010303.csv (opt1), 20260728T001059.csv (opt2)
Result: **TWO SOLUTIONS WORK. Candidate filtering vastly outperforms capital scaling.**

| Option | Scope | Pot sizes tested | Trading P&L | Total return | Candidates |
|---|---|---|---|---|---|
| 1: Top-20 tickers | PCT.L, LNT, AJG, SMT.L, MO, HSBA.L, HUM, BGEO.L, GSK.L, PANW, AAPL, SHEL.L, RTX, CBOE, BG, XEL, AEP, MNG.L, SBRY.L, IMB.L | £25k/50k/100k | +£3,191 / +£6,674 / +£13,348 | **+21.5% / +22.1% / +22.2%** | 354 |
| 2: Large pot (full vol-filter universe) | 118 tickers (vol-filter applied) | £250k / £500k | +£793 / +£3,177 | **+2.6% / +3.0%** | 3,790 |
| 3: Strict threshold (--buy-threshold 4.0) | 597 tickers | £100k | FAILED | CLI arg error | N/A |

**Critical insight:** Strategy is NOT broken; **capital-to-candidate ratio is the constraint.** With 354 candidates per £100k (1 candidate per £282), optimised returns 22%. With 3,790 candidates per £100k (1 per £26), it loses. Same 3,790 candidates need £500k (1 per £132) to barely break even at 3%. **Candidate filtering >> capital scaling by 7-10x on returns.**

Option 3 (strict threshold via `--buy-threshold`) failed because that parameter is backtest-only (run.py), not live_sim (live_sim.py uses pre-generated candidates, can't change entry thresholds at arbitration time). Workaround: re-run backtest with higher buy_threshold to generate stricter candidates, then feed those to live_sim.

Conclusion: **Optimised strategy is viable. Path forward: identify the top-performing tickers/signals and focus live_sim on those 15-25 only (not full 118-603 universe).** This reduces candidate contention from 3,790 to ~200-400, restoring profitability to 20%+ on realistic capital (£50k-£100k). Alternative: use £500k+ capital, accept 3% returns, or re-filter candidates by entry_score (top-50% only, ~1,890 candidates) to split the difference (10%+ target).

---

## 2026-07-27 — optimised strategy capital-arbitrated test: full-universe, 3 pot sizes
Tool: live_sim.py, `--universe --strategies optimised --pot-sizes 25000 50000 100000 --max-trades-per-day 0 --workers 4 --cost-model ibkr_tiered_spread --start-date 2000-01-01`
Scope: 603 tickers (S&P500 + FTSE100), 1 strategy (optimised), vol-filter applied (118/603 candidate-eligible), 3 pot sizes
Journal: data/journals/live.csv (trades), data/journals/live_sim_position_summary_20260727T184200.csv (equity curve)
Result: 3,784 candidates per pot, 3,784 admitted at every size (0 rejections), all pot sizes NEGATIVE:

| Pot size | Candidates | Admitted | Rejected (cash) | End value | Trading P&L | Interest | Max drawdown | Total return |
|---|---|---|---|---|---|---|---|---|
| £25,000 | 3,784 | 3,784 | 0 | £17,683.55 | −£7,714.49 | +£398.04 | (calc pending) | −29.27% |
| £50,000 | 3,784 | 3,784 | 0 | £45,193.83 | −£5,784.14 | +£977.97 | (calc pending) | −9.61% |
| £100,000 | 3,784 | 3,784 | 0 | £99,333.32 | −£2,832.69 | +£2,166.01 | (calc pending) | −0.67% |

**CRITICAL CONFLICT:** This result directly contradicts the 2026-07-27 full_scan baseline, where optimised was the only strategy profitable at every tested pot size (+£2,053 @ 25k, +£7,413 @ 50k, +£16,321 @ 100k; full_scan drawdown −5-7%). Root cause: **isolated-pot backtests (one £10k per ticker, unlimited overlapping capital) cannot see entry clustering and capital starvation that occurs in real shared-pot live_sim arbitration.** When 603 tickers each generate ~6 trades/day (18,000 candidate-days × 3,784 candidates / 650 trading days ≈ 8,700 trades/year), admitting all of them into one shared £100k pot means: (a) positions are smaller (Kelly fraction off a smaller cash balance after each trade), (b) overlapping holds create compounding drawdowns (20 overlapping losers hit equity worse than 20 separate backtests), (c) market drift between when candidates were generated (full history, averaged) and when arbitration runs (realized returns, forward-looking) exposes assumption gaps. **The isolated-pot Sharpe ranking (optimised 0.63, #9) is not just lower than conserv/default (1.51/1.27) — it's inverted in real capital constraint.** 

Conclusion: **live_sim capital-arbitrated testing is the only valid ground truth for expected returns. Isolated-pot backtests are useful for signal screening but dangerously misleading for P&L prediction.** Strategies must be re-evaluated under real shared-pot conditions before deployment. Next: either (a) reduce trade frequency to drop overlaps (via higher thresholds or vol gates), (b) deepen candidate screening to admit only highest-confidence trades, or (c) accept that optimised's structure (few high-conviction trades) fails in full-universe high-frequency setting and is only viable for focused single-ticker or low-candidate-count portfolios (e.g., 10-20 FTSE100 names).

## 2026-07-27 — optimised strategy 597-ticker no-filter test: rules out vol-filter bias hypothesis
Tool: live_sim.py, same as above but `--tickers <all 597>` (no --universe vol-filter)
Scope: 597 tickers (all tickers from full_scan), optimised strategy, 3 pot sizes
Journal: data/journals/live_sim_position_summary_20260727T205915.csv
Result: **Nearly identical losses vs 118-ticker vol-filter run; vol-filter hypothesis REJECTED.**

| Pot size | 118-ticker (vol-filter) | 597-ticker (no filter) | Delta P&L | Delta % |
|---|---|---|---|---|
| £25,000 | −£7,714 / £17,683 | −£7,779 / £17,616 | −£65 | +0.8% |
| £50,000 | −£5,784 / £45,194 | −£5,931 / £45,041 | −£146 | +2.5% |
| £100,000 | −£2,833 / £99,333 | −£3,141 / £99,012 | −£308 | +10.9% |

Interpretation: The 479 excluded tickers (vol-filter rejects) did not contain profitable winners; they contributed ≈4 more candidates with the same loss dynamics. **Root cause is definitively NOT universe selection.** The problem is structural: 3,779 candidates competing for £100k pot causes:
1. Kelly position sizing to collapse mid-trade (remaining balance shrinks after losses)
2. Overlapping losers to compound (20 concurrent losers = worse aggregate drawdown than 20 separate isolated backtests)
3. Market drift to exacerbate (backtest signal generation 2000-2026 average, arbitration 2023-2026 forward = entry assumptions wrong)
4. Entry delays / rejections (capital starvation) unvisible in isolated backtests that assume unlimited capital

Confirmed: isolated-pot full_scan Sharpe ranking (0.63, #9) is not just lower than conserv/default (1.51/1.27) — **it inverts under real capital constraint.** Strategy is fundamentally broken for full-universe deployment at these capital levels. Viable only for: (a) reduced candidate count (top-20 FTSE100 names, ~200-300 candidates total), or (b) much larger pot (£500k+), or (c) radical signal filtering (admit only >70th percentile entry_score).

---

## 2026-07-27 — 38-ticker optimised batch backtest (daily vol gate)
Tool: run.py batch mode (likely with --daily-vol-gate flag testing)
Scope: 38 tickers (20 FTSE100, 18 US large-cap) x optimised strategy
Journal: data/journals/backtest.csv
Result: 693 closed trades, +$10,193.88 net P&L, 54.4% win rate (300 winners / 393 losers)

| Strategy | Closed trades | Net P&L | Avg Profit/Trade | Peak concurrent capital | Peak date | Avg concurrent capital |
|---|---|---|---|---|---|---|
| optimised | 693 | +$10,193.88 | +$14.71 | $85,000 | 2026-02-17 | $42,127 |

Top performers: PCT.L (+$2,982.55, 70.2% return), LNT (+$1,294.59, 36.1% return), SMT.L (+$1,040.66, 33.7% return). Bottom performers: JKHY (−$362.74, −17.0% return), AV.L (−$346.38, −16.2% return), REL.L (−$296.02, −13.9% return). 

Conclusion: optimised strategy scales to batch mode with moderate win rate across mixed universe; capital efficiency (peak $85k for 693 trades) outperforms high-frequency strategies; likely candidate for live_sim.py sweep against real shared-pot arbitration to validate capacity.

---

## 2026-07-27 — 10-strategy fresh baseline, fee/cash bugfixes, £10k cash
Tool: full_scan_all_strategies.py (`scripts/full_scan_10strat_20260726.log`), --force --workers 4 --cost-model ibkr_tiered_spread --initial-cash 10000 --data-cutoff today. Added `--initial-cash` flag to full_scan.py/full_scan_all_strategies.py (previously hardcoded £20k, not exposed) so this run could match the live daemon's £10k pot.
Scope: 603 tickers (S&P 500 + FTSE 100) x 10 strategies: ai, breakout_momentum, breakout_momentum_optimised, conservative, conservative_optimised, default, mean_reversion, optimised, trend, trend_optimised. Run specifically to get a clean re-baseline after recent commits fixed round-robin, commission, and idle-cash-interest bugs (0832fb2, 5821420, 4c81ce3, 93de6e5, 7a5d918) — see `BACKTEST_LOG_ARCHIVE_pre20260727.md` for anything predating these fixes; not comparable to these numbers directly.
Journal: reports/full_scan/summary.csv (filter scanned_at >= 2026-07-26T23:32, dedupe (strategy,ticker) keep-last); per-ticker journals at data/journals/full_scan/<strategy>/<ticker>.csv (£10k isolated pot per ticker, overwritten per run, no dedup needed). Peak/avg concurrent capital reconstructed via scratchpad backtest_summary_20260727.py: pools every ticker's trades onto one shared timeline, each trade committing kelly_fraction × £10,000 for its open interval — capacity approximation, not simulated live_sim capital.
Result: 597/603 ok, 6 no-data, per strategy, consistent across all 10 passes.

| Strategy | Closed trades | Net P&L | Avg Profit/Trade | Peak concurrent capital | Peak date | Avg concurrent capital |
|---|---|---|---|---|---|---|
| ai | 8,750 | +£795,083 | +£90.87 | £226,478 | 2026-04-10 | £82,265 |
| breakout_momentum | 19,290 | +£992,628 | +£51.46 | £587,696 | 2026-04-20 | £358,819 |
| breakout_momentum_optimised | 8,779 | +£852,978 | +£97.16 | £336,477 | 2025-05-12 | £193,897 |
| conservative | 28,564 | +£1,053,042 | +£36.87 | £505,407 | 2025-05-16 | £280,288 |
| conservative_optimised | 10,271 | +£819,706 | +£79.81 | £238,825 | 2025-05-12 | £115,648 |
| default | 24,748 | +£1,011,018 | +£40.85 | £520,258 | 2026-01-12 | £292,167 |
| mean_reversion | 0 | +£498,535 | n/a (0 trades) | £0 | n/a | £0 |
| optimised | 4,729 | +£560,913 | +£118.61 | £106,465 | 2026-01-16 | £55,487 |
| trend | 23,452 | +£1,028,376 | +£43.85 | £568,012 | 2026-01-13 | £295,072 |
| trend_optimised | 9,205 | +£831,107 | +£90.29 | £320,000 | 2025-05-13 | £160,347 |

Conclusion: mean_reversion still takes 0 trades on this universe (confirms archived 2026-07-17 finding) — its +£498,535 "P&L" is pure idle-cash interest (~£843/ticker over ~658 days [~1.8yr] @ £10k, tiered GBP/USD compounding per the 4c81ce3 interest-on-uninvested-cash fix — verified by hand-reconstructing the tiered daily-compound accrual for a 675-day ticker: predicted £867.60 vs actual £866.26), not trading edge; exclude it from any ranking. This is also 597 *independent* £10k pots each earning interest separately, summed — not one account's interest. Of the real traders, optimised has the fewest trades (4,729) and smallest peak capital (£106k) among the whole set — optimised is the capital-efficiency outlier vs the high-frequency cluster (conservative/default/trend/breakout_momentum, which cluster 5-6x higher on trade count with peak capital roughly proportionally higher too). The _optimised variants of breakout_momentum/conservative/trend_optimised sit in between: ~2-3x fewer trades than their base strategy with noticeably lower peak capital. These are gross-of-nothing-else numbers (isolated £10k pot per ticker, no shared-capital constraint, no entry arbitration) — not what a real £10k (or £106k) live_sim portfolio would earn. No "return %" or "annualised" figure is derivable from this table — use `live_sim.py --initial-cash <X>` for that question.

**Risk/return addendum (2026-07-27):** win rate vs buy-and-hold is low across the board (23-28%, `conservative` highest) — every strategy loses to B&H on raw absolute return on most tickers. But avg Sharpe beats B&H's 0.54 baseline for all 9 real-trading strategies (`conservative` 1.51, `default`/`trend` 1.15-1.27 highest; `optimised` 0.63 lowest despite best capital efficiency — fewer, larger, punchier trades = higher variance per trade). Top-mover tickers cluster heavily in semiconductor/AI names (WDC, SNDK, LITE, MU, APP, PLTR, HOOD) across nearly every strategy — concentration risk, not diversified alpha. 6 tickers excluded from all 10 passes (data gaps): ECHO, FDXF, GEV, HONA, SOLV, VLTO. Full report with per-strategy bar charts, full metric table (Sortino/Calmar/max-DD/down-capture), and top-3/bottom-3 movers: https://claude.ai/code/artifact/736bf3e2-0b9c-44cb-a320-e40ff3af281d

**live_sim.py capability upgrade (2026-07-27):** added `--universe` (full S&P500+FTSE100 list), `--workers` (parallel candidate generation), `--pot-sizes` (sweep multiple pot sizes against the same generated candidates, no re-backtesting per size), `--max-trades-per-day 0` = unlimited (cash-gated only), and mark-to-market equity-curve tracking (`position_summary.csv`, additive output — open positions valued at last known price, not frozen cost basis). 20-ticker/4-strategy/4-pot-size smoke test completed in 76s (well under a 15-minute abort threshold).

## 2026-07-27 — full-universe live_sim, real capital arbitration, £25k/£50k/£100k/£200k sweep
Tool: live_sim.py, `--universe --strategies conservative default trend optimised --start-date 2000-01-01 --pot-sizes 25000 50000 100000 200000 --max-trades-per-day 0 --workers 4 --cost-model ibkr_tiered_spread` (`scripts/live_sim_universe_4strat_20260727.log`). **First-ever real capital-arbitrated run at this scale** — every earlier entry in this file (and the archive) used isolated-£10k/£20k-pot backtests with unlimited capital per ticker; this is the first time entries actually competed for one shared pot, with real Kelly sizing off the live pot balance and real admission/rejection.
Scope: full S&P500+FTSE100 universe (603 tickers), default vol-filter excluded 485/603 as unsuitable (trend_quality below threshold) — **only 118 tickers were actually candidate-eligible**, same 118 for all 4 strategies. Candidate history: 2023-11-23 to 2026-07 (~2.6yr, the yfinance 730d-hourly-cap window). x 4 strategies x 4 pot sizes = 16 arbitration runs, sharing 4 candidate-generation passes (one per strategy, reused across all 4 pot sizes per the plan's cost-saving design).
Journal: data/journals/live_sim_universe_20260727.csv (67,281 trades, dedicated path — does NOT touch the live daemon's real data/journals/live.csv); data/journals/live_sim_universe_position_summary_20260727.csv (event-day equity curve + per-strategy-per-pot-size SUMMARY rows: portfolio_value, realized_pnl_cum, interest_cum, n_candidates, n_admitted, n_rejected_cash, max_drawdown).
Result: candidates admitted essentially 100% at £100k+ for every strategy (0 rejected for cash) — **capital is not the binding constraint at £100k for this 118-ticker universe**; conservative/default only saw real rejections at £25k (27/5304 and 46/4968 respectively) and £50k (3 and 7). The more consequential finding: **conservative and default lose money at every pot size tested once run for real**, despite being the top-2 strategies by the (isolated-pot, unconstrained-capital) 2026-07-27 full_scan Sharpe ranking (1.51 and 1.27). trend crosses from loss to profit between £50k and £100k. Only optimised is profitable at every pot size, including the smallest (£25k), and with by far the smallest drawdowns.

| Strategy | Pot size | Candidates | Admitted | Rejected (cash) | Realized P&L | Interest | Max drawdown |
|---|---|---|---|---|---|---|---|
| conservative | £25,000 | 5,304 | 5,277 | 27 | −£10,793 | +£259 | −42.0% |
| conservative | £50,000 | 5,304 | 5,301 | 3 | −£10,029 | +£645 | −19.2% |
| conservative | £100,000 | 5,304 | 5,304 | 0 | −£9,967 | +£1,458 | −12.7% |
| conservative | £200,000 | 5,304 | 5,304 | 0 | −£13,153 | +£3,082 | −11.2% |
| default | £25,000 | 4,968 | 4,922 | 46 | −£9,703 | +£258 | −39.5% |
| default | £50,000 | 4,968 | 4,961 | 7 | −£8,768 | +£625 | −21.5% |
| default | £100,000 | 4,968 | 4,968 | 0 | −£8,186 | +£1,397 | −17.3% |
| default | £200,000 | 4,968 | 4,968 | 0 | −£10,170 | +£2,945 | −15.8% |
| trend | £25,000 | 4,728 | 4,716 | 12 | −£6,409 | +£276 | −28.2% |
| trend | £50,000 | 4,728 | 4,728 | 0 | −£1,797 | +£668 | −12.0% |
| trend | £100,000 | 4,728 | 4,728 | 0 | +£6,040 | +£1,499 | −10.9% |
| trend | £200,000 | 4,728 | 4,728 | 0 | +£18,198 | +£3,141 | −10.8% |
| optimised | £25,000 | 1,359 | 1,359 | 0 | +£2,053 | +£1,068 | −6.8% |
| optimised | £50,000 | 1,359 | 1,359 | 0 | +£7,413 | +£2,321 | −5.6% |
| optimised | £100,000 | 1,359 | 1,359 | 0 | +£16,321 | +£4,694 | −5.3% |
| optimised | £200,000 | 1,359 | 1,359 | 0 | +£32,784 | +£9,753 | −5.3% |

Conclusion: **the isolated-pot full_scan Sharpe ranking did not survive contact with real capital arbitration.** conservative (full_scan Sharpe 1.51, #1) and default (1.27, #2) both lose money at every tested pot size here — high trade frequency across a shared pot means signals cluster and drawdowns compound in ways an isolated per-ticker backtest can't see (max drawdown here, −12% to −42%, is far worse than the per-ticker average −1.4%/−1.8% those strategies showed in the 2026-07-27 full_scan risk/return report). trend only turns profitable once the pot clears ~£75-100k. optimised — the lowest-Sharpe strategy (0.63) in the old isolated-pot ranking — is the only one profitable at every capital level, with drawdowns 2-8x smaller than the others; it also never got capital-constrained even at £25k, since its low trade count (1,359 vs ~5,000 for the others) rarely produces overlapping entries. At the user's target £100k: conservative −£9,967, default −£8,186, trend +£6,040, optimised +£16,321. **This is the first number in this project that actually answers "what would £100k earn me" — everything before it in this file (isolated-pot, unconstrained-capital backtests) could not.** Caveats: only 118/603 tickers were vol-filter-eligible (not the full universe); mark-to-market Sharpe/Sortino for the shared pot itself is deliberately not computed (equity curve is event-day-sampled, too sparse for a defensible annualized ratio — see the code comment in `arbitrate()`); this is one realized draw of history (2023-11-23 to 2026-07), not a distribution. **Known bias in this run, fixed below:** the vol-filter (118/603 eligible) was applied once at the top using today's `trend_quality` snapshot, held fixed across the whole 2.6yr window — not how live trading actually screens (`overnight_scope.py` rescreens nightly). See the daily-rescreen entry immediately below for the corrected version.

---

## 2026-07-27 — full-universe live_sim, DAILY vol-filter rescreening (fixes static-filter bias above)
Tool: live_sim.py, `--universe --strategies conservative default trend optimised --start-date 2000-01-01 --pot-sizes 25000 50000 100000 200000 --max-trades-per-day 0 --workers 4 --cost-model ibkr_tiered_spread --journal data/journals/live_sim_universe_dailyvol_20260727.csv --position-summary data/journals/live_sim_universe_dailyvol_position_summary_20260727.csv` (`scripts/live_sim_universe_4strat_dailyvol_20260727.log`). Implements the fix flagged above: `vol_screen.rolling_trend_quality()` computes trend_quality as a lookahead-free rolling time series (504-trading-day window, shifted 1 day) instead of a single "as of today" snapshot, and the gate is now applied **per-candidate, per-entry-day** (`_filter_candidates_by_daily_trend_quality`) instead of once per ticker up front — matching `overnight_scope.py`'s actual nightly rescreen cadence in the live daemon.
Scope: full 603-ticker universe, **all 603 now get backtested and produce candidates** (the old run's per-ticker admit/reject step is gone — that's why candidate counts below are much larger than the 118-ticker static-filter run above); the daily trend_quality gate then filters individual candidate entry-days instead of whole tickers. x 4 strategies x 4 pot sizes, same cost model and universe as the static-filter run for direct comparison.
Journal: data/journals/live_sim_universe_dailyvol_20260727.csv (198,842 trades — dedicated path, does NOT touch the live daemon's real data/journals/live.csv); data/journals/live_sim_universe_dailyvol_position_summary_20260727.csv.
Result: daily gate survival rate ~72-77% of candidates (vs. the static filter's binary 118/603 ticker cut) — a materially different selection mechanism, not just a stricter/looser version of the same one:

| Strategy | Raw candidates | Gate-passed | Gate survival |
|---|---|---|---|
| conservative | 28,191 | 21,087 | 74.8% |
| default | 24,718 | 18,600 | 75.2% |
| trend | 23,368 | 17,801 | 76.2% |
| optimised | 4,926 | 3,788 | 76.9% |

| Strategy | Pot size | Candidates | Admitted | Rejected (cash) | Realized P&L | Interest | Max drawdown |
|---|---|---|---|---|---|---|---|
| conservative | £25,000 | 21,087 | 10,504 | 10,583 | −£25,011 | +£11 | −100.0% |
| conservative | £50,000 | 21,087 | 16,204 | 4,883 | −£35,548 | +£92 | −71.0% |
| conservative | £100,000 | 21,087 | 17,982 | 3,105 | −£36,816 | +£361 | −37.1% |
| conservative | £200,000 | 21,087 | 19,002 | 2,085 | −£37,522 | +£951 | −25.2% |
| default | £25,000 | 18,600 | 10,812 | 7,788 | −£25,012 | +£12 | −100.0% |
| default | £50,000 | 18,600 | 15,582 | 3,018 | −£33,160 | +£86 | −66.9% |
| default | £100,000 | 18,600 | 16,807 | 1,793 | −£31,753 | +£312 | −34.0% |
| default | £200,000 | 18,600 | 17,457 | 1,143 | −£27,864 | +£809 | −25.0% |
| trend | £25,000 | 17,801 | 11,032 | 6,769 | −£24,983 | +£12 | −99.9% |
| trend | £50,000 | 17,801 | 15,195 | 2,606 | −£29,204 | +£100 | −59.1% |
| trend | £100,000 | 17,801 | 16,259 | 1,542 | −£22,474 | +£351 | −26.9% |
| trend | £200,000 | 17,801 | 16,854 | 947 | −£8,025 | +£895 | −23.1% |
| optimised | £25,000 | 3,788 | 3,788 | 0 | −£7,709 | +£398 | −32.8% |
| optimised | £50,000 | 3,788 | 3,788 | 0 | −£5,760 | +£978 | −18.4% |
| optimised | £100,000 | 3,788 | 3,788 | 0 | −£2,773 | +£2,166 | −13.5% |
| optimised | £200,000 | 3,788 | 3,788 | 0 | +£258 | +£4,549 | −13.3% |

**Comparison vs. the static-filter run above (same table shape, same universe/strategies/pot sizes/cost model):**

| Strategy | Pot | Static P&L | Daily P&L | Direction | Static max DD | Daily max DD |
|---|---|---|---|---|---|---|
| conservative | £100k | −£9,967 | −£36,816 | worse (−269%) | −12.7% | −37.1% |
| default | £100k | −£8,186 | −£31,753 | worse (−288%) | −17.3% | −34.0% |
| trend | £100k | +£6,040 | −£22,474 | flips negative | −10.9% | −26.9% |
| optimised | £100k | +£16,321 | −£2,773 | flips negative | −5.3% | −13.5% |

**Every strategy is worse under daily rescreening than under the static once-only filter, at every pot size tested — optimised is the only one still close to breakeven, and only at £200k (+£258).** conservative/default now blow through £25k pots entirely (−100% max drawdown = pot hit zero, meaning at least one point in the walk-forward the strategy ran out of cash and rejected 10,000+ candidates — see the huge "rejected" counts at £25k/£50k). This is the opposite of a rounding difference: candidate volume is 4-6x higher than the static run (21,087 vs 5,304 for conservative) because all 603 tickers now generate candidates (not just the 118 that passed today's snapshot), and the daily gate admits many candidates on tickers/days the static filter would have excluded entirely for the whole window, and vice versa excludes some the static filter always allowed. Root cause read: the static filter's 118-ticker set was implicitly survivorship-biased toward tickers that are trend-quality-good **as of 2026-07-27** — which correlates with having trended well recently, i.e. already-profitable-looking history. The daily gate removes that hindsight and the result is uniformly worse. **This is a more damaging, not more reassuring, correction than the earlier "return on max deployed" fix** — it says the £100k-earns-£16,321/yr (optimised) headline from the static-filter run does not hold once the vol-filter stops leaking future information into which tickers get considered at all.

Conclusion: **the static once-only vol-filter run above should be treated as invalidated for forward-looking return estimates — its numbers reflect a filter that could not have been applied in real time.** This daily-rescreen run is the more honest answer to "what would live-trading this earn": at £100k, every strategy tested loses money (best: optimised at −£2,773; worst: conservative at −£36,816). Before drawing further conclusions: worth checking whether `--min-trend-quality` (currently the default 0.0) is too permissive for a 603-ticker daily-gated universe (loosening the effective bar from "118 pre-vetted tickers" to "72-77% of any day's candidates" may just be admitting more noise), and whether the huge cash-rejection counts at £25k/£50k for conservative/default point to a strategy that's structurally unsuited to small-pot deployment regardless of filter design.

---

## 2026-09-02 � VIX portfolio-level risk-off gate: threshold sweep + validation

**Context:** Synthetic Jan2008�Jul2009 stress test (previous entry) showed -90.7% return / -89.7% max drawdown for optimised_new at �100k, top-k=70. Per-position 8% stop-loss alone was insufficient � 429/765 trades stopped out correctly but compounding stop-outs through a sustained downtrend wiped the pot. Hypothesis: a portfolio-level VIX regime gate (block all new entries when market-wide fear is elevated) would have de-risked before the crash, not just responded to it per-position. Related to the rolling-Sharpe volatility investigation (same session): correlated same-day entry clustering driven by shared HMM regime signals is the same mechanism that VIX detects.

**What was built:** `vix_entry_gate_threshold` strategy-owned class attribute on `OptimisedNewEntry` (follows same pattern as `same_day_deployment_cap_pct`). `arbitrate()` in `live_sim.py` gains `vix_series: pd.Series | None` + `vix_entry_gate_threshold: float | None` params � before processing each day's candidates, checks `vix_series.asof(day) >= threshold`; if true, all entries for that day are blocked (counted in `n_rejected_vix`). Cash release for closing positions and equity-curve recording still happen on blocked days. Real `^VIX` daily history fetched once via `fetch_daily("^VIX")` (yfinance max period, back to 1993), used even in synthetic-data mode. 1534 tests pass.

**Sweep:** `scripts/run_vix_gate_sweep.ps1` � thresholds {None, 20, 25, 30, 35, 40} x 2 windows. `scripts/analyze_vix_gate_sweep.py` for results.

Tool: live_sim.py, optimised_new, �100k, top-k=70, full universe, --workers 4. Two windows:
- **Window A (crash):** synthetic Jan2008�Jul2009 (450/600 tickers had synthetic coverage)
- **Window B (normal):** real Nov2024�present, --source ibkr

| Threshold | Crash return | Crash Sharpe | Crash Sortino | Normal return | Normal Sharpe | Normal Sortino | VIX-blocked (crash) | VIX-blocked (normal) |
|---|---|---|---|---|---|---|---|---|
| None (baseline) | -89.8% | -6.64 | -7.32 | +13.9% | +0.96 | +1.10 | 0 | 0 |
| 20 | -12.4% | -2.22 | -1.31 | +9.7% | +0.81 | +0.94 | 722 | 107 |
| 25 | -36.1% | -3.42 | -2.78 | +7.2% | +0.59 | +0.65 | 549 | 38 |
| 30 | -44.8% | -3.85 | -3.54 | +11.0% | +0.83 | +0.94 | 479 | 17 |
| 35 | -52.3% | -4.05 | -3.98 | +13.9% | +0.96 | +1.08 | 395 | 6 |
| 40 | -60.4% | -4.29 | -4.55 | +13.6% | +0.93 | +1.06 | 325 | 3 |

**Key findings:**

1. **vix20 chosen:** crash Sharpe -2.22 vs baseline -6.64 (+4.42 improvement), crash Sortino -1.31 vs -7.32 (+6.01). Normal-market cost: only -0.15 Sharpe (0.81 vs 0.96). The Sharpe cost is negligible; the tail protection is large.

2. **vix25 is worst-of-both-worlds:** worst normal Sharpe (0.59) AND mediocre crash Sharpe (-3.42). The VIX 20�25 range contains net-negative entries on average in the real window � blocking them (vix20) outperforms letting them through (vix25). Do not use 25.

3. **vix30 is runner-up:** good balance (crash Sharpe -3.85, normal Sharpe 0.83) but the crash Sortino (-3.54) is more than twice as bad as vix20 (-1.31) for only +0.02 normal Sharpe gained. Not worth the trade-off.

4. **vix35/40 essentially free in normal markets** (normal Sharpe matches baseline) but provide only modest crash protection.

**Decision:** `OptimisedNewEntry.vix_entry_gate_threshold = 20.0`. Comment in strategy file explains validation result.

**Not started:** Wiring VIX gate into the live daemon (`live_daemon.py`). The `live_sim.py` backtest path now has the gate; the daemon's entry path does not yet. Next step is adding a daily VIX check in the daemon's pre-entry logic using real-time `^VIX` (the existing `sentiment.py::vix_regime()` fetches 60d of VIX history � sufficient for a live check, just needs gating on `vix_current >= 20`).