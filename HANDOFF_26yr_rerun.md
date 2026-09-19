# Handoff — re-run the 26-year tier backtest with the corrected cost model

Written 2026-09-19 (evening). Read this first, then `BACKTEST_LOG.md` (top four entries) and `CLAUDE.md`.

## 0. Correction to the premise — read before doing anything

The stamp-duty fix (commit `aa32dca`) changed `plugins/costs.py` (`IbkrTieredCost`), the portfolio ledger, and `commission_pct` sizing in `allocation_manager.py`. **None of the tier backtests use that cost model.** Every script in `allocation/` applies a flat per-switch cost in bps (default 13, `--cost-bps` / `intraday_engine.DEFAULT_COST_BPS`).
Verify in one line: `grep -ln "IbkrTieredCost\|make_cost_model" Strategy_Auto_Trader/allocation/*.py` prints nothing.

So **re-running the 26-year test unchanged gives identical numbers.** Say so to the user rather than presenting an unchanged table as a new result. What is worth doing (section 6):
1. Replace the flat 13 bps with a per-switch cost derived from the corrected model and show sensitivity. At a £20k pot, one switch (sell one fund, buy another) costs **10.0 bps commission-only** and **16.0 bps with the 3 bps ETF half-spread**. Before the fix the buy leg would have carried an extra ~50 bps (0.5% stamp duty + £1 PTM) — that is what the bug would have done had a tier backtest used it. Flat 13 sits between the two (measured half-spreads EQGB 2.7 + CSH2 0.8 bps give ~13.5).
2. Re-run the simple comparators (HANDOFF decision D1) on the same engine and metric. This is the highest-value item and has not been done on the corrected model.

## 1. State of the repo

`main`, pushed through `cf3d33c`. Relevant commits: `f7a988d` research scripts + Sharpe scale fix; `9b1c043` intraday tier allocation (IndexFeed, VXN deadband, config-driven thresholds, intraday engine, spliced dataset); `f83f7bc` log (13 old entries marked `[SUSPECT]`); `aa32dca` cost fix; `cf3d33c` log entry.
**Uncommitted (the user commits — do not commit or push unless asked):**
- Raw-bar fix: `broker/ibkr_data.py` (`fetch_index_hourly(aligned=...)`), `quant_hmm/sentiment.py` (`fetch_vix_hourly_raw`, `fetch_vxn_hourly_raw`), `markov_cli/live_daemon.py` (tier block uses the raw feeds), plus tests. Reason: the default return re-labels bars onto a :30 grid (label 13:30 holds the raw bar that ends 15:00), which made the live "completed bar" test an hour early. Guard test: `test_daemon_tier_block_feeds_the_allocator_raw_bars`.
- `scripts/tier_analysis/` (four analysis scripts, see section 4), this file.
- `Strategy_Auto_Trader/allocation/multi_tier_intraday_backtest.py` — an old, superseded script, deliberately not committed. Ignore it or ask the user before deleting.
Tests: 1,791 pass (`uv run python -m pytest tests -q`).
**Live daemon:** still running old code; the user says it auto-bounces tonight (2026-09-19). Not your task; do not touch the daemon or `state/`.

## 2. The strategy under test (current config)

`config/overnight_strategy.json` → `tier_allocation`: enter Nasdaq when VXN ≤ 23, hold until VXN > 24, otherwise cash; `lower_tiers_enabled: false` (S&P/FTSE tiers off); `commission_pct` 0.05.
Backtest form: `intraday_engine.tiers_vxn_deadband(inp.vxn, 23.0, 24.0)`; decisions on the LSE hourly grid from the latest COMPLETED VIX/VXN bar; **same-bar fill** is the headline, **next-bar** the conservative bound; 13 bps per switch; Nasdaq leg = EQQQ (unhedged, price-only). Live holds EQGB.L (GBP-hedged; its share class is not confirmed accumulating — IBKR just says "GBP HDG"; hourly EQGB data is stale/laggy before ~2024, so an EQGB leg is an upper bound for older years).

## 3. Data (all gitignored — rebuild if missing)

- `data_synthetic/hourly_spliced/{VIX,VXN,NASDAQ,SP500,FTSE,CASH}.csv` + `_meta.json` (columns: `Close`, `bar_end_utc`, `source` real|bridged). Rebuild: `uv run python -m Strategy_Auto_Trader.synthetic_backtest_data.build_intraday_dataset` (~5 min, seed 20260919). Inputs: `data/cache/ibkr_hourly_deep/` (EQQQ, IUSA, ISF.L, CSH2.L, EQGB.L), `data/cache/ibkr_hourly/INDEX_{VIX,VXN}.csv`, `data/cache/ibkr_daily/`, `data/cache/csh2_daily_returns_extended.csv`.
- If the deep cache is gone, refetch read-only from IBKR with a spare client id (the daemon uses 1 and 2; use ≥17): `IBKRDataClient(client_id=N)._fetch_pages(Stock(sym, "LSEETF", "GBP"), min_days=365*30, page_duration="6 M")`. CSH2 is on exchange `LSE`, not `LSEETF`. Gateway is port 4002.
- Real hourly starts: EQQQ 2005-08-17, IUSA and ISF.L 2004-04-16, VIX 2005-10-03 (London-morning bars from 2016), VXN 2007-11-20 (US hours only), CSH2.L 2015-09-02. **Before those dates the series are correlated-Brownian-bridge (intraday shape assumed; VXN 1999–2001 is chart estimates).** Treat 1999–2007 as a stress test: report it separately and never choose parameters on it. Known gaps: pre-2004/05 Nasdaq and S&P legs are USD proxies (no FX); early EQQQ bars are illiquid.

## 4. Code map

- `allocation/intraday_engine.py` — `load_inputs`, `tiers_vxn_vix` / `tiers_vix_only` / `tiers_vxn_deadband`, `simulate(inp, tiers, fill, cost_bps)`, `buy_and_hold`, `window_stats` (includes `xsharpe`), `annual_returns`. `WINDOWS`: train 2007-11-20..2019-12-31, test 2020-01-01.., bridged ..2007-11-19, real 2007-11-20.., morning_vix 2016...
- `allocation/multi_tier_intraday_grid.py` — threshold grid + report (`--variants vxn_vix vix_only vxn_deadband`), writes `data/intraday_backtest/<ts>/`.
- `scripts/tier_analysis/` — `annual_tier_current.py` (year-by-year table, reproduces the baseline below), `three_windows.py`, `ftse_role.py`, `panic_tier.py`. Run as `uv run python scripts/tier_analysis/<name>.py`.
- `plugins/costs.py` (fixed cost model) and `broker/symbols.is_uk_listed_etf`; `core/trading_sessions.py` (bar-end rule); `allocation/index_feed.py` (live completed-bar reader).
- The older `allocation/multi_tier_26yr_backtest.py`, `multi_tier_comparators_lse_lag.py`, etc. use daily / "run B" timing and are the suspect ones. Reuse their comparator definitions (static blend, vol-target, SMA200) if useful, but not their timing or data.

## 5. Baseline to reproduce (same-bar fill, 13 bps) — your sanity check

`uv run python scripts/tier_analysis/annual_tier_current.py` must print, for the 26-year window (1999-03-11 to 2026-09-18, 27.6 yrs, 6,952 days): strategy +2,242%, +12.1%/yr, xSharpe +0.76, max DD −21.5%, 7.4 switches/yr; Nasdaq buy and hold +1,785%, +11.2%/yr, xSharpe +0.44, DD −83.0%; 26 of 28 calendar years positive.
Other reference points: since VXN data (2007-11-20) two-state +1,001%, xSharpe +0.96, DD −14.9%; recent real (2024-03-25) two-state +24%, xSharpe +0.37 vs Nasdaq +52%; deployed with lower tiers on +2,136% / +0.74 / −21.2% / 11.0 sw/yr; old 18/15/17.5 thresholds +24% / −0.15. Full annual table is in `BACKTEST_LOG.md`, entry `2026-09-19 (night)`.
If these do not reproduce, stop and find out why before going further.

## 6. The task

A. **Confirm section 0** (the grep) and tell the user the stamp-duty fix does not alter tier backtests.
B. **Cost sensitivity.** Add a small helper (with tests) that turns `IbkrTieredCost` into per-switch bps for a given pot and fund pair (sell + buy, with and without `include_spread=True`). Re-run the 26-year, since-2007 and recent windows at 10, 13, 16 and 26 bps for: the current rule, plain VXN ≤ 24, and Nasdaq buy and hold. Report return, per year, xSharpe, max DD, switches/yr. Expect small changes (the strategy switches ~7–12 times a year: 10 → 16 bps is ~0.2 pts/yr of drag); say so if that is what you find.
C. **Comparators on the same engine (decision D1).** Static 30/70 Nasdaq/cash, vol-target 5% and 10% (trailing 20-day vol, weekly rebalance with a small deadband — cost per rebalance from the corrected model), and SMA200 + vol-target, over the same three windows and the same fill/cost assumptions. Compare on xSharpe and drawdown against the current rule. The 2026-09-18 log figures (static 30/70 raw Sharpe 0.94, VT10% 0.98, VT5% 1.11, SMA200+VT5% 1.15, deployed 0.79) are `[SUSPECT]` and used raw Sharpe including cash carry — use them only for orientation. Decide the result on whether the gap survives the corrected baseline.
D. **Record.** New entry at the top of `BACKTEST_LOG.md` (above `2026-09-19 (night)`) following the log's own rules: exact commands, actual data range, a results table, conclusion, caveats; no chart (not a `live_sim` run). Add tests for new code (test files mirror source). Update the D1/D4 notes in `HANDOFF.md`.

## 7. Gotchas

- `CLAUDE.md`: do not read `data/`, `logs/`, `reports/`, `state/`, `.venv/`, `uv.lock` unless asked. The user handles all git commits and pushes.
- Do not restart or modify the live daemon. It owns `state/execution_state.json` (which still holds the 2026-09-17 ISF.L entry with the old phantom £110.97 cost; repair only with the daemon stopped and the user's say-so).
- The Bash tool evaluates commands inside single quotes: avoid apostrophes in shell heredocs; write files with the Write tool. Set `PYTHONIOENCODING=utf-8` when printing non-ASCII.
- xSharpe = Sharpe of returns over the cash asset; raw Sharpe flatters cash-heavy rules. All fund returns are PRICE returns (EQQQ distributes ~0.6%/yr; VWRL ~1.9 pts/yr gap to VWRP was measured — see the log).
- Same-bar fill assumes an order at the bar-end price; live prices used for sizing are 15-minute delayed and orders are market orders. Always show next-bar alongside.
- Thresholds (23/24) were chosen on 2007–2019 only and eyeballed from a coarse grid; there is no walk-forward run. Do not present in-sample windows as out-of-sample.

## 8. Open items (not part of this task)

D5 (persist IBKR's real commission report); repair of the ISF.L ledger entry; a UK-listed UCITS S&P fund for tier 2 (S&P/FTSE tiers are off; SPY is not tradable on this account); confirm EQGB's share class; ~22 mangled characters in the 2026-09-02/03 log entries; the daemon restart and first-cycle ISF.L → EQGB.L rotation (user-managed).
