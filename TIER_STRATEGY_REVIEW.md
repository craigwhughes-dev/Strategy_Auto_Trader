# Tier allocation strategy: how it was built, what the 2026-09-18 audit found, current state, open decisions

Written 2026-09-18 for the owner and for an independent reviewer. Facts are marked **[verified]** (read in code / output this session), **[reconstructed]** (inferred from docs, commits and memory notes; may differ from what the owner actually believed) or **[unverified]** (hypothesis, needs a check). No decision below has been taken; nothing described as "candidate" is deployed.

---

## 1. What the strategy is (as deployed)

A single-asset macro rotation. One holding at a time, chosen by VIX/VXN level. Paper account DUR166977 (IBKR UK retail), pot GBP 20,000 (`config/overnight_strategy.json` `capital_pot`). Code: `Strategy_Auto_Trader/allocation/allocation_manager.py` (`MultiTierAllocationManager`), signal logic in `allocation/multi_tier_allocator_4tier.py` (`MultiTierAllocator4Tier.signal`), wired into `markov_cli/live_daemon.py` (thresholds hard-coded at daemon start: `vxn_threshold=18.0, vix_tier1=15.0, vix_tier2=17.5`). **[verified]**

Decision order (first passing tier wins):

| Tier | Test | Asset | Exchange / currency |
|---|---|---|---|
| 1 | VXN <= 18 | EQGB.L (Invesco Nasdaq-100 UCITS, GBP-hedged) | LSE, GBP |
| 2 | else VIX <= 15 | SPY | NYSE Arca, USD |
| 3 | else VIX <= 17.5 | ISF.L (iShares FTSE 100 UCITS) | LSE, GBP |
| 4 | else (also if VIX missing) | CSH2.L (Amundi overnight-rate UCITS) | LSE, GBP |

The FTSE tier is gated by VIX, not VFTSE (no VFTSE data; deliberate). Rebalance = sell the current asset, buy the target whole-share quantity; hold otherwise. **[verified]**

Live timing **[verified in code, effect unverified]**: `_get_vix_current` / `_get_vxn_current` fetch the hourly VIX/VXN once per calendar day, cache the DataFrame, and use `Close.iloc[-1]` from that cache for every cycle for the rest of the day (cache is refreshed only when `date.today()` changes, or when the daemon restarts). So the live signal is effectively a once-a-day snapshot taken at the first allocation cycle of the day, not a re-read each hour. Orders are gated to LSE hours per ticker (commit b10be9a).

## 2. How it was believed to work **[reconstructed]**

Sources: `HANDOFF_ALLOCATION.md`, BACKTEST_LOG.md 2026-09-16/17 entries, memory notes `project_tier_strategy_vftse_decision`, `project_allocation_tier_strategy_live_20260917`, git history.

1. **Origin (2026-09-16):** a VIX-driven 3-asset rotation, SPY / ISF.L / SHV (later CSH2.L), on **daily closes**. Validation: 10-year real window Sharpe 42-53, max DD -2..-4.7%; walk-forward 2015-2023 train -> 2024 test showed no overfit; HMM gating tried and rejected (VIX-only 53 vs VIX+HMM 14); 26/27-year synthetic stress test dropped Sharpe to 17.6-19.5, DD -13.5%, and was recorded as "the realistic expectation".
2. **Nasdaq tier added (2026-09-16/17):** VXN<=18 on EQGB.L, 27-year validation "Sharpe 38.84, +16,464%". Hysteresis and buy-cadence filters were tested and **rejected** (same day) because they reduced those numbers.
3. **Go-live (2026-09-17):** paper trading, superseding the per-ticker optimised_new daemon as the "deployed approach".
4. **The implicit mental model:** each day's VIX/VXN close picks the asset; the switch happens at that close; the asset then earns returns. Headline Sharpe 38-53 was read as roughly what live would deliver (with 27yr synthetic Sharpe ~17-19 as the pessimistic anchor). The backtest engine (`multi_tier_allocator_4tier.backtest`) credited **day T's return to the tier chosen from day T's close** (same-day, look-ahead).
5. **Why this looked fine originally:** the first versions traded SPY and SHV. Those trade in the same hours that VIX is computed, so "decide at the close, own the asset at the close" is nearly true (small gap: VIX final print vs SPY close). Live execution then moved to **LSE-listed UCITS ETFs** (account is UK retail; US-domiciled ETFs are PRIIPs-blocked, see 5.6). LSE closes 16:30 London, mid-way through the US session, so the day's VIX/VXN close does not exist yet when the LSE closes. The backtest was carried over unchanged, so the timing gap that had been negligible for SPY became structural for three of the four assets. The realisation that the ETFs do not trade in NY hours is what triggered the 2026-09-18 audit.

## 3. What the 2026-09-18 audit did and found

All timing runs use daily asset closes from `data_synthetic/hourly/*` via `load_synthetic_daily` (real IBKR + Brownian-bridge blend; EQGB history is the proxy `EQGB_COMPLETE.csv`; CSH2 real history only from 2024-03-25, earlier is synthetic/proxy) and **real hourly IBKR VIX/VXN** (`data/cache/ibkr_hourly/INDEX_{VIX,VXN}.csv`, bar-start stamped; a bar counts only once it has ended). Window 2007-11-20 .. 2026-09-15, 4,452 days. Sharpe is annualised (mean/std * sqrt(252)). **[verified]**

Runs (all pick the tier with the unchanged `MultiTierAllocator4Tier.signal`):
- **A** look-ahead (the old backtests): tier from full-day close on T earns day T.
- **A'** ideal-executable: same tier earns day T+1 (assumes you could trade at the US close; impossible for LSE assets).
- **B** live-faithful (assumed): tier from the last VIX/VXN bar ended by 16:30 London on T (LSE close), earns T+1.

### 3.1 Look-ahead and LSE hours (no costs)

| Run | Sharpe | Return | Max DD |
|---|---|---|---|
| A look-ahead | 2.66 | +4,787% | -9.4% |
| A' ideal | 0.82 | +232% | -13.6% |
| B live-faithful | 0.66 | +162% | -16.5% |

Removing look-ahead (A -> A') is the big effect. The LSE-hours constraint (A' -> B) costs Sharpe 0.82 -> 0.66. Cutoff tier differs from full-close tier on 320 of 4,452 days (7.2%); on 79 of them the close said defensive but 16:30 did not. Every previously logged headline (Sharpe 38-53, +16,464%, +1,093%, +549%) is invalid as a live expectation.

### 3.2 Buy-and-hold references (same window)

| | Sharpe | Return | Max DD |
|---|---|---|---|
| Nasdaq (EQGB proxy) | 0.77 | +1,224% | -51.4% |
| SPY | 0.57 | +434% | -54.7% |
| ISF.L | 0.27 | +76% | -46.5% |
| CSH2.L (synthetic, near-zero vol) | n/m | +34% | 0% |

Without these the 0.66 was uninterpretable. Run B does not beat buy-and-hold Nasdaq on Sharpe, even before costs.

### 3.3 Cost of a tier switch (measured / modelled)

- One real tier fill exists: BUY ISF.L 1,902 @ 10.512 on 2026-09-17 (paper). IBKR execution history is wiped by the nightly Gateway restart (`reqExecutions` returned 0) and the adapter never stores `commissionReport`, so **commission is modelled, not observed**. The recorded `slippage_bps` (9.5) is against a delayed (type 3) signal price and is not real slippage. **[verified]**
- Quoted spread measured from IBKR `reqHistoricalTicks(BID_ASK)`, last 1,000 ticks before ~16:30 London, 09-17 and 09-18 (full spread bps): ISF 1.9-3.9, CSH2 0.8-1.6, EQGB 5.4-7.2, SPY 0.3. Two days only.
- Commission: IBKR UK 0.05% per side (= 10 bps round trip at GBP 20k; the GBP 1 minimum is irrelevant). SPY legs add ~2 bps FX conversion (22% of run-B switches touch SPY).
- **Round trip per switch ~13 bps of NAV.** A pasted Google summary of IBKR pricing corroborated the 0.05% and the ETF stamp-duty exemption.
- Bug found, not fixed: `plugins/costs.py` `IbkrTieredCost` adds 0.5% stamp duty to every `.L` buy. UCITS ETFs (ISF/EQGB/CSH2) should be exempt (ISINs unchecked). The ISF entry cost of GBP 110.97 contains ~GBP 100 phantom duty; live P&L reporting subtracts it.

### 3.4 Churn kills the raw signal

Run B switches 562 times in 18.8 years (31.8/yr). Net of 13 bps per switch: Sharpe 0.19, +26% (10 bps: 0.30 / +49%; 20 bps: -0.06 / -15%). Roughly cash-like return with a worse drawdown.

### 3.5 Whipsaw filters (revisiting the 09-16 rejection, which was made on the biased model)

Filters are pure functions over the tier series (`allocation/tier_filters.py`): a switch needs the *same* new target to persist. "Asymmetric" = move to a more defensive tier immediately, move to a riskier tier only after N consecutive days. Net of 13 bps:

| Filter | Sw/yr | Sharpe | Return | Max DD |
|---|---|---|---|---|
| none | 31.8 | 0.19 | +26% | -27.8% |
| hysteresis 5d | 5.3 | 0.61 | +160% | -20.7% |
| **asymmetric 10d** | 5.7 | **0.79** | +132% | **-14.2%** |
| asymmetric 60d (mostly cash) | 0.8 | 0.73 | +47% | -7.0% |

Every filter beats the raw signal net of cost; the earlier rejection reversed. Symmetric filters of 20d+ collapse into holding one tier and should be ignored. The optimum (10d) is noisy (5d 0.44, 10d 0.79, 15d 0.71).

### 3.6 Threshold sweep (VXN, VIX1, cash trigger), run B, 13 bps, 150 sets x raw/asym10d

- Raising the cash trigger (VIX 17.5) to 20-35 does not help Sharpe at VXN 18 (asym10d: 0.79 -> 0.38-0.57) and deepens DD (-14% -> -20..-46%).
- VXN 25 looks steadier across halves (pre-2019 0.74 / 2019+ 0.90 vs 0.55 / 1.36 for VXN 18) and returns more (+363% vs +132%) at DD -21%.
- SPY's VIX gate (`vix_tier1`) barely matters: SPY is only reached when VXN>18 and VIX<=15, i.e. 3% of days.
- The whole surface sits at Sharpe 0.5-0.8 with pure-Nasdaq (0.77) in the middle; best of 300 rows 0.83 is within noise of buy-and-hold Nasdaq. Thresholds trade drawdown vs return; none is clearly better risk-adjusted than holding Nasdaq.
- Correction to an owner hypothesis: raising VXN gives SPY *less* room (Nasdaq is tested first), not more.

### 3.7 Balanced Nasdaq / SPY / FTSE split (SPY given its own VXN band; not the deployed logic)

Nasdaq VXN<=vxn1; SPY vxn1<VXN<=vxn2 and VIX<=vix1; FTSE else VIX<=vix2; cash else. `vxn2=inf` reproduces deployed logic exactly (unit-tested). 290 configs.

- A real 3-way split (e.g. 18/25/25/35 -> 18% / 39% / 33% / 10% cash) needs SPY's VIX gate at 25 and cash only at VIX>35: Sharpe 0.74, +446%, DD -33%. Same Sharpe as the deployed cash-heavy setup (0.79, +132%, -14%), more return, much worse drawdown.
- **FTSE tier ablation:** re-mapping ISF.L days to SPY gives equal or better Sharpe, return and drawdown (0.75 / +510% / -28.8% vs 0.74 / +446% / -33.1%). ISF.L earns its slot on no metric. Re-mapping to cash lowers DD but discards return.

### 3.8 Sharpe formula

`live_sim.py` and the engine were already correct (`* sqrt(252)`). Six copies of `_compute_summary` in `allocation/` (hmm_gated, multi_tier_allocator, _4tier, _cadence, _hysteresis, rotator) used `mean*252/std` (~15.9x too high); all six now annualise correctly. All older allocation Sharpe/Sortino in BACKTEST_LOG are on the old scale and, being look-ahead, invalid anyway.

## 4. Current state

**Live (paper) [verified from `state/execution_state.json` and logs]:**
- Holding ISF.L, 1,902 shares, cost value GBP 19,993.82, bought 2026-09-17 at 10.512. Cash GBP 6.18. It is the only tier trade ever made.
- On 2026-09-18 the daemon wanted ISF.L -> CSH2.L (12:07 and 12:17 "ACTION: ISF.L -> CSH2.L (REBALANCE)") but later cycles logged "Allocation: missing required prices for ['ISF.L'] - skipping rebalance" (gateway-down / NaN-price group from LogSentinel; unresolved). No SELL has happened.
- The daemon restarted many times on 09-18 (16 daemon logs), which also resets the in-memory VIX/VXN cache.
- Live daemon still switches on the raw signal (about 32 switches/yr implied by the backtest), with costs ~13 bps each and no whipsaw filter.

**Code (all in the working tree; nothing deployed to the daemon):**
- `allocation/multi_tier_backtest_lse_lag.py` (runs A/A'/B, benchmarks, switch-cost columns, shared `load_inputs`)
- `allocation/tier_filters.py`
- `allocation/multi_tier_filter_sweep_lse_lag.py`, `multi_tier_threshold_sweep_lse_lag.py`, `multi_tier_split_sweep_lse_lag.py`
- `live_daemon.py`: per-ticker exchange-hours gate for allocation orders (`_exchange_market_name_for_ticker`); test-suite fixes (mocked screens, `**kwargs` fakes)
- Sharpe fix in 6 `allocation/` modules
- Tests: 1,603 passing (~90s). Docs: BACKTEST_LOG.md (six 2026-09-18 entries), HANDOFF.md ("Decisions pending" section).
- Reproduce: `uv run python -m Strategy_Auto_Trader.allocation.<module>` for `multi_tier_backtest_lse_lag`, `multi_tier_filter_sweep_lse_lag`, `multi_tier_threshold_sweep_lse_lag`, `multi_tier_split_sweep_lse_lag`.

## 5. Things I (the assistant) think are shaky or unchecked

1. **Run B may not model the daemon.** The daemon reads VIX/VXN once per day and holds it. If that first read happens before LSE trading (~08:00 London), the live signal is "full previous-day US close, trade today in LSE hours" = A'-like information (Sharpe 0.82, better than B's 0.66), but earning day T's return *from the LSE morning*, not from the previous close, so the overnight gap (T-1 close to T open) is not captured. Neither A' nor B is the live process. **[unverified]** Needs: actual timestamp of the first allocation fetch each day, which bar `iloc[-1]` returns at that time (in-progress bar vs last completed), and a run C using open-to-close (or first-LSE-hour) returns. Hourly LSE bars exist for only ~730 days, so a long-history C needs daily open data.
2. **SPY is probably untradeable on this account.** Memory notes record IBKR Error 201 (UK retail PRIIPs/KID) for SPY on account DUR166977. The deployed tier 2 and the "SPY room" analysis (3.6, 3.7) assume a working SPY tier, and `MultiTierAllocationManager.__init__` even seeds `current_asset = "SPY"`. UCITS equivalents (CSPX.L / VUSA.L) would be needed and change hours (LSE only, no FX leg), currency and spread. **[unverified, high priority]**
3. **Everything sits on one 19-year path, ~600 config-filter rows, few independent regime events** (2008, 2011, 2015, 2018, 2020, 2022). The best-of-N are selection-biased; sub-period halves often disagree (e.g. 0.55 vs 1.36).
4. **Data provenance.** EQGB history is a proxy; CSH2 is synthetic before 2024-03; asset daily closes are partly Brownian-bridge; CSH2's implied return (+34%, ~1.5%/yr) drives the value of every cash-tier day and was not compared with actual SONIA. ISF.L B&H Sharpe 0.27 may partly reflect the synthetic blend.
5. **Costs.** 13 bps is a flat NAV haircut per switch; no market-impact, no limit-order behaviour, no failed/late fills (the 09-18 gateway outage shows execution failures are real), no FX on SPY beyond an average, no bid-ask asymmetry at the 16:30 cutoff, no tax. Spread sample is two days near the cutoff.
6. **Benchmark fairness.** Comparison is vs buy-and-hold with no drawdown-aware rule, in GBP-hedged Nasdaq (EQGB) vs unhedged others; hedging carry/cost is not modelled and the base currency effects of SPY (USD) vs GBP assets are ignored in the daily series.
7. **Mixed frequency.** VIX/VXN arrive hourly, assets are daily. Days where VIX has no prints in the LSE window (pre-2015 London mornings, all VXN) use stale readings by construction, which is also what live would see, but it inflates 320-day differences in a period-dependent way.
8. **No live evidence yet.** One fill, no exit, no P&L. The IBKR paper account (DUR166977) is not real money.
9. **Sharpe on a near-cash strategy** is dominated by low-volatility periods (near-cash days have tiny variance), which is why "2019+ Sharpe 14" style outliers appear; Sortino/return-per-DD are better lenses.

## 6. Decisions

### Made (implemented; all reversible)
- Audit approach: compare A vs A' vs B on real hourly VIX/VXN with the LSE cutoff read from `config/overnight_strategy.json`.
- Per-ticker exchange-hours gate for allocation orders (LSE orders deferred to LSE open even when raised from the sp500 cycle).
- Cost basis for analysis: 13 bps round trip per switch.
- Filters implemented as pure functions over the tier series, not as new allocator variants.
- Sharpe/Sortino formula fixed in `allocation/`.
- SHV comparison not re-run (owner: tested and discarded earlier; underperformed in high-VIX periods).
- Deployed thresholds and raw-signal switching left unchanged.

### Pending (owner has not decided)
- **D1** keep the tier strategy live at all, vs buy-and-hold Nasdaq/SPY (UCITS). Trade: similar Sharpe to B&H Nasdaq, 1/4 to 1/2 the drawdown, far less return.
- **D2** add an asymmetric re-entry delay (~10 trading days, defensive immediate) to the daemon.
- **D3** raise VXN threshold 18 -> ~25 (more stable, more return, DD -21% vs -14%).
- **D4** fix the stamp-duty line in `IbkrTieredCost` for UCITS ETFs (verify ISINs first).
- **D5** persist `commissionReport` from each fill in `IbkrAdapter.place_order`.
- **D6** tier structure: banded Nasdaq/SPY/cash (drop or de-emphasise FTSE) vs deployed vs keep FTSE.
- Also open: execute the pending ISF.L -> CSH2.L switch / gateway-down price guard; SPY tradability (5.2); live-timing model (5.1).

## 7. Specific questions for the reviewer

1. What are the biggest methodological holes in the audit above (timing, data, cost, statistics)? Verify against the code, do not just trust this document.
2. Is run B the right model of the daemon, given the once-a-day VIX/VXN cache (section 1 / 5.1)? Which run (A', B, or a new C) best represents live, and what would you compute to settle it?
3. Given SPY's apparent PRIIPs block, which parts of the conclusions survive?
4. What would you check before ever trusting 13 bps, and before trusting any single "best" configuration?
5. What alternatives to a hand-tuned VIX/VXN tier rotation should be on the table (e.g. simple vol-targeting, trend filter on the same assets, fewer tiers, holding Nasdaq with a drawdown stop)? Be specific about what evidence would discriminate.
6. What has been missed entirely (operational, regulatory/tax, UCITS hedging, cash-yield, data snooping, execution)?

---

## 8. Independent review (Opus subagent, 2026-09-18) — findings NOT yet reproduced by the assistant

Everything in this section is the reviewer's own reading and its own ad-hoc computations. The assistant has not re-run the numbers; treat as leads to verify, not results.

**Corrections to this document**
- **Run B is not the live process.** Live decides from the previous US close and trades at the LSE morning: the daemon's first allocation cycle is ~08:00 London (`process_cycle` only runs inside `is_trading_hours`, ftse first), and `_get_vix_current`/`_get_vxn_current` cache on `date.today()` so a 07:00-London read returns exactly the prior day's final value (reviewer checked 2026-09-11 -> VIX 18.01 = 09-10 close). Run B uses a fresher signal (T's US morning) and a later execution (T's LSE close), two opposing errors, not a conservative bound. 16 restarts on 09-18 also reset the cache, so the realised live signal depends on restart times (non-reproducible).
- Reviewer's timing band (Sharpe; raw / net 13 bps / asym10d net 13 bps): A' 0.82 / 0.36 / 0.75; B 0.66 / 0.19 / 0.79; **C1** (T-1 signal earns T, upper bound) 1.02 / 0.55 / 0.78; **C2** (T-1 signal, execute T close, lower bound) 0.60 / 0.14 / 0.58. Raw-signal result is timing-fragile (0.60-1.02); the asymmetric-filter result is timing-robust (0.58-0.79). A proper run C needs daily Opens, which `load_synthetic_daily` drops for the Close-only EQGB proxy file.
- `plugins/costs.py` has three wrong lines: stamp duty on every `.L` buy (:72), PTM levy (:73-74, also ETF-exempt) and `_UK_SPREAD = 0.0015` (:46, 4-8x the measured ISF/CSH2 spreads). Also live sizing uses `commission_pct=0.1` (`allocation_manager.py:65`) = 20 bps round trip, vs 13 bps in the study and 2x IBKR's 0.05%.
- `allocation_manager.py:79` seeds `current_asset="SPY"`; the ledger reseed only fires when a tier asset is held, so an empty portfolio believes it holds SPY. `allocation_mgr` is built unconditionally (`live_daemon.py:2334`), so tier rebalancing runs even without `--tier-mode` and shares `portfolio.available_cash` with the per-ticker strategy.
- `bars_by_end_time` stamps a bar's end as the next bar's start, so a session's last bar is invisible until the next session: conservative, no residual look-ahead. Run A still reproduces the old model (2.66 x sqrt(252) = 42.2 vs logged 38.8-42.4).
- Doc/code nits: tool prints B&H CSH2.L Sharpe 10.88 (doc says n/m); n_days is 4,451; HANDOFF's "add `broker.is_connected()` guard ~:1223" is stale (guard exists at :1266; the 09-18 skip came from the price-None branch at :1365).
- "Filter instability overstated": reviewer's asym sweep is a plateau (8d 0.74, 10d 0.79, 12d 0.71, 16d 0.73, 24d 0.76, 26d 0.77); implement as >=8 days, not exactly 10.

**The decisive test the audit never ran (reviewer's numbers, same data, 13 bps, same window)**

| | Sharpe (pre / post 2019) | Return | Max DD |
|---|---|---|---|
| Deployed + asym10d | 0.79 (0.55 / 1.36) | +132% | -14.2% |
| Static 30% Nasdaq / 70% CSH2, no trading | 0.95 (0.70 / 1.29) | +191% | -16.3% |
| 20d vol-target 10% | 0.95 (0.84 / 1.09) | +447% | -17.8% |
| vol-target 15% | 0.91 (0.80 / 1.05) | +787% | -26.3% |
| Nasdaq SMA200 else cash | 0.89 (0.67 / 1.15) | +814% | -23.8% |
| VXN<=18 2-tier Nasdaq/cash, asym10d | 0.81 (0.72 / 1.18) | +110% | -9.2% |

If reproduced, the tier strategy is dominated on Sharpe, return and half-stability by parameter-free alternatives. Its Sharpe is also mechanically flattered: tier 4 holds ~51.6% of days at ~0.15% annualised vol, so the fair comparator is an equal-exposure static blend, not 100% Nasdaq.

**Data risk not mentioned in section 5:** the CSH2 proxy pays a flat 0.50%/yr 2010-2015 and 4.2-5.2% 2023-25, i.e. it is a BoE base-rate patch, not fund NAV (no TER, no tracking difference). If the LSE line traded is the EUR overnight-rate class, the cash tier earns EUR-STR (negative 2015-22) with unhedged EUR/GBP risk. Unverified (ISIN not checked). Whether the EQGB proxy is total return is also unknown; if price-only, ISF's ~3.5% yield is missing and its 0.27 is understated. ~51.6% of days rest on the CSH2 series.

**SPY:** only 4.0% of days under the deployed thresholds, so deployed conclusions barely move; but D6's best balanced config (SPY 39% of days) and the FTSE->SPY ablation depend entirely on SPY, so **D6 is unusable until SPY tradability is verified**. A UCITS substitute (CSPX.L/VUSA.L) is LSE-hours, no FX leg, different spread and unhedged USD (unlike GBP-hedged EQGB), changing timing, cost and currency models at once.

**Missed entirely:** currency/hedging carry (everything in asset-local returns); total return vs price; UK tax (32 switches/yr outside an ISA/SIPP = CGT event stream, s.104 pooling, 30-day rule); depth/partial fills on a GBP 20k EQGB order (2 days of top-of-book data only); order-type policy (market orders into a ~6 bps spread); gateway outages as a first-class risk (the strategy's only claimed edge is drawdown control, which an outage removes, and outages correlate with volatility); single-position design (a failed SELL blocks the BUY); no kill-switch or max-switch guard.

**Discriminating evidence proposed:** stationary-block bootstrap (1,000 resamples) reporting Sharpe distributions; deflated Sharpe with N~600 trials; purged walk-forward (fit <=2016, score 2017+, one number per candidate); same comparison table on real EQGB/ISF/CSH2 history from 2024-03. Also parse daemon logs for the actual first-fetch and restart timestamps per day and replay the realised signal.

**Reviewer's recommendation on decisions:**
- First (5 minutes): D4 widened (stamp duty, PTM levy and the 15 bps UK spread in `costs.py`, after checking ISINs); then D5 (persist commissionReport).
- If the strategy stays live: D2 only (survives every timing model tested, 0.58-0.79 net), as >=8 days.
- Refuse until verified: D1 as posed (comparator should be the static blend and vol-target); D3 (0.81 vs 0.79 is noise and the "stable across halves" argument is in-sample); D6 (depends on SPY tradability); any capital decision before the CSH2 share class/currency and the EQGB proxy construction are confirmed.
- Reviewer unsure about: CSH2 ISIN/share class, whether EQGB proxy is total-return, whether `--tier-mode` is set on the running daemon.
