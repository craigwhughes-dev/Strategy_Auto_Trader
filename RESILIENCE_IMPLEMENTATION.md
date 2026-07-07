# Auto-Reconnect & Retry Implementation Summary

## Problem
FTSE100 daemon experienced socket disconnect during trade execution (2026-07-07 09:01:45), resulting in **zero trades over 1.5 trading days** despite generating normal signals. The socket error was logged but not retried, blocking all trades for that cycle.

**Expected behavior**: ~93% of trading days should generate trades (validated against 675-day backtest).
**Actual behavior**: 0 trades / 1.5 days due to transient connection failure.

## Solution: Three-Layer Resilience

### 1. **Live Daemon Retry Logic** (`markov_cli/live_daemon.py`)
New function `execute_signals_with_retry()` wraps signal execution with:
- **Automatic detection** of connection errors:
  - `ConnectionError` (explicit, from IBKRAdapter)
  - `OSError`, `TimeoutError` (socket-level)
  - String pattern matching ("socket", "disconnect", "ib_insync")
- **Exponential backoff**: 1s, 2s, 4s delays between retries
- **Broker reconnection**: Full disconnect/reconnect cycle before each retry
- **Graceful degradation**: After 3 retries, returns empty results (no trades that cycle) instead of crashing
- **Daemon continuity**: Socket errors never halt the daemon; cycle completes and next iteration runs normally

**Behavior**:
```
09:01:45 [ERROR] Connection error: Socket disconnect
09:01:45 [WARNING] Reconnecting in 1s... (attempt 1/3)
09:01:46 [INFO] Broker reconnected successfully
09:01:46 [INFO] Executing signals for 23 tickers...
09:01:47 [INFO] BUY: 2, SELL: 1, Skipped: 20
✓ Trades executed on retry
```

### 2. **IBKR Adapter Health Checks** (`broker/ibkr_adapter.py`)
Enhanced `IBKRAdapter` with:
- **`is_connected()`**: Safe connection status check (never raises)
- **Pre-flight validation** in `place_order()`: Checks connection before attempting orders
- **Defensive disconnect()**: Safely handles already-disconnected state
- **Better error messages**: Explicitly mentions socket/connection context in exceptions

**Impact**: Orders fail fast with clear `ConnectionError` that the retry logic recognizes.

### 3. **Comprehensive Test Coverage** (`tests/markov_cli/test_live_daemon.py`)
Added 9 new test cases covering:
- ✅ Success path (no retry needed)
- ✅ Socket error triggers reconnect
- ✅ Reconnect failures don't block retry
- ✅ Max retries exhausted → graceful fallback
- ✅ TimeoutError and OSError detection
- ✅ Non-socket errors raised immediately (no retry)
- ✅ String pattern detection ("socket", "disconnect", "ib_insync")
- ✅ Exponential backoff timing

All 32 existing daemon tests still pass; 9 new tests added (100% pass rate).

## Key Design Decisions

1. **Detect, don't suppress**: Socket errors are logged with full context, not silently swallowed
2. **Don't halt the daemon**: Failed batch skips trades that cycle; daemon continues to next cycle
3. **Aggressive reconnection**: Full disconnect/reconnect cycle, not just a resend (more robust against stale connections)
4. **Exponential backoff**: Avoids hammering a struggling connection while still being responsive to transient glitches
5. **Graceful degradation**: After 3 attempts (≈7 seconds total), accept the failure for that cycle rather than blocking indefinitely

## Testing Instructions

### To verify the retry logic:
```bash
# Run all daemon tests
python -m pytest tests/markov_cli/test_live_daemon.py -v

# Run just the retry tests
python -m pytest tests/markov_cli/test_live_daemon.py::TestExecuteSignalsWithRetry -v
```

### To test live (manual):
1. Start the FTSE daemon: `uv run python -m Strategy_Auto_Trader.markov_cli.live_daemon`
2. While running, kill TWS/Gateway (simulate network outage)
3. Observe daemon logs:
   - Detects socket error
   - Logs reconnect attempt
   - Waits 1s, then retries
   - On success: trades execute and portfolio saves
   - On failure (no TWS): cycles continue, next hour retries again

## Configuration
Hardcoded for robustness (no config needed):
- `max_retries = 3`
- Exponential backoff: 2^attempt seconds (1s, 2s, 4s)
- Inter-attempt disconnect/reconnect delay: 0.5s

To adjust, modify the call in `live_daemon.py:process_cycle()`:
```python
buys, sells, skipped = execute_signals_with_retry(
    market_name, ticker_list, DATA_DIR, portfolio, limit_tracker, broker,
    daily_buy_limit, daily_sell_limit, logger,
    max_retries=5  # Change here
)
```

## What's NOT Changed
- Backtest logic (unaffected)
- Signal generation (unaffected)
- Portfolio/position tracking (unaffected)
- Position sizing/limits (unaffected)
- All 105 broker tests pass (no regression)

## Impact on FTSE Daemon
Previously: Socket disconnect → 0 trades that hour
Now: Socket disconnect → Automatic retry → Trades execute (if connection recovers)

Expected outcome: Transient connection blips no longer lose trades; daemon self-heals within ~7 seconds.
