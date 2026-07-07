# Daemon Resilience & Auto-Reconnect

## Overview
Added automatic reconnection and retry logic to handle temporary socket disconnects with IBKR (TWS/Gateway) without losing trades or halting the daemon.

## Changes

### 1. Live Daemon (`markov_cli/live_daemon.py`)

**New Function:** `execute_signals_with_retry()`
- Wraps signal execution with automatic retry logic
- Max 3 attempts (configurable) with exponential backoff: 1s, 2s, 4s
- Detects connection errors:
  - `ConnectionError` (raised by IBKRAdapter)
  - `OSError`, `TimeoutError` (socket-level errors)
  - String patterns: "socket", "disconnect", "ib_insync" in exception message
- On connection error:
  1. Waits exponentially (2^attempt seconds)
  2. Disconnects broker
  3. Waits 0.5s
  4. Reconnects broker
  5. Retries signal execution
- On persistent failure (all retries exhausted):
  - Logs detailed error
  - Returns empty results (BUY=0, SELL=0, Skipped=all)
  - **Does NOT halt daemon** — continues to next cycle
  - Does NOT raise exception — daemon keeps running

**Integration:**
- Replaces direct `execute_signals()` call at line ~317
- Passes market_name and logger for better error context

### 2. IBKR Adapter (`broker/ibkr_adapter.py`)

**New Method:** `is_connected()`
- Checks if `_ib.isConnected()` without raising
- Returns `False` if IB object is `None` or any exception occurs
- Safe to call anytime

**Enhanced Method:** `disconnect()`
- Wrapped in try-except to suppress errors
- Safe to call even if already disconnected or in error state

**Enhanced Method:** `place_order()`
- Pre-flight check: raises `ConnectionError` if not connected before attempting order
- Wraps order logic in try-except
- On any exception, checks connection and re-raises as `ConnectionError` if socket is down
- Provides explicit error messages mentioning socket/connection context

## Behavior

### Before
```
09:01:45 [ERROR] Error executing signals: Socket disconnect
[trades skipped, no retry, daemon log shows error but continues]
```

### After
```
09:01:45 [ERROR] Socket error (attempt 1/3): Socket disconnect: not connected to 127.0.0.1:7497
09:01:45 [WARNING] Reconnecting in 1s...
09:01:46 [INFO] Broker reconnected successfully
09:01:46 [INFO] Executing signals for 23 processed tickers...
09:01:47 [INFO] BUY: 2, SELL: 1, Skipped: 20
[trades executed successfully]
```

## Testing

### To test reconnection:
```bash
# While daemon is running, kill TWS/Gateway connection (e.g., restart TWS)
# Observe daemon logs:
#   1. Detects socket error
#   2. Logs retry attempt
#   3. Waits and reconnects
#   4. Executes signals successfully
```

### Behavior on repeated connection failures:
```
[ERROR] Connection error (attempt 1/3): Socket disconnect...
[WARNING] Reconnecting in 1s...
[WARNING] Reconnect attempt failed: connection refused
[ERROR] Connection error (attempt 2/3): Socket disconnect...
[WARNING] Reconnecting in 2s...
[WARNING] Reconnect attempt failed: connection refused
[ERROR] Connection error (attempt 3/3): Socket disconnect...
[ERROR] Connection error persists after 3 attempts. Skipping signals for this cycle.
[INFO] Cycle done: 23 tickers processed, 0 skipped (budget), 15s elapsed
```

## Configuration

Retry behavior is hardcoded for robustness:
- `max_retries = 3` (adjustable in `execute_signals_with_retry()` call)
- Exponential backoff: 1s, 2s, 4s
- Pre-order connection check: `is_connected()` call in `place_order()`

To adjust, modify the call at `process_cycle()` line ~317:
```python
buys, sells, skipped = execute_signals_with_retry(
    ...,
    max_retries=5  # Change here
)
```

## Impact

- **No breaking changes** — all existing tests pass
- **Daemon stability** — socket errors no longer block cycles or lose state
- **Trade preservation** — successful orders before disconnect are saved; failed batch is retried
- **Logging clarity** — detailed retry/reconnect messages aid debugging
- **Manual intervention prevention** — common transient issues now self-heal
