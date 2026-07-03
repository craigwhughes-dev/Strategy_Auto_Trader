# Strategy_Auto_Trader

Hourly HMM + composite-signal trading strategy backtester and live simulator.

## Entry points
- `Strategy_Auto_Trader/markov_cli/run.py` — main backtest CLI (`uv run python -m Strategy_Auto_Trader.markov_cli.run --ticker SPY`)
- `Strategy_Auto_Trader/markov_cli/live_sim.py` — multi-ticker live simulation

## Strategies
- `Strategy_Auto_Trader/strategy/base/registry.py` — `STRATEGY_REGISTRY`, maps name -> Entry/Exit classes
- `Strategy_Auto_Trader/strategy/{default,conservative,trend_follow}.py` — individual strategies

## Tests
`python -m pytest tests/test_all.py -q --tb=short`

## Work tracking
- `HANDOFF.md` — state left by the last session, read this first when resuming work
- `todo.md` — longer-running roadmap/open items

## Do not read unless specifically asked
`data/`, `logs/`, `reports/`, `state/`, `.venv/`, `uv.lock`, `*.egg-info/` — generated/runtime artifacts, already gitignored and excluded via `.claudeignore` / `permissions.deny`.
