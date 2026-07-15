# Backtest all registered strategies against the 20 biggest S&P 500 stocks.
#
# Prep steps (run once, generates config/watchlist_sp20_<strategy>.json):
#   uv run python scripts/make_watchlist_variants.py --source config/watchlist_sp20.json `
#     default conservative trend optimised optimised_aggressive choppy_vol mean_reversion breakout_momentum ai
#
# Then run this script. Each strategy is a separate batch.py pass (9 passes x 20 tickers
# = 180 runs), writing timestamped run dirs under data/ per cli.md's backtest contract.
#
# Usage:
#   powershell -File scripts/run_sp20_backtest.ps1

$strategies = @(
    "default", "conservative", "trend", "optimised", "optimised_aggressive",
    "choppy_vol", "mean_reversion", "breakout_momentum", "ai"
)

foreach ($strat in $strategies) {
    $watchlist = "config/watchlist_sp20_$strat.json"
    if (-not (Test-Path $watchlist)) {
        Write-Host "SKIP $strat - $watchlist not found (run make_watchlist_variants.py first)"
        continue
    }
    Write-Host "=== $strat ==="
    uv run python -m Strategy_Auto_Trader.markov_cli.batch --watchlist $watchlist --no-email
    if (-not $?) {
        Write-Host "FAILED: $strat (continuing)"
    }
}
