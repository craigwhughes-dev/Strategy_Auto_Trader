$ErrorActionPreference = "Stop"

$common = @(
    "--universe"
    "--strategies", "optimised_new"
    "--start-date", "2000-01-01"
    "--synthetic-data-dir", "data_synthetic/hourly"
    "--synthetic-end-date", "2026-09-01"
    "--pot-sizes", "100000"
    "--top-k", "70"
    "--vol-weight", "0.7"
    "--win-rate-weight", "0.3"
    "--lookback-days", "60"
    "--workers", "4"
    "--cost-model", "ibkr_tiered_spread"
    "--seasonal-volume"
)

Write-Host "=== 26yr synthetic out-of-sample validation: optimised_new ==="
uv run python -m Strategy_Auto_Trader.markov_cli.live_sim @common `
    --journal data_synthetic/journals/synth_26yr.csv `
    --position-summary data_synthetic/journals/synth_26yr_equity.csv `
    | Tee-Object -FilePath scripts/synth_26yr.log
