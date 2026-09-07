$ErrorActionPreference = "Stop"
$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$original = Get-Content $strategyFile -Raw

function Set-Threshold { param([string]$v)
    $content = Get-Content $strategyFile -Raw
    $content = $content -replace "vix_entry_gate_threshold: float(\s*\|\s*None)? = \S+", ("vix_entry_gate_threshold: float | None = " + $v)
    Set-Content -Path $strategyFile -Value $content -NoNewline
}

$common = "--universe --strategies optimised_new --start-date 2023-01-01 --pot-sizes 10000 100000 --top-k 70 --vol-weight 0.7 --win-rate-weight 0.3 --lookback-days 60 --workers 4 --cost-model ibkr_tiered_spread --seasonal-volume"

try {
    Write-Host "`n=== baseline (no VIX gate) ==="
    Set-Threshold "None"
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim $common.Split(" ") --journal data/journals/fullhist_baseline.csv --position-summary data/journals/fullhist_baseline_equity.csv | Tee-Object -FilePath scripts/fullhist_baseline.log
    Write-Host "`n=== vix20 ==="
    Set-Threshold "20.0"
    uv run python -m Strategy_Auto_Trader.markov_cli.live_sim $common.Split(" ") --journal data/journals/fullhist_vix20.csv --position-summary data/journals/fullhist_vix20_equity.csv | Tee-Object -FilePath scripts/fullhist_vix20.log
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "Restored $strategyFile"
}