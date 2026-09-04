# Follow-up to item #4 of the exit-parameter audit (trend/sma200 weight
# grid, BACKTEST_LOG.md 2026-09-04): the grid was real-window-only and not
# monotonic, so its standout cell (t1s3: trend=1.0, sma200=3.0, +20.2% real
# return vs current t2s3's +12.2%) could be a single-window fluke. Runs the
# crash window for BOTH t1s3 and current (t2s3) for direct comparison —
# no crash-window weight-grid data existed before this.
#
# Usage:
#   powershell -File scripts/run_weight_grid_crash_followup.ps1

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$original = Get-Content $strategyFile -Raw

$sweep = [ordered]@{
    "t1s3" = @{ trend = 1.0; sma200 = 3.0 }
    "t2s3" = @{ trend = 2.0; sma200 = 3.0 }  # current
}

function Set-Weights {
    param([double]$Trend, [double]$Sma200)
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace '"trend":\s*[\d.]+,', ('"trend":  ' + $Trend.ToString("0.0") + ',')
    $updated = $updated -replace '"sma200":\s*[\d.]+,', ('"sma200": ' + $Sma200.ToString("0.0") + ',')
    Set-Content -Path $strategyFile -Value $updated -NoNewline
}

function Invoke-LiveSimWithRetry {
    param([string[]]$LiveSimArgs, [string]$LogPath, [int]$MaxAttempts = 4)
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim @LiveSimArgs | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -eq 0) { return $true }
        Write-Host "    attempt $i failed (exit $LASTEXITCODE)$(if ($i -lt $MaxAttempts) { ' -- retrying' } else { ' -- giving up' })"
    }
    return $false
}

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

try {
    foreach ($label in $sweep.Keys) {
        $trend = $sweep[$label].trend
        $sma200 = $sweep[$label].sma200
        Write-Host "`n=== $label (trend=$trend, sma200=$sma200) - crash window ==="
        Set-Weights $trend $sma200

        $journal = "data/journals/weight_grid_sweep_${label}_synthetic.csv"
        $equity  = "data/journals/weight_grid_sweep_${label}_synthetic_equity.csv"
        $log     = "scripts/weight_grid_sweep_${label}_synthetic.log"

        Invoke-LiveSimWithRetry -LiveSimArgs @(
            "--universe", "--strategies", "optimised_new",
            "--initial-cash", "100000", "--top-k", "70", "--workers", "4",
            "--start-date", "2008-01-01",
            "--synthetic-data-dir", "data_synthetic/hourly",
            "--synthetic-end-date", "2009-07-31",
            "--journal", $journal, "--position-summary", $equity
        ) -LogPath $log
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to baseline"
}

Write-Host "`nDone. Compare data/journals/weight_grid_sweep_t1s3_synthetic_equity.csv vs weight_grid_sweep_t2s3_synthetic_equity.csv"
