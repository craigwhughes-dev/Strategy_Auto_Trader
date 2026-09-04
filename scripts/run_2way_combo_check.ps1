# Isolates which of the other 2 exit-parameter-audit changes (trend/sma200
# weights, RSI_OVERBOUGHT) is eating sell_threshold=-6.0's real-return edge
# when all 3 are combined (BACKTEST_LOG.md 2026-09-04 "combined winners"
# entry: alone +25.2% real return, combined +15.3%).
#
# combo_st_weights: sell_threshold=-6.0 + trend=1.0/sma200=3.0, RSI stays 70 (current)
# combo_st_rsi:     sell_threshold=-6.0 + RSI=60, weights stay trend=2.0/sma200=3.0 (current)
#
# Both windows, both configs. Compare against already-run current/sell_threshold-
# alone/combined-all-3 journals from run_combined_winners_check.ps1.
#
# Usage:
#   powershell -File scripts/run_2way_combo_check.ps1

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$original = Get-Content $strategyFile -Raw

$configs = [ordered]@{
    "combo_st_weights" = @{ sellT = -6.0; trend = 1.0; sma200 = 3.0; rsi = 70.0 }
    "combo_st_rsi"      = @{ sellT = -6.0; trend = 2.0; sma200 = 3.0; rsi = 60.0 }
}

function Set-Config {
    param([double]$SellThreshold, [double]$Trend, [double]$Sma200, [double]$Rsi)
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace 'sell_threshold: float = \S+', ("sell_threshold: float = " + $SellThreshold.ToString("0.0"))
    $updated = $updated -replace '"trend":\s*[\d.]+,', ('"trend":  ' + $Trend.ToString("0.0") + ',')
    $updated = $updated -replace '"sma200":\s*[\d.]+,', ('"sma200": ' + $Sma200.ToString("0.0") + ',')
    $updated = $updated -replace '_RSI_OVERBOUGHT = [\d.]+', ('_RSI_OVERBOUGHT = ' + $Rsi.ToString("0.0"))
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
    foreach ($label in $configs.Keys) {
        $c = $configs[$label]
        Write-Host "`n=== $label (sell_threshold=$($c.sellT), trend=$($c.trend), sma200=$($c.sma200), RSI_OVERBOUGHT=$($c.rsi)) ==="
        Set-Config -SellThreshold $c.sellT -Trend $c.trend -Sma200 $c.sma200 -Rsi $c.rsi

        $journalA = "data/journals/combo_2way_${label}_synthetic.csv"
        $equityA  = "data/journals/combo_2way_${label}_synthetic_equity.csv"
        $logA     = "scripts/combo_2way_${label}_synthetic.log"
        Write-Host "  [crash]"
        Invoke-LiveSimWithRetry -LiveSimArgs @(
            "--universe", "--strategies", "optimised_new",
            "--initial-cash", "100000", "--top-k", "70", "--workers", "4",
            "--start-date", "2008-01-01",
            "--synthetic-data-dir", "data_synthetic/hourly",
            "--synthetic-end-date", "2009-07-31",
            "--journal", $journalA, "--position-summary", $equityA
        ) -LogPath $logA

        $journalB = "data/journals/combo_2way_${label}_real.csv"
        $equityB  = "data/journals/combo_2way_${label}_real_equity.csv"
        $logB     = "scripts/combo_2way_${label}_real.log"
        Write-Host "  [real]"
        Invoke-LiveSimWithRetry -LiveSimArgs @(
            "--universe", "--strategies", "optimised_new",
            "--initial-cash", "100000", "--top-k", "70", "--source", "ibkr", "--workers", "4",
            "--start-date", "2024-09-03",
            "--journal", $journalB, "--position-summary", $equityB
        ) -LogPath $logB
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to true baseline (vol_stop_mult=1.0 adopted, everything else unchanged)"
}

Write-Host "`nDone."
