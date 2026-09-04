# Combined check: the 3 exit-parameter-audit items validated on both windows
# but not yet applied (sell_threshold, trend/sma200 weights, _RSI_OVERBOUGHT)
# tested TOGETHER as one config, vs current, both windows. Each item was only
# tested in isolation (one changed at a time) — this checks they don't
# interact badly before any of them gets adopted.
#
# combined: sell_threshold=-6.0, trend=1.0, sma200=3.0, _RSI_OVERBOUGHT=60.0
#   (the outright-best value for each item; 65 was noted as a milder
#   RSI alternative but 60 is what's being checked here)
# current:  sell_threshold=-4.5, trend=2.0, sma200=3.0, _RSI_OVERBOUGHT=70.0
#   (baseline, all current values — re-run here so the comparison is against
#   this exact universe/window pull, not an older journal)
#
# vol_stop_mult stays at its already-adopted 1.0 in both configs (untouched).
#
# Usage:
#   powershell -File scripts/run_combined_winners_check.ps1

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$original = Get-Content $strategyFile -Raw

$configs = [ordered]@{
    "combined" = @{ sellT = -6.0; trend = 1.0; sma200 = 3.0; rsi = 60.0 }
    "current"  = @{ sellT = -4.5; trend = 2.0; sma200 = 3.0; rsi = 70.0 }
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

        # crash window
        $journalA = "data/journals/combined_winners_${label}_synthetic.csv"
        $equityA  = "data/journals/combined_winners_${label}_synthetic_equity.csv"
        $logA     = "scripts/combined_winners_${label}_synthetic.log"
        Write-Host "  [crash]"
        Invoke-LiveSimWithRetry -LiveSimArgs @(
            "--universe", "--strategies", "optimised_new",
            "--initial-cash", "100000", "--top-k", "70", "--workers", "4",
            "--start-date", "2008-01-01",
            "--synthetic-data-dir", "data_synthetic/hourly",
            "--synthetic-end-date", "2009-07-31",
            "--journal", $journalA, "--position-summary", $equityA
        ) -LogPath $logA

        # real window
        $journalB = "data/journals/combined_winners_${label}_real.csv"
        $equityB  = "data/journals/combined_winners_${label}_real_equity.csv"
        $logB     = "scripts/combined_winners_${label}_real.log"
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

Write-Host "`nDone. Compare combined_winners_combined_* vs combined_winners_current_* equity/journal CSVs."
