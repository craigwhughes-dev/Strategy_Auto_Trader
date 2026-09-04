# Runs the remaining 5 items of the 6-item exit-parameter audit (HANDOFF.md
# 2026-09-03) in sequence: #1 min_hold_bars, #3 sell_threshold, #4 trend/
# sma200 weight grid, #5 profit_stop_scale/min_stop_pct, #6 _RSI_OVERBOUGHT.
# Item #2 (vol_stop_mult) already done, tightened 2.0 -> 1.0, adopted.
#
# Each parameter's sweep edits optimised_new.py directly between runs (no CLI
# flag for these — same pattern as every prior sweep in this project), always
# restoring to the CURRENT baseline (captured once at the top of this script,
# so it already includes the adopted vol_stop_mult=1.0) between parameters —
# never carrying one parameter's temporary sweep value into the next
# parameter's sweep.
#
# Retries each individual live_sim invocation up to 4 attempts on non-zero
# exit — the live StrategyAutoTraderDaemon task runs continuously and
# periodically collides on the shared data/cache/hmm_cache dir (a different
# ticker's .tmp->.pkl replace each time, confirmed transient across today's
# earlier sweeps, not a bug in this code).
#
# Usage:
#   powershell -File scripts/run_exit_param_audit.ps1
#
# After it finishes, each parameter has its own analyze_<name>_sweep.py.

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$trueBaseline = Get-Content $strategyFile -Raw

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

function Restore-Baseline {
    Set-Content -Path $strategyFile -Value $trueBaseline -NoNewline
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

function Run-CrashWindow {
    param([string]$JournalPrefix)
    $journal = "data/journals/${JournalPrefix}_synthetic.csv"
    $equity  = "data/journals/${JournalPrefix}_synthetic_equity.csv"
    $log     = "scripts/${JournalPrefix}_synthetic.log"
    Write-Host "  [crash] $JournalPrefix"
    Invoke-LiveSimWithRetry -LiveSimArgs @(
        "--universe", "--strategies", "optimised_new",
        "--initial-cash", "100000", "--top-k", "70", "--workers", "4",
        "--start-date", "2008-01-01",
        "--synthetic-data-dir", "data_synthetic/hourly",
        "--synthetic-end-date", "2009-07-31",
        "--journal", $journal, "--position-summary", $equity
    ) -LogPath $log
}

function Run-RealWindow {
    param([string]$JournalPrefix, [string]$StartDate = "2024-09-03")
    $journal = "data/journals/${JournalPrefix}_real.csv"
    $equity  = "data/journals/${JournalPrefix}_real_equity.csv"
    $log     = "scripts/${JournalPrefix}_real.log"
    Write-Host "  [real] $JournalPrefix"
    Invoke-LiveSimWithRetry -LiveSimArgs @(
        "--universe", "--strategies", "optimised_new",
        "--initial-cash", "100000", "--top-k", "70", "--source", "ibkr", "--workers", "4",
        "--start-date", $StartDate,
        "--journal", $journal, "--position-summary", $equity
    ) -LogPath $log
}

# ============================================================
# #1 min_hold_bars: {0, 3, 6, 48 (current)}, both windows
# ============================================================
Write-Host "`n########## #1 min_hold_bars ##########"
$minHoldSweep = [ordered]@{ "mhb0" = 0; "mhb3" = 3; "mhb6" = 6; "mhb48" = 48 }
foreach ($label in $minHoldSweep.Keys) {
    $v = $minHoldSweep[$label]
    Write-Host "`n=== $label (min_hold_bars=$v) ==="
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace 'min_hold_bars: int = \S+', "min_hold_bars: int = $v"
    Set-Content -Path $strategyFile -Value $updated -NoNewline
    Run-CrashWindow -JournalPrefix "min_hold_sweep_$label"
    Run-RealWindow -JournalPrefix "min_hold_sweep_$label"
}
Restore-Baseline

# ============================================================
# #3 sell_threshold: {-6.0, -4.5 (current), -3.0, -1.5}, both windows
# ============================================================
Write-Host "`n########## #3 sell_threshold ##########"
$sellThreshSweep = [ordered]@{ "st6" = -6.0; "st45" = -4.5; "st3" = -3.0; "st15" = -1.5 }
foreach ($label in $sellThreshSweep.Keys) {
    $v = $sellThreshSweep[$label]
    Write-Host "`n=== $label (sell_threshold=$v) ==="
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace 'sell_threshold: float = \S+', "sell_threshold: float = $v"
    Set-Content -Path $strategyFile -Value $updated -NoNewline
    Run-CrashWindow -JournalPrefix "sell_thresh_sweep_$label"
    Run-RealWindow -JournalPrefix "sell_thresh_sweep_$label"
}
Restore-Baseline

# ============================================================
# #4 trend/sma200 weight grid: trend in {1,2,3}, sma200 in {2,3,4},
# real window only (prev-2yr)
# ============================================================
Write-Host "`n########## #4 trend/sma200 weight grid ##########"
$trendVals = @(1.0, 2.0, 3.0)
$sma200Vals = @(2.0, 3.0, 4.0)
foreach ($t in $trendVals) {
    foreach ($s in $sma200Vals) {
        $label = "t$([int]$t)s$([int]$s)"
        Write-Host "`n=== $label (trend=$t, sma200=$s) ==="
        $content = Get-Content $strategyFile -Raw
        $updated = $content -replace '"trend":\s*[\d.]+,', ('"trend":  ' + $t.ToString("0.0") + ',')
        $updated = $updated -replace '"sma200":\s*[\d.]+,', ('"sma200": ' + $s.ToString("0.0") + ',')
        Set-Content -Path $strategyFile -Value $updated -NoNewline
        Run-RealWindow -JournalPrefix "weight_grid_sweep_$label"
    }
}
Restore-Baseline

# ============================================================
# #5 profit_stop_scale/min_stop_pct: current (0.30/0.03) vs off
# (0.0/0.04, optimised's original), both windows
# ============================================================
Write-Host "`n########## #5 profit_stop_scale/min_stop_pct ##########"
$ratchetSweep = [ordered]@{
    "ratchet_current" = @{ pss = 0.30; msp = 0.03 }
    "ratchet_off"      = @{ pss = 0.0;  msp = 0.04 }
}
foreach ($label in $ratchetSweep.Keys) {
    $pss = $ratchetSweep[$label].pss
    $msp = $ratchetSweep[$label].msp
    Write-Host "`n=== $label (profit_stop_scale=$pss, min_stop_pct=$msp) ==="
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace '"profit_stop_scale":\s*[\d.]+,', ('"profit_stop_scale": ' + $pss.ToString("0.00") + ',')
    $updated = $updated -replace '"min_stop_pct":\s*[\d.]+,', ('"min_stop_pct": ' + $msp.ToString("0.00") + ',')
    Set-Content -Path $strategyFile -Value $updated -NoNewline
    Run-CrashWindow -JournalPrefix "ratchet_sweep_$label"
    Run-RealWindow -JournalPrefix "ratchet_sweep_$label"
}
Restore-Baseline

# ============================================================
# #6 _RSI_OVERBOUGHT: {60, 65, 70 (current), 75, 999 (no veto)},
# real window only
# ============================================================
Write-Host "`n########## #6 _RSI_OVERBOUGHT ##########"
$rsiSweep = [ordered]@{ "rsi60" = 60.0; "rsi65" = 65.0; "rsi70" = 70.0; "rsi75" = 75.0; "rsiOff" = 999.0 }
foreach ($label in $rsiSweep.Keys) {
    $v = $rsiSweep[$label]
    Write-Host "`n=== $label (_RSI_OVERBOUGHT=$v) ==="
    $content = Get-Content $strategyFile -Raw
    $updated = $content -replace '_RSI_OVERBOUGHT = [\d.]+', ('_RSI_OVERBOUGHT = ' + $v.ToString("0.0"))
    Set-Content -Path $strategyFile -Value $updated -NoNewline
    Run-RealWindow -JournalPrefix "rsi_overbought_sweep_$label"
}
Restore-Baseline

Write-Host "`n`n########## ALL 5 SWEEPS COMPLETE ##########"
Write-Host "optimised_new.py restored to true baseline (vol_stop_mult=1.0, everything else unchanged)."
