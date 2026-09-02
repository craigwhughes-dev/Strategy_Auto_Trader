# Validate the strategy-owned same_day_deployment_cap_pct concentration gate
# (Strategy_Auto_Trader/strategy/optimised_new.py's OptimisedNewEntry) against
# the full-universe optimised_new backtest whose rolling-30d Sharpe ratio was
# found to swing wildly (2026-09-02 investigation: 22 same-day trade closes
# on 2025-03-11 during the worst stretch, corr(cluster size, Sharpe) = -0.37).
# Plan: C:\Users\Craig\.claude\plans\tender-chasing-iverson.md
#
# The cap is strategy-owned, not a CLI flag (see .claude/rules/strategy.md) —
# this script edits its value directly in optimised_new.py between runs,
# restoring the file to baseline (None/off) when done or on any failure.
#
# Sweeps cap_pct in {None (baseline), 0.10, 0.20, 0.35, 0.50} against the
# same £100k pot, full universe, top-k=70, 2024-11-21 start window used for
# the original diagnosis. Runs sequentially, ~600+ tickers per pass — this
# is built to run unattended overnight.
#
# Usage:
#   powershell -File scripts/run_same_day_cap_sweep.ps1
#
# After it finishes, compare results with:
#   uv run python scripts/analyze_same_day_cap_sweep.py

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$anchor = "same_day_deployment_cap_pct: float | None = "
$original = Get-Content $strategyFile -Raw

# label -> cap_pct value ($null = baseline, no cap)
$sweep = [ordered]@{
    "baseline" = $null
    "cap10"    = 0.10
    "cap20"    = 0.20
    "cap35"    = 0.35
    "cap50"    = 0.50
}

function Set-Cap {
    param([string]$PyValue)
    $content = Get-Content $strategyFile -Raw
    $pattern = [regex]::Escape($anchor) + '\S+'
    $updated = $content -replace $pattern, ($anchor + $PyValue)
    Set-Content -Path $strategyFile -Value $updated -NoNewline
}

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

try {
    foreach ($label in $sweep.Keys) {
        $capValue = $sweep[$label]
        $pyValue = if ($null -eq $capValue) { "None" } else { $capValue.ToString("0.00") }

        Write-Host "`n=== $label (same_day_deployment_cap_pct=$pyValue) ==="
        Set-Cap $pyValue

        $journal = "data/journals/same_day_cap_sweep_$label.csv"
        $posSummary = "data/journals/same_day_cap_sweep_${label}_equity.csv"
        $log = "scripts/same_day_cap_sweep_$label.log"

        # No 2>&1 here: redirecting a native command's stderr in PowerShell 5.1
        # wraps every stderr line (including benign warnings, e.g. IBKR
        # reqHistoricalData retry-timeout messages logged during a 600-ticker
        # fetch) in a terminating NativeCommandError under $ErrorActionPreference
        # = Stop, which aborted the very first run of this sweep on nothing
        # more than a normal transient warning. $LASTEXITCODE is the reliable
        # native-process success signal instead.
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
            --universe --strategies optimised_new `
            --initial-cash 100000 --top-k 70 --source ibkr --workers 4 `
            --start-date 2024-11-21 `
            --journal $journal --position-summary $posSummary `
            | Tee-Object -FilePath $log

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $label (exit $LASTEXITCODE, continuing to next cap value)"
        }
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to baseline (same_day_deployment_cap_pct=None)"
}

Write-Host "`nSweep complete. Compare results with:"
Write-Host "  uv run python scripts/analyze_same_day_cap_sweep.py"
