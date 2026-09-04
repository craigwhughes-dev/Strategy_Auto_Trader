# Validate the strategy-owned max_correlation_to_admitted_today gate
# (Strategy_Auto_Trader/strategy/optimised_new.py's OptimisedNewEntry) against
# the same rolling-30d-Sharpe-volatility problem the (rejected)
# same_day_deployment_cap_pct $-cap sweep already tried and failed to fix
# (2026-09-02 investigation: 22 same-day trade closes on 2025-03-11 during the
# worst stretch, corr(cluster size, Sharpe) = -0.37). Sector-bucket gating was
# considered as a simpler alternative and ruled out first (2026-09-03 spot
# check: the largest same-day clusters span 15+ sectors — a per-sector cap
# wouldn't bind on the real problem days). A direct correlation check on the
# same clusters showed real elevation vs random tickers instead (~2x on some
# days), so this sweep tests a trailing-return-correlation admission gate.
# Plan: C:\Users\Craig\.claude\plans\cosmic-bouncing-pretzel.md
#
# The cap is strategy-owned, not a CLI flag (see .claude/rules/strategy.md) —
# this script edits its value directly in optimised_new.py between runs,
# restoring the file to baseline (None/off) when done or on any failure.
#
# Sweeps max_correlation_to_admitted_today in {None (baseline), 0.3, 0.5, 0.7}
# against the SAME £100k pot, full universe, top-k=70, 2024-11-21 start window
# used for the $-cap sweep — apples-to-apples against the mechanism that
# already failed this exact test. Runs sequentially, ~600+ tickers per pass —
# built to run unattended overnight.
#
# Usage:
#   powershell -File scripts/run_correlation_cap_sweep.ps1
#
# After it finishes, compare results with:
#   uv run python scripts/analyze_correlation_cap_sweep.py

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$anchor = "max_correlation_to_admitted_today: float | None = "
$original = Get-Content $strategyFile -Raw

# label -> threshold value ($null = baseline, no gate)
$sweep = [ordered]@{
    "baseline" = $null
    "corr30"   = 0.30
    "corr50"   = 0.50
    "corr70"   = 0.70
}

function Set-Threshold {
    param([string]$PyValue)
    $content = Get-Content $strategyFile -Raw
    $pattern = [regex]::Escape($anchor) + '\S+'
    $updated = $content -replace $pattern, ($anchor + $PyValue)
    Set-Content -Path $strategyFile -Value $updated -NoNewline
}

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

try {
    foreach ($label in $sweep.Keys) {
        $threshold = $sweep[$label]
        $pyValue = if ($null -eq $threshold) { "None" } else { $threshold.ToString("0.00") }

        Write-Host "`n=== $label (max_correlation_to_admitted_today=$pyValue) ==="
        Set-Threshold $pyValue

        $journal = "data/journals/correlation_cap_sweep_$label.csv"
        $posSummary = "data/journals/correlation_cap_sweep_${label}_equity.csv"
        $log = "scripts/correlation_cap_sweep_$label.log"

        # No 2>&1 here: redirecting a native command's stderr in PowerShell 5.1
        # wraps every stderr line (including benign warnings, e.g. IBKR
        # reqHistoricalData retry-timeout messages logged during a 600-ticker
        # fetch) in a terminating NativeCommandError under $ErrorActionPreference
        # = Stop, which aborted the same_day_cap sweep's first run on nothing
        # more than a normal transient warning. $LASTEXITCODE is the reliable
        # native-process success signal instead.
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
            --universe --strategies optimised_new `
            --initial-cash 100000 --top-k 70 --source ibkr --workers 4 `
            --start-date 2024-11-21 `
            --journal $journal --position-summary $posSummary `
            | Tee-Object -FilePath $log

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $label (exit $LASTEXITCODE, continuing to next threshold)"
        }
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to baseline (max_correlation_to_admitted_today=None)"
}

Write-Host "`nSweep complete. Compare results with:"
Write-Host "  uv run python scripts/analyze_correlation_cap_sweep.py"
