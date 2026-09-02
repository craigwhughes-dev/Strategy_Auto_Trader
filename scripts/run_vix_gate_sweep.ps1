# Sweep vix_entry_gate_threshold levels for optimised_new against two windows:
#   A) Jan2008-Jul2009 synthetic crash  — measures crash protection
#   B) Nov2024-Aug2026 real data        — measures cost to normal upside
#
# Baseline (None = gate off) is included first so all windows have a reference.
# Thresholds tried: None, 20, 25, 30, 35, 40
#   20 = very aggressive (blocks even "elevated" VIX)
#   25 = normal→high_vol boundary (current default on OptimisedNewEntry)
#   30 = only clearly high-vol days blocked
#   35 = only crisis-level VIX blocked
#   40 = extreme-crisis only
#
# The attribute is strategy-owned, not a CLI flag (.claude/rules/strategy.md).
# This script edits it directly in optimised_new.py between runs, restoring
# the original value on completion or failure.
#
# Usage:
#   powershell -File scripts/run_vix_gate_sweep.ps1
#
# After it finishes, compare results with:
#   uv run python scripts/analyze_vix_gate_sweep.py

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$anchor = "vix_entry_gate_threshold: float = "
$original = Get-Content $strategyFile -Raw

# label -> threshold value ($null = baseline, gate off)
$sweep = [ordered]@{
    "baseline" = $null
    "vix20"    = 20.0
    "vix25"    = 25.0
    "vix30"    = 30.0
    "vix35"    = 35.0
    "vix40"    = 40.0
}

function Set-Threshold {
    param([string]$PyValue)
    $content = Get-Content $strategyFile -Raw
    if ($PyValue -eq "None") {
        # Swap the float attribute line for a None-typed one
        $content = $content -replace "vix_entry_gate_threshold: float = \S+", "vix_entry_gate_threshold: float | None = None"
    } else {
        # Ensure the line is the float variant
        $content = $content -replace "vix_entry_gate_threshold: float(\s*\|\s*None)? = \S+", ("vix_entry_gate_threshold: float = " + $PyValue)
    }
    Set-Content -Path $strategyFile -Value $content -NoNewline
}

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

try {
    foreach ($label in $sweep.Keys) {
        $threshold = $sweep[$label]
        $pyValue = if ($null -eq $threshold) { "None" } else { $threshold.ToString("0.0") }

        Write-Host "`n=== $label (vix_entry_gate_threshold=$pyValue) ==="
        Set-Threshold $pyValue

        # --- Window A: synthetic 2008 crash ---
        $journalA   = "data/journals/vix_sweep_${label}_synthetic.csv"
        $equityA    = "data/journals/vix_sweep_${label}_synthetic_equity.csv"
        $logA       = "scripts/vix_sweep_${label}_synthetic.log"

        Write-Host "  [A] Synthetic 2008 crash window..."
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
            --universe --strategies optimised_new `
            --initial-cash 100000 --top-k 70 --workers 4 `
            --start-date 2008-01-01 `
            --synthetic-data-dir data_synthetic/hourly `
            --synthetic-end-date 2009-07-31 `
            --journal $journalA --position-summary $equityA `
            | Tee-Object -FilePath $logA

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED window A: $label (exit $LASTEXITCODE)"
        }

        # --- Window B: real data 2024-Nov to present ---
        $journalB   = "data/journals/vix_sweep_${label}_real.csv"
        $equityB    = "data/journals/vix_sweep_${label}_real_equity.csv"
        $logB       = "scripts/vix_sweep_${label}_real.log"

        Write-Host "  [B] Real 2024-Nov window..."
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
            --universe --strategies optimised_new `
            --initial-cash 100000 --top-k 70 --source ibkr --workers 4 `
            --start-date 2024-11-21 `
            --journal $journalB --position-summary $equityB `
            | Tee-Object -FilePath $logB

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED window B: $label (exit $LASTEXITCODE)"
        }
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to baseline (vix_entry_gate_threshold=25.0)"
}

Write-Host "`nSweep complete. Compare results with:"
Write-Host "  uv run python scripts/analyze_vix_gate_sweep.py"
