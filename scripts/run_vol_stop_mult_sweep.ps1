# Sweep vol_stop_mult (the vol-scaled trailing-stop multiplier) for
# optimised_new against two windows, mirroring run_vix_gate_sweep.ps1's shape:
#   A) Jan2008-Jul2009 synthetic crash — does a tighter multiplier reduce the
#      rr_stop_loss-dominated whipsaw loss found in the 2026-09-02 crash
#      stress test (429/765 trades, -£83k, median hold 1-2 days)?
#   B) prev-2yr real data (2024-09-03 to present) — cost/benefit in normal
#      markets.
#
# Item #2 of the 6-item exit-parameter audit (HANDOFF.md 2026-09-03): the
# effective trailing-stop distance is vol_stop_mult * realised_vol *
# sqrt(vol_stop_window) — it WIDENS during high-vol periods, the opposite of
# what you'd want in a crash. vol_stop_window held fixed at 20 (unchanged) —
# this sweep isolates the multiplier only, the more direct "how wide" lever.
#
# baseline (2.0 = current) included first so all windows have a reference.
# Values tried: 1.0, 1.5, 2.0 (current), 2.5
#
# The attribute is a nested-dict default inside OptimisedNewExit.__init__,
# not a class attribute (no CLI flag — .claude/rules/strategy.md's Strategy-
# Owned Admission Attributes section covers ADMISSION knobs; this is an EXIT
# knob, always been per-strategy code, same edit-between-runs approach as
# every other sweep in this project). Restores optimised_new.py on exit.
#
# Usage:
#   powershell -File scripts/run_vol_stop_mult_sweep.ps1
#
# After it finishes, compare results with:
#   uv run python scripts/analyze_vol_stop_mult_sweep.py

$ErrorActionPreference = "Stop"

$strategyFile = "Strategy_Auto_Trader/strategy/optimised_new.py"
$original = Get-Content $strategyFile -Raw

$sweep = [ordered]@{
    "mult10" = 1.0
    "mult15" = 1.5
    "mult20" = 2.0
    "mult25" = 2.5
}

function Set-Mult {
    param([double]$Value)
    $content = Get-Content $strategyFile -Raw
    $pyValue = $Value.ToString("0.0")
    $updated = $content -replace '"vol_stop_mult":\s*[\d.]+,', ('"vol_stop_mult": ' + $pyValue + ',')
    Set-Content -Path $strategyFile -Value $updated -NoNewline
}

New-Item -ItemType Directory -Force -Path "data/journals" | Out-Null

try {
    foreach ($label in $sweep.Keys) {
        $value = $sweep[$label]

        Write-Host "`n=== $label (vol_stop_mult=$value) ==="
        Set-Mult $value

        # --- Window A: synthetic 2008 crash ---
        $journalA = "data/journals/vol_stop_mult_sweep_${label}_synthetic.csv"
        $equityA  = "data/journals/vol_stop_mult_sweep_${label}_synthetic_equity.csv"
        $logA     = "scripts/vol_stop_mult_sweep_${label}_synthetic.log"

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

        # --- Window B: real data, prev 2 years ---
        $journalB = "data/journals/vol_stop_mult_sweep_${label}_real.csv"
        $equityB  = "data/journals/vol_stop_mult_sweep_${label}_real_equity.csv"
        $logB     = "scripts/vol_stop_mult_sweep_${label}_real.log"

        Write-Host "  [B] Real prev-2yr window..."
        uv run python -m Strategy_Auto_Trader.markov_cli.live_sim `
            --universe --strategies optimised_new `
            --initial-cash 100000 --top-k 70 --source ibkr --workers 4 `
            --start-date 2024-09-03 `
            --journal $journalB --position-summary $equityB `
            | Tee-Object -FilePath $logB

        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED window B: $label (exit $LASTEXITCODE)"
        }
    }
} finally {
    Set-Content -Path $strategyFile -Value $original -NoNewline
    Write-Host "`nRestored $strategyFile to baseline (vol_stop_mult=2.0)"
}

Write-Host "`nSweep complete. Compare results with:"
Write-Host "  uv run python scripts/analyze_vol_stop_mult_sweep.py"
