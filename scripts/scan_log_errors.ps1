<#
.SYNOPSIS
    Scans the log directory for ERROR/WARNING lines in files modified within the last N days.

.PARAMETER Days
    Look back window in days. Default 1.

.PARAMETER LogDir
    Directory to scan. Default "logs" relative to repo root.

.EXAMPLE
    .\scan_log_errors.ps1 -Days 3
#>

param(
    [int]$Days = 1,
    [string]$LogDir = (Join-Path $PSScriptRoot "..\logs")
)

if (-not (Test-Path $LogDir)) {
    Write-Error "Log directory not found: $LogDir"
    exit 1
}

$cutoff = (Get-Date).AddDays(-$Days)
$pattern = 'ERROR|WARN|WARNING|Traceback|Exception'

$files = Get-ChildItem -Path $LogDir -Filter *.log -File -Recurse |
    Where-Object { $_.LastWriteTime -ge $cutoff }

if (-not $files) {
    Write-Host "No log files modified in the last $Days day(s) in $LogDir"
    exit 0
}

# Digits are masked so one recurring message (retry counters, timestamps,
# port numbers) collapses to a single signature.
$results = foreach ($file in $files) {
    $relPath = $file.FullName.Substring((Resolve-Path $LogDir).Path.Length + 1)
    Select-String -Path $file.FullName -Pattern $pattern -CaseSensitive:$false |
        Where-Object { $_.Line -notmatch 'NativeCommandError|FullyQualifiedErrorId\s*:\s*NativeCommandError|CategoryInfo.*NativeCommandError' } |
        Select-Object @{n='File';e={$relPath}}, LineNumber,
            @{n='Text';e={$_.Line.Trim()}},
            @{n='Signature';e={ ($_.Line.Trim() -replace '\d+(\.\d+)?', '#') }}
}

if (-not $results) {
    Write-Host "No errors/warnings found in $($files.Count) file(s) from the last $Days day(s)."
    exit 0
}

$summary = $results | Group-Object File, Signature | ForEach-Object {
    $first = $_.Group[0]
    $last = $_.Group[-1]
    [pscustomobject]@{
        Count  = $_.Count
        File   = $first.File
        Lines  = "$($first.LineNumber)-$($last.LineNumber)"
        Sample = $first.Text.Substring(0, [Math]::Min(160, $first.Text.Length))
    }
} | Sort-Object Count -Descending

$summary | Format-Table -AutoSize -Wrap
Write-Host "`n$($results.Count) match(es) in $(@($summary).Count) distinct message(s) across $(@($files | Group-Object Name).Count) file(s)."
