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

$files = Get-ChildItem -Path $LogDir -Filter *.log -File |
    Where-Object { $_.LastWriteTime -ge $cutoff }

if (-not $files) {
    Write-Host "No log files modified in the last $Days day(s) in $LogDir"
    exit 0
}

$results = foreach ($file in $files) {
    Select-String -Path $file.FullName -Pattern $pattern -CaseSensitive:$false |
        Select-Object @{n='File';e={$file.Name}}, LineNumber, @{n='Text';e={$_.Line.Trim()}}
}

if (-not $results) {
    Write-Host "No errors/warnings found in $($files.Count) file(s) from the last $Days day(s)."
    exit 0
}

$results | Format-Table -AutoSize -Wrap
Write-Host "`n$($results.Count) match(es) across $($files | Group-Object { $_.Name } | Measure-Object).Count file(s)."
