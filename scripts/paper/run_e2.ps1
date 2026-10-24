<#
.SYNOPSIS
Runs the gauge-regime grid.
.DESCRIPTION
Builds in release mode and sweeps every gauge learning rate, momentum and
seed the cell file declares. Pass -Smoke for the shortened form, which
checks itself against the documented reference and exits non-zero on any
deviation.
#>
[CmdletBinding()]
param(
    [switch]$Smoke,
    [string]$Out = "runs",
    [string]$DataDir = "data",
    [switch]$NoDownload,
    [int]$Threads = 1
)

$ErrorActionPreference = "Stop"
$arguments = @("run", "-p", "cosmos-paper", "--release", "--", "e2",
               "--out", $Out, "--data-dir", $DataDir, "--threads", $Threads)
if ($Smoke) { $arguments += "--smoke" }
if ($NoDownload) { $arguments += "--no-download" }

Write-Host "cargo $($arguments -join ' ')"
& cargo @arguments
exit $LASTEXITCODE
