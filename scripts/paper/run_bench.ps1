<#
.SYNOPSIS
Runs the benchmark cell.
.DESCRIPTION
Builds in release mode and measures the timing and memory comparison set
the cell file declares. Pass -Quick or -Smoke for the shortened measurement
set.
#>
[CmdletBinding()]
param(
    [switch]$Quick,
    [switch]$Smoke,
    [string]$Out = "runs/bench",
    [string]$Config
)

$ErrorActionPreference = "Stop"
$arguments = @("run", "-p", "cosmos-paper", "--release", "--", "bench",
               "--out", $Out)
if ($Quick) { $arguments += "--quick" }
if ($Smoke) { $arguments += "--smoke" }
if ($Config) { $arguments += @("--config", $Config) }

Write-Host "cargo $($arguments -join ' ')"
& cargo @arguments
exit $LASTEXITCODE
