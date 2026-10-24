<#
.SYNOPSIS
Runs the flat-versus-patchwise comparison cells.
.DESCRIPTION
Builds in release mode and runs every flat-versus-patchwise comparison cell
file in order. Pass -Smoke for the shortened form, which checks itself
against the documented reference and exits non-zero on any deviation.
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
$arguments = @("run", "-p", "cosmos-paper", "--release", "--", "e5",
               "--out", $Out, "--data-dir", $DataDir, "--threads", $Threads)
if ($Smoke) { $arguments += "--smoke" }
if ($NoDownload) { $arguments += "--no-download" }

Write-Host "cargo $($arguments -join ' ')"
& cargo @arguments
exit $LASTEXITCODE
