<#
.SYNOPSIS
Runs the margin-ablation and baseline cells.
.DESCRIPTION
Builds in release mode and runs the baseline cell file, then the
margin-ablation cell file. Pass -Smoke for the shortened form, which
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
$arguments = @("run", "-p", "cosmos-paper", "--release", "--", "e3",
               "--out", $Out, "--data-dir", $DataDir, "--threads", $Threads)
if ($Smoke) { $arguments += "--smoke" }
if ($NoDownload) { $arguments += "--no-download" }

Write-Host "cargo $($arguments -join ' ')"
& cargo @arguments
exit $LASTEXITCODE
