<#
.SYNOPSIS
Measures the resident-set size of a driver run from outside the process.
.DESCRIPTION
Starts the given command, samples its working set every -IntervalSeconds
until it exits, and writes poll_log.csv and summary.json into -Out. The
measurement is an operating-system measurement, not a library one.
.EXAMPLE
.\rss_probe.ps1 -Probe cosmos_flat -Out runs\rss\cosmos_flat -- `
  cargo run -p cosmos-paper --release -- e3 --smoke --no-download
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Probe,
    [Parameter(Mandatory = $true)][string]$Out,
    [double]$IntervalSeconds = 2.0,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Command
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

$log = Join-Path $Out "poll_log.csv"
Set-Content -Path $log -Encoding utf8 -Value "timestamp_iso,elapsed_s,workingset_bytes,workingset_mb"

$started = Get-Date
$process = Start-Process -FilePath $Command[0] -ArgumentList ($Command[1..($Command.Length - 1)]) -PassThru -NoNewWindow
$peak = 0
$samples = 0

while (-not $process.HasExited) {
    try {
        $process.Refresh()
        $bytes = $process.WorkingSet64
        if ($bytes -gt $peak) { $peak = $bytes }
        $elapsed = ((Get-Date) - $started).TotalSeconds
        $line = "{0},{1:F3},{2},{3:F2}" -f (Get-Date -Format "o"), $elapsed, $bytes, ($bytes / 1MB)
        Add-Content -Path $log -Encoding utf8 -Value $line
        $samples++
    } catch { }
    Start-Sleep -Seconds $IntervalSeconds
}
$process.WaitForExit()

$summary = [ordered]@{
    probe          = $Probe
    cli            = ($Command -join " ")
    peak_rss_bytes = $peak
    peak_rss_gb    = [math]::Round($peak / 1GB, 4)
    n_samples      = $samples
    duration_s     = [math]::Round(((Get-Date) - $started).TotalSeconds, 3)
    exit_code      = $process.ExitCode
}
$summary | ConvertTo-Json | Set-Content -Path (Join-Path $Out "summary.json") -Encoding utf8
exit $process.ExitCode
