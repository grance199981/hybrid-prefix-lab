# Capture three kinds of kernels with Nsight Compute on a local Windows GPU.
#
# Reading GPU performance counters on Windows needs either the "Manage GPU
# Performance Counters" setting or an administrator terminal. This script is
# meant to be started from an administrator PowerShell; it changes no setting.
#
# What it does: runs tools\ncu_probe.py three times under `ncu --set full`
# (decode-like, prefill-like, attention), a few kernels each, and writes
#   <OutDir>\<case>.ncu-repz   the Nsight Compute report
#   <OutDir>\<case>.log        what ncu and the probe printed
# Nothing else is installed, changed or uploaded. About 5 minutes.
# Afterwards (no administrator needed), per case:
#   ncu --import <case>.ncu-repz --csv --page raw > <case>.raw.csv
# and then tools\ncu_extract.py on the three CSV files.
#
# Usage (administrator PowerShell):
#   powershell -ExecutionPolicy Bypass -File <repo>\experiments\e4_profiling\tools\run_ncu_local.ps1 [-Python <python.exe>]
param(
    [string]$Python = "python",             # an interpreter with torch for CUDA
    [string]$Ncu = "",                      # default: the newest ncu.bat under Program Files
    [string]$OutDir = "",                   # default: <repo>\ncu_reports
    [double]$GiB = 0.25
)
$ErrorActionPreference = "Stop"
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Host "Not an administrator terminal: ncu will report ERR_NVGPUCTRPERM. Open PowerShell with 'Run as administrator' and start this script again."
    exit 2
}
if (-not $Ncu) {
    $Ncu = Get-ChildItem "$env:ProgramFiles\NVIDIA Corporation\Nsight Compute*\ncu.bat" |
        Sort-Object FullName | Select-Object -Last 1 -ExpandProperty FullName
}
if (-not $Ncu -or -not (Test-Path $Ncu)) { Write-Host "ncu not found"; exit 3 }
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { Write-Host "python not found: $Python"; exit 3 }
if (-not $OutDir) {
    $root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
    $OutDir = Join-Path $root "ncu_reports"
}
New-Item -ItemType Directory -Force $OutDir | Out-Null
$probe = Join-Path $PSScriptRoot "ncu_probe.py"
$ErrorActionPreference = "Continue"
foreach ($case in "decode", "prefill", "attention") {
    Write-Host "== $case"
    & $Ncu --target-processes all --set full -f -o (Join-Path $OutDir $case) `
        $Python $probe --case $case --warmup 1 --iters 2 --gib $GiB *> (Join-Path $OutDir "$case.log")
    Get-Content (Join-Path $OutDir "$case.log") | Select-Object -Last 3
}
Write-Host "done. Reports are in $OutDir"
