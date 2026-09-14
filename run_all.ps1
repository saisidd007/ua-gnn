<#
PowerShell convenience wrapper to run the project's run_all.py.

Usage (PowerShell):
    .venv\Scripts\Activate.ps1
    .\run_all.ps1

This script just invokes the Python wrapper; edit the Python script for behavior.
#>
param()

Write-Host "Activating existing venv is recommended before running this script.`nRun: .venv\Scripts\Activate.ps1" -ForegroundColor Yellow

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "python not found in PATH; ensure venv activated or python is installed." -ForegroundColor Red
    exit 1
}

python .\scripts\run_all.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "run_all.py exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "run_all completed (exit code 0)" -ForegroundColor Green
