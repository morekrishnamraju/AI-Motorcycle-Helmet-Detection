# PowerShell Launcher for AI Helmet Detection System
# Automatically handles Execution Policy for this session

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  AI Helmet Detection & Rider Safety Monitoring System" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan

# Use virtual environment python directly (avoids Activate.ps1 execution policy restrictions)
$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[*] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

Write-Host "[*] Installing/Verifying dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")

Write-Host "[*] Verifying model weights & generating sample data..." -ForegroundColor Yellow
& $VenvPython (Join-Path $PSScriptRoot "models\download_weights.py")

Write-Host "[✓] Launching Streamlit Dashboard at http://localhost:8501 ..." -ForegroundColor Green
& $VenvPython -m streamlit run (Join-Path $PSScriptRoot "app.py")
