@echo off
title AI Helmet Detection System - Full Stack
color 0A

:: Ensure virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [!] Virtual environment not found. Creating one...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [!] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

set VENV_PYTHON=.venv\Scripts\python.exe

:: Upgrade pip and install dependencies
%VENV_PYTHON% -m pip install --upgrade pip >nul
%VENV_PYTHON% -m pip install -r requirements.txt >nul

:: Verify model weights
echo [✓] Checking model weights...
%VENV_PYTHON% models\download_weights.py

:: Start Flask API in background
start "Flask API" /B %VENV_PYTHON% server.py

:: Give Flask a couple of seconds to initialise
ping -n 5 127.0.0.1 >nul

:: Launch Streamlit UI
echo [✓] Launching Streamlit UI at http://localhost:8501
%VENV_PYTHON% -m streamlit run app.py

pause