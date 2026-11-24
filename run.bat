@echo off
title AI Motorcycle Helmet Detection ^& Rider Safety Monitoring System
color 0A

:: Check existing virtual environment
if exist "%CD%\.venv\Scripts\python.exe" (
    set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
    set ULTRALYTICS_HOME=%CD%\data
    echo [✓] Found established virtual environment (.venv).
) else (
    echo [ERROR] Virtual environment not found. Creating one...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [!] Could not create virtual environment. Exiting.
        exit /b 1
    ) else (
        set "VENV_PYTHON=%CD%\.venv\Scripts\python.exe"
        set ULTRALYTICS_HOME=%CD%\data
    )
)

:: Install / verify dependencies
%VENV_PYTHON% -m pip install --upgrade pip
%VENV_PYTHON% -m pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [!] Package installation issues, attempting direct install of core packages...
    %VENV_PYTHON% -m pip install streamlit ultralytics opencv-python-headless pillow numpy pandas plotly
)

:: Verify model weights
%VENV_PYTHON% models\download_weights.py

:: Launch Flask backend (which serves the full UI)
%VENV_PYTHON% server.py

pause