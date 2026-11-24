@echo off
title Setup - AI Helmet Detection System
color 0B

echo ===============================================================================
echo   Setup & Dependency Installer
echo   AI-Based Motorcycle Helmet Detection and Rider Safety Monitoring System
echo ===============================================================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo [✓] Existing virtual environment found.
    set TARGET_PY=.venv\Scripts\python.exe
    goto INSTALL_DEPS
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set PYTHON_CMD=py
) else (
    set PYTHON_CMD=python
)

echo [*] Initializing virtual environment in .venv ...
%PYTHON_CMD% -m venv .venv
if exist ".venv\Scripts\python.exe" (
    set TARGET_PY=.venv\Scripts\python.exe
) else (
    set TARGET_PY=%PYTHON_CMD%
)

:INSTALL_DEPS
echo [*] Upgrading pip...
%TARGET_PY% -m pip install --upgrade pip

echo [*] Installing all project dependencies from requirements.txt...
%TARGET_PY% -m pip install -r requirements.txt

echo [*] Downloading YOLOv8 model weights...
%TARGET_PY% models\download_weights.py

echo.
echo ===============================================================================
echo   [SUCCESS] Setup is complete!
echo   Double-click 'run.bat' to start the application.
echo ===============================================================================
echo.
pause
