@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

echo ======================================
echo   Machine Learning Platform - Startup Script
echo ======================================
echo.

REM Set project directory
set PROJECT_DIR=%~dp0
cd /d "%PROJECT_DIR%"

echo [INFO] Project directory: %PROJECT_DIR%
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 or higher
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version

REM Check uv installation
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] uv not found, installing uv...
    python -m pip install uv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install uv
        echo [TIP] Try manual installation: pip install uv
        pause
        exit /b 1
    )
    echo [OK] uv installed successfully
) else (
    echo [OK] uv is available
    uv --version
)

REM Create virtual environment if not exists
if not exist ".venv" (
    echo.
    echo [INFO] Creating virtual environment...
    uv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created successfully
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [OK] Virtual environment activated

REM Install dependencies
echo.
echo [INFO] Installing project dependencies...
uv pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    echo.
    echo [TIP] Try installing manually:
    echo   1. Activate venv: .venv\Scripts\activate
    echo   2. Install: uv pip install -r requirements.txt
    pause
    exit /b 1
)

echo [OK] Dependencies installed successfully

REM Initialize database if not exists
if not exist "data\databases\ml_platform.db" (
    echo.
    echo [INFO] Initializing database...
    python backend\database\models.py
    if %errorlevel% neq 0 (
        echo [ERROR] Database initialization failed
        echo [TIP] Check if data directory exists and has write permissions
        pause
        exit /b 1
    )
    echo [OK] Database initialized successfully
) else (
    echo [OK] Database already exists
)

REM Start Flask server
echo.
echo ======================================
echo   Starting Flask Server...
echo   Access URL: http://localhost:5000
echo   Press Ctrl+C to stop server
echo ======================================
echo.

cd backend
python app.py

pause