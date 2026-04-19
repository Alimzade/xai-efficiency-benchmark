@echo off
setlocal

echo [INFO] Searching for a compatible Python installation (3.9+)...

:: --- DISCOVERY ---
set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel% equ 0 (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
        goto :python_found
    )
)

where py >nul 2>nul
if %errorlevel% equ 0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=py -3"
        goto :python_found
    )
)

echo [ERROR] No compatible Python 3.9+ installation was found.
echo Please install Python from https://www.python.org/downloads/
pause
exit /b 1

:python_found
%PYTHON_CMD% -c "import sys; print(f'[INFO] Discovered Python {sys.version.split()[0]}')"

:: --- VENV CHECK ---
if not exist "venv\" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
    
    echo [INFO] First-time setup...
    call venv\Scripts\activate
    python smart_setup.py
) else (
    echo [INFO] Activating environment...
    call venv\Scripts\activate
)

:: --- SILENCE STREAMLIT ---
python -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

:: --- SYNC ---
python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing Streamlit...
    python -m pip install -r requirements.txt
)

:: --- LAUNCH ---
echo [INFO] Launching UI...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
python -m streamlit run gui/app.py --browser.gatherUsageStats=false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
