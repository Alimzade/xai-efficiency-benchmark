@echo off
setlocal

echo [INFO] Searching for a compatible Python installation (3.9+)...

:: --- DISCOVERY ---
set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel% neq 0 goto :try_py

cmd /c python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :python_found
)

:try_py
where py >nul 2>nul
if %errorlevel% neq 0 goto :no_python

cmd /c py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3"
    goto :python_found
)

:no_python
echo [ERROR] No compatible Python 3.9+ installation was found.
echo Please install Python from https://www.python.org/downloads/
pause
exit /b 1

:python_found
call %PYTHON_CMD% -c "import sys; print(f'[INFO] Discovered Python {sys.version.split()[0]}')"

:: --- VENV CHECK ---
if exist "venv\" goto :activate_venv

echo [INFO] Creating virtual environment...
call %PYTHON_CMD% -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)

echo [INFO] First-time setup...
call venv\Scripts\activate
call python smart_setup.py
goto :run_streamlit

:activate_venv
echo [INFO] Activating environment...
call venv\Scripts\activate

:run_streamlit
:: --- SILENCE STREAMLIT ---
call python -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

:: --- SYNC ---
call python -c "import streamlit" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing Streamlit...
    call python -m pip install -r requirements.txt
)

:: --- LAUNCH ---
echo [INFO] Launching UI...
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
call python -m streamlit run gui/app.py --browser.gatherUsageStats=false --logger.level=error

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)


