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
if not exist "%~dp0venv\" goto :create_venv

"%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>nul
if %errorlevel% equ 0 goto :activate_venv

echo [WARNING] Existing virtual environment seems corrupted or broken.
echo [INFO] Removing corrupted environment...
rmdir /s /q "%~dp0venv"

:create_venv
echo [INFO] Creating virtual environment...
call %PYTHON_CMD% -m venv %~dp0venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)

echo [INFO] First-time setup...
call %~dp0venv\Scripts\activate
call python %~dp0utils\setup_env.py
goto :run_streamlit

:activate_venv
echo [INFO] Activating environment...
call %~dp0venv\Scripts\activate

:run_streamlit
:: --- SILENCE STREAMLIT ---
call python -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

:: --- SYNC ---
if not exist "%~dp0venv\.setup_complete" (
    echo [INFO] Environment incomplete. Running smart setup...
    call python "%~dp0utils\setup_env.py"
)

:: --- LAUNCH ---
echo [INFO] Testing PyTorch installation (Checking for DLL/Segfaults)...
call python -c "import torch; print('PyTorch loaded successfully!')"
if %errorlevel% neq 0 (
    echo [ERROR] PyTorch failed to load! This is likely a Python 3.13 compatibility issue or a missing DLL.
    pause
    exit /b 1
)

echo [INFO] Launching UI...
call python -X faulthandler -m streamlit run %~dp0main.py --server.runOnSave=true --browser.gatherUsageStats=false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)


