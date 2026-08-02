@echo off
setlocal

:: If the virtual environment is already activated (e.g. by a previous run or manually), skip all checks!
if defined VIRTUAL_ENV (
    goto :run_cli
)

:: --- DISCOVERY ---
set "PYTHON_CMD="

where python >nul 2>nul
if %errorlevel% neq 0 goto :try_py

cmd /c python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :venv_check
)

:try_py
where py >nul 2>nul
if %errorlevel% neq 0 goto :no_python

cmd /c py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3"
    goto :venv_check
)

:no_python
echo [ERROR] No compatible Python 3.9+ installation was found.
echo Please install Python from https://www.python.org/downloads/
pause
exit /b 1

:: --- VENV CHECK ---
:venv_check
:: We check both current directory and parent directory for flexibility during restructuring
set "VENV_DIR="
if exist "venv\" set "VENV_DIR=venv"
if exist "%~dp0..\venv\" set "VENV_DIR=%~dp0..\venv"

if not defined VENV_DIR goto :create_venv

"%VENV_DIR%\Scripts\python.exe" -c "import sys" >nul 2>nul
if %errorlevel% equ 0 goto :activate_venv

echo [WARNING] Existing virtual environment seems corrupted or broken.
echo [INFO] Removing corrupted environment...
rmdir /s /q "%VENV_DIR%"

:create_venv

echo [INFO] Creating virtual environment...
call %PYTHON_CMD% -m venv %~dp0..\venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
)
set "VENV_DIR=%~dp0..\venv"

echo [INFO] First-time setup...
call %VENV_DIR%\Scripts\activate
if exist "%~dp0..\utils\setup_env.py" (
    call python %~dp0..\utils\setup_env.py
) else if exist "utils\setup_env.py" (
    call python utils\setup_env.py
)
goto :run_cli

:activate_venv
call %VENV_DIR%\Scripts\activate

:run_cli
:: --- SYNC ---
if not exist "%VENV_DIR%\.setup_complete" (
    echo [INFO] Environment incomplete. Running smart setup...
    if exist "%~dp0..\utils\setup_env.py" (
        call python "%~dp0..\utils\setup_env.py"
    ) else if exist "utils\setup_env.py" (
        call python "utils\setup_env.py"
    )
)

:: --- LAUNCH ---
:: Execute cli.py which is housed in the same directory as this batch script
set LAUNCHED_VIA_WRAPPER=1
call python %~dp0cli.py %*

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CLI execution failed.
)
