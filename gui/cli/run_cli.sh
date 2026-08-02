#!/bin/bash

# Set TMPDIR to a local directory to bypass /tmp RAM-disk size limits on Linux/macOS
mkdir -p .pip_tmp
export TMPDIR="$PWD/.pip_tmp"
trap "rm -rf .pip_tmp" EXIT

# If the virtual environment is already activated, skip all checks!
if [ -n "$VIRTUAL_ENV" ]; then
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    python "$DIR/cli.py" "$@"
    exit $?
fi

# 1. Check if the virtual environment exists and is healthy
VENV_DIR=""
if [ -f "venv/bin/activate" ]; then
    VENV_DIR="venv"
elif [ -f "$DIR/../venv/bin/activate" ]; then
    VENV_DIR="$DIR/../venv"
fi

if [ -n "$VENV_DIR" ]; then
    if ! "$VENV_DIR/bin/python" -c "import sys" >/dev/null 2>&1; then
        echo "[WARNING] Existing virtual environment seems corrupted. Deleting to recreate..."
        rm -rf "$VENV_DIR"
        VENV_DIR=""
    fi
fi

if [ -z "$VENV_DIR" ]; then
    echo "[INFO] Linux-compatible virtual environment not found. Preparing venv..."
    if [ -d "venv" ]; then
        echo "[WARNING] Found existing 'venv' directory. Deleting to recreate..."
        rm -rf venv
    fi
    echo "[INFO] Creating fresh virtual environment..."
    python3 -m venv "$DIR/../venv"
    if [ $? -ne 0 ]; then
        echo "[WARNING] Virtual environment creation failed."
        if command -v apt-get >/dev/null 2>&1; then
            echo "[INFO] Detected Debian/Ubuntu. Attempting to install missing venv package..."
            PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            echo "[INFO] Running: sudo apt-get update && sudo apt-get install -y python${PY_VERSION}-venv"
            sudo apt-get update && sudo apt-get install -y "python${PY_VERSION}-venv"
            if [ $? -eq 0 ]; then
                echo "[INFO] Installation successful. Retrying virtual environment creation..."
                rm -rf "$DIR/../venv"
                python3 -m venv "$DIR/../venv"
            fi
        fi
    fi
    
    if [ ! -f "$DIR/../venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment could not be created. Please install python3-venv manually."
        exit 1
    fi
    
    source "$DIR/../venv/bin/activate"
    python -m pip install --upgrade pip && \
    echo "[INFO] Starting Smart Setup..."
    if [ -f "$DIR/../utils/setup_env.py" ]; then
        python "$DIR/../utils/setup_env.py"
    else
        python utils/setup_env.py
    fi
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Installation or smart setup failed. Cleaning up incomplete environment..."
        rm -rf "$DIR/../venv"
        exit 1
    fi
    VENV_DIR="$DIR/../venv"
else
    # 2. Activate environment
    source $VENV_DIR/bin/activate
fi

# 3. SYNC
if [ ! -f "$VENV_DIR/.setup_complete" ]; then
    echo "[INFO] Environment incomplete. Running smart setup..."
    if [ -f "$DIR/../utils/setup_env.py" ]; then
        python "$DIR/../utils/setup_env.py"
    else
        python utils/setup_env.py
    fi
fi

# 4. Launch CLI
export LAUNCHED_VIA_WRAPPER=1
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python "$DIR/cli.py" "$@"

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] CLI execution failed."
fi
