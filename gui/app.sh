#!/bin/bash

# Set TMPDIR to a local directory to bypass /tmp RAM-disk size limits on Linux/macOS
mkdir -p .pip_tmp
export TMPDIR="$PWD/.pip_tmp"
trap "rm -rf .pip_tmp" EXIT

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# 1. Check if the virtual environment exists and is healthy
if [ -f "$DIR/venv/bin/activate" ]; then
    if ! "$DIR/venv/bin/python" -c "import sys" >/dev/null 2>&1; then
        echo "[WARNING] Existing virtual environment seems corrupted. Deleting to recreate..."
        rm -rf "$DIR/venv"
    fi
fi

if [ ! -f "$DIR/venv/bin/activate" ]; then
    echo "[INFO] Linux-compatible virtual environment not found. Preparing venv..."
    if [ -d "$DIR/venv" ]; then
        echo "[WARNING] Found existing 'venv' directory. Deleting to recreate..."
        rm -rf "$DIR/venv"
    fi
    echo "[INFO] Creating fresh virtual environment..."
    python3 -m venv "$DIR/venv"
    if [ $? -ne 0 ]; then
        echo "[WARNING] Virtual environment creation failed."
        if command -v apt-get >/dev/null 2>&1; then
            echo "[INFO] Detected Debian/Ubuntu. Attempting to install missing venv package..."
            PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            echo "[INFO] Running: sudo apt-get update && sudo apt-get install -y python${PY_VERSION}-venv"
            sudo apt-get update && sudo apt-get install -y "python${PY_VERSION}-venv"
            if [ $? -eq 0 ]; then
                echo "[INFO] Installation successful. Retrying virtual environment creation..."
                rm -rf "$DIR/venv"
                python3 -m venv "$DIR/venv"
            fi
        fi
    fi
    
    if [ ! -f "$DIR/venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment could not be created. Please install python3-venv manually."
        exit 1
    fi
    
    source "$DIR/venv/bin/activate"
    python -m pip install --upgrade pip && \
    echo "[INFO] Starting Environment Setup..." && \
    python "$DIR/utils/setup_env.py"
    
    if [ $? -ne 0 ]; then
        echo "[ERROR] Installation or smart setup failed. Cleaning up incomplete environment..."
        rm -rf "$DIR/venv"
        exit 1
    fi
else
    # 2. Activate environment
    echo "[INFO] Activating virtual environment..."
    source "$DIR/venv/bin/activate"
    
    if [ ! -f "$DIR/venv/.setup_complete" ]; then
        echo "[INFO] Environment incomplete. Running smart setup..."
        python "$DIR/utils/setup_env.py"
    fi
fi

# 3. SILENCE STREAMLIT (The Final Fix):
# This creates the "already seen" credentials file in the user's home folder.
python -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

# 4. Set environment variables
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=false

# 5. Launch Streamlit UI
echo "[INFO] Launching XAI Efficiency Benchmark UI..."
python -m streamlit run "$DIR/main.py" --browser.gatherUsageStats=false --logger.level=error 2>&1 | grep -v "components.v1.html\|will be removed after"
