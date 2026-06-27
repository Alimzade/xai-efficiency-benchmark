#!/bin/bash

# 1. Check if the virtual environment exists and is Linux-compatible
if [ ! -f "venv/bin/activate" ]; then
    echo "[INFO] Linux-compatible virtual environment not found. Preparing venv..."
    if [ -d "venv" ]; then
        echo "[WARNING] Found existing 'venv' directory. Deleting to recreate..."
        rm -rf venv
    fi
    echo "[INFO] Creating fresh virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[WARNING] Virtual environment creation failed."
        if command -v apt-get >/dev/null 2>&1; then
            echo "[INFO] Detected Debian/Ubuntu. Attempting to install missing venv package..."
            PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            echo "[INFO] Running: sudo apt-get update && sudo apt-get install -y python3-${PY_VERSION}-venv"
            sudo apt-get update && sudo apt-get install -y "python3-${PY_VERSION}-venv"
            if [ $? -eq 0 ]; then
                echo "[INFO] Installation successful. Retrying virtual environment creation..."
                rm -rf venv
                python3 -m venv venv
            fi
        fi
    fi
    
    if [ ! -f "venv/bin/activate" ]; then
        echo "[ERROR] Virtual environment could not be created. Please install python3-venv manually."
        exit 1
    fi
    
    source venv/bin/activate
    python -m pip install --upgrade pip
    echo "[INFO] Starting Smart Setup..."
    python smart_setup.py
else
    # 2. Activate environment
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
fi

# 3. SILENCE STREAMLIT (The Final Fix):
# This creates the "already seen" credentials file in the user's home folder.
python -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

# 4. Set environment variables
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=false

# 5. Launch Streamlit UI
echo "[INFO] Launching XAI Efficiency Benchmark UI..."
python -m streamlit run gui/app.py --browser.gatherUsageStats=false
