#!/bin/bash

# 1. Check if the virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Virtual environment not found. Starting Smart Setup..."
    python3 smart_setup.py
fi

# 2. Activate environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# 3. SILENCE STREAMLIT (The Final Fix):
# This creates the "already seen" credentials file in the user's home folder.
python3 -c "import os; path=os.path.expanduser('~/.streamlit'); os.makedirs(path, exist_ok=True); f=open(os.path.join(path, 'credentials.toml'), 'w'); f.write('[general]\nemail=\"\"\n'); f.close()"

# 4. Set environment variables
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=false

# 5. Launch Streamlit UI
echo "[INFO] Launching XAI Efficiency Benchmark UI..."
python3 -m streamlit run gui/app.py --browser.gatherUsageStats=false
