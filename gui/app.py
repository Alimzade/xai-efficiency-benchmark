"""
Main entry point for the Streamlit-based GUI application.
Handles page routing, UI initialization, and overall application state.
"""
import os
import re
import time
import json
import logging
import warnings

# --- SILENCE KNOWN WARNINGS (targeted only) ---
logging.getLogger("streamlit").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="captum.*")
warnings.filterwarnings("ignore", message=".*components.v1.html.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")
warnings.filterwarnings("ignore", message=".*Passing.*palette.*without assigning.*hue.*")

import streamlit as st
# Silence sub-logger warnings that get reset during streamlit import
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner.script_run_context").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)

import streamlit.components.v1 as components

# --- APP CONFIGURATION ---
st.set_page_config(page_title="XAI Efficiency Benchmark", page_icon="🔍", layout="wide")

# --- INITIALIZATION SPLASH SCREEN ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    def render_splash(step):
        def get_status(item_step, active_step):
            if item_step < active_step:
                return '<div class="task-icon check-glow">✓</div>', 'color: #34d399; font-weight: 500; opacity: 1;'
            elif item_step == active_step:
                return '<div class="task-icon"><div class="pulse-loader"></div></div>', 'color: #38bdf8; font-weight: 600; opacity: 1;'
            else:
                return '<div class="task-icon pending"></div>', 'color: #475569; opacity: 0.4;'
        status1, style1 = get_status(1, step)
        status2, style2 = get_status(2, step)
        status3, style3 = get_status(3, step)
        status4, style4 = get_status(4, step)
        
        opacity = 1 if step == 5 else 0
        launching_html = f"""<div style="margin-top: 1.5rem; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 1rem; opacity: {opacity}; transition: opacity 0.3s ease;">
    <div class="flicker-text">Starting Benchmark...</div>
</div>"""

        html = f"""<style>
[data-testid="stSidebarNav"] {{ display: none !important; }}
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMainBlockContainer"] {{
    overflow: hidden !important;
    height: 100vh !important;
}}
[data-testid="stMainBlockContainer"] {{
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
}}
.splash-container {{
    max-width: 680px;
    width: 100%;
    margin: auto !important;
    padding: 2.2rem 2.5rem;
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.75) 0%, rgba(15, 23, 42, 0.85) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    box-shadow: 0 0 50px rgba(45, 212, 191, 0.03), 0 25px 50px -12px rgba(0, 0, 0, 0.6);
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #f1f5f9;
    text-align: center;
}}
.splash-title {{
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.15rem;
    background: linear-gradient(90deg, #2dd4bf, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}}
.splash-subtitle {{
    font-size: 1rem;
    color: #64748b;
    margin-bottom: 1.25rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}
.splash-tasks {{
    text-align: left;
    margin: 1.5rem auto;
    max-width: 540px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 12px;
    padding: 1.5rem;
}}
.task-item {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
    font-size: 0.92rem;
    transition: all 0.3s ease;
    white-space: nowrap;
}}
.task-item:last-child {{
    margin-bottom: 0;
}}
.task-icon {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    flex-shrink: 0;
}}
.task-icon.check-glow {{
    color: #34d399;
    background: rgba(52, 211, 153, 0.12);
    border-radius: 50%;
    font-size: 0.85rem;
    font-weight: bold;
    box-shadow: 0 0 8px rgba(52, 211, 153, 0.2);
}}
.task-icon.pending {{
    border: 1.5px solid rgba(148, 163, 184, 0.2);
    border-radius: 50%;
    width: 20px;
    height: 20px;
}}
.pulse-loader {{
    display: inline-block;
    width: 14px;
    height: 14px;
    border: 2px solid #38bdf8;
    border-radius: 50%;
    border-top-color: transparent;
    animation: spin 1s linear infinite;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.3);
}}
.flicker-text {{
    font-size: 0.95rem;
    color: #2dd4bf;
    font-weight: 600;
    text-align: center;
    letter-spacing: 0.02em;
    animation: flicker 1.8s infinite ease-in-out;
}}
@keyframes flicker {{
    0%, 100% {{ opacity: 1; filter: drop-shadow(0 0 3px rgba(45, 212, 191, 0.4)); }}
    50% {{ opacity: 0.25; filter: drop-shadow(0 0 0px transparent); }}
}}
@keyframes spin {{
    0% {{ transform: rotate(0deg); }}
    100% {{ transform: rotate(360deg); }}
}}
</style>
<div class="splash-container">
    <div class="splash-title">XAI Efficiency Benchmark</div>
    <div class="splash-subtitle">Initializing System Environment</div>
    <div class="splash-tasks">
        <div class="task-item" style="{style1}">
            {status1}
            <span><b>Loading AI Engine</b>: Importing PyTorch, NumPy, & Pandas...</span>
        </div>
        <div class="task-item" style="{style2}">
            {status2}
            <span><b>Loading XAI Suite</b>: Initializing Captum explainability algorithms...</span>
        </div>
        <div class="task-item" style="{style3}">
            {status3}
            <span><b>Hardware Check</b>: Detecting GPU (CUDA/MPS) acceleration...</span>
        </div>
        <div class="task-item" style="{style4}">
            {status4}
            <span><b>Workspace Setup</b>: Creating cache and session database...</span>
        </div>
    </div>
    {launching_html}
</div>"""
        return html

    placeholder = st.empty()

    # --- STEP 1: LOAD FRAMEWORKS ---
    placeholder.markdown(render_splash(1), unsafe_allow_html=True)
    import torch
    import matplotlib.pyplot as plt
    time.sleep(0.4)

    # --- STEP 2: LOAD XAI CORE ---
    placeholder.markdown(render_splash(2), unsafe_allow_html=True)
    from gui.core import sm, default_device_mode
    time.sleep(0.4)

    # --- STEP 3: SYSTEM CHECK ---
    placeholder.markdown(render_splash(3), unsafe_allow_html=True)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    time.sleep(0.1)

    # --- STEP 4: WORKSPACE CHECK ---
    placeholder.markdown(render_splash(4), unsafe_allow_html=True)
    from gui.views.configure import render_configure_page
    from gui.views.active_run import render_active_run_page
    from gui.views.history import render_history_page
    from gui.views.documentation import render_documentation_page
    time.sleep(0.1)

    # --- STEP 5: FINAL LAUNCH TRANSITION ---
    placeholder.markdown(render_splash(5), unsafe_allow_html=True)
    time.sleep(0.4)

    st.session_state.initialized = True
    st.rerun()

# --- MAIN SCRIPTS TOP-LEVEL IMPORTS (Instantaneous on rerun!) ---
import torch
import matplotlib.pyplot as plt
from gui.core import sm, default_device_mode, model_opts, xai_opts, PROJECT_ROOT
from gui.backend.benchmark_runner import run_benchmark_task
from gui.utils.helpers import get_device_string, timestamp_now
from gui.utils.state import get_or_create_result_group, write_current_batch_results_json
from gui.views.configure import render_configure_page
from gui.views.active_run import render_active_run_page
from gui.views.history import render_history_page
from gui.views.documentation import render_documentation_page

# CSS
st.markdown("""
    <style>
    :root {
        --xai-bg-top: #151b2b;
        --xai-bg-mid: #111827;
        --xai-bg-bottom: #0f172a;
        --xai-border: rgba(148, 163, 184, 0.2);
        --xai-muted: #a8b3c3;
        --xai-text: #e5edf6;
        --xai-panel: rgba(255, 255, 255, 0.055);
        --xai-panel-strong: rgba(255, 255, 255, 0.09);
        --xai-accent: #2dd4bf;
        --xai-accent-2: #60a5fa;
        --xai-accent-warm: #fbbf24;
        --xai-sidebar-top: #252b3a;
        --xai-sidebar-mid: #1f2937;
        --xai-sidebar-bottom: #18252d;
        --xai-sidebar-panel: rgba(255, 255, 255, 0.055);
        --xai-sidebar-panel-strong: rgba(255, 255, 255, 0.09);
        --xai-sidebar-border: rgba(148, 163, 184, 0.22);
        --xai-sidebar-text: #e5edf6;
        --xai-sidebar-muted: #a8b3c3;
        --xai-sidebar-accent: #2dd4bf;
        --xai-sidebar-accent-2: #60a5fa;
    }
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(180deg, var(--xai-bg-top) 0%, var(--xai-bg-mid) 42%, var(--xai-bg-bottom) 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(17, 24, 39, 0.78);
        border-bottom: 1px solid rgba(148, 163, 184, 0.08);
    }
    .stApp * {
        caret-color: transparent;
    }
    .stApp input,
    .stApp textarea,
    .stApp [contenteditable="true"],
    .stApp [role="textbox"] {
        caret-color: auto;
    }
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.45rem;
    }
    .app-title {
        text-align: center !important;
        margin-bottom: 2.2rem !important;
        padding: 0.35rem 0 0.2rem 0;
    }
    .app-title h1 {
        font-size: 1.75rem;
        letter-spacing: 0;
        margin: 0 !important;
        color: var(--xai-text);
    }
    .app-title p {
        color: var(--xai-muted);
        margin: -0.15rem 0 0 0 !important;
        font-size: 0.95rem;
    }
    .run-card {
        background: linear-gradient(180deg, var(--xai-panel-strong) 0%, var(--xai-panel) 100%);
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 1rem;
    }
    .run-card-title {
        color: var(--xai-text);
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .run-card-copy {
        color: var(--xai-muted);
        font-size: 0.86rem;
        line-height: 1.4;
    }
    .run-summary-bar {
        display: flex;
        flex-wrap: nowrap;
        align-items: center;
        justify-content: space-between;
        gap: 0;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.035));
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        padding: 0.88rem 1rem;
        margin-bottom: 0.2rem;
        color: var(--xai-muted);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    .run-summary-item {
        display: flex;
        min-width: 0;
        flex: 1 1 0;
        flex-direction: column;
        align-items: flex-start;
        gap: 0.18rem;
        padding: 0 0.85rem;
        font-size: 0.83rem;
        line-height: 1.15;
    }
    .run-summary-item strong {
        color: var(--xai-text);
        font-size: 1.55rem;
        line-height: 1;
        font-weight: 760;
    }
    .run-summary-separator {
        color: rgba(148, 163, 184, 0.45);
        align-self: stretch;
        display: flex;
        align-items: center;
        font-size: 1.35rem;
        padding: 0 0.1rem;
    }
    .run-summary-action-spacer {
        height: 0.85rem;
    }
    .settings-hint {
        color: var(--xai-muted);
        font-size: 0.82rem;
        line-height: 1.35;
        margin-top: -0.25rem;
    }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, var(--xai-sidebar-top) 0%, var(--xai-sidebar-mid) 48%, var(--xai-sidebar-bottom) 100%);
        border-right: 1px solid var(--xai-sidebar-border);
    }
    [data-testid="stSidebarContent"] {
        padding: 1.15rem 1rem 1.5rem 1rem;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.7rem;
        color: var(--xai-sidebar-text);
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-size: 0.86rem;
        font-weight: 800;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-top: 0.65rem;
        color: #c7d2fe;
    }
    [data-testid="stSidebar"] h3::before {
        content: "";
        display: inline-block;
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 999px;
        background: linear-gradient(135deg, var(--xai-sidebar-accent), var(--xai-sidebar-accent-2));
        margin-right: 0.45rem;
        vertical-align: 0.08rem;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] small {
        color: var(--xai-sidebar-muted);
    }
    [data-testid="stSidebar"] label {
        font-weight: 650;
        color: var(--xai-sidebar-text);
    }
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(148, 163, 184, 0.18);
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] textarea,
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    textarea,
    [role="radiogroup"] {
        border-radius: 8px;
        background: var(--xai-sidebar-panel);
        border-color: rgba(148, 163, 184, 0.2);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    [data-baseweb="tag"] {
        border-radius: 999px;
        background: rgba(45, 212, 191, 0.16);
        border: 1px solid rgba(45, 212, 191, 0.24);
        color: #d8fff8;
        max-width: none;
        height: auto;
        min-height: 1.65rem;
        white-space: normal;
    }
    [data-baseweb="tag"] span {
        max-width: none;
        overflow: visible;
        text-overflow: clip;
        white-space: normal;
        line-height: 1.15;
    }
    [data-baseweb="select"] > div > div {
        flex-wrap: wrap;
        row-gap: 0.3rem;
    }
    [data-baseweb="popover"] {
        background-color: var(--xai-bg-mid) !important;
        border: 1px solid var(--xai-border) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="menu"] {
        background-color: var(--xai-bg-mid) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="menu"] li {
        color: var(--xai-text) !important;
        background-color: transparent !important;
    }
    [data-baseweb="menu"] li:hover {
        background-color: rgba(45, 212, 191, 0.16) !important;
        color: #d8fff8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--xai-sidebar-border);
        background: var(--xai-sidebar-panel-strong);
    }
    [data-testid="stSidebar"] pre {
        border-radius: 8px;
        border: 1px solid var(--xai-sidebar-border);
        background: rgba(11, 18, 26, 0.36);
        font-size: 0.76rem;
    }
    [data-testid="stTabs"] [role="tablist"] {
        justify-content: center;
        gap: 0.35rem;
        border-bottom: 1px solid var(--xai-border);
    }
    [data-testid="stTabs"] [role="tab"] {
        border-radius: 8px 8px 0 0;
        color: var(--xai-muted);
        padding: 0.6rem 0.9rem;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.18), rgba(45, 212, 191, 0.08));
        color: var(--xai-text);
        border-bottom: 2px solid var(--xai-accent);
    }
    div[data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.035));
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        padding: 0.8rem 0.9rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--xai-muted);
    }
    [data-testid="stMetricValue"] {
        color: var(--xai-text);
        font-weight: 750;
    }
    /* Subtle modern styling for documentation reference tables */
    .doc-reference-table table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid var(--xai-border) !important;
        margin-top: 8px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    .doc-reference-table th {
        background: linear-gradient(180deg, rgba(45, 212, 191, 0.12), rgba(96, 165, 250, 0.06)) !important;
        color: var(--xai-text) !important;
        font-weight: 600 !important;
        border-bottom: 2px solid rgba(45, 212, 191, 0.3) !important;
        padding: 8px 12px !important;
        font-size: 0.85rem !important;
        text-align: left !important;
    }
    .doc-reference-table td {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 8px 12px !important;
        font-size: 0.82rem !important;
        line-height: 1.45 !important;
        color: var(--xai-muted) !important;
    }
    .doc-reference-table td:first-child {
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }
    .doc-reference-table tr:hover td {
        background-color: rgba(45, 212, 191, 0.05) !important;
    }
    /* Specific column widths based on column counts (using CSS selector math to keep it simple and robust) */
    /* 4-column tables (e.g. XAI methods) */
    .doc-reference-table table th:nth-child(1):nth-last-child(4),
    .doc-reference-table table td:nth-child(1):nth-last-child(4) { width: 18% !important; }
    .doc-reference-table table th:nth-child(2):nth-last-child(3),
    .doc-reference-table table td:nth-child(2):nth-last-child(3) { width: 28% !important; }
    .doc-reference-table table th:nth-child(3):nth-last-child(2),
    .doc-reference-table table td:nth-child(3):nth-last-child(2) { width: 14% !important; }
    .doc-reference-table table th:nth-child(4):nth-last-child(1),
    .doc-reference-table table td:nth-child(4):nth-last-child(1) { width: 40% !important; }
    /* 3-column tables (e.g. Models, Metrics) */
    .doc-reference-table table th:nth-child(1):nth-last-child(3),
    .doc-reference-table table td:nth-child(1):nth-last-child(3) { width: 25% !important; }
    .doc-reference-table table th:nth-child(2):nth-last-child(2),
    .doc-reference-table table td:nth-child(2):nth-last-child(2) { width: 20% !important; }
    .doc-reference-table table th:nth-child(3):nth-last-child(1),
    .doc-reference-table table td:nth-child(3):nth-last-child(1) { width: 55% !important; }
    div.stButton button,
    div.stDownloadButton button {
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.18), rgba(45, 212, 191, 0.12)) !important;
        color: var(--xai-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    div.stButton button:hover,
    div.stDownloadButton button:hover {
        border-color: rgba(45, 212, 191, 0.55) !important;
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18)) !important;
    }
    /* Reddish theme for delete buttons containing .delete-marker */
    div[class*="st-key-del"] button {
        border-radius: 8px !important;
        border: 1px solid rgba(239, 68, 68, 0.22) !important;
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.08), rgba(220, 38, 38, 0.04)) !important;
        color: rgba(252, 165, 165, 0.8) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    div[class*="st-key-del"] button:hover {
        border-color: rgba(239, 68, 68, 0.45) !important;
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.16), rgba(220, 38, 38, 0.08)) !important;
        color: #fca5a5 !important;
    }
    /* Increase font-size of locked size warning caption */
    div[data-testid="column"]:has(input[disabled]) [data-testid="stCaptionContainer"] {
        font-size: 0.88rem !important;
    }
    div[data-testid="column"]:has(input[disabled]) code {
        font-size: 1.0rem !important;
    }
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.045);
        border: 1px dashed rgba(96, 165, 250, 0.36);
        border-radius: 8px;
        padding: 0.75rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.26);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    [data-testid="stExpander"] {
        background: rgba(11, 18, 27, 0.25) !important;
        border: 1px solid var(--xai-border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        transition: border-color 0.2s ease !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(96, 165, 250, 0.3) !important;
    }
    [data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stExpander"] details summary {
        padding: 0.6rem 1rem !important;
        background-color: rgba(255, 255, 255, 0.04) !important;
        transition: none !important;
        animation: none !important;
        opacity: 1 !important;
    }
    [data-testid="stExpander"] details summary:hover {
        background-color: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stExpander"] details summary,
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary span,
    [data-testid="stExpander"] details summary svg {
        color: var(--xai-text) !important;
        fill: var(--xai-text) !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        transition: none !important;
        animation: none !important;
        opacity: 1 !important;
    }
    [data-testid="stExpander"] details[open] summary {
        border-bottom: 1px solid var(--xai-border) !important;
    }
    [data-testid="stExpander"] details [data-testid="stExpanderDetails"] {
        padding: 1rem !important;
        background: rgba(11, 18, 27, 0.45) !important;
    }
    /* Custom TDP override card styled with greenish gradient and pattern */
    div[data-testid="stVerticalBlock"]:has(> div[class*="st-key-custom_cpu_tdp"]),
    div[data-testid="stVerticalBlock"]:has(> div[class*="st-key-custom_gpu_tdp"]) {
        border: 1px solid var(--xai-border) !important;
        border-radius: 8px !important;
        padding: 12px 14px 16px 14px !important;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.14) 0%, rgba(45, 212, 191, 0.08) 100%),
                    repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.015) 0px, rgba(255, 255, 255, 0.015) 2px, transparent 2px, transparent 10px) !important;
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        gap: 4px !important;
        height: auto !important;
        min-height: min-content !important;
        overflow: visible !important;
    }
    div[data-testid="stVerticalBlock"]:has(> div[class*="st-key-custom_cpu_tdp"]) div[class*="st-key-custom_cpu_tdp"],
    div[data-testid="stVerticalBlock"]:has(> div[class*="st-key-custom_gpu_tdp"]) div[class*="st-key-custom_gpu_tdp"] {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    div.tdp-caption-wrapper {
        word-wrap: break-word !important;
        white-space: normal !important;
        line-height: 1.45 !important;
        margin-top: 6px !important;
        margin-bottom: 6px !important;
    }

    [data-testid="stAlert"] {
        border-radius: 8px;
        border: 1px solid var(--xai-border);
        background: rgba(255, 255, 255, 0.07);
    }
    [data-testid="stTable"],
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--xai-border);
    }
    [data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: var(--xai-text);
        letter-spacing: 0;
    }
    .compact-preview { width: 100%; }
    div[data-testid="column"]:has(.compact-preview) {
        margin-top: -3.2rem !important;
    }
    .preview-image-frame {
        height: 310px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.22);
        border: 1px solid rgba(148, 163, 184, 0.16);
    }
    .preview-image-frame img {
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border: 0;
        border-radius: 7px;
    }
    div[data-testid="column"]:nth-child(3) { display: flex; flex-direction: column; align-items: flex-end; }
    .stButton button { padding: 2px 10px !important; font-size: 0.9em !important; }
    [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; }
    .status-pulse { color: #22d3ee; font-weight: bold; animation: pulse 1.5s infinite; font-size: 1.1em; }
    /* Device toggle buttons — selected/unselected distinction */
    div[class*="st-key-device_btn_"] button[kind="primary"] {
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.14) 0%, rgba(45, 212, 191, 0.08) 100%),
                    repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.015) 0px, rgba(255, 255, 255, 0.015) 2px, transparent 2px, transparent 10px) !important;
        border: 1px solid rgba(45, 212, 191, 0.45) !important;
        color: var(--xai-text) !important;
        box-shadow: 0 0 8px rgba(45, 212, 191, 0.15) !important;
    }
    div[class*="st-key-device_btn_"] button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        background-image: repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.012) 0px, rgba(255, 255, 255, 0.012) 2px, transparent 2px, transparent 10px) !important;
        border: 1px solid var(--xai-border) !important;
        color: rgba(229, 237, 246, 0.75) !important;
    }
    div[class*="st-key-device_btn_"] button[kind="secondary"]:hover {
        border-color: rgba(96, 165, 250, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
    }
    .custom-hw-label {
        display: block !important;
        margin-top: 0px !important;
        margin-bottom: 2px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        color: var(--xai-text) !important;
    }
    div[class*="st-key-device_btn_"] {
        margin-top: -10.5px !important;
    }
    div[class*="st-key-device_btn_"] button[disabled] {
        opacity: 1.0 !important;
        cursor: default !important;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    /* Segmented Radio Buttons — full-width pills */
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] > div {
        width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        justify-content: center !important;
        align-items: stretch !important;
        gap: 10px !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > div,
    div[data-testid="stRadio"] [data-testid="stRadioOption"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
        display: flex !important;
        align-items: stretch !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        flex: 1 !important;
        width: 100% !important;
        min-width: 0 !important;
        background: rgba(255, 255, 255, 0.03) !important;
        background-image: repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.012) 0px, rgba(255, 255, 255, 0.012) 2px, transparent 2px, transparent 10px) !important;
        border: 1px solid var(--xai-border) !important;
        border-radius: 6px !important;
        padding: 6px 18px !important;
        cursor: pointer !important;
        transition: background-color 0.2s, border-color 0.2s !important;
        margin: 0 !important;
        color: rgba(229, 237, 246, 0.75) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        border-color: rgba(45, 212, 191, 0.45) !important;
        background: linear-gradient(135deg, rgba(96, 165, 250, 0.14) 0%, rgba(45, 212, 191, 0.08) 100%),
                    repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.015) 0px, rgba(255, 255, 255, 0.015) 2px, transparent 2px, transparent 10px) !important;
        color: var(--xai-text) !important;
        box-shadow: 0 0 8px rgba(45, 212, 191, 0.15) !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        border-color: rgba(96, 165, 250, 0.3) !important;
        background-color: rgba(255, 255, 255, 0.06) !important;
        color: var(--xai-text) !important;
    }
    /* Style Text Inputs, Number Inputs and Select Dropdowns globally */
    [data-testid="stTextInput"] [data-baseweb="input"],
    div[data-testid="stNumberInputContainer"] {
        background-color: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: var(--xai-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
        transition: all 0.2s ease !important;
    }
    
    /* Remove duplicate inner border/background on text inputs */
    [data-testid="stTextInput"] [data-baseweb="input"] > div {
        border: none !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    div[data-testid="stNumberInputContainer"] input {
        color: var(--xai-text) !important;
        background-color: transparent !important;
    }
    
    /* Hover states */
    [data-baseweb="select"] > div:hover,
    [data-testid="stTextInput"] [data-baseweb="input"]:hover,
    div[data-testid="stNumberInputContainer"]:hover,
    textarea:hover {
        border-color: rgba(45, 212, 191, 0.35) !important;
        background-color: rgba(255, 255, 255, 0.07) !important;
    }
    
    /* Focus states */
    [data-baseweb="select"] > div:focus-within,
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    div[data-testid="stNumberInputContainer"]:focus-within,
    div[data-testid="stNumberInputContainer"].focused,
    textarea:focus {
        border-color: rgba(45, 212, 191, 0.55) !important;
        background-color: rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 0 0 1px rgba(45, 212, 191, 0.55) !important;
        outline: none !important;
    }
    
    /* Number input step buttons */
    div[data-testid="stNumberInputContainer"] button {
        background-color: rgba(255, 255, 255, 0.02) !important;
        color: var(--xai-text) !important;
        border: none !important;
        border-left: 1px solid rgba(148, 163, 184, 0.15) !important;
        border-radius: 0px !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stNumberInputContainer"] button:hover {
        background-color: rgba(45, 212, 191, 0.15) !important;
        color: #fff !important;
    }
    div[data-testid="stNumberInputContainer"] button:last-of-type {
        border-top-right-radius: 7px !important;
        border-bottom-right-radius: 7px !important;
    }
    
    /* Disabled text input styles */
    [data-testid="stTextInput"]:has(input:disabled) [data-baseweb="input"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-color: rgba(148, 163, 184, 0.1) !important;
        cursor: not-allowed !important;
    }
    [data-testid="stTextInput"] input:disabled {
        color: rgba(229, 237, 246, 0.45) !important;
        cursor: not-allowed !important;
    }
    
    /* Styled step headers */
    .step-header {
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.055) 0%, rgba(255, 255, 255, 0.015) 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.12) !important;
        border-left: 4px solid var(--xai-accent) !important;
        padding: 0.55rem 1rem !important;
        border-radius: 6px !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        color: var(--xai-text) !important;
        margin-top: 1.8rem !important;
        margin-bottom: 1.2rem !important;
        display: block !important;
    }
    
    /* Nice modern bullets styling */
    .nice-bullets {
        list-style: none !important;
        padding-left: 0 !important;
        margin: 0 !important;
        margin-left: -0.9rem !important;
    }
    .nice-bullets li {
        position: relative !important;
        padding-left: 0.9rem !important;
        margin-bottom: 0.55rem !important;
        color: var(--xai-muted) !important;
        font-size: 0.84rem !important;
        line-height: 1.45 !important;
    }
    .nice-bullets li::before {
        content: "•" !important;
        position: absolute !important;
        left: 0.02rem !important;
        top: -0.05rem !important;
        color: var(--xai-accent) !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
    }
    /* Disabled multiselect tags/circles turned-off styling */
    [data-testid="stMultiSelect"]:has(input:disabled) [data-baseweb="tag"],
    [data-baseweb="select"]:has(input:disabled) [data-baseweb="tag"],
    div[aria-disabled="true"] [data-baseweb="tag"] {
        background-color: rgba(255, 255, 255, 0.035) !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        color: rgba(229, 237, 246, 0.65) !important;
        opacity: 0.75 !important;
        box-shadow: none !important;
    }
    
    [data-testid="stMultiSelect"]:has(input:disabled) [data-baseweb="select"] > div {
        background-color: rgba(15, 23, 42, 0.25) !important;
        border-color: rgba(148, 163, 184, 0.12) !important;
    }
    
    [data-testid="stMultiSelect"]:has(input:disabled) label {
        color: rgba(148, 163, 184, 0.65) !important;
    }

    /* Turned-off toggle switch circle & track */
    [data-testid="stToggle"] input:not(:checked) + div {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
    }
    [data-testid="stToggle"] input:not(:checked) + div > div {
        background-color: #94a3b8 !important;
        box-shadow: none !important;
    }

    /* Align nested columns with their parent's left boundary */
    div[data-testid="column"] div[data-testid="stHorizontalBlock"] {
        margin-left: -12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="app-title">
        <h1>XAI Efficiency Benchmark</h1>
        <p>Compare attribution runtime, peak memory, and image-size behavior across models, methods and hardwares.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'selected_history_batch' not in st.session_state: st.session_state.selected_history_batch = None
if 'first_run_render' not in st.session_state: st.session_state.first_run_render = False
if 'img_idx' not in st.session_state: st.session_state.img_idx = 0
if 'persisted_urls' not in st.session_state: st.session_state.persisted_urls = ""
if 'last_run_results' not in st.session_state: st.session_state.last_run_results = []
if 'stop_requested' not in st.session_state: st.session_state.stop_requested = False
if 'is_finished' not in st.session_state: st.session_state.is_finished = False
if 'benchmark_running' not in st.session_state: st.session_state.benchmark_running = False
if 'run_progress_idx' not in st.session_state: st.session_state.run_progress_idx = 0
if 'current_batch_id' not in st.session_state: st.session_state.current_batch_id = ""
if 'last_run_batch_id' not in st.session_state: st.session_state.last_run_batch_id = ""
if 'completed_batch_id' not in st.session_state: st.session_state.completed_batch_id = ""
if 'completion_notice_batch_id' not in st.session_state: st.session_state.completion_notice_batch_id = ""
if 'benchmark_ready_to_run' not in st.session_state: st.session_state.benchmark_ready_to_run = False
if 'current_img_base64' not in st.session_state: st.session_state.current_img_base64 = ""
if 'batch_start_time' not in st.session_state: st.session_state.batch_start_time = None
if 'batch_started_at' not in st.session_state: st.session_state.batch_started_at = ""
if 'batch_completed_at' not in st.session_state: st.session_state.batch_completed_at = ""
if 'total_execution_time' not in st.session_state: st.session_state.total_execution_time = 0
if 'task_queue' not in st.session_state: st.session_state.task_queue = []
if 'prepared_img_sources' not in st.session_state: st.session_state.prepared_img_sources = []
if 'current_run_order' not in st.session_state: st.session_state.current_run_order = "Balanced"
if 'current_batch_methods' not in st.session_state: st.session_state.current_batch_methods = []
if 'current_batch_models' not in st.session_state: st.session_state.current_batch_models = []
if 'current_batch_sizes' not in st.session_state: st.session_state.current_batch_sizes = []
if 'selected_models' not in st.session_state: st.session_state.selected_models = ["resnet50"]
if 'input_sizes_options' not in st.session_state:
    st.session_state.input_sizes_options = [
        "32", "64", "96", "128", "160", "192", "224", "256", "288", "299", 
        "320", "352", "384", "416", "448", "480", "512", "576", "640", "704", 
        "768", "832", "896", "960", "1024"
    ]
if 'selected_input_sizes' not in st.session_state:
    st.session_state.selected_input_sizes = ["224"]
if 'input_size_str' not in st.session_state:
    st.session_state.input_size_str = "224"
if 'selected_methods' not in st.session_state: st.session_state.selected_methods = ["Saliency", "Integrated_Gradients"]
if 'selected_warmups' not in st.session_state: st.session_state.selected_warmups = 3
if 'selected_repeats' not in st.session_state: st.session_state.selected_repeats = 5
if 'selected_memory_runs' not in st.session_state: st.session_state.selected_memory_runs = 1
if 'selected_run_order' not in st.session_state: st.session_state.selected_run_order = "Balanced"
if 'enable_quality_metrics' not in st.session_state: st.session_state.enable_quality_metrics = False
if 'selected_quality_metrics' not in st.session_state: st.session_state.selected_quality_metrics = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]

if 'selected_device_mode' not in st.session_state: st.session_state.selected_device_mode = default_device_mode
if 'current_device_mode' not in st.session_state: st.session_state.current_device_mode = default_device_mode
if 'current_warmups' not in st.session_state: st.session_state.current_warmups = 1
if 'current_repeats' not in st.session_state: st.session_state.current_repeats = 5
if 'current_memory_runs' not in st.session_state: st.session_state.current_memory_runs = 1
if 'current_enable_quality_metrics' not in st.session_state: st.session_state.current_enable_quality_metrics = False
if 'current_selected_quality_metrics' not in st.session_state: st.session_state.current_selected_quality_metrics = ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]
if 'current_page' not in st.session_state: st.session_state.current_page = "Configure"

if 'custom_cpu_tdp' not in st.session_state: st.session_state.custom_cpu_tdp = None
if 'custom_gpu_tdp' not in st.session_state: st.session_state.custom_gpu_tdp = None
if 'current_cpu_tdp' not in st.session_state: st.session_state.current_cpu_tdp = None
if 'current_gpu_tdp' not in st.session_state: st.session_state.current_gpu_tdp = None

if 'ig_steps_str' not in st.session_state: st.session_state.ig_steps_str = "50"
if 'ig_internal_batch_str' not in st.session_state: st.session_state.ig_internal_batch_str = "2"
if 'ig_baseline_mode' not in st.session_state: st.session_state.ig_baseline_mode = "Zeros (Black)"
if 'gs_samples_str' not in st.session_state: st.session_state.gs_samples_str = "10"
if 'gs_stdevs_str' not in st.session_state: st.session_state.gs_stdevs_str = "0.0001"
if 'gs_baseline_mode' not in st.session_state: st.session_state.gs_baseline_mode = "Zeros & Mean"
if 'occlusion_window_str' not in st.session_state: st.session_state.occlusion_window_str = "15"
if 'occlusion_stride_str' not in st.session_state: st.session_state.occlusion_stride_str = "8"
if 'occlusion_value_str' not in st.session_state: st.session_state.occlusion_value_str = "0"
if 'lime_samples_str' not in st.session_state: st.session_state.lime_samples_str = "500"
if 'lime_batch_str' not in st.session_state: st.session_state.lime_batch_str = "10"
if 'lime_segments_str' not in st.session_state: st.session_state.lime_segments_str = "50"

# --- SIDEBAR ---

# --- Restore Config Request (Must happen before any widgets are instantiated) ---
if st.session_state.get("restore_config"):
    restore_data = st.session_state.restore_config
    settings = restore_data.get("settings", {})
    environment = restore_data.get("environment", {})
    
    # Restore algorithm-specific parameter values if present
    xai_params = settings.get("xai_params", {})
    for k, v in xai_params.items():
        st.session_state[k] = v
        
    st.session_state.selected_models = settings.get("models", ["resnet50"])
    
    # Map and restore each method version individually from methods_info
    methods_map = {m.lower().replace("_", ""): m for m in xai_opts}
    restored_methods = []
    
    # We retrieve methods_info from settings or top-level restore data
    methods_info = settings.get("methods_info") or restore_data.get("methods_info")
    
    if methods_info:
        version_counts = {}
        for run in methods_info:
            base_lower = run.get("base_name", "")
            base_norm = base_lower.lower().replace("_", "").replace("-", "")
            
            # Find matching base name from available XAI options
            official_base = None
            for m_opt in xai_opts:
                if m_opt.lower().replace("_", "") == base_norm:
                    official_base = m_opt
                    break
                    
            if not official_base:
                continue
                
            version_counts[official_base] = version_counts.get(official_base, 0) + 1
            count = version_counts[official_base]
            restored_name = f"{official_base}_{count}" if count > 1 else official_base
            
            if restored_name not in restored_methods:
                restored_methods.append(restored_name)
                
            # Restore settings into session state for this restored_name
            params = run.get("params", {})
            if official_base == "Integrated_Gradients":
                st.session_state[f"ig_steps_str_{restored_name}"] = str(params.get("n_steps", 50))
                st.session_state[f"ig_internal_batch_str_{restored_name}"] = str(params.get("internal_batch_size", 2))
                st.session_state[f"ig_baseline_mode_{restored_name}"] = params.get("baseline_mode", "Zeros (Black)")
            elif official_base == "Gradient_Shap":
                st.session_state[f"gs_samples_str_{restored_name}"] = str(params.get("n_samples", 10))
                st.session_state[f"gs_stdevs_str_{restored_name}"] = str(params.get("stdevs", 0.0001))
                st.session_state[f"gs_baseline_mode_{restored_name}"] = params.get("baseline_mode", "Zeros & Mean")
            elif official_base == "Occlusion":
                sliding = params.get("sliding_window_shapes", (3, 15, 15))
                strds = params.get("strides", (3, 8, 8))
                st.session_state[f"occlusion_window_str_{restored_name}"] = str(sliding[1] if len(sliding) > 1 else 15)
                st.session_state[f"occlusion_stride_str_{restored_name}"] = str(strds[1] if len(strds) > 1 else 8)
                st.session_state[f"occlusion_value_str_{restored_name}"] = str(params.get("occlude_color", 0))
            elif official_base == "Lime":
                st.session_state[f"lime_samples_str_{restored_name}"] = str(params.get("n_samples", 500))
                st.session_state[f"lime_batch_str_{restored_name}"] = str(params.get("perturbations_per_eval", 10))
                st.session_state[f"lime_segments_str_{restored_name}"] = str(params.get("n_segments", 50))
    else:
        # Fallback parsing for legacy configs that only have expanded current_batch_methods
        legacy_methods = settings.get("methods", [])
        for base in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]:
            has_v1 = False
            for m in legacy_methods:
                if m.startswith(base):
                    if m == base or re.match(rf"^{base}_1(?:_\d+)?$", m):
                        has_v1 = True
                        break
            if not has_v1:
                for m in legacy_methods:
                    if m.startswith(base) and not any(m.startswith(f"{base}_{v}") for v in [2, 3, 4, 5]):
                        has_v1 = True
                        break
            if has_v1:
                if base in methods_map.values():
                    restored_methods.append(base)
                
            for v in range(2, 6):
                for m in legacy_methods:
                    if m.startswith(f"{base}_{v}"):
                        restored_name = f"{base}_{v}"
                        if restored_name not in restored_methods:
                            restored_methods.append(restored_name)
                        break
                        
        for m in legacy_methods:
            base_m = re.sub(r'_\d+$', '', m)
            if base_m not in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]:
                base_norm = base_m.lower().replace("_", "")
                if base_norm in methods_map:
                    restored_name = methods_map[base_norm]
                    if restored_name not in restored_methods:
                        restored_methods.append(restored_name)
                    
    st.session_state.selected_methods = restored_methods if restored_methods else ["Saliency", "Integrated_Gradients"]
    
    raw_sizes = settings.get("input_sizes") or settings.get("input_size_str") or [224]
    if isinstance(raw_sizes, list) and len(raw_sizes) > 0:
        st.session_state.selected_input_sizes = [str(s) for s in raw_sizes]
    elif isinstance(raw_sizes, (str, int)):
        if isinstance(raw_sizes, str):
            st.session_state.selected_input_sizes = [s.strip() for s in raw_sizes.split(",") if s.strip()]
        else:
            st.session_state.selected_input_sizes = [str(raw_sizes)]
    else:
        st.session_state.selected_input_sizes = ["224"]
    
    # Ensure options contains all selected ones
    for s in st.session_state.selected_input_sizes:
        if s not in st.session_state.input_sizes_options:
            st.session_state.input_sizes_options.append(s)
            
    st.session_state.input_size_str = ", ".join(st.session_state.selected_input_sizes)
    st.session_state.selected_repeats = settings.get("repeat_count", 5)
    st.session_state.selected_warmups = settings.get("warmup_runs", 3)
    st.session_state.selected_memory_runs = settings.get("memory_runs", 1)
    st.session_state.selected_run_order = settings.get("run_order", "Balanced")
    st.session_state.enable_quality_metrics = settings.get("enable_quality_metrics", False)
    st.session_state.selected_quality_metrics = settings.get("selected_quality_metrics") or ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Sensitivity (Max)"]
    
    env_dev = environment.get("selected_device")
    if env_dev:
        env_dev_str = str(env_dev).lower()
        if "cuda" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (CUDA)"
        elif "mps" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (MPS)"
        else:
            st.session_state.selected_device_mode = "CPU"
            
    st.session_state.custom_cpu_tdp = settings.get("custom_cpu_tdp")
    st.session_state.custom_gpu_tdp = settings.get("custom_gpu_tdp")
    st.session_state.current_cpu_tdp = settings.get("custom_cpu_tdp")
    st.session_state.current_gpu_tdp = settings.get("custom_gpu_tdp")

    # Clear the restoration request so it only runs once
    del st.session_state.restore_config
    st.session_state.config_restored_toast = True

if st.session_state.get("config_restored_toast"):
    st.toast("Configuration successfully loaded! Switched to 'Benchmark Workspace'.", icon="✅")
    st.session_state.config_restored_toast = False
    
    # Inject JavaScript to automatically switch active tab to "Benchmark Workspace" (Index 0)
    js_switch = """
    <script>
        function clickTabByIndex(index) {
            var selectors = [
                'button[role="tab"]',
                'button[data-baseweb="tab"]',
                'button[data-testid="stWidgetTab"]'
            ];
            for (var s = 0; s < selectors.length; s++) {
                var tabs = window.parent.document.querySelectorAll(selectors[s]);
                if (tabs.length > index) {
                    var tab = tabs[index];
                    if (tab.getAttribute("aria-selected") === "true") {
                        return false;
                    }
                    tab.click();
                    return true;
                }
            }
            return false;
        }
        setTimeout(function() { clickTabByIndex(0); }, 150);
    </script>
    """
    components.html(js_switch, height=0)

fragment_api = getattr(st, "fragment", getattr(st, "experimental_fragment", None))

# --- TABS WORKSPACE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🚀 Benchmark Workspace", 
    "📜 Results History & Evaluation", 
    "📚 Documentation", 
    "📖 Citation"
])

with tab1:
    if st.session_state.benchmark_running or st.session_state.benchmark_ready_to_run or st.session_state.is_finished:
        render_active_run_page()
    else:
        render_configure_page()

with tab2:
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        st.warning("⚡ **Benchmark is running in the background.** This page will refresh automatically as tasks complete. We recommend staying on the **Benchmark Workspace** tab to monitor progress.")
    render_history_page()

with tab3:
    render_documentation_page()

with tab4:
    if st.session_state.benchmark_running and not st.session_state.is_finished:
        st.warning("⚡ **Benchmark is running in the background.** This page will refresh automatically as tasks complete. We recommend staying on the **Benchmark Workspace** tab to monitor progress.")
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("If you use this benchmark in your research, papers, or projects, please cite it using the following BibTeX entry:")
    
    st.code("""@software{alimzade2025xai,
  author  = {Anar Alimzade},
  title   = {Efficiency Benchmark for XAI},
  year    = {2025},
  month   = {May},
  version = {1.0.0}
}""", language="bibtex")

# --- ENGINE ---
if st.session_state.benchmark_running and not st.session_state.is_finished and not st.session_state.get("first_run_render"):
    idx = st.session_state.run_progress_idx
    task_queue = st.session_state.task_queue
    run_img_sources = st.session_state.prepared_img_sources

    if st.session_state.last_run_batch_id != st.session_state.current_batch_id:
        st.session_state.last_run_results = []
        st.session_state.last_run_batch_id = st.session_state.current_batch_id

    if idx < len(task_queue):
        task = task_queue[idx]
        img_i = task["img_i"]
        
        src = run_img_sources[img_i]
        model_name = task["model_name"]
        target_size = task["target_size"]
        method_name = task["method_name"]
        
        target_group = get_or_create_result_group(st.session_state.last_run_results, img_i, run_img_sources)
        model_label = f"{model_name} ({target_size}px)"
        model_entry = next((m for m in target_group["models"] if m.get("model_label") == model_label), None)
        
        if not model_entry:
            s_dir = sm.get_task_path(st.session_state.current_batch_id, img_i + 1, f"{model_name}_{target_size}")
            if hasattr(src, 'getbuffer'):
                tp = os.path.join(s_dir, "input_image.jpg")
                with open(tp, "wb") as f:
                    f.write(src.getbuffer())
                fs = tp
            else:
                fs = src
            
            # Check if prediction and original resolution are already computed (e.g. from a resumed run)
            pred_val = "Unknown"
            orig_res_val = "Unknown"
            config_path = os.path.normpath(os.path.join(s_dir, "config.json"))
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        t_cfg = json.load(f)
                    pred_val = t_cfg.get("prediction", "Unknown")
                    orig_res_val = t_cfg.get("original_resolution", "Unknown")
                except Exception:
                    pass
                    
            model_entry = {"model": model_name, "model_label": model_label, "input_size": target_size, "results": [], "session_dir": s_dir, "src_path": fs, "prediction": pred_val, "original_resolution": orig_res_val}
            target_group["models"].append(model_entry)
            
            # If the prediction is unknown, run a fast prediction-only pass to get it immediately
            if pred_val == "Unknown":
                run_benchmark_task({
                    "model_name": model_name, 
                    "image_source": model_entry["src_path"], 
                    "methods": [], 
                    "method_params": {},
                    "force_device": get_device_string(st.session_state.current_device_mode), 
                    "input_size": target_size,
                    "warmup_runs": 0,
                    "memory_runs": 0,
                    "repeat_count": 1,
                    "run_order": st.session_state.current_run_order,
                    "enable_quality_metrics": False,
                    "selected_quality_metrics": [],
                    "custom_cpu_tdp": st.session_state.get("current_cpu_tdp"),
                    "custom_gpu_tdp": st.session_state.get("current_gpu_tdp")
                }, s_dir)
                
                # Load the newly saved prediction and original resolution
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            t_cfg = json.load(f)
                        model_entry["prediction"] = t_cfg.get("prediction", "Unknown")
                        model_entry["original_resolution"] = t_cfg.get("original_resolution", "Unknown")
                    except Exception:
                        pass
                
                st.rerun()

        results = run_benchmark_task({
            "model_name": model_name, 
            "image_source": model_entry["src_path"], 
            "methods": [method_name], 
            "method_params": task.get("method_params", {}),
            "force_device": get_device_string(st.session_state.current_device_mode), 
            "input_size": target_size,
            "warmup_runs": st.session_state.current_warmups,
            "memory_runs": st.session_state.current_memory_runs,
            "repeat_count": st.session_state.current_repeats,
            "run_order": st.session_state.current_run_order,
            "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
            "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
            "custom_cpu_tdp": st.session_state.get("current_cpu_tdp"),
            "custom_gpu_tdp": st.session_state.get("current_gpu_tdp")
        }, model_entry["session_dir"])
        
        # Explicit composite Task ID (Image + Model + Resolution + Method) to guarantee 100% mathematical uniqueness
        task_id = f"img{img_i}_{model_name}_{target_size}px_{method_name.lower()}"
        for res in results:
            res["_task_id"] = task_id
            
        existing_tasks = {r.get("_task_id"): idx_r for idx_r, r in enumerate(model_entry["results"]) if r.get("_task_id")}
        for res in results:
            t_id = res.get("_task_id")
            if t_id and t_id in existing_tasks:
                model_entry["results"][existing_tasks[t_id]] = res
            else:
                model_entry["results"].append(res)
                
        # Read the latest prediction and original resolution from config.json and update model_entry
        config_path = os.path.normpath(os.path.join(model_entry["session_dir"], "config.json"))
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    t_cfg = json.load(f)
                model_entry["prediction"] = t_cfg.get("prediction", "Unknown")
                model_entry["original_resolution"] = t_cfg.get("original_resolution", "Unknown")
            except Exception:
                pass
                
        st.session_state.run_progress_idx += 1
        
        if st.session_state.run_progress_idx >= len(task_queue):
            st.session_state.is_finished = True
            st.session_state.benchmark_running = False
            st.session_state.completed_batch_id = st.session_state.current_batch_id
            st.session_state.total_execution_time = time.time() - st.session_state.batch_start_time
            st.session_state.batch_completed_at = timestamp_now()
            write_current_batch_results_json()
        st.rerun()

    else:
        st.session_state.is_finished = True
        st.session_state.benchmark_running = False
        st.session_state.completed_batch_id = st.session_state.current_batch_id
        if not st.session_state.batch_completed_at:
            st.session_state.batch_completed_at = timestamp_now()
        write_current_batch_results_json()
        st.rerun()


plt.close('all')
