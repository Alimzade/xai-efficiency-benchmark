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
import os
import time

# --- Page Config ---
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
    import pandas as pd
    import numpy as np
    import torch
    import platform
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- STEP 2: LOAD XAI CORE ---
    placeholder.markdown(render_splash(2), unsafe_allow_html=True)
    from captum.attr import (
        DeepLift,
        DeepLiftShap,
        GradientShap,
        GuidedBackprop,
        InputXGradient,
        IntegratedGradients,
        LayerAttribution,
        LayerGradCam,
        Saliency,
    )

    # --- STEP 3: SYSTEM CHECK ---
    placeholder.markdown(render_splash(3), unsafe_allow_html=True)
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    time.sleep(0.1)

    # --- STEP 4: WORKSPACE CHECK ---
    placeholder.markdown(render_splash(4), unsafe_allow_html=True)
    import json
    import random
    from datetime import datetime
    from PIL import Image
    import requests
    from io import BytesIO
    from session_manager import SessionManager
    from benchmark_runner import collect_environment_metadata, run_benchmark_task, get_cpu_name
    from exporter import generate_pdf_report, generate_csv_report
    
    sm = SessionManager()
    time.sleep(0.1)

    # --- STEP 5: FINAL LAUNCH TRANSITION ---
    placeholder.markdown(render_splash(5), unsafe_allow_html=True)
    time.sleep(0.12)

    st.session_state.initialized = True
    st.rerun()

# --- MAIN SCRIPTS TOP-LEVEL IMPORTS (Instantaneous on rerun!) ---
import pandas as pd
import numpy as np
import torch
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from session_manager import SessionManager
from benchmark_runner import collect_environment_metadata, run_benchmark_task, get_cpu_name
from exporter import generate_pdf_report, generate_csv_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    /* Subtle modern styling ONLY for documentation reference tables */
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
    /* Specific column widths for XAI methods table */
    .col-xai-name { width: 18% !important; }
    .col-xai-form { width: 28% !important; }
    .col-xai-comp { width: 14% !important; }
    .col-xai-char { width: 40% !important; }
    /* Specific column widths for Models table */
    .col-model-name { width: 25% !important; }
    .col-model-params { width: 20% !important; }
    .col-model-char { width: 55% !important; }
    /* Specific column widths for Metrics table */
    .col-metric-name { width: 25% !important; }
    .col-metric-unit { width: 20% !important; }
    .col-metric-def { width: 55% !important; }
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
        background: rgba(255, 255, 255, 0.045);
        border: 1px solid var(--xai-border);
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stExpander"] details summary {
        color: var(--xai-text);
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
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    /* Segmented Radio Buttons styling for modern Target Device Selection */
    div[data-testid="stRadio"] {
        width: 100% !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 10px !important;
        min-height: 40px !important;
        width: 100% !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] > div,
    div[data-testid="stRadio"] [role="radiogroup"] [data-testid="stRadioOption"] {
        flex: 1 1 0% !important;
        min-width: 0 !important;
        width: 100% !important;
        max-width: 100% !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        flex: 1 1 0% !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        background: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        padding: 6px 18px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
        color: rgba(229, 237, 246, 0.75) !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18)) !important;
        border-color: rgba(45, 212, 191, 0.55) !important;
        color: var(--xai-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        border-color: rgba(45, 212, 191, 0.35) !important;
        background: rgba(255, 255, 255, 0.08) !important;
        color: var(--xai-text) !important;
    }
    /* Style Text Inputs, Number Inputs and Select Dropdowns globally */
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="input"] > div,
    div[data-testid="stNumberInputContainer"] {
        background-color: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        border-radius: 8px !important;
        color: var(--xai-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04) !important;
        transition: all 0.2s ease !important;
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
    [data-testid="stTextInput"] [data-baseweb="input"]:hover > div,
    div[data-testid="stNumberInputContainer"]:hover,
    textarea:hover {
        border-color: rgba(45, 212, 191, 0.35) !important;
        background-color: rgba(255, 255, 255, 0.07) !important;
    }
    
    /* Focus states */
    [data-baseweb="select"] > div:focus-within,
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
    [data-testid="stTextInput"] [data-baseweb="input"]:focus-within > div,
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
    [data-testid="stTextInput"]:has(input:disabled) [data-baseweb="input"],
    [data-testid="stTextInput"]:has(input:disabled) [data-baseweb="input"] > div {
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
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="app-title">
        <h1>XAI Efficiency Benchmark</h1>
        <p>Compare attribution runtime, peak memory, and image-size behavior across models, methods and hardwares.</p>
    </div>
    """, unsafe_allow_html=True)

sm = SessionManager()

# --- HELPER FUNCTIONS ---
def get_device_string(mode_str):
    if "CUDA" in mode_str:
        return "cuda"
    elif "MPS" in mode_str:
        return "mps"
    return "cpu"

ATTR_RUNTIME_COL = "Attribution Runtime (sec)"
ATTR_MEMORY_COL = "Peak Attribution Memory (MB)"
LEGACY_RUNTIME_COL = "Runtime (sec)"
LEGACY_MEMORY_COL = "Peak Memory (MB)"
METADATA_COLS = ["Timing Scope", "Memory Scope", "Model Cache"]
PRESENTATION_COL_ORDER = [
    "Method",
    "Model",
    "Resolution",
    "Original Resolution",
    "Prediction",
    "Device",
    "Warmup Runs",
    "Measured Runs",
    ATTR_RUNTIME_COL,
    "Attribution Runtime Median (sec)",
    "Attribution Runtime Mean (sec)",
    "Attribution Runtime Std (sec)",
    "Attribution Runtime Min (sec)",
    "Attribution Runtime Max (sec)",
    ATTR_MEMORY_COL,
    "Gini Index",
    "Deletion AUC",
    "Insertion AUC",
    "Infidelity",
    "Quality Eval Time (sec)",
    "Status",
]

def metric_col(df, preferred, legacy):
    return preferred if preferred in df.columns else legacy

def normalize_metric_columns(df):
    df = df.copy()
    # Remove duplicate column names if any exist from legacy schemas and guarantee unique row index
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    
    if ATTR_RUNTIME_COL not in df.columns and LEGACY_RUNTIME_COL in df.columns:
        df[ATTR_RUNTIME_COL] = df[LEGACY_RUNTIME_COL]
    if ATTR_MEMORY_COL not in df.columns and LEGACY_MEMORY_COL in df.columns:
        df[ATTR_MEMORY_COL] = df[LEGACY_MEMORY_COL]
        
    # Coerce metric columns to float64 numeric dtypes so historical JSON string nulls ('-', '.', 'None') convert cleanly to np.nan
    non_numeric_text_cols = {
        "Method", "Model", "Resolution", "Original Resolution", "Prediction", 
        "Status", "Device", "Model Cache", "Timing Scope", "Memory Scope", "_task_id",
        "started_at", "completed_at"
    }
    for col in df.columns:
        if col not in non_numeric_text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # For timing std columns, replace exactly 0.0 with np.nan because a runtime std of 0.0 is a single-run artifact
    std_cols = [c for c in df.columns if any(k in c for k in ["Runtime Std", "Across Images", "Run Std"])]
    for col in std_cols:
        df[col] = df[col].replace(0.0, np.nan)
        
    return df

def add_input_size_column(df):
    df = df.copy()
    if "Input Size (px)" not in df.columns and "Resolution" in df.columns:
        df["Input Size (px)"] = pd.to_numeric(df["Resolution"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return df

def presentation_df(df):
    df = add_input_size_column(normalize_metric_columns(df))
    duplicate_cols = [
        LEGACY_RUNTIME_COL, "Runtime Median (sec)", "Runtime Mean (sec)",
        "Runtime Std (sec)", "Runtime Min (sec)", "Runtime Max (sec)",
        LEGACY_MEMORY_COL, "Input Size (px)"
    ] + METADATA_COLS
    df = df.drop(columns=[c for c in duplicate_cols if c in df.columns], errors="ignore")
    ordered_cols = [c for c in PRESENTATION_COL_ORDER if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]

def parse_comma_sep_ints(val_str, default_val):
    try:
        vals = [int(x.strip()) for x in val_str.split(",") if x.strip().isdigit()]
        return vals if vals else [default_val]
    except Exception:
        return [default_val]

def parse_comma_sep_floats(val_str, default_val):
    try:
        vals = []
        for x in val_str.split(","):
            try:
                vals.append(float(x.strip()))
            except ValueError:
                pass
        return vals if vals else [default_val]
    except Exception:
        return [default_val]

def expand_xai_methods_with_params(selected_methods):
    expanded = []
    for method in selected_methods:
        import re
        base_name = re.sub(r'_\d+$', '', method)
        
        if base_name == "Integrated_Gradients":
            steps_list = parse_comma_sep_ints(st.session_state.get(f"ig_steps_str_{method}", st.session_state.get("ig_steps_str", "50")), 50)
            batch_list = parse_comma_sep_ints(st.session_state.get(f"ig_internal_batch_str_{method}", st.session_state.get("ig_internal_batch_str", "2")), 2)
            base_mode = st.session_state.get(f"ig_baseline_mode_{method}", st.session_state.get("ig_baseline_mode", "Zeros (Black)"))
            
            combos = []
            for s in steps_list:
                for b in batch_list:
                    combos.append({"n_steps": s, "internal_batch_size": b, "baseline_mode": base_mode})
                    
            if len(combos) > 1:
                for idx, params in enumerate(combos):
                    name = f"{method}_{idx + 1}"
                    expanded.append({
                        "display_name": name,
                        "base_name": "integrated_gradients",
                        "params": params
                    })
            else:
                expanded.append({
                    "display_name": method,
                    "base_name": "integrated_gradients",
                    "params": combos[0]
                })
                
        elif base_name == "Gradient_Shap":
            samples_list = parse_comma_sep_ints(st.session_state.get(f"gs_samples_str_{method}", st.session_state.get("gs_samples_str", "10")), 10)
            stdevs_list = parse_comma_sep_floats(st.session_state.get(f"gs_stdevs_str_{method}", st.session_state.get("gs_stdevs_str", "0.0001")), 0.0001)
            base_mode = st.session_state.get(f"gs_baseline_mode_{method}", st.session_state.get("gs_baseline_mode", "Zeros & Mean"))
            
            combos = []
            for s in samples_list:
                for std in stdevs_list:
                    combos.append({"n_samples": s, "stdevs": std, "baseline_mode": base_mode})
                    
            if len(combos) > 1:
                for idx, params in enumerate(combos):
                    name = f"{method}_{idx + 1}"
                    expanded.append({
                        "display_name": name,
                        "base_name": "gradient_shap",
                        "params": params
                    })
            else:
                expanded.append({
                    "display_name": method,
                    "base_name": "gradient_shap",
                    "params": combos[0]
                })
                
        elif base_name == "Occlusion":
            window_list = parse_comma_sep_ints(st.session_state.get(f"occlusion_window_str_{method}", st.session_state.get("occlusion_window_str", "15")), 15)
            stride_list = parse_comma_sep_ints(st.session_state.get(f"occlusion_stride_str_{method}", st.session_state.get("occlusion_stride_str", "8")), 8)
            occ_val = st.session_state.get(f"occlusion_value_str_{method}", st.session_state.get("occlusion_value_str", "0"))
            
            combos = []
            for w in window_list:
                for strd in stride_list:
                    combos.append({
                        "sliding_window_shapes": (3, w, w),
                        "strides": (3, strd, strd),
                        "occlude_color": occ_val
                    })
                    
            if len(combos) > 1:
                for idx, params in enumerate(combos):
                    name = f"{method}_{idx + 1}"
                    expanded.append({
                        "display_name": name,
                        "base_name": "occlusion",
                        "params": params
                    })
            else:
                expanded.append({
                    "display_name": method,
                    "base_name": "occlusion",
                    "params": combos[0]
                })
                
        elif base_name == "Lime":
            samples_list = parse_comma_sep_ints(st.session_state.get(f"lime_samples_str_{method}", st.session_state.get("lime_samples_str", "500")), 500)
            batch_list = parse_comma_sep_ints(st.session_state.get(f"lime_batch_str_{method}", st.session_state.get("lime_batch_str", "10")), 10)
            seg_list = parse_comma_sep_ints(st.session_state.get(f"lime_segments_str_{method}", st.session_state.get("lime_segments_str", "50")), 50)
            
            combos = []
            for s in samples_list:
                for b in batch_list:
                    for seg in seg_list:
                        combos.append({"n_samples": s, "perturbations_per_eval": b, "n_segments": seg})
                        
            if len(combos) > 1:
                for idx, params in enumerate(combos):
                    name = f"{method}_{idx + 1}"
                    expanded.append({
                        "display_name": name,
                        "base_name": "lime",
                        "params": params
                    })
            else:
                expanded.append({
                    "display_name": method,
                    "base_name": "lime",
                    "params": combos[0]
                })
        else:
            expanded.append({
                "display_name": method,
                "base_name": method.lower().replace("-", "_"),
                "params": {}
            })
    return assign_display_name_suffixes(expanded)

def assign_display_name_suffixes(expanded_methods):
    display_names = [m["display_name"] for m in expanded_methods]
    counts = {}
    for name in display_names:
        counts[name] = counts.get(name, 0) + 1
        
    current_counts = {}
    for m in expanded_methods:
        name = m["display_name"]
        if counts[name] > 1:
            current_counts[name] = current_counts.get(name, 0) + 1
            m["display_name"] = f"{name}_{current_counts[name]}"
    return expanded_methods

def assign_numbered_suffixes(items_list):
    counts = {}
    for item in items_list:
        counts[item] = counts.get(item, 0) + 1
        
    result = []
    current_counts = {}
    for item in items_list:
        if counts[item] > 1:
            current_counts[item] = current_counts.get(item, 0) + 1
            result.append(f"{item}_{current_counts[item]}")
        else:
            result.append(item)
    return result

def build_task_queue(num_images, models, sizes, methods, run_order, seed=None):
    # Note: methods here is a list of dicts: [{"display_name": "...", "base_name": "...", "params": {...}}, ...]
    tasks = []
    if run_order == "Balanced":
        for img_i in range(num_images):
            for mod_i, model_name in enumerate(models):
                for met_i, method_info in enumerate(methods):
                    rotation = (img_i + mod_i + met_i) % len(sizes)
                    ordered_sizes = sizes[rotation:] + sizes[:rotation]
                    for target_size in ordered_sizes:
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_info["display_name"],
                            "method_base": method_info["base_name"],
                            "method_params": method_info["params"]
                        })
    else:
        for img_i in range(num_images):
            for model_name in models:
                for method_info in methods:
                    for target_size in sizes:
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_info["display_name"],
                            "method_base": method_info["base_name"],
                            "method_params": method_info["params"]
                        })

    if run_order == "Randomized":
        rng = random.Random(seed)
        rng.shuffle(tasks)
    return tasks

def sorted_result_groups(groups):
    return sorted(groups, key=lambda g: g.get("img_idx", 0))

def get_or_create_result_group(results, img_i, img_sources):
    img_idx = img_i + 1
    group = next((g for g in results if g.get("img_idx") == img_idx), None)
    if group is None:
        group = {"img_idx": img_idx, "models": [], "source": img_sources[img_i]}
        results.append(group)
        results.sort(key=lambda g: g.get("img_idx", 0))
    return group

def methods_from_result_groups(groups):
    methods = []
    for group in groups:
        for model_entry in group.get("models", []):
            for result in model_entry.get("results", []):
                method = result.get("Method")
                if method and method not in methods:
                    methods.append(method)
    return methods

def get_cpu_info(): return get_cpu_name()
def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f}s"
    elif seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else: return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"

def timestamp_now():
    return datetime.now().astimezone().isoformat(timespec="seconds")

def display_timestamp(value):
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S %Z")
    except ValueError:
        return str(value)

def format_run_timestamps(started_iso, completed_iso):
    if not started_iso:
        return ""
    try:
        dt_start = datetime.fromisoformat(started_iso)
        tz_str = dt_start.strftime("%Z")
        if not tz_str:
            offset = dt_start.utcoffset()
            if offset is not None:
                offset_hours = int(offset.total_seconds() / 3600)
                tz_str = f"UTC{'+' if offset_hours >= 0 else ''}{offset_hours}"
            else:
                tz_str = "UTC"
            
        date_start = dt_start.strftime("%b %d, %Y")
        time_start = dt_start.strftime("%H:%M:%S")
        
        if completed_iso:
            dt_end = datetime.fromisoformat(completed_iso)
            date_end = dt_end.strftime("%b %d, %Y")
            time_end = dt_end.strftime("%H:%M:%S")
            
            if date_start == date_end:
                return f"""
                <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Date:</span>
                    <code>{date_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Time:</span>
                    <code>{time_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 3px;">&rarr;</span>
                    <code>{time_end}</code>
                    <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
                </div>
                """
            else:
                return f"""
                <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                    <code>{date_start} {time_start}</code>
                    <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                    <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Completed:</span>
                    <code>{date_end} {time_end}</code>
                    <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
                </div>
                """
        else:
            return f"""
            <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                <code>{date_start} {time_start}</code>
                <span style="font-size: 0.76rem; color: var(--xai-muted); margin-left: 4px;">({tz_str})</span>
            </div>
            """
    except Exception:
        start_display = display_timestamp(started_iso)
        if completed_iso:
            end_display = display_timestamp(completed_iso)
            return f"""
            <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
                <code>{start_display}</code>
                <span style="color: rgba(148, 163, 184, 0.35); margin: 0 6px;">|</span>
                <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Completed:</span>
                <code>{end_display}</code>
            </div>
            """
        return f"""
        <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.8rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
            <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Started:</span>
            <code>{start_display}</code>
        </div>
        """

def format_run_duration(duration):
    if not duration:
        return ""
    return f"""
    <div style="font-size: 1.0rem; color: var(--xai-muted); margin-bottom: 0.2rem; display: flex; align-items: center; justify-content: center; gap: 4px;">
        <span style="color: var(--xai-muted); font-weight: 500; margin-right: 2px;">Total Duration:</span>
        <code>{duration}</code>
    </div>
    """

def parse_input_sizes(size_str):
    try:
        sizes = [int(s.strip()) for s in size_str.split(",") if s.strip().isdigit()]
        return sizes or [224]
    except Exception:
        return [224]

def current_image_sources():
    uploaded_files = st.session_state.get("uploaded_files") or []
    url_list = [u.strip() for u in st.session_state.persisted_urls.split("\n") if u.strip()]
    
    # Auto-load local images from gui/images/ folder if it exists
    local_images = []
    images_dir = os.path.join(PROJECT_ROOT, "gui", "images")
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        try:
            active_files = st.session_state.get("active_local_filenames")
            for f in sorted(os.listdir(images_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    if active_files is None or f in active_files:
                        local_images.append(os.path.join(images_dir, f))
        except Exception as e:
            st.error(f"Error loading local images: {str(e)}")
            
    return local_images + list(uploaded_files) + url_list

def serialize_and_persist_image_sources(img_sources, batch_id, base_dir="gui/sessions"):
    persisted_sources = []
    uploaded_dir = os.path.join(base_dir, batch_id, "uploaded_images")
    
    for idx, src in enumerate(img_sources):
        if isinstance(src, str):
            persisted_sources.append(src)
        else:
            # It's a Streamlit UploadedFile or file-like object
            if not os.path.exists(uploaded_dir):
                os.makedirs(uploaded_dir, exist_ok=True)
            
            ext = "jpg"
            if hasattr(src, "name") and src.name:
                parts = src.name.rsplit(".", 1)
                if len(parts) > 1:
                    ext = parts[1].lower()
            
            filename = f"img_{idx}.{ext}"
            dest_path = os.path.normpath(os.path.join(uploaded_dir, filename))
            
            # Save the file content
            with open(dest_path, "wb") as f:
                if hasattr(src, "getbuffer"):
                    f.write(src.getbuffer())
                else:
                    f.write(src.read())
            
            persisted_sources.append(dest_path)
            
    return persisted_sources

def check_and_reset_session_after_delete(deleted_bid):
    if (st.session_state.get("current_batch_id") == deleted_bid or 
        st.session_state.get("last_run_batch_id") == deleted_bid or 
        st.session_state.get("completed_batch_id") == deleted_bid):
        st.session_state.is_finished = False
        st.session_state.benchmark_running = False
        st.session_state.benchmark_ready_to_run = False
        st.session_state.run_progress_idx = 0
        st.session_state.task_queue = []
        st.session_state.prepared_img_sources = []
        st.session_state.current_batch_id = ""
        st.session_state.batch_started_at = ""
        st.session_state.batch_completed_at = ""
        st.session_state.total_execution_time = 0
        st.session_state.batch_start_time = None
        st.session_state.last_run_results = []
        st.session_state.completed_batch_id = ""
        st.session_state.last_run_batch_id = ""
        st.session_state.current_batch_methods = []
        st.session_state.current_batch_models = []
        st.session_state.current_batch_sizes = []

def write_current_batch_results_json():
    batch_id = st.session_state.current_batch_id
    if not batch_id:
        return
    clean_results = []
    for g in sorted_result_groups(st.session_state.last_run_results):
        cg = g.copy()
        if hasattr(cg["source"], 'name'): cg["source"] = cg["source"].name
        clean_results.append(cg)
        
    results_path = os.path.join(sm.base_dir, batch_id, "batch_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": clean_results, 
            "methods": st.session_state.current_batch_methods,
            "benchmark_settings": {
                "warmup_runs": st.session_state.current_warmups,
                "memory_runs": st.session_state.current_memory_runs,
                "repeat_count": st.session_state.current_repeats,
                "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
                "run_order": st.session_state.current_run_order,
                "task_count": len(st.session_state.task_queue),
                "models": st.session_state.current_batch_models,
                "input_sizes": st.session_state.current_batch_sizes,
                "methods": st.session_state.current_batch_methods
            },
            "environment": collect_environment_metadata(get_device_string(st.session_state.current_device_mode)),
            "started_at": st.session_state.batch_started_at,
            "completed_at": st.session_state.batch_completed_at,
            "total_execution_time": st.session_state.total_execution_time,
            "methods_info": st.session_state.get("current_batch_methods_info", [])
        }, f, indent=4)

def resume_batch(batch_id):
    cfg = sm.load_batch_config(batch_id)
    if not cfg:
        st.error(f"Failed to load configuration for batch {batch_id}")
        return
        
    st.session_state.current_batch_id = batch_id
    st.session_state.last_run_batch_id = batch_id
    st.session_state.current_run_order = cfg.get("run_order", "Balanced")
    st.session_state.current_batch_methods = list(cfg.get("methods", []))
    st.session_state.current_batch_methods_info = cfg.get("methods_info", [])
    st.session_state.current_batch_models = list(cfg.get("models", []))
    st.session_state.current_batch_sizes = list(cfg.get("input_sizes", []))
    st.session_state.current_device_mode = cfg.get("device_mode", "CPU")
    st.session_state.current_warmups = cfg.get("warmup_runs", 1)
    st.session_state.current_repeats = cfg.get("repeat_count", 5)
    st.session_state.current_memory_runs = cfg.get("memory_runs", 1)
    st.session_state.current_enable_quality_metrics = cfg.get("enable_quality_metrics", False)
    st.session_state.current_selected_quality_metrics = list(cfg.get("selected_quality_metrics", []))
    st.session_state.prepared_img_sources = list(cfg.get("image_sources", []))
    st.session_state.batch_started_at = cfg.get("started_at", "")
    
    # Rebuild the exact same task queue using parameter-expanded methods
    expanded_methods = cfg.get("methods_info", [])
    if not expanded_methods:
        expanded_methods = [{"display_name": m, "base_name": m.lower().replace("-", "_"), "params": {}} for m in cfg.get("methods", [])]
        
    st.session_state.task_queue = build_task_queue(
        len(st.session_state.prepared_img_sources),
        st.session_state.current_batch_models,
        st.session_state.current_batch_sizes,
        expanded_methods,
        st.session_state.current_run_order,
        seed=batch_id
    )
    
    # Reload completed results from the filesystem
    st.session_state.last_run_results = []
    st.session_state.run_progress_idx = 0
    
    for idx, task in enumerate(st.session_state.task_queue):
        img_i = task["img_i"]
        model_name = task["model_name"]
        target_size = task["target_size"]
        method_name = task["method_name"]
        
        s_dir = sm.get_task_path(batch_id, img_i + 1, f"{model_name}_{target_size}")
        csv_path = os.path.normpath(os.path.join(s_dir, "results.csv"))
        config_path = os.path.normpath(os.path.join(s_dir, "config.json"))
        
        pred_val = "Unknown"
        orig_res_val = "Unknown"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    t_cfg = json.load(f)
                pred_val = t_cfg.get("prediction", "Unknown")
                orig_res_val = t_cfg.get("original_resolution", "Unknown")
            except Exception:
                pass
        
        if os.path.exists(csv_path):
            try:
                df_res = pd.read_csv(csv_path)
                method_rows = df_res[df_res["Method"].str.lower() == method_name.lower()]
                if not method_rows.empty:
                    target_group = get_or_create_result_group(st.session_state.last_run_results, img_i, st.session_state.prepared_img_sources)
                    model_label = f"{model_name} ({target_size}px)"
                    model_entry = next((m for m in target_group["models"] if m.get("model_label") == model_label), None)
                    if not model_entry:
                        model_entry = {
                            "model": model_name,
                            "model_label": model_label,
                            "input_size": target_size,
                            "results": [],
                            "session_dir": s_dir,
                            "src_path": st.session_state.prepared_img_sources[img_i],
                            "prediction": pred_val,
                            "original_resolution": orig_res_val
                        }
                        target_group["models"].append(model_entry)
                    else:
                        model_entry["prediction"] = pred_val
                        model_entry["original_resolution"] = orig_res_val
                    
                    res_dicts = method_rows.to_dict(orient="records")
                    task_id = f"img{img_i}_{model_name}_{target_size}px_{method_name.lower()}"
                    for r in res_dicts:
                        r["_task_id"] = task_id
                        
                    existing_tasks = {r.get("_task_id"): idx_r for idx_r, r in enumerate(model_entry["results"]) if r.get("_task_id")}
                    for r in res_dicts:
                        t_id = r.get("_task_id")
                        if t_id and t_id in existing_tasks:
                            model_entry["results"][existing_tasks[t_id]] = r
                        else:
                            model_entry["results"].append(r)
                            
                    if idx == st.session_state.run_progress_idx:
                        st.session_state.run_progress_idx = idx + 1
                    continue
            except Exception:
                pass
        break
        
    st.session_state.is_finished = (st.session_state.run_progress_idx >= len(st.session_state.task_queue))
    if st.session_state.is_finished:
        st.session_state.benchmark_running = False
        st.session_state.benchmark_ready_to_run = False
        st.session_state.completed_batch_id = batch_id
        if not st.session_state.batch_completed_at:
            st.session_state.batch_completed_at = timestamp_now()
        write_current_batch_results_json()
    else:
        st.session_state.benchmark_running = True
        st.session_state.benchmark_ready_to_run = True
        
    st.session_state.stop_requested = False
    st.session_state.current_page = "Active Run"
    st.session_state.batch_start_time = time.time()


def render_live_elapsed_timer(start_time):
    start_ms = int((start_time or time.time()) * 1000)
    components.html(f"""
        <div id="elapsed-badge" style="
            font-family: monospace;
            font-size: 1.2em;
            font-weight: 700;
            color: #6b7280;
            text-align: right;
            white-space: nowrap;
            padding-top: 2px;
        ">0.0s</div>
        <script>
            const startMs = {start_ms};
            const badge = document.getElementById("elapsed-badge");
            function formatElapsed(seconds) {{
                if (seconds < 60) return seconds.toFixed(1) + "s";
                if (seconds < 3600) return Math.floor(seconds / 60) + "m " + Math.floor(seconds % 60) + "s";
                return Math.floor(seconds / 3600) + "h " + Math.floor((seconds % 3600) / 60) + "m";
            }}
            function tick() {{
                badge.textContent = formatElapsed((Date.now() - startMs) / 1000);
            }}
            tick();
            setInterval(tick, 500);
        </script>
    """, height=34)

def get_base64(img):
    buffered = BytesIO(); img.save(buffered, format="PNG")
    import base64; return base64.b64encode(buffered.getvalue()).decode()

def style_dataframe(df, raw_precision=False):
    df = normalize_metric_columns(df.copy())
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.index = pd.RangeIndex(len(df))
    
    # Identify which columns to style with background gradient.
    possible_style_cols = [
        ATTR_RUNTIME_COL, ATTR_MEMORY_COL,
        "Attribution Runtime Median (sec)", "Attribution Runtime Mean (sec)",
        "Attribution Runtime Std (sec)", "Attribution Runtime Min (sec)",
        "Attribution Runtime Max (sec)", "Mean Attribution Runtime (sec)",
        "Std Across Images (sec)", "Repeat Run Std (sec)",
        "Mean Peak Attribution Memory (MB)", "Peak Memory (MB)", "Peak Attribution Memory (MB)",
        "Std Peak Memory (MB)", "Peak Memory Std (MB)", "Attribution Memory Std (MB)", "Repeat Memory Std (MB)",
        "Gini Index", "Mean Gini Index",
        "Deletion AUC", "Mean Deletion AUC",
        "Insertion AUC", "Mean Insertion AUC",
        "Infidelity", "Mean Infidelity",
        "Quality Eval Time (sec)", "Mean Quality Eval Time (sec)"
    ]
    
    subset_cols = [
        c for c in possible_style_cols 
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().shape[0] > 0
    ]
            
    def make_formatter(col_name):
        def _fmt(val):
            if pd.isna(val) or val is None:
                return "–"
            try:
                v = float(val)
            except Exception:
                return str(val)

            if raw_precision:
                if col_name == "Resolution":
                    return f"{int(round(v))} x {int(round(v))}"
                elif col_name in ["Input Size (px)", "Samples", "Methods", "Resolutions", "Images", "Repeats", "Total Attribution Runs"]:
                    return f"{int(round(v))}"
                s = f"{v:.6f}".rstrip('0').rstrip('.')
                return s if s else "0"
            else:
                if "MB" in col_name or col_name in [ATTR_MEMORY_COL, "Mean Peak Attribution Memory (MB)", "Std Peak Memory (MB)", "Peak Memory Std (MB)", "Attribution Memory Std (MB)"]:
                    return f"{v:.2f}"
                elif "sec" in col_name or "Infidelity" in col_name or col_name in ["Gini Index", "Mean Gini Index", "Deletion AUC", "Mean Deletion AUC", "Insertion AUC", "Mean Insertion AUC"]:
                    return f"{v:.4f}"
                elif col_name == "Resolution":
                    return f"{int(round(v))} x {int(round(v))}"
                elif col_name in ["Input Size (px)", "Samples", "Methods", "Resolutions", "Images", "Repeats", "Total Attribution Runs"]:
                    return f"{int(round(v))}"
                else:
                    return f"{v:.4f}"
        return _fmt

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    formatters = {c: make_formatter(c) for c in numeric_cols}

    styler = df.style
    if subset_cols:
        lower_is_better_cols = [c for c in subset_cols if not any(k in c for k in ["Gini", "Insertion"])]
        higher_is_better_cols = [c for c in subset_cols if any(k in c for k in ["Gini", "Insertion"])]

        try:
            for col in lower_is_better_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    styler = styler.background_gradient(cmap="coolwarm", subset=[col])
            for col in higher_is_better_cols:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    styler = styler.background_gradient(cmap="coolwarm_r", subset=[col])
        except Exception as e:
            logger.warning(f"Background gradient styling skipped due to Pandas Styler incompatibility: {e}")
        
    # Right-align all columns containing metrics or numeric dimensions to keep dashes aligned with numbers
    right_align_cols = [c for c in df.columns if c not in ["Method", "Model", "Resolution", "Prediction", "Status", "Device", "Model_Size"]]
    if right_align_cols:
        styler = styler.set_properties(**{"text-align": "right !important"}, subset=right_align_cols)
        
        # Also right-align the column headers (th) for these columns to align with data cells
        header_styles = []
        for idx, col in enumerate(df.columns):
            if col in right_align_cols:
                header_styles.append({
                    "selector": f"th.col{idx}",
                    "props": [("text-align", "right !important")]
                })
        if header_styles:
            styler = styler.set_table_styles(header_styles, overwrite=False)
        
    styler = styler.format(formatters, na_rep="–")
    if hasattr(styler, "hide"):
        styler = styler.hide()
    elif hasattr(styler, "hide_index"):
        styler = styler.hide_index()
    return styler

def plot_method_runtime_log(df, title="Runtime Comparison (Log Scale)"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    
    # Sort methods by median runtime
    order = list(df.groupby("Method")[runtime_col].median().sort_values(ascending=True).index)
    
    stats = []
    for method in order:
        vals = df[df["Method"] == method][runtime_col].values
        vals = vals[vals > 0]  # Positive values only for log scale
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        q1_val = float(np.percentile(vals, 25))
        q3_val = float(np.percentile(vals, 75))
        
        stats.append({
            "label": method,
            "med": mean_val,
            "q1": q1_val,
            "q3": q3_val,
            "whislo": max(min_val, mean_val - std_val),
            "whishi": min(max_val, mean_val + std_val),
            "fliers": []
        })
        
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats:
        bp = ax.bxp(stats, vert=False, patch_artist=True, showmeans=False)
        
        # Style elements to match the requested look
        for patch in bp['boxes']:
            patch.set_facecolor('#8ecae6')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
            
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
            
        for line in bp['whiskers']:
            line.set_color('black')
            line.set_linestyle('--')
            
        for line in bp['caps']:
            line.set_color('black')
            
        def format_label(val):
            if val == 0: return "0.00"
            if val >= 10: return f"{val:.1f}"
            if val >= 0.1: return f"{val:.2f}"
            if val >= 0.001: return f"{val:.3f}"
            return f"{val:.4f}"
            
        for i, s in enumerate(stats):
            y = i + 1
            ax.text(s["med"], y + 0.30, format_label(s["med"]), ha='center', va='bottom', fontsize=8, color='black', alpha=0.9, fontweight='bold')
            if s["whislo"] < s["med"]:
                ax.text(s["whislo"], y - 0.33, format_label(s["whislo"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)
            if s["whishi"] > s["med"]:
                ax.text(s["whishi"], y - 0.33, format_label(s["whishi"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)

    ax.set_xscale("log")
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Runtime (log scale, seconds)")
    ax.set_ylabel("XAI Method")
    ax.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    return fig

def plot_method_memory(df, title="Peak Memory Overhead"):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    valid_df = df[df[memory_col].notna()].copy()
    if valid_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Memory profiling disabled or not available", ha='center', va='center', fontsize=12, color='gray')
        ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
        ax.axis('off')
        return fig

    # Sort methods by median memory
    order = list(valid_df.groupby("Method")[memory_col].median().sort_values(ascending=True).index)
    
    stats = []
    for method in order:
        vals = valid_df[valid_df["Method"] == method][memory_col].values
        vals = np.array([v for v in vals if v is not None and pd.notna(v) and v >= 0], dtype=float)
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))
        q1_val = float(np.percentile(vals, 25))
        q3_val = float(np.percentile(vals, 75))
        
        stats.append({
            "label": method,
            "med": mean_val,
            "q1": q1_val,
            "q3": q3_val,
            "whislo": max(min_val, mean_val - std_val),
            "whishi": min(max_val, mean_val + std_val),
            "fliers": []
        })
        
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats:
        bp = ax.bxp(stats, vert=False, patch_artist=True, showmeans=False)
        
        # Style elements
        for patch in bp['boxes']:
            patch.set_facecolor('#ffb5a7')
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
            
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.5)
            
        for line in bp['whiskers']:
            line.set_color('black')
            line.set_linestyle('--')
            
        for line in bp['caps']:
            line.set_color('black')
            
        def format_label(val):
            if val == 0: return "0.0"
            if val >= 10: return f"{val:.1f}"
            return f"{val:.2f}"
            
        # Draw labels: std below, mean above
        for i, s in enumerate(stats):
            y = i + 1
            ax.text(s["med"], y + 0.30, format_label(s["med"]), ha='center', va='bottom', fontsize=8, color='black', alpha=0.9, fontweight='bold')
            if s["whislo"] < s["med"]:
                ax.text(s["whislo"], y - 0.33, format_label(s["whislo"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)
            if s["whishi"] > s["med"]:
                ax.text(s["whishi"], y - 0.33, format_label(s["whishi"]), ha='center', va='top', fontsize=7, color='black', alpha=0.75)

    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Peak Attribution Memory (MB)")
    ax.set_ylabel("XAI Method")
    ax.grid(True, ls="-", alpha=0.2)
    plt.tight_layout()
    return fig



def plot_model_comparison_grouped(df, title="Architecture Efficiency Comparison"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort methods (X-axis) by mean runtime (small to big)
    methods_order = list(df.groupby("Method")[runtime_col].mean().sort_values(ascending=True).index)
    # Sort models (Hue/Legend) by mean runtime (small to big)
    hue_order = list(df.groupby("Model")[runtime_col].mean().sort_values(ascending=True).index)
    
    sns.barplot(
        data=df, 
        x="Method", 
        order=methods_order,
        y=runtime_col, 
        hue="Model", 
        hue_order=hue_order,
        palette="colorblind", 
        ax=ax, 
        edgecolor="black",
        errorbar="sd"
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean Runtime (log scale, seconds)")
    ax.set_xlabel("XAI Method")
    
    # Put values on top of the bars dynamically based on height with a background mask
    import math
    bg_color = ax.get_facecolor()
    
    # Adaptive label rotation to prevent collisions
    num_hues = len(ax.containers)
    rot = 90 if num_hues > 3 else 0
    font_sz = 7 if num_hues > 4 else 8
    pad_val = 5 if rot == 90 else 3
    
    # Add top margin to prevent rotated labels clipping
    ax.margins(y=0.25 if rot == 90 else 0.15)
    
    for container in ax.containers:
        labels = []
        for rect in container:
            height = rect.get_height()
            if math.isnan(height) or height <= 0:
                labels.append("")
            elif height < 0.1:
                labels.append(f"{height:.3f}s")
            else:
                labels.append(f"{height:.2f}s")
        bar_labels = ax.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
        for label in bar_labels:
            label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.85))
            
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.legend(title="Model Architecture", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_model_memory_comparison_grouped(df, title="Architecture Peak Memory Comparison"):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Sort methods (X-axis) by mean memory (small to big)
    methods_order = list(df.groupby("Method")[memory_col].mean().sort_values(ascending=True).index)
    # Sort models (Hue/Legend) by mean memory (small to big)
    hue_order = list(df.groupby("Model")[memory_col].mean().sort_values(ascending=True).index)
    
    sns.barplot(
        data=df, 
        x="Method", 
        order=methods_order,
        y=memory_col, 
        hue="Model", 
        hue_order=hue_order,
        palette="colorblind", 
        ax=ax, 
        edgecolor="black",
        errorbar="sd"
    )
    ax.set_ylabel("Mean Peak Memory (MB)")
    ax.set_xlabel("XAI Method")
    
    # Put values on top of the bars dynamically based on height with a background mask
    import math
    bg_color = ax.get_facecolor()
    
    # Adaptive label rotation to prevent collisions
    num_hues = len(ax.containers)
    rot = 90 if num_hues > 3 else 0
    font_sz = 7 if num_hues > 4 else 8
    pad_val = 5 if rot == 90 else 3
    
    # Add top margin to prevent rotated labels clipping
    ax.margins(y=0.25 if rot == 90 else 0.15)
    
    for container in ax.containers:
        labels = []
        for rect in container:
            height = rect.get_height()
            if math.isnan(height) or height <= 0:
                labels.append("")
            else:
                labels.append(f"{height:.1f}MB")
        bar_labels = ax.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
        for label in bar_labels:
            label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.85))
            
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.legend(title="Model Architecture", loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, ls="-", alpha=0.2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def image_size_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    if "Input Size (px)" not in df.columns or df["Input Size (px)"].nunique() < 2:
        return pd.DataFrame()

    summary = df.groupby(["Input Size (px)"]).agg(
        **{
            "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
            "Std Across Images (sec)": (runtime_col, "std"),
            "Samples": (runtime_col, "count"),
            "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
            "Std Peak Memory (MB)": (memory_col, "std"),
        }
    ).reset_index()
    # Rename Column to Resolution
    summary = summary.rename(columns={"Input Size (px)": "Resolution"})
    return summary.sort_values("Resolution")

def plot_image_size_runtime_scaling(summary_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df = summary_df.sort_values("Resolution")
    ax.errorbar(
        summary_df["Resolution"],
        summary_df["Mean Attribution Runtime (sec)"],
        yerr=summary_df["Std Across Images (sec)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        color="#1f77b4"
    )
    ax.set_xticks(summary_df["Resolution"])
    ax.set_xticklabels([str(r) for r in summary_df["Resolution"]])
    
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("Mean Attribution Runtime (sec)")
    ax.set_title("Overall Runtime Scaling by Resolution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def plot_image_size_memory_scaling(summary_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    summary_df = summary_df.sort_values("Resolution")
    ax.errorbar(
        summary_df["Resolution"],
        summary_df["Mean Peak Attribution Memory (MB)"],
        yerr=summary_df["Std Peak Memory (MB)"],
        marker="o",
        linestyle="-",
        linewidth=2,
        capsize=4,
        color="#d62728"
    )
    ax.set_xticks(summary_df["Resolution"])
    ax.set_xticklabels([str(r) for r in summary_df["Resolution"]])
    
    ax.set_xlabel("Resolution (px)")
    ax.set_ylabel("Mean Peak Attribution Memory (MB)")
    ax.set_title("Overall Memory Scaling by Resolution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


def method_detail_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    n_images = int(df["Image Index"].nunique()) if "Image Index" in df.columns else 1
    std_runtime_src = "Attribution Runtime Std (sec)" if (n_images == 1 and "Attribution Runtime Std (sec)" in df.columns) else runtime_col
    std_runtime_func = "mean" if n_images == 1 else "std"

    std_mem_src = "Attribution Memory Std (MB)" if (n_images == 1 and "Attribution Memory Std (MB)" in df.columns) else memory_col
    std_mem_func = "mean" if n_images == 1 else "std"

    agg_dict = {
        "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
        "Attribution Runtime Std (sec)": (std_runtime_src, std_runtime_func),
        "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
        "Peak Memory Std (MB)": (std_mem_src, std_mem_func),
        "Samples": (runtime_col, "count"),
    }
    if "Gini Index" in df.columns and df["Gini Index"].notna().any():
        agg_dict["Mean Gini Index"] = ("Gini Index", "mean")
    if "Deletion AUC" in df.columns and df["Deletion AUC"].notna().any():
        agg_dict["Mean Deletion AUC"] = ("Deletion AUC", "mean")
    if "Insertion AUC" in df.columns and df["Insertion AUC"].notna().any():
        agg_dict["Mean Insertion AUC"] = ("Insertion AUC", "mean")
    if "Infidelity" in df.columns and df["Infidelity"].notna().any():
        agg_dict["Mean Infidelity"] = ("Infidelity", "mean")
    if "Quality Eval Time (sec)" in df.columns and df["Quality Eval Time (sec)"].notna().any():
        agg_dict["Mean Quality Eval Time (sec)"] = ("Quality Eval Time (sec)", "mean")
    summary = df.groupby("Method", sort=False).agg(**agg_dict).reset_index()
    return summary

def fastest_slowest_rows(df, count=5):
    df = presentation_df(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    cols = [c for c in ["Method", "Model", "Resolution", "Prediction", runtime_col, ATTR_MEMORY_COL, "Gini Index"] if c in df.columns]
    fastest = df.nsmallest(count, runtime_col)[cols]
    slowest = df.nlargest(count, runtime_col)[cols]
    return fastest, slowest

def plot_runtime_distribution(df):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="Method", y=runtime_col, ax=ax, color="#8ecae6")
    ax.set_title("Attribution Runtime Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel("Attribution Runtime (sec)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_runtime_memory_scatter(df, figsize=(9, 6)):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    # Group by configuration (Method, Model, Resolution) and average the metrics (across all images)
    group_cols = ["Method", "Model", "Resolution"]
    df_config = df.groupby(group_cols).agg({runtime_col: "mean", memory_col: "mean"}).reset_index()
    
    # Combine Model and Resolution into a single column for marker style grouping
    df_config["Model (Resolution)"] = df_config["Model"] + " (" + df_config["Resolution"] + ")"
    
    fig, ax = plt.subplots(figsize=figsize)
    # Style represents Model (Resolution), Hue represents Method. Constant size (s=110)
    sns.scatterplot(
        data=df_config, 
        x=runtime_col, 
        y=memory_col, 
        hue="Method", 
        style="Model (Resolution)", 
        s=110, 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Runtime vs Peak Attribution Memory", fontsize=14, fontweight='bold')
    ax.set_xlabel("Mean Attribution Runtime (sec)")
    ax.set_ylabel("Mean Peak Attribution Memory (MB)")
    
    # Set axis limits starting at 0 with 10% padding on the max side to prevent cramped borders
    x_max = df_config[runtime_col].max()
    y_max = df_config[memory_col].max()
    x_limit = (x_max * 1.1) if not pd.isna(x_max) and x_max > 0 else 1.0
    y_limit = (y_max * 1.1) if not pd.isna(y_max) and y_max > 0 else 100.0
    ax.set_xlim(left=0.0, right=x_limit)
    ax.set_ylim(bottom=0.0, top=y_limit)
    
    # Extract legend handles and insert an empty spacer row between different categories
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    from matplotlib.patches import Patch
    
    for h, l in zip(handles, labels):
        # Prepend a blank row when moving to the Model (Resolution) section
        if l in ["Model (Resolution)", "Model", "Input Size (px)"] and len(new_labels) > 0:
            new_handles.append(Patch(color='none', label=''))
            new_labels.append('')
        new_handles.append(h)
        new_labels.append(l)
        
    ax.legend(new_handles, new_labels, loc='upper left', bbox_to_anchor=(1, 1), labelspacing=0.65)
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig

def render_analytics_sections(fdf, result_groups):
    """Render adaptive, collapsible analytics sections based on which dimensions vary."""
    fdf = normalize_metric_columns(fdf)
    runtime_col = metric_col(fdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(fdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    n_methods = fdf["Method"].nunique()
    n_models = fdf["Model"].nunique()
    n_resolutions = fdf["Resolution"].nunique() if "Resolution" in fdf.columns else 1

    # ── Shared workload metric card helper ────────────────────────────────
    def _stat_card(label, value, is_result=False, suffix=""):
        if is_result:
            bg, border = "rgba(16,185,129,0.08)", "1.5px solid rgba(16,185,129,0.45)"
            lbl_clr, val_clr = "#6ee7b7", "#34d399"
            suffix_html = f'<div style="font-size:0.62rem;color:#6ee7b7;opacity:0.7;margin-top:2px;">{suffix}</div>' if suffix else ""
        else:
            bg, border = "rgba(99,102,241,0.07)", "1.5px solid rgba(99,102,241,0.35)"
            lbl_clr, val_clr = "#a5b4fc", "inherit"
            suffix_html = ""
        return (
            f'<div style="flex:1;background:{bg};border:{border};border-radius:8px;'
            f'padding:8px 12px 6px 12px;text-align:center;min-width:0;">'
            f'<div style="font-size:0.68rem;color:{lbl_clr};letter-spacing:0.07em;'
            f'text-transform:uppercase;margin-bottom:2px;">{label}</div>'
            f'<div style="font-size:1.35rem;font-weight:700;line-height:1.1;color:{val_clr};">{value}</div>'
            + suffix_html + '</div>'
        )

    def _sep(symbol):
        clr = "#34d399" if symbol == "=" else "#6366f1"
        return (
            f'<span style="font-size:1.2rem;font-weight:500;color:{clr};'
            f'flex-shrink:0;padding:0 1px;opacity:0.8;">{symbol}</span>'
        )

    def _cards_row(*items):
        """items: list of (label, value) or (label, value, is_result) or (label, value, is_result, suffix) or (sep_symbol,)"""
        html = '<div style="display:flex;align-items:center;gap:6px;width:100%;margin-bottom:12px;">'
        for item in items:
            if len(item) == 1:
                html += _sep(item[0])
            else:
                label, value = item[0], item[1]
                is_result = item[2] if len(item) > 2 else False
                suffix = item[3] if len(item) > 3 else ""
                html += _stat_card(label, value, is_result, suffix)
        html += "</div>"
        return html
    # ─────────────────────────────────────────────────────────────────────

    # If nothing varies, the per-image results table is sufficient
    if n_methods <= 1 and n_models <= 1 and n_resolutions <= 1:
        return
    
    with st.expander("📊 Performance Analysis", expanded=True):
        # 1. Configuration Averages Table at the very top
        group_cols = ["Method", "Model", "Resolution"]
        n_images = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
        std_runtime_src = "Attribution Runtime Std (sec)" if (n_images == 1 and "Attribution Runtime Std (sec)" in fdf.columns) else runtime_col
        std_runtime_func = "mean" if n_images == 1 else "std"

        std_mem_src = "Attribution Memory Std (MB)" if (n_images == 1 and "Attribution Memory Std (MB)" in fdf.columns) else memory_col
        std_mem_func = "mean" if n_images == 1 else "std"

        agg_dict_config = {
            "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
            "Attribution Runtime Std (sec)": (std_runtime_src, std_runtime_func),
            "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
            "Peak Memory Std (MB)": (std_mem_src, std_mem_func),
            "Samples": (runtime_col, "count"),
        }
        if "Gini Index" in fdf.columns and fdf["Gini Index"].notna().any():
            agg_dict_config["Mean Gini Index"] = ("Gini Index", "mean")
        if "Deletion AUC" in fdf.columns and fdf["Deletion AUC"].notna().any():
            agg_dict_config["Mean Deletion AUC"] = ("Deletion AUC", "mean")
        if "Insertion AUC" in fdf.columns and fdf["Insertion AUC"].notna().any():
            agg_dict_config["Mean Insertion AUC"] = ("Insertion AUC", "mean")
        if "Infidelity" in fdf.columns and fdf["Infidelity"].notna().any():
            agg_dict_config["Mean Infidelity"] = ("Infidelity", "mean")
        if "Quality Eval Time (sec)" in fdf.columns and fdf["Quality Eval Time (sec)"].notna().any():
            agg_dict_config["Mean Quality Eval Time (sec)"] = ("Quality Eval Time (sec)", "mean")
        summary_df = fdf.groupby(group_cols).agg(**agg_dict_config).reset_index()
        with st.expander("📋 Configuration Averages", expanded=True):
            _c_images  = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
            _c_repeats = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
            _c_config  = _c_images * _c_repeats
            _c_total   = n_models * n_methods * n_resolutions * _c_images * _c_repeats
            st.html(_cards_row(
                ("Images", _c_images),
                ("×",),
                ("Repeats", _c_repeats),
                ("=",),
                ("Runs per Config", _c_config, True),
            ))
            st.table(style_dataframe(summary_df))
        
            # 2. Side-by-side Configuration Charts of matching height (12, 7)
            cs1, cs2 = st.columns(2)
            with cs1:
                # Architecture & Resolution Efficiency Chart (Log Scale with values)
                fdf_plot = fdf.copy()
                fdf_plot["Model_Size"] = fdf_plot["Model"] + " (" + fdf_plot["Resolution"] + ")"
                
                # Build custom palette: different base color per model, shades per resolution (larger is darker)
                import colorsys
                import matplotlib.colors as mcolors
                import re
                
                def parse_res_num(res_str):
                    match = re.search(r'\d+', str(res_str))
                    return int(match.group()) if match else 0
                
                base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                unique_models = sorted(fdf_plot["Model"].unique())
                model_to_base = {model: base_colors[idx % len(base_colors)] for idx, model in enumerate(unique_models)}
                
                custom_palette = {}
                hue_order = []
                for model in unique_models:
                    model_sizes = fdf_plot[fdf_plot["Model"] == model]["Model_Size"].unique()
                    sorted_sizes = sorted(model_sizes, key=parse_res_num)
                    hue_order.extend(sorted_sizes)
                    V = len(sorted_sizes)
                    base_color = model_to_base[model]
                    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(base_color))
                    
                    for i, ms in enumerate(sorted_sizes):
                        mult = 1.35 - (i / (V - 1)) * 0.7 if V > 1 else 1.0
                        new_l = max(0.1, min(0.9, l * mult))
                        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
                        custom_palette[ms] = mcolors.to_hex(new_rgb)
                
                # Order X-axis Methods from fastest to slowest
                methods_order = list(fdf_plot.groupby("Method")[runtime_col].mean().sort_values(ascending=True).index)
                
                fig_bar, ax_bar = plt.subplots(figsize=(12, 7))
                sns.barplot(
                    data=fdf_plot, 
                    x="Method", 
                    order=methods_order,
                    y=runtime_col, 
                    hue="Model_Size", 
                    hue_order=hue_order,
                    palette=custom_palette, 
                    ax=ax_bar, 
                    edgecolor="black", 
                    errorbar="sd"
                )
                
                # Apply log scale
                ax_bar.set_yscale("log")
                ax_bar.set_ylabel("Mean Attribution Runtime (sec, log scale)")
                
                # Put values on top of the bars dynamically based on height with a background mask
                import math
                bg_color = ax_bar.get_facecolor()
                
                # Adaptive label rotation to prevent collisions
                num_hues = len(ax_bar.containers)
                rot = 90 if num_hues > 3 else 0
                font_sz = 7 if num_hues > 4 else 8
                pad_val = 5 if rot == 90 else 3
                
                # Add top margin to prevent rotated labels clipping
                ax_bar.margins(y=0.25 if rot == 90 else 0.15)
                
                for container in ax_bar.containers:
                    labels = []
                    for rect in container:
                        height = rect.get_height()
                        if math.isnan(height) or height <= 0:
                            labels.append("")
                        elif height < 0.1:
                            labels.append(f"{height:.3f}s")
                        else:
                            labels.append(f"{height:.2f}s")
                    bar_labels = ax_bar.bar_label(container, labels=labels, padding=pad_val, fontsize=font_sz, rotation=rot)
                    for label in bar_labels:
                        label.set_bbox(dict(facecolor=bg_color, edgecolor='none', pad=0.8, alpha=0.9))
                    
                ax_bar.set_title("Architecture & Resolution Efficiency", fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                ax_bar.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.tight_layout()
                st.pyplot(fig_bar)
                
            with cs2:
                # Runtime vs Memory Scatter Plot (matching height 12, 7)
                st.pyplot(plot_runtime_memory_scatter(fdf, figsize=(12, 7)))
            
        st.markdown('<div style="margin-top: 1.5rem; border-top: 1px solid var(--xai-border); padding-top: 1rem;"></div>', unsafe_allow_html=True)
        render_detailed_comparisons_header("comparison_section")
        
        # 3. Nested Collapsible Sections for varying dimensions
        
        # --- Method Comparison: multiple XAI methods were benchmarked ---
        if n_methods > 1:
            with st.expander("📊 XAI Method Comparison", expanded=False):
                _m_models      = int(fdf["Model"].nunique())
                _m_resolutions = int(fdf["Resolution"].nunique())
                _m_images      = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                _m_repeats     = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                _m_total       = _m_models * _m_resolutions * _m_images * _m_repeats
                st.html(_cards_row(
                    ("Models", _m_models),
                    ("×",),
                    ("Resolutions", _m_resolutions),
                    ("×",),
                    ("Images", _m_images),
                    ("×",),
                    ("Repeats", _m_repeats),
                    ("=",),
                    ("Runs per Method", _m_total, True),
                ))
                st.table(style_dataframe(method_detail_summary(fdf)))
                cm1, cm2 = st.columns(2)
                with cm1:
                    st.pyplot(plot_method_runtime_log(fdf))
                with cm2:
                    st.pyplot(plot_method_memory(fdf))
        
        # --- Model Comparison: multiple architectures were benchmarked ---
        if n_models > 1:
            with st.expander("🏗️ Model Comparison", expanded=False):
                unique_methods     = int(fdf["Method"].nunique())
                unique_resolutions = int(fdf["Resolution"].nunique())
                unique_images      = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                repeats            = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                total_runs         = unique_methods * unique_resolutions * unique_images * repeats
                st.html(_cards_row(
                    ("Methods", unique_methods),
                    ("×",),
                    ("Resolutions", unique_resolutions),
                    ("×",),
                    ("Images", unique_images),
                    ("×",),
                    ("Repeats", repeats),
                    ("=",),
                    ("Runs per Model", total_runs, True),
                ))
                
                # Performance comparison table
                model_agg_dict = {
                    "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
                    "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
                    "Samples": (runtime_col, "count"),
                }
                if "Gini Index" in fdf.columns and fdf["Gini Index"].notna().any():
                    model_agg_dict["Mean Gini Index"] = ("Gini Index", "mean")
                if "Deletion AUC" in fdf.columns and fdf["Deletion AUC"].notna().any():
                    model_agg_dict["Mean Deletion AUC"] = ("Deletion AUC", "mean")
                if "Insertion AUC" in fdf.columns and fdf["Insertion AUC"].notna().any():
                    model_agg_dict["Mean Insertion AUC"] = ("Insertion AUC", "mean")
                if "Infidelity" in fdf.columns and fdf["Infidelity"].notna().any():
                    model_agg_dict["Mean Infidelity"] = ("Infidelity", "mean")
                if "Quality Eval Time (sec)" in fdf.columns and fdf["Quality Eval Time (sec)"].notna().any():
                    model_agg_dict["Mean Quality Eval Time (sec)"] = ("Quality Eval Time (sec)", "mean")
                model_summary = fdf.groupby("Model").agg(**model_agg_dict).reset_index().sort_values("Mean Attribution Runtime (sec)")
                st.table(style_dataframe(model_summary))
                
                mc1, mc2 = st.columns(2)
                with mc1:
                    st.pyplot(plot_model_comparison_grouped(fdf))
                with mc2:
                    st.pyplot(plot_model_memory_comparison_grouped(fdf))
        
        # --- Resolution Comparison: multiple input resolutions were benchmarked ---
        if n_resolutions > 1:
            size_summary_df = image_size_summary(fdf)
            if not size_summary_df.empty:
                with st.expander("📐 Resolution Comparison", expanded=False):
                    _r_models  = int(fdf["Model"].nunique())
                    _r_methods = int(fdf["Method"].nunique())
                    _r_images  = int(fdf["Image Index"].nunique()) if "Image Index" in fdf.columns else 1
                    _r_repeats = int(fdf["Measured Runs"].iloc[0]) if "Measured Runs" in fdf.columns else 1
                    _r_total   = _r_models * _r_methods * _r_images * _r_repeats
                    st.html(_cards_row(
                        ("Models", _r_models),
                        ("×",),
                        ("Methods", _r_methods),
                        ("×",),
                        ("Images", _r_images),
                        ("×",),
                        ("Repeats", _r_repeats),
                        ("=",),
                        ("Runs per Resolution", _r_total, True),
                    ))
                    st.table(style_dataframe(size_summary_df))
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.pyplot(plot_image_size_runtime_scaling(size_summary_df))
                    with rc2:
                        st.pyplot(plot_image_size_memory_scaling(size_summary_df))
    


def render_environment_summary(environment):
    if not environment:
        return
        
    # Helper to clean strings and handle missing
    def clean_val(v):
        if v is None or str(v).strip() in ["", "nan", "None", ".", "unknown", "not available"]:
            return "-"
        return str(v)

    env_rows = [
        ("Platform", clean_val(environment.get("platform"))),
        ("CPU", clean_val(environment.get("processor"))),
    ]

    cuda_devices = environment.get("cuda_devices", [])
    selected_device = str(environment.get("selected_device", "")).lower()
    if cuda_devices:
        gpu_names = ", ".join([d.get("name", "Unknown GPU") for d in cuda_devices])
        env_rows.append(("GPU Device(s)", clean_val(gpu_names)))
    elif "mps" in selected_device:
        env_rows.append(("GPU Device", "Apple Silicon MPS"))

    exec_dev = clean_val(environment.get("selected_device", "cpu")).upper()
    if "CUDA" in exec_dev or "MPS" in exec_dev:
        exec_dev = f"{exec_dev} (GPU)"
    env_rows.append(("Execution Device", exec_dev))

    python_v = clean_val(environment.get("python_version"))
    torch_v = clean_val(environment.get("torch_version"))
    cuda_v = environment.get("torch_cuda_version")
    
    if cuda_v and str(cuda_v).strip().lower() not in ["", "none", "nan", "."]:
        stack_str = f"Python {python_v} | PyTorch {torch_v} (CUDA {cuda_v})"
    else:
        stack_str = f"Python {python_v} | PyTorch {torch_v}"
        
    env_rows.append(("Software Stack", stack_str))
    env_rows.append(("Git Commit", clean_val(environment.get("git_commit"))))
    
    # Build HTML table with controlled column widths and style matching Config Metadata
    table_html = """
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88rem; color: var(--xai-text); margin-bottom: 0.5rem;">
      <thead>
        <tr style="border-bottom: 2px solid var(--xai-border); text-align: left; color: var(--xai-muted);">
          <th style="padding: 8px 10px; width: 160px; font-weight: 600;">Field</th>
          <th style="padding: 8px 10px; font-weight: 600;">Value</th>
        </tr>
      </thead>
      <tbody>
    """
    
    for field, value in env_rows:
        table_html += f"""
        <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.12);">
          <td style="padding: 8px 10px; font-weight: 500; color: var(--xai-text);">{field}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{value}</td>
        </tr>
        """
        
    table_html += """
      </tbody>
    </table>
    """
    
    # Clean up whitespace to prevent markdown block parsing issues
    clean_html = table_html.replace("\n", "").replace("    ", "").strip()
    st.markdown(clean_html, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_batch_display_name(bid, base_dir):
    batch_meta_p = os.path.join(base_dir, bid, "batch_results.json")
    if os.path.exists(batch_meta_p):
        try:
            with open(batch_meta_p, 'r') as f:
                meta = json.load(f)
            
            # Format date & time from: Batch_YYYYMMDD_HHMMSS or Batch_YYYY-MM-DD_HH-MM-SS
            parts = bid.split("_")
            if len(parts) >= 3:
                date_part = parts[1]
                time_part = parts[2]
                
                # Format date to DD-MM-YYYY
                if len(date_part) == 8: # YYYYMMDD
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:]
                    date_str = f"{day}-{month}-{year}"
                elif "-" in date_part: # YYYY-MM-DD
                    y_m_d = date_part.split("-")
                    if len(y_m_d) == 3:
                        date_str = f"{y_m_d[2]}-{y_m_d[1]}-{y_m_d[0]}"
                    else:
                        date_str = date_part
                else:
                    date_str = date_part
                    
                # Format time
                if len(time_part) == 6: # HHMMSS
                    time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:]}"
                else:
                    time_str = time_part.replace("-", ":")
                    
                # 4 non-breaking spaces between date and time
                display_time = f"{date_str}\u00A0\u00A0\u00A0\u00A0{time_str}"
            else:
                display_time = bid
                
            results = meta.get("results", [])
            img_count = len(results)
            
            settings = meta.get("benchmark_settings", {}) or {}
            models = settings.get("models", []) or []
            methods = settings.get("methods", []) or meta.get("methods", []) or []
            
            # Fallback if settings are empty
            if not models or not methods:
                scanned_models = set()
                scanned_methods = set()
                for g in results:
                    for m in g.get("models", []):
                        if m.get("model_name"):
                            scanned_models.add(m.get("model_name"))
                        for r in m.get("results", []):
                            if r.get("Method"):
                                scanned_methods.add(r.get("Method"))
                if not models:
                    models = list(scanned_models)
                if not methods:
                    methods = list(scanned_methods)
                
            img_lbl = f"{img_count} img" if img_count == 1 else f"{img_count} imgs"
            model_lbl = f"{len(models)} model" if len(models) == 1 else f"{len(models)} models"
            method_lbl = f"{len(methods)} method" if len(methods) == 1 else f"{len(methods)} methods"
            
            # 4 non-breaking spaces before details parenthesis
            return f"{display_time}\u00A0\u00A0\u00A0\u00A0({img_lbl}, {model_lbl}, {method_lbl})"
        except Exception:
            return bid
    return bid

def render_parameters_mapping_table(methods_info):
    if not methods_info:
        return
        
    # Filter only methods that have configured parameters
    parameterized = [m for m in methods_info if m.get("params")]
    if not parameterized:
        return
        
    st.markdown('<div style="font-weight: 600; margin-top: 15px; margin-bottom: 8px; color: var(--xai-text);">Method Parameters</div>', unsafe_allow_html=True)
    
    table_html = (
        '<table style="width:100%; border-collapse:collapse; font-size:0.85em; margin-bottom:20px; color:var(--xai-text);">'
        '<thead>'
        '<tr style="border-bottom:2px solid rgba(255,255,255,0.15); text-align:left;">'
        '<th style="padding:8px; width:25%;">Method Identifier</th>'
        '<th style="padding:8px; width:25%;">Base Algorithm</th>'
        '<th style="padding:8px; width:50%;">Configured Parameters</th>'
        '</tr>'
        '</thead>'
        '<tbody>'
    )
    for m in parameterized:
        disp_name = m["display_name"]
        base_name = m["base_name"].replace("_", " ").title()
        params = m["params"]
        
        # Format the parameters dictionary into a readable string
        formatted_params = []
        for k, v in params.items():
            k_pretty = k.replace("_", " ").title()
            if isinstance(v, list) or isinstance(v, tuple):
                # Format sliding window shapes/strides tuple cleanly
                if len(v) == 3:
                    v_str = f"({v[0]}, {v[1]}, {v[2]})"
                else:
                    v_str = str(v)
            else:
                v_str = str(v)
            formatted_params.append(f"<b>{k_pretty}</b>: {v_str}")
        
        params_str = ", ".join(formatted_params)
        table_html += (
            '<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">'
            f'<td style="padding:8px; font-family:monospace; font-weight:bold; color:var(--xai-text);">{disp_name}</td>'
            f'<td style="padding:8px;">{base_name}</td>'
            f'<td style="padding:8px; line-height:1.4;">{params_str}</td>'
            '</tr>'
        )
        
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)

def render_configuration_summary(settings, results):
    if not results and not settings:
        return
    # Fallback to scanning results if settings is empty/missing
    if not settings:
        settings = {}
        
    models = settings.get("models", [])
    methods = settings.get("methods", [])
    sizes = settings.get("input_sizes", [])
    repeats = settings.get("repeat_count")
    warmups = settings.get("warmup_runs")
    memory_runs = settings.get("memory_runs", 1)
    
    if not models or not methods or not sizes:
        scanned_models = set()
        scanned_methods = set()
        scanned_sizes = set()
        if results:
            for g in results:
                for m in g.get("models", []):
                    if m.get("model_name"):
                        scanned_models.add(m.get("model_name"))
                    for r in m.get("results", []):
                        if r.get("Method"):
                            scanned_methods.add(r.get("Method"))
                        if "Resolution" in r and r.get("Resolution"):
                            scanned_sizes.add(str(r.get("Resolution")))
        if not models:
            models = sorted(list(scanned_models))
        if not methods:
            methods = sorted(list(scanned_methods))
        if not sizes:
            sizes = sorted(list(scanned_sizes))
            
    if results:
        img_count = len(results)
    elif settings.get("image_sources"):
        img_count = len(settings.get("image_sources"))
    else:
        img_count = len(st.session_state.get("prepared_img_sources", []))
    
    # Helper to clean strings and handle missing
    def clean_val(v):
        if v is None or str(v).strip() in ["", "nan", "None", "."]:
            return "-"
        return str(v)
        
    quality_metrics = settings.get("selected_quality_metrics") or settings.get("quality_metrics") or []
    if not quality_metrics and settings.get("enable_quality_metrics") is False:
        quality_metrics = []
    elif not quality_metrics and results:
        scanned_qm = set()
        for g in results:
            for m in g.get("models", []):
                for r in m.get("results", []):
                    for qm_name in ["Gini Index", "Deletion AUC", "Insertion AUC", "Infidelity"]:
                        if qm_name in r and r.get(qm_name) is not None:
                            scanned_qm.add(qm_name)
        if scanned_qm:
            quality_metrics = sorted(list(scanned_qm))

    qm_count = len(quality_metrics) if quality_metrics else 0
    qm_detail = ", ".join(quality_metrics) if quality_metrics else "None (Disabled)"

    config_rows = [
        ("Images Analyzed", clean_val(img_count), "-"),
        ("Models", clean_val(len(models)), ", ".join(models) if models else "-"),
        ("Methods", clean_val(len(methods)), ", ".join(methods) if methods else "-"),
        ("Input Resolutions", clean_val(len(sizes)), ", ".join([str(s) for s in sizes]) if sizes else "-"),
        ("Repeats per Config", clean_val(repeats), "-"),
        ("Warmup Runs", clean_val(warmups), "-"),
        ("Memory Runs", clean_val(memory_runs), "-"),
        ("Quality Metrics", clean_val(qm_count), qm_detail),
    ]
    
    # Build HTML table with controlled column widths
    table_html = """
    <table style="width:100%; border-collapse: collapse; font-family: sans-serif; font-size: 0.88rem; color: var(--xai-text); margin-bottom: 0.5rem;">
      <thead>
        <tr style="border-bottom: 2px solid var(--xai-border); text-align: left; color: var(--xai-muted);">
          <th style="padding: 8px 10px; width: 160px; font-weight: 600;">Parameter</th>
          <th style="padding: 8px 10px; width: 80px; font-weight: 600;">Count</th>
          <th style="padding: 8px 10px; font-weight: 600;">Detail</th>
        </tr>
      </thead>
      <tbody>
    """
    
    for param, count, detail in config_rows:
        table_html += f"""
        <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.12);">
          <td style="padding: 8px 10px; font-weight: 500; color: var(--xai-text);">{param}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{count}</td>
          <td style="padding: 8px 10px; color: var(--xai-muted);">{detail}</td>
        </tr>
        """
        
    table_html += """
      </tbody>
    </table>
    """
    
    # Clean up whitespace to prevent the markdown parser from treating indented HTML as code
    clean_html = table_html.replace("\n", "").replace("    ", "").strip()
    st.markdown(clean_html, unsafe_allow_html=True)

def render_detailed_results_header(key_prefix, title_text="Detailed Per-Image Results"):
    components.html(f"""
        <style>
            body {{
                margin: 0;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 100%;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .xai-title {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #e5edf6;
                white-space: nowrap;
                line-height: 1;
                display: flex;
                align-items: center;
            }}
            .xai-expander-controls {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .xai-expander-controls button {{
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(45, 212, 191, 0.1));
                color: #e5edf6;
                cursor: pointer;
                font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 0.35rem 0.7rem;
                white-space: nowrap;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                margin: 0;
                box-sizing: border-box;
            }}
            .xai-expander-controls button:hover {{
                border-color: rgba(45, 212, 191, 0.55);
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18));
            }}
            .xai-expander-controls button:disabled {{
                opacity: 0.32 !important;
                cursor: not-allowed !important;
                border-color: rgba(148, 163, 184, 0.1) !important;
                background: rgba(255, 255, 255, 0.02) !important;
                color: rgba(229, 237, 246, 0.35) !important;
                pointer-events: none !important;
            }}
        </style>
        <div class="xai-title">{title_text}</div>
        <div class="xai-expander-controls">
            <button type="button" id="expand-btn">Expand All</button>
            <button type="button" id="collapse-btn">Collapse All</button>
        </div>
        <script>
            const root = window.parent.document;
            const expandBtn = document.getElementById('expand-btn');
            const collapseBtn = document.getElementById('collapse-btn');
            let isTransitioning = false;
            
            function imageResultDetails() {{
                return Array.from(root.querySelectorAll('details')).filter((details) => {{
                    // Filter out elements in hidden tabs (not currently visible on screen)
                    if (details.offsetParent === null) return false;
                    const summary = details.querySelector('summary');
                    return summary && summary.textContent.includes('Results for Image');
                }});
            }}
            
            function updateButtonStates() {{
                if (isTransitioning) return;
                const details = imageResultDetails();
                if (details.length === 0) {{
                    expandBtn.disabled = true;
                    collapseBtn.disabled = true;
                    return;
                }}
                const anyOpen = details.some(d => d.open);
                const anyClosed = details.some(d => !d.open);
                
                expandBtn.disabled = !anyClosed;
                collapseBtn.disabled = !anyOpen;
            }}
            
            function toggleAll(openState) {{
                const details = imageResultDetails();
                details.forEach(d => {{
                    const summary = d.querySelector('summary');
                    if (summary) {{
                        if ((openState && !d.open) || (!openState && d.open)) {{
                            summary.click();
                        }}
                    }}
                }});
            }}
            
            expandBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                // Save scroll positions of all elements in the parent document that are scrolled
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                // Lock states immediately for transitions (expand is now disabled, collapse is enabled)
                isTransitioning = true;
                expandBtn.disabled = true;
                collapseBtn.disabled = false;

                // Expand all
                toggleAll(true);

                expandBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                // Restore all scroll positions (immediate and deferred)
                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                // Release lock and re-evaluate once React animations settle
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            collapseBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                // Save scroll positions of all elements in the parent document that are scrolled
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                // Lock states immediately for transitions (collapse is now disabled, expand is enabled)
                isTransitioning = true;
                collapseBtn.disabled = true;
                expandBtn.disabled = false;

                // Collapse all
                toggleAll(false);

                collapseBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                // Restore all scroll positions (immediate and deferred)
                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                // Release lock and re-evaluate once React animations settle
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            // Listen to toggle events on the parent document (capture phase because toggle does not bubble)
            root.addEventListener('toggle', (e) => {{
                if (e.target.tagName && e.target.tagName.toLowerCase() === 'details') {{
                    // Delay state check slightly to let browser layout and React render cycles settle
                    setTimeout(updateButtonStates, 100);
                }}
            }}, true);
            
            setTimeout(updateButtonStates, 200);
            setInterval(updateButtonStates, 2000);
        </script>
    """, height=38)

def render_detailed_comparisons_header(key_prefix, title_text="Detailed Comparisons"):
    components.html(f"""
        <style>
            body {{
                margin: 0;
                background: transparent;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 100%;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .xai-title {{
                font-size: 1.2rem;
                font-weight: 600;
                color: #e5edf6;
                white-space: nowrap;
                line-height: 1;
                display: flex;
                align-items: center;
            }}
            .xai-expander-controls {{
                display: flex;
                gap: 8px;
                align-items: center;
            }}
            .xai-expander-controls button {{
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(45, 212, 191, 0.1));
                color: #e5edf6;
                cursor: pointer;
                font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 0.35rem 0.7rem;
                white-space: nowrap;
                transition: all 0.2s ease;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                line-height: 1;
                margin: 0;
                box-sizing: border-box;
            }}
            .xai-expander-controls button:hover {{
                border-color: rgba(45, 212, 191, 0.55);
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18));
            }}
            .xai-expander-controls button:disabled {{
                opacity: 0.32 !important;
                cursor: not-allowed !important;
                border-color: rgba(148, 163, 184, 0.1) !important;
                background: rgba(255, 255, 255, 0.02) !important;
                color: rgba(229, 237, 246, 0.35) !important;
                pointer-events: none !important;
            }}
        </style>
        <div class="xai-title">{title_text}</div>
        <div class="xai-expander-controls">
            <button type="button" id="expand-btn">Expand All</button>
            <button type="button" id="collapse-btn">Collapse All</button>
        </div>
        <script>
            const root = window.parent.document;
            const expandBtn = document.getElementById('expand-btn');
            const collapseBtn = document.getElementById('collapse-btn');
            let isTransitioning = false;
            
            function comparisonExpanderDetails() {{
                return Array.from(root.querySelectorAll('details')).filter((details) => {{
                    if (details.offsetParent === null) return false;
                    const summary = details.querySelector('summary');
                    return summary && summary.textContent.includes('Comparison');
                }});
            }}
            
            function updateButtonStates() {{
                if (isTransitioning) return;
                const details = comparisonExpanderDetails();
                if (details.length === 0) {{
                    expandBtn.disabled = true;
                    collapseBtn.disabled = true;
                    return;
                }}
                const anyOpen = details.some(d => d.open);
                const anyClosed = details.some(d => !d.open);
                
                expandBtn.disabled = !anyClosed;
                collapseBtn.disabled = !anyOpen;
            }}
            
            function toggleAll(openState) {{
                const details = comparisonExpanderDetails();
                details.forEach(d => {{
                    const summary = d.querySelector('summary');
                    if (summary) {{
                        if ((openState && !d.open) || (!openState && d.open)) {{
                            summary.click();
                        }}
                    }}
                }});
            }}
            
            expandBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                isTransitioning = true;
                expandBtn.disabled = true;
                collapseBtn.disabled = false;

                toggleAll(true);

                expandBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            collapseBtn.addEventListener('click', (e) => {{
                e.preventDefault();
                const parentWin = window.parent;
                const parentDoc = parentWin.document;
                
                const scrollStates = [];
                const allElements = parentDoc.querySelectorAll('*');
                allElements.forEach(el => {{
                    if (el.scrollTop > 0 || el.scrollLeft > 0) {{
                        scrollStates.push({{
                            element: el,
                            scrollTop: el.scrollTop,
                            scrollLeft: el.scrollLeft
                        }});
                    }}
                }});
                const docEl = parentDoc.documentElement;
                const bodyEl = parentDoc.body;
                const winScrollX = parentWin.scrollX || docEl.scrollLeft || bodyEl.scrollLeft;
                const winScrollY = parentWin.scrollY || docEl.scrollTop || bodyEl.scrollTop;

                isTransitioning = true;
                collapseBtn.disabled = true;
                expandBtn.disabled = false;

                toggleAll(false);

                collapseBtn.blur();
                if (parentDoc.activeElement) {{
                    parentDoc.activeElement.blur();
                }}

                const restoreScrolls = () => {{
                    scrollStates.forEach(state => {{
                        state.element.scrollTop = state.scrollTop;
                        state.element.scrollLeft = state.scrollLeft;
                    }});
                    parentWin.scrollTo(winScrollX, winScrollY);
                }};
                restoreScrolls();
                setTimeout(restoreScrolls, 10);
                setTimeout(restoreScrolls, 50);
                
                setTimeout(() => {{
                    isTransitioning = false;
                    updateButtonStates();
                }}, 1200);
            }});
            
            root.addEventListener('toggle', (e) => {{
                if (e.target.tagName && e.target.tagName.toLowerCase() === 'details') {{
                    setTimeout(updateButtonStates, 100);
                }}
            }}, true);
            
            setTimeout(updateButtonStates, 200);
            setInterval(updateButtonStates, 2000);
        </script>
    """, height=38)

def get_image_thumbnail_base64(img_path, size=(24, 24)):
    if not img_path or not os.path.exists(img_path):
        return ""
    try:
        from PIL import Image
        import base64
        from io import BytesIO
        
        with Image.open(img_path) as img:
            img_rgb = img.convert("RGB")
            
            # Crop to center square to avoid aspect ratio distortion
            width, height = img_rgb.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            
            img_square = img_rgb.crop((left, top, right, bottom))
            img_square.thumbnail(size, Image.LANCZOS)
            
            buffer = BytesIO()
            img_square.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""

def render_result_group(group, selected_methods, expanded=True, key_suffix=""):
    # Find input image path to generate thumbnail
    img_path = None
    for m in group.get("models", []):
        s_dir = m.get("session_dir", "")
        # Try absolute path first
        p = os.path.join(s_dir, "input_image.jpg")
        if os.path.exists(p):
            img_path = p
            break
        # Fallback to relative path resolution under base_dir
        parts = s_dir.replace("\\", "/").split("/")
        if len(parts) >= 2:
            p_rel = os.path.join(sm.base_dir, parts[-2], parts[-1], "input_image.jpg")
            if os.path.exists(p_rel):
                img_path = p_rel
                break
                
    thumb_b64 = get_image_thumbnail_base64(img_path, size=(24, 24))
    
    if thumb_b64:
        label = f"![Thumbnail]({thumb_b64})&nbsp; **Results for Image {group['img_idx']}**"
    else:
        label = f"🖼️ **Results for Image {group['img_idx']}**"
        
    exp_key = f"res_group_exp_{group['img_idx']}_{key_suffix}" if key_suffix else None
    with st.expander(label, expanded=expanded, key=exp_key):
        # Group entries by base model architecture
        architectures = []
        for m in group["models"]:
            if m["model"] not in architectures: architectures.append(m["model"])
        
        for arch in architectures:
            arch_models = sorted(
                [m for m in group["models"] if m["model"] == arch],
                key=lambda m: m.get("input_size", 0)
            )
            sample_m = arch_models[0]
            pred_class = sample_m.get("prediction", "Unknown")
            if pred_class == "Unknown" and sample_m.get("results") and len(sample_m["results"]) > 0:
                pred_class = sample_m["results"][0].get("Prediction", "Unknown")
            
            st.markdown(f"#### Model: `{arch}`")
            st.markdown(f"<div style='margin-top: -12px; margin-bottom: 12px; font-size: 0.9rem; color: #94a3b8;'>Prediction: <strong style='color: #e5edf6;'>{pred_class}</strong></div>", unsafe_allow_html=True)
            
            # Layout: Input Image (Left) | Method Collage (Right)
            col_left, col_right = st.columns([1, 3])
            
            # 1. Show Input Image once for this Architecture
            img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
            if os.path.exists(img_path):
                # Pull original resolution from the model_entry or fallback to the first result entry
                orig_res = sample_m.get("original_resolution", "Unknown")
                if orig_res == "Unknown" and sample_m["results"]:
                    orig_res = sample_m["results"][0].get("Original Resolution", "Unknown")
                col_left.image(img_path, caption=f"Input Image ({orig_res})", use_container_width=True)
            
            # 2. Show Heatmap Thumbnails (Grid layout when 1 size, Row layout when multiple sizes)
            with col_right:
                if len(arch_models) == 1:
                    # --- Single Resolution Mode: Optimal 3 or 4 column grid based on method count ---
                    m_data = arch_models[0]
                    n_meth = len(selected_methods)
                    if n_meth % 4 == 0:
                        GRID_COLUMNS = 4
                    elif n_meth % 3 == 0:
                        GRID_COLUMNS = 3
                    else:
                        # Compare row fill efficiency (remainder): pick 3 or 4 whichever leaves a fuller final row.
                        # For n_meth = 1 or 2, this naturally selects 4, keeping image sizes at a clean 25% width.
                        GRID_COLUMNS = 3 if (n_meth % 3) > (n_meth % 4) else 4
                    
                    for idx_batch in range(0, len(selected_methods), GRID_COLUMNS):
                        method_chunk = selected_methods[idx_batch:idx_batch + GRID_COLUMNS]
                        grid_cols = st.columns(GRID_COLUMNS)
                        for i_col, method in enumerate(method_chunk):
                            res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                            with grid_cols[i_col]:
                                st.markdown(f"**{method}**")
                                if res:
                                    h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                                    if os.path.exists(h_p):
                                        st.image(h_p, caption=f"{m_data['input_size']}px", use_container_width=True)
                                else:
                                    st.markdown("<div style='height: 60px; border: 1px dashed rgba(148, 163, 184, 0.32); border-radius: 8px; background: rgba(255, 255, 255, 0.035); text-align: center; padding-top: 20px; color: #94a3b8; font-size: 0.7em;'>...</div>", unsafe_allow_html=True)
                else:
                    # --- Multiple Resolutions Mode: Method per row, sizes side-by-side ---
                    for method in selected_methods:
                        st.markdown(f"**{method}**")
                        num_sizes = len(arch_models)
                        cols_to_make = max(num_sizes, 4) 
                        size_cols = st.columns(cols_to_make)
                        
                        for i, m_data in enumerate(arch_models):
                            res = next((r for r in m_data["results"] if r["Method"].lower() == method.lower()), None)
                            with size_cols[i]:
                                if res:
                                    h_p = os.path.join(m_data["session_dir"], "heatmaps", f"{res['Method']}.png")
                                    if os.path.exists(h_p):
                                        st.image(h_p, caption=f"{m_data['input_size']}px", use_container_width=True)
                                else:
                                    st.markdown("<div style='height: 60px; border: 1px dashed rgba(148, 163, 184, 0.32); border-radius: 8px; background: rgba(255, 255, 255, 0.035); text-align: center; padding-top: 20px; color: #94a3b8; font-size: 0.7em;'>...</div>", unsafe_allow_html=True)
            
            # 3. Consolidated Table for all sizes of this Architecture
            # Reorder arch_results to match the visual flow (Method first, then all sizes)
            arch_results = []
            for method in selected_methods:
                for m in arch_models:
                    res = next((r for r in m["results"] if r["Method"].lower() == method.lower()), None)
                    if res: arch_results.append(res)
            
            if arch_results:
                raw_df = presentation_df(pd.DataFrame(arch_results))
                display_cols = [c for c in ["Method", "Resolution", ATTR_RUNTIME_COL, "Attribution Runtime Std (sec)", ATTR_MEMORY_COL, "Attribution Memory Std (MB)", "Gini Index", "Deletion AUC", "Insertion AUC", "Infidelity", "Quality Eval Time (sec)"] if c in raw_df.columns]
                if "Status" in raw_df.columns and raw_df["Status"].astype(str).str.startswith("Failed").any():
                    display_cols.append("Status")
                st.table(style_dataframe(raw_df[display_cols], raw_precision=True))

@st.dialog("Image Viewer", width="large")
def show_lightbox(img):
    st.markdown(f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{st.session_state.current_img_base64}" style="max-height: 80vh; max-width: 100%; object-fit: contain;"></div>', unsafe_allow_html=True)

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
if 'input_size_str' not in st.session_state: st.session_state.input_size_str = "224"
if 'selected_methods' not in st.session_state: st.session_state.selected_methods = ["Saliency", "Integrated_Gradients"]
if 'selected_warmups' not in st.session_state: st.session_state.selected_warmups = 1
if 'selected_repeats' not in st.session_state: st.session_state.selected_repeats = 5
if 'selected_memory_runs' not in st.session_state: st.session_state.selected_memory_runs = 1
if 'selected_run_order' not in st.session_state: st.session_state.selected_run_order = "Balanced"
if 'enable_quality_metrics' not in st.session_state: st.session_state.enable_quality_metrics = False
if 'selected_quality_metrics' not in st.session_state: st.session_state.selected_quality_metrics = []
has_cuda = torch.cuda.is_available()
has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
default_device_mode = "CPU"
if has_cuda:
    default_device_mode = "GPU (CUDA)"
elif has_mps:
    default_device_mode = "GPU (MPS)"

if 'selected_device_mode' not in st.session_state: st.session_state.selected_device_mode = default_device_mode
if 'current_device_mode' not in st.session_state: st.session_state.current_device_mode = default_device_mode
if 'current_warmups' not in st.session_state: st.session_state.current_warmups = 1
if 'current_repeats' not in st.session_state: st.session_state.current_repeats = 5
if 'current_memory_runs' not in st.session_state: st.session_state.current_memory_runs = 1
if 'current_enable_quality_metrics' not in st.session_state: st.session_state.current_enable_quality_metrics = False
if 'current_selected_quality_metrics' not in st.session_state: st.session_state.current_selected_quality_metrics = ["Gini Index (Sparsity)"]
if 'current_page' not in st.session_state: st.session_state.current_page = "Configure"

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
model_opts = ['resnet50', 'convnext-t', 'efficientnet-b0', 'swin-t', 'regnet-y-8gf', 'mobilenet-v3-large', 'densenet121', 'vit-b-16']
xai_opts = [
    "Saliency",
    "Integrated_Gradients",
    "Guided_Backprop",
    "Input_X_Gradient",
    "Gradient_Shap",
    "DeepLift",
    "DeepLift_Shap",
    "Grad_CAM",
    "Occlusion",
    "Lime",
]

# --- Restore Config Request (Must happen before any widgets are instantiated) ---
if st.session_state.get("restore_config"):
    restore_data = st.session_state.restore_config
    settings = restore_data.get("settings", {})
    environment = restore_data.get("environment", {})
    
    st.session_state.selected_models = settings.get("models", ["resnet50"])
    
    methods_map = {m.lower().replace("_", ""): m for m in xai_opts}
    restored_methods = []
    for m in settings.get("methods", []):
        m_norm = m.lower().replace("_", "")
        if m_norm in methods_map:
            restored_methods.append(methods_map[m_norm])
    st.session_state.selected_methods = restored_methods if restored_methods else ["Saliency", "Integrated_Gradients"]
    
    raw_sizes = settings.get("input_sizes") or settings.get("input_size_str") or [224]
    if isinstance(raw_sizes, list) and len(raw_sizes) > 0:
        st.session_state.input_size_str = ", ".join([str(s) for s in raw_sizes])
    elif isinstance(raw_sizes, (str, int)):
        st.session_state.input_size_str = str(raw_sizes)
    else:
        st.session_state.input_size_str = "224"
    st.session_state.selected_repeats = settings.get("repeat_count", 5)
    st.session_state.selected_warmups = settings.get("warmup_runs", 1)
    st.session_state.selected_memory_runs = settings.get("memory_runs", 1)
    st.session_state.selected_run_order = settings.get("run_order", "Balanced")
    st.session_state.enable_quality_metrics = settings.get("enable_quality_metrics", False)
    st.session_state.selected_quality_metrics = settings.get("selected_quality_metrics", ["Gini Index (Sparsity)"])
    
    env_dev = environment.get("selected_device")
    if env_dev:
        env_dev_str = str(env_dev).lower()
        if "cuda" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (CUDA)"
        elif "mps" in env_dev_str:
            st.session_state.selected_device_mode = "GPU (MPS)"
        else:
            st.session_state.selected_device_mode = "CPU"
            
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

def rerun_app():
    try:
        st.rerun(scope="app")
    except TypeError:
        st.rerun()

def rerun_fragment():
    if fragment_api:
        st.rerun(scope="fragment")
    else:
        rerun_app()

# --- COMPARISON PLOTS FOR COMBINED BATCH EVALUATION ---
def plot_combined_batches_comparison(df):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    df["Model_Method"] = df["Model"] + "\n(" + df["Method"] + ")"
    df_sorted = df.sort_values(by=["Model_Method", "Batch"])
    
    sns.barplot(
        data=df_sorted, 
        x="Model_Method", 
        y=runtime_col, 
        hue="Batch", 
        palette="viridis", 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Attribution Runtime Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Model & Method")
    ax.set_ylabel("Attribution Runtime (sec)")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig

def plot_combined_batches_memory(df):
    df = normalize_metric_columns(df)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    
    valid_df = df[df[memory_col].notna()].copy()
    if valid_df.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "Memory profiling disabled or not available", ha='center', va='center', fontsize=12, color='gray')
        ax.set_title("Peak Attribution Memory Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(12, 7))
    valid_df["Model_Method"] = valid_df["Model"] + "\n(" + valid_df["Method"] + ")"
    df_sorted = valid_df.sort_values(by=["Model_Method", "Batch"])
    
    sns.barplot(
        data=df_sorted, 
        x="Model_Method", 
        y=memory_col, 
        hue="Batch", 
        palette="viridis", 
        ax=ax, 
        edgecolor="black"
    )
    ax.set_title("Peak Attribution Memory Comparison Across Batches", fontsize=14, fontweight='bold', family='serif')
    ax.set_xlabel("Model & Method")
    ax.set_ylabel("Peak Memory (MB)")
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, ls="-", alpha=0.15)
    plt.tight_layout()
    return fig

# --- IMAGE PREVIEW FRAGMENT ---
def render_image_preview_gallery(img_sources):
    if img_sources:
        num_imgs = len(img_sources)
        if st.session_state.img_idx >= num_imgs:
            st.session_state.img_idx = 0
        current_src = img_sources[st.session_state.img_idx]
        try:
            if hasattr(current_src, 'name'):
                img_view = Image.open(current_src)
                src_name = f"Uploaded: {current_src.name}"
            elif isinstance(current_src, str) and (current_src.startswith("http://") or current_src.startswith("https://")):
                response = requests.get(current_src)
                img_view = Image.open(BytesIO(response.content))
                src_name = "URL Link"
            else:
                img_view = Image.open(current_src)
                src_name = f"Local: {os.path.basename(current_src)}"
            st.session_state.current_img_base64 = get_base64(img_view)
            
            html_content = f"""
            <div class="compact-preview">
                <div style='text-align: center; color: #94a3b8; font-size: 0.8em; margin-bottom: 2px;'>Resolution: {img_view.size[0]}x{img_view.size[1]} px | {src_name}</div>
                <div class="preview-image-frame"><img src="data:image/png;base64,{st.session_state.current_img_base64}" alt="Selected input preview"></div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 8px;"></div>', unsafe_allow_html=True)
            if st.button("View", use_container_width=True):
                show_lightbox(img_view)
        except Exception:
            st.markdown("<div class='compact-preview'><div style='height: 330px; text-align: center; padding-top: 100px; color: #94a3b8;'>Preview unavailable</div></div>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 0.8, 1])
        with n1:
            if st.button("⬅️ Prev", key="prev_btn", use_container_width=True):
                st.session_state.img_idx = (st.session_state.img_idx - 1) % num_imgs
                rerun_app()
        with n2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; font-weight: bold;'>{st.session_state.img_idx + 1}/{num_imgs}</div>", unsafe_allow_html=True)
        with n3:
            if st.button("Next ➡️", key="next_btn", use_container_width=True):
                st.session_state.img_idx = (st.session_state.img_idx + 1) % num_imgs
                rerun_app()



def keep_local_images_expanded():
    st.session_state.local_images_expanded = True

# --- PAGE 1: CONFIGURE ---
def render_configure_page():
    # Detect and initialize auto-loaded folder images state (Safe from widget lock here!)
    local_all = []
    images_dir = os.path.join(PROJECT_ROOT, "gui", "images")
    if os.path.exists(images_dir) and os.path.isdir(images_dir):
        try:
            for f in sorted(os.listdir(images_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    local_all.append(f)
            
            if local_all:
                if 'active_local_filenames' not in st.session_state:
                    st.session_state.active_local_filenames = local_all.copy()
                    st.session_state.previous_local_all = local_all.copy()
                else:
                    prev_all = st.session_state.get("previous_local_all", [])
                    new_files = [f for f in local_all if f not in prev_all]
                    if new_files:
                        st.session_state.active_local_filenames = list(set(st.session_state.active_local_filenames + new_files))
                    st.session_state.previous_local_all = local_all.copy()
                
                # Filter out deleted files
                st.session_state.active_local_filenames = [
                    f for f in st.session_state.active_local_filenames if f in local_all
                ]
        except Exception as e:
            st.error(f"Error initializing local images: {str(e)}")

    if st.session_state.get("stop_requested") and st.session_state.get("sh_models"):
        st.session_state.selected_models = st.session_state.sh_models
        st.session_state.selected_methods = st.session_state.sh_methods
        st.session_state.input_size_str = st.session_state.sh_input_size_str
        st.session_state.selected_repeats = st.session_state.sh_repeats
        st.session_state.selected_warmups = st.session_state.sh_warmups
        st.session_state.selected_run_order = st.session_state.sh_run_order
        st.session_state.selected_device_mode = st.session_state.sh_device_mode
        st.session_state.stop_requested = False

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

    incomplete_batches = sm.list_incomplete_batches()
    if incomplete_batches:
        st.markdown("### ⚡ Interrupted Benchmarks")
        st.warning("The following benchmarks were interrupted (e.g. due to system sleep/restart). You can resume them from where they left off.")
        
        for b in incomplete_batches:
            col_b1, col_b2, col_b3 = st.columns([3, 1, 1])
            with col_b1:
                st.markdown(f"**Batch:** `{b['id']}` ({b['created']})  \n`{b['info']}`")
            with col_b2:
                if st.button("Resume 🚀", key=f"resume_{b['id']}", use_container_width=True):
                    resume_batch(b['id'])
                    rerun_app()
            with col_b3:
                if st.button("Delete 🗑️", key=f"del_inc_{b['id']}", use_container_width=True):
                    sm.delete_batch(b['id'])
                    check_and_reset_session_after_delete(b['id'])
                    st.success(f"Deleted {b['id']}")
                    rerun_app()
        st.markdown('<div style="margin-top: 1.5rem; margin-bottom: 1.5rem; border-top: 1px solid rgba(148, 163, 184, 0.2); padding-top: 0.5rem;"></div>', unsafe_allow_html=True)

    # Row 1
    row1_left, row1_right = st.columns([2, 1])
    with row1_left:
        selected_models_widget = st.multiselect(
            "Model Architectures",
            model_opts,
            key="selected_models",
        )
    with row1_right:
        col_w, col_m = st.columns(2)
        with col_w:
            st.number_input(
                "Warmup runs",
                min_value=0,
                max_value=20,
                key="selected_warmups",
                step=1,
                help="Untimed runs before measurement. Useful for CUDA/model warmup.",
            )
        with col_m:
            st.number_input(
                "Memory runs",
                min_value=0,
                max_value=100,
                key="selected_memory_runs",
                step=1,
                help="Dedicated memory measurement runs (default 1). Set to 0 to skip memory profiling entirely.",
            )

    # Row 2
    row2_left, row2_right = st.columns([2, 1])
    with row2_left:
        fixed_models = [m for m in selected_models_widget if m in ["vit-b-16", "swin-t"]]
        if fixed_models:
            st.text_input("Input Sizes (px)", value="224", disabled=True)
            if len(fixed_models) == 1:
                st.caption(f"⚠️ *Fixed-size architecture selected (Locked to 224px):* **`{fixed_models[0]}`**")
            else:
                formatted_models = ", ".join([f"**`{m}`**" for m in fixed_models])
                st.caption(f"⚠️ *Fixed-size architectures selected (Locked to 224px):* {formatted_models}")
        else:
            st.text_input(
                "Input Sizes (px)",
                key="input_size_str",
                help="Only applicable to CNN-based architectures."
            )
            st.markdown('<div style="margin-top: -15px; margin-bottom: 15px; font-size: 0.85em; color: gray;">Separate by commas (e.g., 224, 448, 512).</div>', unsafe_allow_html=True)
            parsed_sizes = parse_input_sizes(st.session_state.input_size_str)
            if parsed_sizes == [224] and st.session_state.input_size_str.strip() not in ["", "224"]:
                st.error("Invalid size format. Using 224.")
            elif any(s < 32 for s in parsed_sizes):
                st.error("⚠️ Input size must be at least 32px. CNN models will crash at lower resolutions.")
    with row2_right:
        st.number_input(
            "Measured repeats",
            min_value=1,
            max_value=1000,
            key="selected_repeats",
            step=1,
            help="Timed attribution repeats per image/model/size/method. Use 30-100 for stronger size studies when methods are fast enough.",
        )

    # Row 3
    row3_left, row3_right = st.columns([2, 1])
    with row3_left:
        # Dynamically build options for XAI Methods.
        # This allows users to select the same method multiple times (e.g. Integrated_Gradients, Integrated_Gradients_2)
        base_xai_opts = [
            "Saliency",
            "Integrated_Gradients",
            "Guided_Backprop",
            "Input_X_Gradient",
            "Gradient_Shap",
            "DeepLift",
            "DeepLift_Shap",
            "Grad_CAM",
            "Occlusion",
            "Lime",
        ]
        
        selected_methods = st.session_state.get("selected_methods", ["Saliency", "Integrated_Gradients"])
        
        dynamic_xai_opts = list(base_xai_opts)
        parameterized_bases = ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]
        for base in base_xai_opts:
            if base not in parameterized_bases:
                continue
                
            import re
            pattern = re.compile(rf"^{base}(?:_(\d+))?$")
            versions = []
            for m in selected_methods:
                match = pattern.match(m)
                if match:
                    val = match.group(1)
                    versions.append(int(val) if val else 1)
            
            if versions:
                next_version = max(versions) + 1
                dynamic_xai_opts.append(f"{base}_{next_version}")
                for v in sorted(versions):
                    name = f"{base}_{v}" if v > 1 else base
                    if name not in dynamic_xai_opts:
                        dynamic_xai_opts.append(name)

        st.multiselect(
            "XAI Methods",
            dynamic_xai_opts,
            key="selected_methods",
        )
        
        # Compact XAI Method Parameters rendered right under the selector in the same column
        modifiable_selected = []
        for m in st.session_state.selected_methods:
            import re
            base_name = re.sub(r'_\d+$', '', m)
            if base_name in ["Integrated_Gradients", "Gradient_Shap", "Occlusion", "Lime"]:
                modifiable_selected.append(m)
        
        if modifiable_selected:
            # Force defaults back into session state if they were deleted during streamlit's widget cleanup
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
            
            # Pre-fill method-specific parameter defaults
            for method in modifiable_selected:
                import re
                base = re.sub(r'_\d+$', '', method)
                if base == "Integrated_Gradients":
                    if f"ig_steps_str_{method}" not in st.session_state: st.session_state[f"ig_steps_str_{method}"] = st.session_state.get("ig_steps_str", "50")
                    if f"ig_internal_batch_str_{method}" not in st.session_state: st.session_state[f"ig_internal_batch_str_{method}"] = st.session_state.get("ig_internal_batch_str", "2")
                    if f"ig_baseline_mode_{method}" not in st.session_state: st.session_state[f"ig_baseline_mode_{method}"] = st.session_state.get("ig_baseline_mode", "Zeros (Black)")
                elif base == "Gradient_Shap":
                    if f"gs_samples_str_{method}" not in st.session_state: st.session_state[f"gs_samples_str_{method}"] = st.session_state.get("gs_samples_str", "10")
                    if f"gs_stdevs_str_{method}" not in st.session_state: st.session_state[f"gs_stdevs_str_{method}"] = st.session_state.get("gs_stdevs_str", "0.0001")
                    if f"gs_baseline_mode_{method}" not in st.session_state: st.session_state[f"gs_baseline_mode_{method}"] = st.session_state.get("gs_baseline_mode", "Zeros & Mean")
                elif base == "Occlusion":
                    if f"occlusion_window_str_{method}" not in st.session_state: st.session_state[f"occlusion_window_str_{method}"] = st.session_state.get("occlusion_window_str", "15")
                    if f"occlusion_stride_str_{method}" not in st.session_state: st.session_state[f"occlusion_stride_str_{method}"] = st.session_state.get("occlusion_stride_str", "8")
                    if f"occlusion_value_str_{method}" not in st.session_state: st.session_state[f"occlusion_value_str_{method}"] = st.session_state.get("occlusion_value_str", "0")
                elif base == "Lime":
                    if f"lime_samples_str_{method}" not in st.session_state: st.session_state[f"lime_samples_str_{method}"] = st.session_state.get("lime_samples_str", "500")
                    if f"lime_batch_str_{method}" not in st.session_state: st.session_state[f"lime_batch_str_{method}"] = st.session_state.get("lime_batch_str", "10")
                    if f"lime_segments_str_{method}" not in st.session_state: st.session_state[f"lime_segments_str_{method}"] = st.session_state.get("lime_segments_str", "50")

            with st.expander("⚙️ Algorithm Parameter Tuning", expanded=False):
                st.markdown('<div style="font-size: 0.8em; color: var(--xai-muted); margin-bottom: 12px;">Customize algorithm inputs below. Use comma-separated values (e.g. <b>50, 100</b>) to test multiple parameter variations.</div>', unsafe_allow_html=True)
                
                for idx_m, method in enumerate(modifiable_selected):
                    import re
                    base_method = re.sub(r'_\d+$', '', method)
                    
                    if idx_m > 0:
                        st.markdown("<hr style='margin: 4px 0 16px 0; border: 0; border-top: 1px solid rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
                        
                    st.markdown(f"<div style='font-size: 0.85em; font-weight: 600; color: var(--xai-text); margin-top: 0px; margin-bottom: 8px;'>{method.replace('_', ' ')}</div>", unsafe_allow_html=True)
                    
                    if base_method == "Integrated_Gradients":
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.text_input(
                                "Steps",
                                value=st.session_state.get(f"ig_steps_str_{method}", st.session_state.get("ig_steps_str", "50")),
                                key=f"ig_steps_str_{method}",
                                help="Number of steps along the path from baseline to input.\n\n- Default: 50\n- Higher is more mathematically accurate but slower."
                            )
                        with col_p2:
                            st.text_input(
                                "Internal Batch Size",
                                value=st.session_state.get(f"ig_internal_batch_str_{method}", st.session_state.get("ig_internal_batch_str", "2")),
                                key=f"ig_internal_batch_str_{method}",
                                help="Mini-batch size to chunk the steps.\n\n- Default: 2\n- Set to 2-4 to prevent Out-Of-Memory (OOM) on large models."
                            )
                        with col_p3:
                            ig_opts = ["Zeros (Black)", "Ones (White)", "Input Mean"]
                            st.selectbox(
                                "Baseline Mode",
                                options=ig_opts,
                                index=ig_opts.index(st.session_state.get(f"ig_baseline_mode_{method}", st.session_state.get("ig_baseline_mode", "Zeros (Black)"))) if st.session_state.get(f"ig_baseline_mode_{method}", st.session_state.get("ig_baseline_mode", "Zeros (Black)")) in ig_opts else 0,
                                key=f"ig_baseline_mode_{method}",
                                help="Baseline reference image for attribution.\n\n- Default: Zeros (Black)"
                            )
                            
                    elif base_method == "Gradient_Shap":
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.text_input(
                                "Samples",
                                value=st.session_state.get(f"gs_samples_str_{method}", st.session_state.get("gs_samples_str", "10")),
                                key=f"gs_samples_str_{method}",
                                help="Number of baseline samples to average.\n\n- Default: 10\n- Suggest 10-20 for stable results."
                            )
                        with col_p2:
                            st.text_input(
                                "Stdevs (Noise)",
                                value=st.session_state.get(f"gs_stdevs_str_{method}", st.session_state.get("gs_stdevs_str", "0.0001")),
                                key=f"gs_stdevs_str_{method}",
                                help="Standard deviation of Gaussian noise added to inputs.\n\n- Default: 0.0001"
                            )
                        with col_p3:
                            gs_opts = ["Zeros & Mean", "Zeros Only", "Ones Only"]
                            st.selectbox(
                                "Baseline Mode",
                                options=gs_opts,
                                index=gs_opts.index(st.session_state.get(f"gs_baseline_mode_{method}", st.session_state.get("gs_baseline_mode", "Zeros & Mean"))) if st.session_state.get(f"gs_baseline_mode_{method}", st.session_state.get("gs_baseline_mode", "Zeros & Mean")) in gs_opts else 0,
                                key=f"gs_baseline_mode_{method}",
                                help="Distribution of baselines to sample from.\n\n- Default: Zeros & Mean"
                            )
                            
                    elif base_method == "Occlusion":
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.text_input(
                                "Patch Size (px)",
                                value=st.session_state.get(f"occlusion_window_str_{method}", st.session_state.get("occlusion_window_str", "15")),
                                key=f"occlusion_window_str_{method}",
                                help="Size of the square occlusion patch.\n\n- Default: 15\n- Larger is faster but coarser."
                            )
                        with col_p2:
                            st.text_input(
                                "Stride (px)",
                                value=st.session_state.get(f"occlusion_stride_str_{method}", st.session_state.get("occlusion_stride_str", "8")),
                                key=f"occlusion_stride_str_{method}",
                                help="Step size of the sliding window.\n\n- Default: 8\n- Smaller is more detailed but much slower."
                            )
                        with col_p3:
                            st.text_input(
                                "Occlude Color",
                                value=st.session_state.get(f"occlusion_value_str_{method}", st.session_state.get("occlusion_value_str", "0")),
                                key=f"occlusion_value_str_{method}",
                                help="Pixel value to fill the occlusion patch.\n\n- Default: 0 (Black)\n- Can be 0, 1 (White), or mean."
                            )
                            
                    elif base_method == "Lime":
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.text_input(
                                "Perturbation Samples",
                                value=st.session_state.get(f"lime_samples_str_{method}", st.session_state.get("lime_samples_str", "500")),
                                key=f"lime_samples_str_{method}",
                                help="Number of samples to perturb and fit the surrogate model on.\n\n- Default: 500\n- Warning: High values (>1000) are very slow."
                            )
                        with col_p2:
                            st.text_input(
                                "Batch Size",
                                value=st.session_state.get(f"lime_batch_str_{method}", st.session_state.get("lime_batch_str", "10")),
                                key=f"lime_batch_str_{method}",
                                help="Internal batch size to run pertubations through the model.\n\n- Default: 10\n- Prevents memory overflow."
                            )
                        with col_p3:
                            st.text_input(
                                "Superpixels Count",
                                value=st.session_state.get(f"lime_segments_str_{method}", st.session_state.get("lime_segments_str", "50")),
                                key=f"lime_segments_str_{method}",
                                help="Target number of superpixels for SLIC segmentation.\n\n- Default: 50\n- Fewer superpixels speeds up LIME."
                            )
                    st.markdown('<div style="margin-bottom: 5px;"></div>', unsafe_allow_html=True)
        
        # Measurement Details directly below parameters inside left column
        st.markdown('<div style="margin-top: 25px; font-weight: bold; margin-bottom: 8px; font-size: 1.1em; color: var(--xai-text);">Measurement Details</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul class="nice-bullets">
            <li><b>Warmups</b> are not reported in statistics. Measured repeats are timed and summarized with median, mean, and standard deviation.</li>
            <li><b>Task Ordering</b>: Benchmark runs are executed in a <b>Balanced</b> order (automatically rotating resolutions and model architectures) to mitigate PyTorch/CUDA caching allocator and execution-order bias.</li>
            <li><b>Separate Timing & Memory</b>: By default, CPU memory is measured once in a dedicated run. The timed repeats are then executed cleanly without the memory profiler to ensure accurate speed statistics.</li>
        </ul>
        """, unsafe_allow_html=True)

    with row3_right:
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        device_options = ["CPU"]
        if has_cuda:
            device_options.insert(0, "GPU (CUDA)")
        elif has_mps:
            device_options.insert(0, "GPU (MPS)")

        if st.session_state.selected_device_mode not in device_options:
            st.session_state.selected_device_mode = device_options[0]
        st.radio(
            "Hardware Device",
            device_options,
            key="selected_device_mode",
            horizontal=True,
        )
        
        if "GPU" in st.session_state.selected_device_mode:
            if torch.cuda.is_available():
                gpu_desc = torch.cuda.get_device_name(0)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                gpu_desc = "Apple Silicon (MPS)"
            else:
                gpu_desc = "Active"
            status_text = f"🟢 {gpu_desc}"
        else:
            status_text = f"💻 {get_cpu_info()}"
            
        import sys
        cuda_line = f"<p>• <strong>CUDA Version</strong>: {torch.version.cuda}</p>" if torch.cuda.is_available() else ""
        
        status_html = f"""
        <style>
            .hw-status-container {{
                border: 1px solid var(--xai-border) !important;
                border-radius: 8px !important;
                overflow: hidden !important;
                background: linear-gradient(135deg, rgba(96, 165, 250, 0.14) 0%, rgba(45, 212, 191, 0.08) 100%),
                            repeating-linear-gradient(-45deg, rgba(255, 255, 255, 0.015) 0px, rgba(255, 255, 255, 0.015) 2px, transparent 2px, transparent 10px) !important;
                transition: border-color 0.2s ease !important;
            }}
            .hw-status-container:hover {{
                border-color: rgba(45, 212, 191, 0.4) !important;
            }}
            .hw-status-summary {{
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                padding: 0.6rem 0.9rem !important;
                cursor: pointer !important;
                color: var(--xai-text) !important;
                font-weight: 500 !important;
                font-size: 0.88rem !important;
                list-style: none !important;
                outline: none !important;
                box-sizing: border-box !important;
            }}
            .hw-status-summary::-webkit-details-marker {{
                display: none !important;
            }}
            .hw-status-container[open] .hw-chevron {{
                transform: rotate(180deg) !important;
            }}
            .hw-status-details {{
                padding: 0.9rem !important;
                border-top: 1px solid var(--xai-border) !important;
                font-size: 0.85rem !important;
                color: var(--xai-muted) !important;
                background: rgba(11, 18, 27, 0.45) !important;
            }}
            .hw-status-details p {{
                margin: 0 0 6px 0 !important;
                line-height: 1.4 !important;
            }}
            .hw-status-details p:last-child {{
                margin-bottom: 0 !important;
            }}
        </style>
        <details class="hw-status-container">
            <summary class="hw-status-summary">
                <span>{status_text}</span>
                <span class="hw-chevron" style="transition: transform 0.2s; font-size: 0.75rem; display: inline-block;">▼</span>
            </summary>
            <div class="hw-status-details">
                <p><strong>Hardware & Environment Details:</strong></p>
                <p>• <strong>OS/Platform</strong>: {platform.platform()}</p>
                <p>• <strong>Python Version</strong>: {sys.version.split()[0]}</p>
                <p>• <strong>PyTorch Version</strong>: {torch.__version__}</p>
                {cuda_line}
            </div>
        </details>
        """
        st.markdown(status_html, unsafe_allow_html=True)

    # Row 5: Quality Metrics (Post-Processing)
    st.divider()
    is_quality_enabled = st.toggle(
        "Explanation Quality Evaluation",
        key="enable_quality_metrics",
        help="Evaluates explanation quality (e.g. Gini Index / Sparsity) in post-processing outside the timing and memory benchmarking clock."
    )

    st.multiselect(
        "Select Quality Metrics",
        ["Gini Index (Sparsity)", "Deletion AUC", "Insertion AUC", "Infidelity"],
        key="selected_quality_metrics",
        disabled=not is_quality_enabled,
        help="Post-hoc quality metrics evaluated outside the timing clock. Gini Index (sparsity), Deletion AUC (faithfulness upon removal), Insertion AUC (faithfulness upon addition), and Infidelity (perturbation sensitivity)."
    )

    st.divider()

    col_input, col_spacer, col_preview = st.columns([2.5, 0.2, 0.8])
    with col_input:
        uploaded_files = st.file_uploader(
            "Drag and drop images",
            type=["jpg", "jpeg", "png", "webp", "gif"],
            accept_multiple_files=True,
            key="uploaded_files"
        )
        
        # Auto-loaded folder images expander
        if local_all:
            active_count = len(st.session_state.active_local_filenames)
            expanded_val = st.session_state.get("local_images_expanded", False)
            with st.expander(f"Auto-Loaded Images ({active_count}/{len(local_all)})", expanded=expanded_val):
                st.session_state.local_images_expanded = False  # Reset state after rendering
                st.markdown('<div style="font-size: 0.85em; color: var(--xai-muted); margin-bottom: 8px;">Images loaded automatically from <code>gui/images/</code>. Select or remove images to customize the benchmark workload.</div>', unsafe_allow_html=True)
                st.multiselect(
                    "Included Images",
                    options=local_all,
                    default=local_all,
                    key="active_local_filenames",
                    label_visibility="collapsed",
                    on_change=keep_local_images_expanded
                )
                
                st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                del_col_select, del_col_btn = st.columns([3.2, 0.8])
                with del_col_select:
                    file_to_del = st.selectbox(
                        "Delete file from disk:",
                        options=["-- Choose file to delete from disk --"] + local_all,
                        key="local_file_to_delete",
                        label_visibility="collapsed",
                        on_change=keep_local_images_expanded
                    )
                with del_col_btn:
                    if file_to_del != "-- Choose file to delete from disk --":
                        if st.button("🗑️ Delete", key="delete_local_file_btn", help="Permanently delete this file from your local disk.", use_container_width=True):
                            path = os.path.join(images_dir, file_to_del)
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                    st.session_state.local_images_expanded = True  # Keep expander open on rerun!
                                    st.toast(f"Deleted {file_to_del} from disk.", icon="🗑️")
                                    if file_to_del in st.session_state.active_local_filenames:
                                        st.session_state.active_local_filenames.remove(file_to_del)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                    else:
                        st.button("🗑️ Delete", disabled=True, use_container_width=True)
        
        with st.expander("Paste Image URLs", expanded=False):
            st.text_area(
                "Image URLs",
                height=100,
                help="Please put each URL on a new line, without commas or other separators.",
                key="persisted_urls"
            )
    with col_preview:
        render_image_preview_gallery(current_image_sources())

    # Dynamic CSS to replace default file list icons with actual image thumbnails
    if uploaded_files:
        css_rules = []
        for idx, file in enumerate(uploaded_files):
            try:
                file.seek(0)
                img = Image.open(file)
                img.thumbnail((60, 60))
                b64 = get_base64(img)
                css_rules.append(f"""
                /* Target file icon specifically (exclude deletion cross inside button) */
                [data-testid="stFileUploader"] [role="list"] > *:nth-child({idx + 1}) svg:not(button svg),
                [data-testid="stFileUploader"] li:nth-child({idx + 1}) svg:not(button svg),
                .stFileUploaderFile:nth-child({idx + 1}) svg:not(button svg),
                .stFileUploaderFile:nth-of-type({idx + 1}) svg:not(button svg) {{
                    display: none !important;
                }}
                
                /* Prepend the actual image thumbnail */
                [data-testid="stFileUploader"] [role="list"] > *:nth-child({idx + 1})::before,
                [data-testid="stFileUploader"] li:nth-child({idx + 1})::before,
                .stFileUploaderFile:nth-child({idx + 1})::before,
                .stFileUploaderFile:nth-of-type({idx + 1})::before {{
                    content: "";
                    display: inline-block;
                    width: 28px;
                    height: 28px;
                    border-radius: 4px;
                    margin-right: 8px;
                    background-image: url('data:image/png;base64,{b64}');
                    background-size: cover;
                    background-position: center;
                    flex-shrink: 0;
                    align-self: center;
                    border: 1px solid rgba(255, 255, 255, 0.15);
                }}
                """)
            except Exception:
                pass
        if css_rules:
            st.markdown(f"<style>{''.join(css_rules)}</style>", unsafe_allow_html=True)
    st.divider()
    
    selected_sizes = [224] if any(m in ["vit-b-16", "swin-t"] for m in st.session_state.selected_models) else parse_input_sizes(st.session_state.input_size_str)
    img_sources = current_image_sources()
    planned_image_count = len(st.session_state.prepared_img_sources) if st.session_state.benchmark_ready_to_run else len(img_sources)
    
    st.markdown(f"""
        <div class="run-summary-bar">
            <span class="run-summary-item">Models <strong>{len(st.session_state.selected_models)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Size variations <strong>{len(selected_sizes)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Methods <strong>{len(st.session_state.selected_methods)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Images <strong>{planned_image_count}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Repeats/config <strong>{st.session_state.selected_repeats}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Warmups/config <strong>{st.session_state.selected_warmups}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Memory runs/config <strong>{st.session_state.selected_memory_runs}</strong></span>
        </div>
        <div class="run-summary-action-spacer"></div>
        """, unsafe_allow_html=True)

    if st.button("Start Multi-Model Benchmark ⚡", type="primary", use_container_width=True):
        st.session_state.last_run_results = []
        st.session_state.last_run_batch_id = ""
        st.session_state.completed_batch_id = ""
        st.session_state.completion_notice_batch_id = ""
        st.session_state.is_finished = False
        st.session_state.total_execution_time = 0
        st.session_state.batch_started_at = ""
        st.session_state.batch_completed_at = ""
        st.session_state.task_queue = []
        st.session_state.prepared_img_sources = []
        st.session_state.current_batch_methods = []
        st.session_state.current_batch_models = []
        st.session_state.current_batch_sizes = []
        
        if not img_sources or not st.session_state.selected_models or not st.session_state.selected_methods:
            st.error("Select at least one image, model, input size, and XAI method.")
        elif any(s < 32 for s in selected_sizes):
            st.error("Cannot start benchmark: All input sizes must be at least 32px to prevent model architecture crashes.")
        else:
            st.session_state.sh_models = list(st.session_state.selected_models)
            st.session_state.sh_methods = list(st.session_state.selected_methods)
            st.session_state.sh_input_size_str = st.session_state.input_size_str
            st.session_state.sh_repeats = st.session_state.selected_repeats
            st.session_state.sh_warmups = st.session_state.selected_warmups
            st.session_state.sh_memory_runs = st.session_state.selected_memory_runs
            st.session_state.sh_run_order = st.session_state.selected_run_order
            st.session_state.sh_device_mode = st.session_state.selected_device_mode

            batch_id = sm.start_batch()
            st.session_state.current_batch_id = batch_id
            st.session_state.last_run_batch_id = batch_id
            st.session_state.current_run_order = st.session_state.selected_run_order
            
            # Expand methods with parameter variations
            expanded_methods = expand_xai_methods_with_params(st.session_state.selected_methods)
            st.session_state.current_batch_methods_info = expanded_methods
            display_methods = [m["display_name"] for m in expanded_methods]
            st.session_state.current_batch_methods = display_methods
            
            st.session_state.current_batch_models = assign_numbered_suffixes(list(st.session_state.selected_models))
            st.session_state.current_batch_sizes = list(selected_sizes)
            st.session_state.current_device_mode = st.session_state.selected_device_mode
            st.session_state.current_warmups = st.session_state.selected_warmups
            st.session_state.current_repeats = st.session_state.selected_repeats
            st.session_state.current_memory_runs = st.session_state.selected_memory_runs
            st.session_state.current_enable_quality_metrics = st.session_state.enable_quality_metrics
            st.session_state.current_selected_quality_metrics = list(st.session_state.get('selected_quality_metrics', ["Gini Index (Sparsity)"]))
            
            # Persist and serialize image sources to survive app restarts and system sleep/shuts
            persisted_imgs = serialize_and_persist_image_sources(img_sources, batch_id, sm.base_dir)
            st.session_state.prepared_img_sources = persisted_imgs
            
            st.session_state.task_queue = build_task_queue(
                len(persisted_imgs),
                st.session_state.current_batch_models,
                st.session_state.current_batch_sizes,
                expanded_methods,
                st.session_state.selected_run_order,
                seed=batch_id
            )
            
            # Save batch configuration for recovery
            batch_config = {
                "batch_id": batch_id,
                "warmup_runs": st.session_state.current_warmups,
                "memory_runs": st.session_state.current_memory_runs,
                "repeat_count": st.session_state.current_repeats,
                "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                "selected_quality_metrics": st.session_state.current_selected_quality_metrics,
                "run_order": st.session_state.current_run_order,
                "models": st.session_state.current_batch_models,
                "input_sizes": st.session_state.current_batch_sizes,
                "methods": display_methods,
                "methods_info": expanded_methods,
                "device_mode": st.session_state.current_device_mode,
                "image_sources": persisted_imgs,
                "started_at": ""
            }
            sm.save_batch_config(batch_id, batch_config)
            
            st.session_state.batch_start_time = None
            st.session_state.total_execution_time = 0
            st.session_state.stop_requested = False
            st.session_state.run_progress_idx = 0
            st.session_state.benchmark_ready_to_run = True
            st.session_state.current_page = "Active Run"
            rerun_app()

    # Interrupted benchmarks moved to top of configure page

# --- PAGE 2: ACTIVE RUN ---
def render_active_run_page():
    title_text = "⚡ Benchmark Execution Engine"
    if st.session_state.is_finished:
        title_text += " | Results"
        
    st.markdown(f"""
        <div style="text-align: center; margin-top: 0.8rem; margin-bottom: -2.5rem !important;">
            <h3 style="background: linear-gradient(90deg, #60a5fa, #2dd4bf); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 1.55rem; font-weight: 700; margin: 0; display: inline-block; letter-spacing: -0.02em;">
                {title_text}
            </h3>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.benchmark_ready_to_run:
        st.info("Preparing benchmark environment... Starting shortly.")
        components.html("""
            <script>
                const targetLabel = "Auto-start benchmark engine";
                let clicked = false;

                function hideAndClick(button) {
                    if (clicked || !button || button.disabled) {
                        return;
                    }
                    clicked = true;
                    const wrapper = button.closest('[data-testid="stButton"]') || button.parentElement;
                    if (wrapper) {
                        wrapper.style.display = "none";
                        wrapper.style.height = "0";
                        wrapper.style.overflow = "hidden";
                    }
                    setTimeout(() => button.click(), 1500);
                }

                function findButton() {
                    const buttons = Array.from(window.parent.document.querySelectorAll("button"));
                    return buttons.find((button) => button.textContent.trim() === targetLabel);
                }

                function scan() {
                    try {
                        hideAndClick(findButton());
                    } catch (error) {
                        clicked = true;
                    }
                }

                const observer = new MutationObserver(scan);
                observer.observe(window.parent.document.body, { childList: true, subtree: true });
                scan();
                setTimeout(() => observer.disconnect(), 5000);
            </script>
        """, height=0)
        
        if st.button("Auto-start benchmark engine", type="primary", use_container_width=True):
            if not st.session_state.get("batch_started_at"):
                st.session_state.batch_started_at = timestamp_now()
                cfg = sm.load_batch_config(st.session_state.current_batch_id)
                if cfg:
                    cfg["started_at"] = st.session_state.batch_started_at
                    sm.save_batch_config(st.session_state.current_batch_id, cfg)
            st.session_state.batch_start_time = time.time()
            st.session_state.batch_completed_at = ""
            st.session_state.total_execution_time = 0
            st.session_state.benchmark_ready_to_run = False
            st.session_state.benchmark_running = True
            st.session_state.first_run_render = True
            rerun_app()
            
    elif st.session_state.benchmark_running and not st.session_state.is_finished:
        task_queue = st.session_state.task_queue
        total_steps = len(task_queue)
        idx = st.session_state.run_progress_idx
        current_task = task_queue[idx] if idx < total_steps else {}
        cur_mod = current_task.get("model_name", "?")
        cur_met = current_task.get("method_name", "?")
        cur_size = current_task.get("target_size", "?")
        img_i = current_task.get("img_i", 0)

        # Render the summary bar of the active run configuration at the top (visually stable anchor)
        st.markdown(f"""
            <div class="run-summary-bar" style="margin-top: 5px; margin-bottom: 8px;">
                <span class="run-summary-item">Models <strong>{len(st.session_state.current_batch_models)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Size variations <strong>{len(st.session_state.current_batch_sizes)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Methods <strong>{len(st.session_state.current_batch_methods)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Images <strong>{len(st.session_state.prepared_img_sources)}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Repeats/config <strong>{st.session_state.current_repeats}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Warmups/config <strong>{st.session_state.current_warmups}</strong></span>
                <span class="run-summary-separator">|</span>
                <span class="run-summary-item">Memory runs/config <strong>{st.session_state.current_memory_runs}</strong></span>
            </div>
            """, unsafe_allow_html=True)

        # Progress bar (loading line)
        st.progress(idx / total_steps if total_steps else 0)
        
        # Step status details and live elapsed timer below the loading line
        status_col, timer_col = st.columns([5, 1])
        with status_col:
            st.markdown(
                f"<div class='status-pulse'>STEP {idx + 1}/{total_steps}: {cur_met} on {cur_mod} @ {cur_size}px (Image {img_i + 1})</div>",
                unsafe_allow_html=True
            )
        with timer_col:
            render_live_elapsed_timer(st.session_state.batch_start_time)
            
        st.divider()
        
        if st.button("🛑 Stop Benchmark", use_container_width=True):
            st.session_state.stop_requested = True
            st.session_state.benchmark_running = False
            st.session_state.last_run_results = []
            st.session_state.task_queue = []
            st.session_state.prepared_img_sources = []
            st.session_state.last_run_batch_id = ""
            st.session_state.completed_batch_id = ""
            st.session_state.benchmark_ready_to_run = False
            st.session_state.current_batch_methods = []
            st.session_state.current_batch_models = []
            st.session_state.current_batch_sizes = []
            st.session_state.current_page = "Configure"
            rerun_app()
            
        st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
        with st.expander("⚙️ Environment & Configuration Details", expanded=False, key=f"env_details_{st.session_state.current_batch_id}_active"):
            meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
            with meta_col1:
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                render_environment_summary(collect_environment_metadata(get_device_string(st.session_state.current_device_mode)))
            with meta_col2:
                st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                active_settings = {
                    "models": st.session_state.current_batch_models,
                    "methods": st.session_state.current_batch_methods,
                    "input_sizes": st.session_state.current_batch_sizes,
                    "repeat_count": st.session_state.current_repeats,
                    "warmup_runs": st.session_state.current_warmups,
                    "memory_runs": st.session_state.current_memory_runs,
                    "selected_quality_metrics": st.session_state.current_selected_quality_metrics if st.session_state.get("current_enable_quality_metrics", False) else []
                }
                render_configuration_summary(active_settings, st.session_state.last_run_results)
            
            # Render parameter mapping table if parameterized methods exist
            render_parameters_mapping_table(st.session_state.get("current_batch_methods_info", []))
            
        if st.session_state.last_run_results:
            render_detailed_results_header("current_live_results", "Completed Results So Far")
            for group in sorted_result_groups(st.session_state.last_run_results):
                render_result_group(group, st.session_state.current_batch_methods, key_suffix=f"live_{st.session_state.current_batch_id}")
                
        if st.session_state.get("first_run_render"):
            components.html("""
                <script>
                    const targetLabel = "Trigger engine step";
                    function scanAndHide() {
                        const buttons = Array.from(window.parent.document.querySelectorAll("button"));
                        const btn = buttons.find((button) => button.textContent.trim() === targetLabel);
                        if (btn) {
                            const wrapper = btn.closest('[data-testid="stElementContainer"]') || btn.closest('[data-testid="stButton"]') || btn.parentElement;
                            if (wrapper) {
                                wrapper.style.display = "none";
                            }
                            // Wait 600ms before clicking to let the page fully render
                            setTimeout(() => {
                                btn.click();
                            }, 600);
                            return true;
                        }
                        return false;
                    }
                    // Poll to hide it as early as possible
                    let interval = setInterval(function() {
                        if (scanAndHide()) {
                            clearInterval(interval);
                        }
                    }, 50);
                    // Disconnect after 2 seconds just in case
                    setTimeout(function() { clearInterval(interval); }, 2000);
                </script>
            """, height=0)
            st.markdown('<div style="display:none;">', unsafe_allow_html=True)
            if st.button("Trigger engine step", key="trigger_engine_step_btn", use_container_width=True):
                st.session_state.first_run_render = False
                rerun_app()
            st.markdown('</div>', unsafe_allow_html=True)
            
    elif st.session_state.is_finished:

        if st.session_state.completion_notice_batch_id != st.session_state.current_batch_id:
            st.toast("Benchmark complete. Results are ready.", icon="✅")
            st.session_state.completion_notice_batch_id = st.session_state.current_batch_id

        all_r = []
        for g in sorted_result_groups(st.session_state.last_run_results):
            img_idx = g["img_idx"]
            for m in g["models"]:
                for r in m["results"]:
                    r_copy = r.copy()
                    r_copy["Image Index"] = img_idx
                    all_r.append(r_copy)
            
        if all_r:
            fdf = normalize_metric_columns(pd.DataFrame(all_r))
            result_groups = sorted_result_groups(st.session_state.last_run_results)
            

            st.markdown('<div style="margin-top: 0.8rem;"></div>', unsafe_allow_html=True)
            if st.session_state.total_execution_time:
                st.markdown(format_run_duration(format_time(st.session_state.total_execution_time)), unsafe_allow_html=True)
            st.markdown(format_run_timestamps(st.session_state.batch_started_at, st.session_state.batch_completed_at), unsafe_allow_html=True)
            st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
            
            # --- EXPORT & NAVIGATION BUTTONS (Above configurations) ---
            ex1, ex2, ex3, ex4 = st.columns([1, 1, 1.4, 1.6])
            with ex1:
                if st.session_state.current_batch_id and os.path.exists(os.path.join(sm.base_dir, st.session_state.current_batch_id)):
                    csv_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.csv")
                    if not os.path.exists(csv_path):
                        generate_csv_report(st.session_state.last_run_results, csv_path)
                    if os.path.exists(csv_path):
                        with open(csv_path, "rb") as f:
                            st.download_button("📥 Export CSV", data=f, file_name=f"{st.session_state.current_batch_id}.csv", mime="text/csv", use_container_width=True)
            with ex2:
                if st.session_state.current_batch_id and os.path.exists(os.path.join(sm.base_dir, st.session_state.current_batch_id)):
                    pdf_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.pdf")
                    if not os.path.exists(pdf_path):
                        with st.spinner("Generating PDF..."):
                            generate_pdf_report(
                                 st.session_state.current_batch_id,
                                 st.session_state.last_run_results,
                                 st.session_state.current_batch_methods,
                                 pdf_path,
                                 st.session_state.total_execution_time,
                                 collect_environment_metadata(get_device_string(st.session_state.current_device_mode))
                            )
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button("📄 Export PDF", data=f, file_name=f"{st.session_state.current_batch_id}.pdf", mime="application/pdf", use_container_width=True)
            with ex3:
                if st.button("Repeat Config", key="repeat_current_active_btn", help="Load this configuration back into your workspace inputs to tweak or run it again.", use_container_width=True):
                    st.session_state.restore_config = {
                        "settings": {
                            "models": st.session_state.current_batch_models if st.session_state.current_batch_models else st.session_state.selected_models,
                            "methods": st.session_state.current_batch_methods if st.session_state.current_batch_methods else st.session_state.selected_methods,
                            "input_sizes": st.session_state.current_batch_sizes if st.session_state.current_batch_sizes else parse_input_sizes(st.session_state.get("input_size_str", "224")),
                            "repeat_count": st.session_state.current_repeats,
                            "warmup_runs": st.session_state.current_warmups,
                            "memory_runs": st.session_state.current_memory_runs,
                            "run_order": st.session_state.current_run_order,
                            "enable_quality_metrics": st.session_state.current_enable_quality_metrics,
                            "selected_quality_metrics": st.session_state.current_selected_quality_metrics
                        },
                        "environment": {
                            "selected_device": st.session_state.current_device_mode
                        }
                    }
                    st.session_state.is_finished = False
                    st.session_state.benchmark_running = False
                    st.session_state.benchmark_ready_to_run = False
                    st.session_state.last_run_results = []
                    st.session_state.completed_batch_id = ""
                    st.session_state.last_run_batch_id = ""
                    st.session_state.current_page = "Configure"
                    rerun_app()
            with ex4:
                if st.button("⬅️ Setup Another Run", key="back_from_run_btn", help="Reset all configuration inputs back to defaults to start a fresh benchmark from scratch.", use_container_width=True):
                    st.session_state.is_finished = False
                    st.session_state.benchmark_running = False
                    st.session_state.benchmark_ready_to_run = False
                    st.session_state.run_progress_idx = 0
                    st.session_state.task_queue = []
                    st.session_state.prepared_img_sources = []
                    st.session_state.current_batch_id = ""
                    st.session_state.batch_started_at = ""
                    st.session_state.batch_completed_at = ""
                    st.session_state.total_execution_time = 0
                    st.session_state.batch_start_time = None
                    st.session_state.current_page = "Configure"
                    st.session_state.selected_models = ["resnet50"]
                    st.session_state.selected_methods = ["Saliency", "Integrated_Gradients"]
                    st.session_state.input_size_str = "224"
                    st.session_state.selected_repeats = 5
                    st.session_state.selected_warmups = 1
                    st.session_state.selected_run_order = "Balanced"
                    st.session_state.selected_device_mode = default_device_mode
                    st.session_state.last_run_results = []
                    st.session_state.completed_batch_id = ""
                    st.session_state.last_run_batch_id = ""
                    st.session_state.current_batch_methods = []
                    st.session_state.current_batch_models = []
                    st.session_state.current_batch_sizes = []
                    rerun_app()
                    
            st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
            with st.expander("⚙️ Environment & Configuration Details", expanded=True, key=f"env_details_{st.session_state.current_batch_id}_finished"):
                meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
                with meta_col1:
                    st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                    render_environment_summary(collect_environment_metadata(get_device_string(st.session_state.current_device_mode)))
                with meta_col2:
                    st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                    active_settings = {
                        "models": st.session_state.current_batch_models,
                        "methods": st.session_state.current_batch_methods,
                        "input_sizes": st.session_state.current_batch_sizes,
                        "repeat_count": st.session_state.current_repeats,
                        "warmup_runs": st.session_state.current_warmups,
                        "memory_runs": st.session_state.current_memory_runs,
                        "selected_quality_metrics": st.session_state.selected_quality_metrics if st.session_state.get("enable_quality_metrics", False) else []
                    }
                    render_configuration_summary(active_settings, st.session_state.last_run_results)
                
                # Render parameter mapping table if parameterized methods exist
                render_parameters_mapping_table(st.session_state.get("current_batch_methods_info", []))

            render_analytics_sections(fdf, result_groups)

        st.markdown('<hr style="margin: 1.4rem 0 1.4rem 0; border: none; border-top: 1px solid var(--xai-border);">', unsafe_allow_html=True)
        render_detailed_results_header("current_final_results")
        for group in sorted_result_groups(st.session_state.last_run_results):
            render_result_group(group, st.session_state.current_batch_methods, key_suffix=f"finished_{st.session_state.current_batch_id}")
    else:
        st.info("No active benchmark run. Go to the **Configure Benchmark** page to set up and launch a run!")

# --- PAGE 3: HISTORY & EVALUATION ---
def render_history_page():
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    batches = sm.list_batches()
    if not batches:
        st.info("No benchmark history found. Start a new run in the 'Configure Benchmark' page!")
        return

    batch_ids = [b["id"] for b in batches]
    default_idx = 0
    if st.session_state.selected_history_batch in batch_ids:
        default_idx = batch_ids.index(st.session_state.selected_history_batch)
    else:
        if batch_ids:
            st.session_state.selected_history_batch = batch_ids[0]

    selected_bid = st.selectbox(
        "Select Benchmark Batch to View & Evaluate",
        batch_ids,
        index=default_idx,
        format_func=lambda bid: get_batch_display_name(bid, sm.base_dir),
        help="Select a batch to view its results, charts, and exports."
    )
    st.session_state.selected_history_batch = selected_bid

    if not selected_bid:
        st.info("Please select a batch from the list above.")
        return

    if True:
        bid = selected_bid
        batch_meta_p = os.path.join(sm.base_dir, bid, "batch_results.json")
        if os.path.exists(batch_meta_p):
            try:
                with open(batch_meta_p, 'r') as f:
                    meta = json.load(f)
                
                all_h_r = []
                for g in meta["results"]:
                    img_idx = g["img_idx"]
                    for m in g["models"]:
                        for r in m["results"]:
                            r_copy = r.copy()
                            r_copy["Image Index"] = img_idx
                            all_h_r.append(r_copy)
                
                if all_h_r:
                    hdf = normalize_metric_columns(pd.DataFrame(all_h_r))
                    hdf["Model_Size"] = hdf["Model"] + " (" + hdf.get("Resolution", "224x224") + ")"
                    h_runtime_col = metric_col(hdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
                    h_memory_col = metric_col(hdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
                    hdf = hdf.sort_values(by=["Method", "Resolution"])
                    
                    h_total_time = meta.get("total_execution_time", 0)
                    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                    
                    # Header row with metadata

                    if h_total_time:
                        st.markdown(format_run_duration(format_time(h_total_time)), unsafe_allow_html=True)
                    st.markdown(format_run_timestamps(meta.get('started_at'), meta.get('completed_at')), unsafe_allow_html=True)
                    st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
                    
                    # --- EXPORT & NAVIGATION BUTTONS (Above configurations) ---
                    hx1, hx2, hx3, hx4 = st.columns([1, 1, 1.4, 1.6])
                    with hx1:
                        h_csv = os.path.join(sm.base_dir, bid, f"{bid}.csv")
                        if not os.path.exists(h_csv):
                            generate_csv_report(meta["results"], h_csv)
                        if os.path.exists(h_csv):
                            with open(h_csv, "rb") as f:
                                st.download_button("📥 Export CSV", data=f, file_name=f"{bid}.csv", mime="text/csv", key=f"csv_{bid}", use_container_width=True)
                    with hx2:
                        h_pdf = os.path.join(sm.base_dir, bid, f"{bid}.pdf")
                        if not os.path.exists(h_pdf):
                            with st.spinner("Generating PDF..."):
                                generate_pdf_report(bid, meta["results"], meta["methods"], h_pdf, h_total_time, meta.get("environment"))
                        if os.path.exists(h_pdf):
                            with open(h_pdf, "rb") as f:
                                st.download_button("📄 Export PDF", data=f, file_name=f"{bid}.pdf", mime="application/pdf", key=f"pdf_{bid}", use_container_width=True)
                    with hx3:
                        if st.button("Repeat Config", key=f"repeat_{bid}", use_container_width=True):
                            settings = meta.get("benchmark_settings", {})
                            if settings:
                                st.session_state.restore_config = {
                                    "settings": settings,
                                    "environment": meta.get("environment", {})
                                }
                                st.session_state.is_finished = False
                                st.session_state.benchmark_running = False
                                st.session_state.benchmark_ready_to_run = False
                                st.session_state.last_run_results = []
                                st.session_state.completed_batch_id = ""
                                st.session_state.last_run_batch_id = ""
                                st.session_state.current_page = "Configure"
                                rerun_app()
                    with hx4:
                        if st.button("Delete Batch", key=f"del_{bid}", use_container_width=True):
                            # Pre-compute next batch selection
                            next_bid = None
                            if bid in batch_ids:
                                idx = batch_ids.index(bid)
                                if idx + 1 < len(batch_ids):
                                    next_bid = batch_ids[idx + 1]
                                elif idx - 1 >= 0:
                                    next_bid = batch_ids[idx - 1]
                            
                            sm.delete_batch(bid)
                            check_and_reset_session_after_delete(bid)
                            st.session_state.selected_history_batch = next_bid
                            st.success(f"Batch {bid} deleted.")
                            st.rerun()
                            
                    st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
                    with st.expander("⚙️ Environment & Configuration Details", expanded=False, key=f"env_details_{bid}_history"):
                        meta_col1, meta_col_spacer, meta_col2 = st.columns([1.8, 0.2, 2.0])
                        with meta_col1:
                            st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Environment Summary</div>', unsafe_allow_html=True)
                            render_environment_summary(meta.get("environment"))
                        with meta_col2:
                            st.markdown('<div style="font-weight: 600; margin-bottom: 8px; color: var(--xai-text);">Benchmark Configuration</div>', unsafe_allow_html=True)
                            render_configuration_summary(meta.get("benchmark_settings"), meta.get("results"))
                            
                        # Render parameter mapping table if parameterized methods exist
                        render_parameters_mapping_table(meta.get("methods_info", []))

                    render_analytics_sections(hdf, meta["results"])

                # Per-image results
                st.markdown('<hr style="margin: 1.4rem 0 1.4rem 0; border: none; border-top: 1px solid var(--xai-border);">', unsafe_allow_html=True)
                render_detailed_results_header(f"history_results_{bid}")
                for group in meta["results"]:
                    render_result_group(group, meta["methods"], key_suffix=f"history_{bid}")

            except Exception as e:
                header_left, header_right = st.columns([4, 1.2])
                with header_left:
                    st.error(f"Error reading historical data: {str(e)}")
                    st.exception(e)
                with header_right:
                    st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                    if st.button("Delete Batch", key=f"del_corr_{bid}", use_container_width=True):
                        # Pre-compute next batch selection
                        next_bid = None
                        if bid in batch_ids:
                            idx = batch_ids.index(bid)
                            if idx + 1 < len(batch_ids):
                                next_bid = batch_ids[idx + 1]
                            elif idx - 1 >= 0:
                                next_bid = batch_ids[idx - 1]
                        
                        sm.delete_batch(bid)
                        check_and_reset_session_after_delete(bid)
                        st.session_state.selected_history_batch = next_bid
                        st.success(f"Batch {bid} deleted.")
                        st.rerun()
        else:
            st.info("Loading metadata for this batch...")
    else:
        pass

def load_docs_reference():
    docs_path = os.path.join(os.path.dirname(__file__), "docs_reference.json")
    if os.path.exists(docs_path):
        try:
            with open(docs_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading documentation reference file: {e}")
    return {}

def render_documentation_page():
    st.markdown('<div style="margin-top: 1.0rem;"></div>', unsafe_allow_html=True)
    st.markdown("## XAI Methods, Models & Metrics")
    st.markdown("Technical specifications and mathematical formulations for supported attribution algorithms, model architectures, and evaluation metrics.")
    
    docs_data = load_docs_reference()
    
    # Section 1: XAI Methods
    with st.expander("🔬 Feature Attribution Methods", expanded=True):
        xai_groups = docs_data.get("xai_methods", [])
        for group in xai_groups:
            st.markdown(f"#### {group.get('category', '')}")
            
            rows = []
            for item in group.get("methods", []):
                name = item.get('name', '')
                formulation = item.get('formulation', '')
                complexity = item.get('complexity', '')
                characteristics = item.get('characteristics', '')
                
                # Convert markdown italic syntax from JSON to HTML tags
                name_html = name.replace("*(", "<i>(").replace(")*", ")</i>").replace("*", "<i>")
                
                row = (
                    "<tr>"
                    f"<td>{name_html}</td>"
                    f"<td>{formulation}</td>"
                    f"<td>{complexity}</td>"
                    f"<td>{characteristics}</td>"
                    "</tr>"
                )
                rows.append(row)
                
            table_html = (
                '<div class="doc-reference-table">'
                '<table>'
                '<thead>'
                '<tr>'
                '<th class="col-xai-name">Algorithm</th>'
                '<th class="col-xai-form">Mathematical Formulation</th>'
                '<th class="col-xai-comp">Complexity</th>'
                '<th class="col-xai-char">Key Characteristics & Properties</th>'
                '</tr>'
                '</thead>'
                '<tbody>'
                + "".join(rows) +
                '</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

    # Section 2: Model Architectures
    with st.expander("🏗️ Vision Model Architectures", expanded=False):
        model_groups = docs_data.get("models", [])
        for group in model_groups:
            st.markdown(f"#### {group.get('category', '')}")
            
            rows = []
            for item in group.get("models", []):
                name = item.get('name', '')
                params = item.get('params', '')
                characteristics = item.get('characteristics', '')
                
                name_html = name.replace("*(", "<i>(").replace(")*", ")</i>").replace("*", "<i>")
                
                row = (
                    "<tr>"
                    f"<td>{name_html}</td>"
                    f"<td>{params}</td>"
                    f"<td>{characteristics}</td>"
                    "</tr>"
                )
                rows.append(row)
                
            table_html = (
                '<div class="doc-reference-table">'
                '<table>'
                '<thead>'
                '<tr>'
                '<th class="col-model-name">Model Backbone</th>'
                '<th class="col-model-params">Parameters (M)</th>'
                '<th class="col-model-char">Design Paradigm & Key Innovations</th>'
                '</tr>'
                '</thead>'
                '<tbody>'
                + "".join(rows) +
                '</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

    # Section 3: Benchmark Metrics
    with st.expander("📊 Evaluation Metrics", expanded=False):
        metric_groups = docs_data.get("metrics", [])
        for group in metric_groups:
            st.markdown(f"#### {group.get('category', '')}")
            
            rows = []
            for item in group.get("metrics", []):
                name = item.get('name', '')
                unit = item.get('unit', '')
                definition = item.get('definition', '')
                
                name_html = name.replace("*(", "<i>(").replace(")*", ")</i>").replace("*", "<i>")
                
                row = (
                    "<tr>"
                    f"<td>{name_html}</td>"
                    f"<td>{unit}</td>"
                    f"<td>{definition}</td>"
                    "</tr>"
                )
                rows.append(row)
                
            table_html = (
                '<div class="doc-reference-table">'
                '<table>'
                '<thead>'
                '<tr>'
                '<th class="col-metric-name">Metric Name</th>'
                '<th class="col-metric-unit">Measurement Unit</th>'
                '<th class="col-metric-def">Definition & Evaluation Logic</th>'
                '</tr>'
                '</thead>'
                '<tbody>'
                + "".join(rows) +
                '</tbody>'
                '</table>'
                '</div>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

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
                    "selected_quality_metrics": []
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
            "selected_quality_metrics": st.session_state.current_selected_quality_metrics
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
