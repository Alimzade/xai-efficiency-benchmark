import streamlit as st
import streamlit.components.v1 as components
import os
import time
import pandas as pd
import torch
import platform
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import random
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from session_manager import SessionManager
from benchmark_runner import collect_environment_metadata, run_benchmark_task
from exporter import generate_pdf_report, generate_csv_report

# --- SILENCE NOISY WARNINGS ---
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="captum.attr._utils.visualization")

# --- Page Config ---
st.set_page_config(page_title="XAI Efficiency Benchmark", page_icon="🔍", layout="wide")

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
        margin-bottom: 1rem;
        padding: 0.35rem 0 0.2rem 0;
    }
    .app-title h1 {
        font-size: 1.75rem;
        letter-spacing: 0;
        margin: 0;
        color: var(--xai-text);
    }
    .app-title p {
        color: var(--xai-muted);
        margin: 0.05rem 0 0 0;
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
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.18), rgba(45, 212, 191, 0.12)) !important;
        color: var(--xai-text) !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        border-color: rgba(45, 212, 191, 0.55) !important;
        background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18)) !important;
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
    .compact-preview { max-width: 300px; margin-left: auto; margin-right: 0; }
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
    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 10px;
        min-height: 40px; /* Align height with multiselect input box */
    }
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-of-type {
        display: none !important;
    }
    div[data-testid="stRadio"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.045) !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        padding: 6px 18px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin: 0 !important;
        color: rgba(229, 237, 246, 0.75) !important;
        display: flex;
        align-items: center;
        justify-content: center;
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
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class="app-title">
        <h1>XAI Efficiency Benchmark</h1>
        <p>Compare attribution runtime, peak memory, and image-size behavior across models and methods.</p>
    </div>
    """, unsafe_allow_html=True)

sm = SessionManager()

# --- HELPER FUNCTIONS ---
ATTR_RUNTIME_COL = "Attribution Runtime (sec)"
ATTR_MEMORY_COL = "Peak Attribution Memory (MB)"
LEGACY_RUNTIME_COL = "Runtime (sec)"
LEGACY_MEMORY_COL = "Peak Memory (MB)"
METADATA_COLS = ["Timing Scope", "Memory Scope", "Model Cache"]
PRESENTATION_COL_ORDER = [
    "Method",
    "Model",
    "Input Size (px)",
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
    "Status",
]

def metric_col(df, preferred, legacy):
    return preferred if preferred in df.columns else legacy

def normalize_metric_columns(df):
    df = df.copy()
    if ATTR_RUNTIME_COL not in df.columns and LEGACY_RUNTIME_COL in df.columns:
        df[ATTR_RUNTIME_COL] = df[LEGACY_RUNTIME_COL]
    if ATTR_MEMORY_COL not in df.columns and LEGACY_MEMORY_COL in df.columns:
        df[ATTR_MEMORY_COL] = df[LEGACY_MEMORY_COL]
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
        LEGACY_MEMORY_COL, "Resolution"
    ] + METADATA_COLS
    df = df.drop(columns=[c for c in duplicate_cols if c in df.columns], errors="ignore")
    ordered_cols = [c for c in PRESENTATION_COL_ORDER if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in ordered_cols]
    return df[ordered_cols + remaining_cols]

def build_task_queue(num_images, models, sizes, methods, run_order, seed=None):
    tasks = []
    if run_order == "Balanced":
        for img_i in range(num_images):
            for mod_i, model_name in enumerate(models):
                for met_i, method_name in enumerate(methods):
                    rotation = (img_i + mod_i + met_i) % len(sizes)
                    ordered_sizes = sizes[rotation:] + sizes[:rotation]
                    for target_size in ordered_sizes:
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_name,
                        })
    else:
        for img_i in range(num_images):
            for model_name in models:
                for method_name in methods:
                    for target_size in sizes:
                        tasks.append({
                            "img_i": img_i,
                            "model_name": model_name,
                            "target_size": target_size,
                            "method_name": method_name,
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

def get_cpu_info(): return platform.processor() or "Generic CPU"
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

def parse_input_sizes(size_str):
    try:
        sizes = [int(s.strip()) for s in size_str.split(",") if s.strip().isdigit()]
        return sizes or [224]
    except Exception:
        return [224]

def current_image_sources():
    uploaded_files = st.session_state.get("uploaded_files") or []
    url_list = [u.strip() for u in st.session_state.persisted_urls.split("\n") if u.strip()]
    return list(uploaded_files) + url_list

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

def style_dataframe(df):
    df = add_input_size_column(normalize_metric_columns(df))
    subset_cols = [c for c in [
        ATTR_RUNTIME_COL, ATTR_MEMORY_COL,
        "Attribution Runtime Median (sec)", "Attribution Runtime Mean (sec)",
        "Attribution Runtime Std (sec)", "Attribution Runtime Min (sec)",
        "Attribution Runtime Max (sec)", "Mean Attribution Runtime (sec)",
        "Std Across Images (sec)", "Mean Peak Attribution Memory (MB)"
    ] if c in df.columns]
    formatters = {c: "{:.4f}" if "sec" in c else "{:.2f}" for c in subset_cols}
    if "Input Size (px)" in df.columns:
        formatters["Input Size (px)"] = "{:.0f}"
    if "Samples" in df.columns:
        formatters["Samples"] = "{:.0f}"
    return df.style.background_gradient(cmap="coolwarm", subset=subset_cols).format(formatters, na_rep="")

def plot_method_runtime_log(df, title="Runtime Comparison (Log Scale)"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(10, 6))
    summary = df.groupby("Method")[runtime_col].mean().sort_values().reset_index()
    sns.barplot(data=summary, x=runtime_col, y="Method", palette="crest", ax=ax, edgecolor="black")
    ax.set_xscale("log"); ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    ax.grid(True, ls="-", alpha=0.2); plt.tight_layout(); return fig

def plot_model_comparison_grouped(df, title="Architecture Efficiency Comparison"):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df, x="Method", y=runtime_col, hue="Model", palette="colorblind", ax=ax, edgecolor="black")
    ax.set_title(title, fontsize=14, fontweight='bold', family='serif')
    plt.xticks(rotation=45); ax.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout(); return fig

def image_size_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    if "Input Size (px)" not in df.columns or df["Input Size (px)"].nunique() < 2:
        return pd.DataFrame()

    summary = df.groupby(["Model", "Method", "Input Size (px)"]).agg(
        **{
            "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
            "Std Across Images (sec)": (runtime_col, "std"),
            "Samples": (runtime_col, "count"),
            "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
        }
    ).reset_index()
    summary["Std Across Images (sec)"] = summary["Std Across Images (sec)"].fillna(0)
    summary["Input Size (px)"] = summary["Input Size (px)"].astype(int)
    return summary.sort_values(["Model", "Method", "Input Size (px)"])

def plot_image_size_scaling(summary_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    for (model_name, method_name), group in summary_df.groupby(["Model", "Method"]):
        group = group.sort_values("Input Size (px)")
        label = f"{model_name} / {method_name}"
        ax.errorbar(
            group["Input Size (px)"],
            group["Mean Attribution Runtime (sec)"],
            yerr=group["Std Across Images (sec)"],
            marker="o",
            capsize=4,
            label=label
        )
    ax.set_xlabel("Input Size (px)")
    ax.set_ylabel("Mean Attribution Runtime (sec)")
    ax.set_title("Image Size Scaling", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig

def method_detail_summary(df):
    df = add_input_size_column(normalize_metric_columns(df))
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    summary = df.groupby("Method").agg(
        **{
            "Mean Attribution Runtime (sec)": (runtime_col, "mean"),
            "Std Across Images (sec)": (runtime_col, "std"),
            "Min Attribution Runtime (sec)": (runtime_col, "min"),
            "Max Attribution Runtime (sec)": (runtime_col, "max"),
            "Mean Peak Attribution Memory (MB)": (memory_col, "mean"),
            "Samples": (runtime_col, "count"),
        }
    ).reset_index()
    summary["Std Across Images (sec)"] = summary["Std Across Images (sec)"].fillna(0)
    return summary.sort_values("Mean Attribution Runtime (sec)")

def fastest_slowest_rows(df, count=5):
    df = presentation_df(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    cols = [c for c in ["Method", "Model", "Input Size (px)", "Resolution", "Prediction", runtime_col, ATTR_MEMORY_COL] if c in df.columns]
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

def plot_runtime_memory_scatter(df):
    df = normalize_metric_columns(df)
    runtime_col = metric_col(df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
    memory_col = metric_col(df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=df, x=runtime_col, y=memory_col, hue="Method", style="Model", s=90, ax=ax)
    ax.set_title("Runtime vs Peak Attribution Memory", fontsize=14, fontweight='bold')
    ax.set_xlabel("Attribution Runtime (sec)")
    ax.set_ylabel("Peak Attribution Memory (MB)")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    return fig

def render_environment_summary(environment):
    if not environment:
        return
    with st.expander("Environment Metadata", expanded=False):
        gpu_names = ", ".join([d.get("name", "Unknown GPU") for d in environment.get("cuda_devices", [])]) or "None"
        env_df = pd.DataFrame([
            {"Field": "Git Commit", "Value": environment.get("git_commit", "unknown")},
            {"Field": "Python", "Value": environment.get("python_version", "unknown")},
            {"Field": "Platform", "Value": environment.get("platform", "unknown")},
            {"Field": "Torch", "Value": environment.get("torch_version", "unknown")},
            {"Field": "Torch CUDA", "Value": environment.get("torch_cuda_version") or "not available"},
            {"Field": "CUDA Available", "Value": environment.get("cuda_available", False)},
            {"Field": "Selected Device", "Value": environment.get("selected_device", "unknown")},
            {"Field": "GPU(s)", "Value": gpu_names},
        ])
        st.table(env_df)

def render_result_view_controls(key_prefix):
    components.html(f"""
        <style>
            body {{ margin: 0; background: transparent; }}
            .xai-expander-controls {{
                display: flex;
                gap: 0.45rem;
                align-items: center;
                margin: 0.1rem 0 0.55rem 0;
            }}
            .xai-expander-controls button {{
                border: 1px solid rgba(148, 163, 184, 0.26);
                border-radius: 8px;
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.16), rgba(45, 212, 191, 0.1));
                color: #e5edf6;
                cursor: pointer;
                font: 600 12px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                padding: 0.35rem 0.7rem;
            }}
            .xai-expander-controls button:hover {{
                border-color: rgba(45, 212, 191, 0.55);
                background: linear-gradient(180deg, rgba(96, 165, 250, 0.24), rgba(45, 212, 191, 0.18));
            }}
        </style>
        <div class="xai-expander-controls" data-key="{key_prefix}">
            <button type="button" data-action="expand">Expand all</button>
            <button type="button" data-action="collapse">Collapse all</button>
        </div>
        <script>
            const root = window.parent.document;
            const controls = document.querySelector('.xai-expander-controls[data-key="{key_prefix}"]');
            function imageResultDetails() {{
                return Array.from(root.querySelectorAll('details')).filter((details) => {{
                    const summary = details.querySelector('summary');
                    return summary && summary.textContent.includes('Results for Image');
                }});
            }}
            controls.addEventListener('click', (event) => {{
                const action = event.target.dataset.action;
                if (!action) return;
                imageResultDetails().forEach((details) => {{
                    details.open = action === 'expand';
                }});
            }});
        </script>
    """, height=45)

def render_result_group(group, selected_methods, expanded=True):
    with st.expander(f"🖼️ Results for Image {group['img_idx']}", expanded=expanded):
        # Group entries by base model architecture
        architectures = []
        for m in group["models"]:
            if m["model"] not in architectures: architectures.append(m["model"])
        
        for arch in architectures:
            st.markdown(f"#### Model: `{arch}`")
            arch_models = sorted(
                [m for m in group["models"] if m["model"] == arch],
                key=lambda m: m.get("input_size", 0)
            )
            
            # Layout: Input Image (Left) | Method Collage (Right)
            col_left, col_right = st.columns([1, 3])
            
            # 1. Show Input Image once for this Architecture
            sample_m = arch_models[0]
            img_path = os.path.join(sample_m["session_dir"], "input_image.jpg")
            if os.path.exists(img_path):
                # Pull original resolution from the first result entry
                orig_res = sample_m["results"][0].get("Original Resolution", "Unknown") if sample_m["results"] else "Unknown"
                col_left.image(img_path, caption=f"Input Image ({orig_res})", use_container_width=True)
            
            # 2. Show Heatmap Rows (One row per Method, sizes side-by-side)
            with col_right:
                for method in selected_methods:
                    st.markdown(f"**{method}**")
                    # Create at least 4 columns to ensure thumbnails stay small and separate
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
                st.table(style_dataframe(presentation_df(pd.DataFrame(arch_results))))

@st.dialog("Image Viewer", width="large")
def show_lightbox(img):
    st.markdown(f'<div style="display: flex; justify-content: center;"><img src="data:image/png;base64,{st.session_state.current_img_base64}" style="max-height: 80vh; max-width: 100%; object-fit: contain;"></div>', unsafe_allow_html=True)

# --- Initialize Session State ---
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
if 'selected_run_order' not in st.session_state: st.session_state.selected_run_order = "Balanced"
default_device_mode = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
if 'selected_device_mode' not in st.session_state: st.session_state.selected_device_mode = default_device_mode
if 'current_page' not in st.session_state: st.session_state.current_page = "Configure"

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
]
run_order_notes = {
    "Balanced": "Rotates size order across images/models/methods; recommended for size studies.",
    "Grouped": "Runs image -> model -> method -> size in fixed order; useful for debugging.",
    "Randomized": "Shuffles all tasks with a batch-specific seed; useful for robustness checks.",
}
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
    
    fig, ax = plt.subplots(figsize=(12, 7))
    df["Model_Method"] = df["Model"] + "\n(" + df["Method"] + ")"
    df_sorted = df.sort_values(by=["Model_Method", "Batch"])
    
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
def render_image_preview_gallery():
    img_sources = current_image_sources()
    if img_sources:
        num_imgs = len(img_sources)
        if st.session_state.img_idx >= num_imgs:
            st.session_state.img_idx = 0
        st.markdown('<div class="compact-preview">', unsafe_allow_html=True)
        current_src = img_sources[st.session_state.img_idx]
        try:
            if hasattr(current_src, 'name'):
                img_view = Image.open(current_src)
            else:
                response = requests.get(current_src)
                img_view = Image.open(BytesIO(response.content))
            st.session_state.current_img_base64 = get_base64(img_view)
            st.markdown(f"<div style='text-align: center; color: #94a3b8; font-size: 0.8em; margin-bottom: 2px;'>Resolution: {img_view.size[0]}x{img_view.size[1]} px</div>", unsafe_allow_html=True)
            st.markdown(f"""
                <style>
                .st-key-image_preview_lightbox_trigger button {{
                    height: 310px !important;
                    width: 100% !important;
                    background-image: url('data:image/png;base64,{st.session_state.current_img_base64}') !important;
                    background-size: contain !important;
                    background-position: center !important;
                    background-repeat: no-repeat !important;
                    border: 1px solid rgba(148, 163, 184, 0.16) !important;
                    background-color: rgba(15, 23, 42, 0.22) !important;
                    border-radius: 8px !important;
                    cursor: pointer !important;
                    padding: 0 !important;
                }}
                .st-key-image_preview_lightbox_trigger button:hover {{
                    border-color: rgba(45, 212, 191, 0.55) !important;
                    background-color: rgba(15, 23, 42, 0.32) !important;
                }}
                .st-key-image_preview_lightbox_trigger button:focus {{
                    box-shadow: 0 0 0 1px rgba(45, 212, 191, 0.55) !important;
                }}
                .st-key-image_preview_lightbox_trigger button div p {{
                    display: none !important;
                }}
                </style>
            """, unsafe_allow_html=True)
            if st.button("Click to view", key="image_preview_lightbox_trigger", use_container_width=True):
                show_lightbox(img_view)
        except Exception:
            st.markdown("<div style='height: 330px; text-align: center; padding-top: 100px; color: #94a3b8;'>Preview unavailable</div>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 0.8, 1])
        with n1:
            if st.button("⬅️ Prev", key="prev_btn", width='stretch'):
                st.session_state.img_idx = (st.session_state.img_idx - 1) % num_imgs
                rerun_fragment()
        with n2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px; font-weight: bold;'>{st.session_state.img_idx + 1}/{num_imgs}</div>", unsafe_allow_html=True)
        with n3:
            if st.button("Next ➡️", key="next_btn", width='stretch'):
                st.session_state.img_idx = (st.session_state.img_idx + 1) % num_imgs
                rerun_fragment()
        st.markdown('</div>', unsafe_allow_html=True)

if fragment_api:
    render_image_preview_gallery = fragment_api(render_image_preview_gallery)

# --- PAGE 1: CONFIGURE ---
def render_configure_page():
    st.markdown('<div class="step-header">Step 1: Configure Benchmark Settings</div>', unsafe_allow_html=True)
    
    # Row 1
    row1_left, row1_right = st.columns([2, 1])
    with row1_left:
        selected_models_widget = st.multiselect(
            "Model Architectures",
            model_opts,
            key="selected_models",
        )
    with row1_right:
        st.number_input(
            "Warmup runs",
            min_value=0,
            max_value=20,
            key="selected_warmups",
            step=1,
            help="Untimed runs before measurement. Useful for CUDA/model warmup.",
        )

    # Row 2
    row2_left, row2_right = st.columns([2, 1])
    with row2_left:
        fixed_size_trigger = any(m in ["vit-b-16", "swin-t"] for m in selected_models_widget)
        if fixed_size_trigger:
            st.text_input("Input Sizes (px)", value="224", disabled=True)
            st.caption("⚠️ *Fixed-size architecture selected (Locked to 224px)*")
        else:
            st.text_input(
                "Input Sizes (px)",
                key="input_size_str",
                help="Only applicable to CNN-based architectures."
            )
            st.markdown('<div style="margin-top: -15px; margin-bottom: 15px; font-size: 0.85em; color: gray;">Separate by commas (e.g., 224, 448, 512).</div>', unsafe_allow_html=True)
            if parse_input_sizes(st.session_state.input_size_str) == [224] and st.session_state.input_size_str.strip() not in ["", "224"]:
                st.error("Invalid size format. Using 224.")
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
        st.multiselect(
            "XAI Methods",
            xai_opts,
            key="selected_methods",
        )
    with row3_right:
        device_options = ["GPU (CUDA)", "CPU"] if torch.cuda.is_available() else ["CPU"]
        if st.session_state.selected_device_mode not in device_options:
            st.session_state.selected_device_mode = device_options[0]
        st.radio(
            "Target Device Selection",
            device_options,
            key="selected_device_mode",
            horizontal=True,
        )

    # Row 4
    row4_left, row4_right = st.columns([2, 1])
    with row4_left:
        st.markdown('<div style="margin-top: 35px; font-weight: bold; margin-bottom: 8px; font-size: 1.1em; color: var(--xai-text);">Measurement Details</div>', unsafe_allow_html=True)
        st.markdown('<div class="settings-hint">Warmups are not reported in statistics. Measured repeats are timed and summarized with median, mean, and standard deviation.<div style="margin-top: 6px;"><b>Task Ordering</b>: Benchmark runs are executed in a <b>Balanced</b> order (automatically rotating resolutions and model architectures) to mitigate PyTorch/CUDA caching allocator and execution-order bias.</div></div>', unsafe_allow_html=True)
    with row4_right:
        st.markdown('<div style="margin-top: 35px; font-weight: bold; margin-bottom: 8px; font-size: 1.1em; color: var(--xai-text);">Hardware Status</div>', unsafe_allow_html=True)
        if "GPU" in st.session_state.selected_device_mode:
            st.success(f"**GPU Active:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Active'}")
        else:
            st.warning(f"**CPU Active:** {get_cpu_info()}")

    st.divider()
    st.markdown('<div class="step-header">Step 2: Select Input Images</div>', unsafe_allow_html=True)
    
    col_input, col_spacer, col_preview = st.columns([2.5, 0.2, 0.8])
    with col_input:
        uploaded_files = st.file_uploader(
            "Drag and drop images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="uploaded_files"
        )
        with st.expander("Paste Image URLs", expanded=False):
            st.text_area(
                "Input URLs here",
                height=100,
                label_visibility="collapsed",
                key="persisted_urls"
            )
    with col_preview:
        render_image_preview_gallery()

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
            <span class="run-summary-item">Images <strong>{planned_image_count}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Size variations <strong>{len(selected_sizes)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Models <strong>{len(st.session_state.selected_models)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Methods <strong>{len(st.session_state.selected_methods)}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Repeats/config <strong>{st.session_state.selected_repeats}</strong></span>
            <span class="run-summary-separator">|</span>
            <span class="run-summary-item">Warmups/config <strong>{st.session_state.selected_warmups}</strong></span>
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
            st.error("Select at least one image, model, and XAI method.")
        else:
            st.session_state.current_batch_id = sm.start_batch()
            st.session_state.last_run_batch_id = st.session_state.current_batch_id
            st.session_state.current_run_order = st.session_state.selected_run_order
            st.session_state.current_batch_methods = list(st.session_state.selected_methods)
            st.session_state.current_batch_models = list(st.session_state.selected_models)
            st.session_state.current_batch_sizes = list(selected_sizes)
            st.session_state.task_queue = build_task_queue(
                len(img_sources),
                st.session_state.current_batch_models,
                st.session_state.current_batch_sizes,
                st.session_state.current_batch_methods,
                st.session_state.selected_run_order,
                seed=st.session_state.current_batch_id
            )
            st.session_state.prepared_img_sources = list(img_sources)
            st.session_state.batch_start_time = None
            st.session_state.total_execution_time = 0
            st.session_state.stop_requested = False
            st.session_state.run_progress_idx = 0
            st.session_state.benchmark_ready_to_run = True
            st.session_state.current_page = "Active Run"
            rerun_app()

# --- PAGE 2: ACTIVE RUN ---
def render_active_run_page():
    st.markdown("## ⚡ Benchmark Execution Engine")
    
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
                    setTimeout(() => button.click(), 650);
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
            st.session_state.batch_start_time = time.time()
            st.session_state.batch_started_at = timestamp_now()
            st.session_state.batch_completed_at = ""
            st.session_state.total_execution_time = 0
            st.session_state.benchmark_ready_to_run = False
            st.session_state.benchmark_running = True
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

        status_col, timer_col = st.columns([5, 1])
        with status_col:
            st.markdown(
                f"<div class='status-pulse'>🚀 STEP {idx + 1}/{total_steps}: Running {cur_met} on {cur_mod} @ {cur_size}px (Image {img_i + 1}, {st.session_state.current_run_order} order)</div>",
                unsafe_allow_html=True
            )
        with timer_col:
            render_live_elapsed_timer(st.session_state.batch_start_time)
            
        st.progress(idx / total_steps if total_steps else 0)
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
            
        if st.session_state.last_run_results:
            st.subheader("Completed Results So Far")
            render_result_view_controls("current_live_results")
            for group in sorted_result_groups(st.session_state.last_run_results):
                render_result_group(group, st.session_state.current_batch_methods)
                
    elif st.session_state.is_finished:
        st.success(f"Benchmark complete in {format_time(st.session_state.total_execution_time)}. Results, charts, and exports are ready.")
        if st.session_state.completion_notice_batch_id != st.session_state.current_batch_id:
            st.toast("Benchmark complete. Results are ready.", icon="✅")
            st.session_state.completion_notice_batch_id = st.session_state.current_batch_id

        all_r = []
        for g in sorted_result_groups(st.session_state.last_run_results):
            for m in g["models"]: all_r.extend(m["results"])
            
        if all_r:
            fdf = normalize_metric_columns(pd.DataFrame(all_r))
            fdf["Model_Size"] = fdf["Model"] + " (" + fdf["Resolution"] + ")"
            runtime_col = metric_col(fdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
            memory_col = metric_col(fdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
            
            st.markdown('<div class="step-header">Batch Summary</div>', unsafe_allow_html=True)
            st.markdown(f"**Batch Wall Time:** `{format_time(st.session_state.total_execution_time)}`")
            st.caption(f"Started: {display_timestamp(st.session_state.batch_started_at)} | Completed: {display_timestamp(st.session_state.batch_completed_at)}")
            render_environment_summary(collect_environment_metadata("cuda" if "GPU" in st.session_state.selected_device_mode else "cpu"))
            
            # --- EXPORT BUTTONS ---
            ex1, ex2, ex3 = st.columns([1, 1, 3])
            with ex1:
                csv_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.csv")
                if not os.path.exists(csv_path):
                    generate_csv_report(st.session_state.last_run_results, csv_path)
                if os.path.exists(csv_path):
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 Export CSV", data=f, file_name=f"{st.session_state.current_batch_id}.csv", mime="text/csv", use_container_width=True)
            with ex2:
                pdf_path = os.path.join(sm.base_dir, st.session_state.current_batch_id, f"{st.session_state.current_batch_id}.pdf")
                if not os.path.exists(pdf_path):
                    with st.spinner("Generating PDF..."):
                        generate_pdf_report(
                            st.session_state.current_batch_id,
                            st.session_state.last_run_results,
                            st.session_state.current_batch_methods,
                            pdf_path,
                            st.session_state.total_execution_time,
                            collect_environment_metadata("cuda" if "GPU" in st.session_state.selected_device_mode else "cpu")
                        )
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button("📄 Export PDF", data=f, file_name=f"{st.session_state.current_batch_id}.pdf", mime="application/pdf", use_container_width=True)
            with ex3:
                if st.button("⬅️ Setup Another Run", key="back_from_run_btn", use_container_width=True):
                    st.session_state.is_finished = False
                    st.session_state.current_page = "Configure"
                    rerun_app()

            cs1, cs2 = st.columns(2)
            with cs1:
                st.subheader("Configuration Averages")
                group_cols = ["Model", "Resolution"]
                if "Original Resolution" in fdf.columns: group_cols.append("Original Resolution")
                summary_df = fdf.groupby(group_cols).agg({runtime_col: "mean", memory_col: "mean"}).reset_index()
                st.table(style_dataframe(summary_df))
                fig1, ax1 = plt.subplots(figsize=(12, 7))
                sns.barplot(data=fdf, x="Method", y=runtime_col, hue="Model_Size", palette="colorblind", ax=ax1, edgecolor="black")
                ax1.set_title("Architecture & Resolution Efficiency", fontsize=14, fontweight='bold')
                plt.xticks(rotation=45); ax1.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
                st.pyplot(fig1)
            with cs2:
                st.subheader("Method Averages"); st.table(style_dataframe(fdf.groupby("Method").agg({runtime_col: "mean", memory_col: "mean"}).reset_index()))
                st.pyplot(plot_method_runtime_log(fdf))
            st.subheader("Method Detail")
            st.table(style_dataframe(method_detail_summary(fdf)))
            fs1, fs2 = st.columns(2)
            fastest_df, slowest_df = fastest_slowest_rows(fdf)
            with fs1:
                st.subheader("Fastest Runs")
                st.table(style_dataframe(fastest_df))
            with fs2:
                st.subheader("Slowest Runs")
                st.table(style_dataframe(slowest_df))
            dist1, dist2 = st.columns(2)
            with dist1:
                st.pyplot(plot_runtime_distribution(fdf))
            with dist2:
                st.pyplot(plot_runtime_memory_scatter(fdf))
            size_summary_df = image_size_summary(fdf)
            if not size_summary_df.empty:
                st.subheader("Image Size Scaling")
                st.table(style_dataframe(size_summary_df))
                st.pyplot(plot_image_size_scaling(size_summary_df))

            st.divider()

        st.markdown('<div class="step-header">Detailed Per-Image Attribution Heatmaps</div>', unsafe_allow_html=True)
        render_result_view_controls("current_final_results")
        for group in sorted_result_groups(st.session_state.last_run_results):
            render_result_group(group, st.session_state.current_batch_methods)
    else:
        st.info("No active benchmark run. Go to the **Configure Benchmark** page to set up and launch a run!")

# --- PAGE 3: HISTORY & EVALUATION ---
def render_history_page():
    st.markdown('<div class="step-header">Benchmark History & Evaluation</div>', unsafe_allow_html=True)
    
    batches = sm.list_batches()
    if not batches:
        st.info("No benchmark history found. Start a new run in the 'Configure Benchmark' page!")
        return

    selected_bids = st.multiselect(
        "Select Benchmark Batch(es) to View & Evaluate",
        [b["id"] for b in batches],
        help="Select one batch to view its standard results, or multiple batches to combine and compare them."
    )

    if not selected_bids:
        st.info("Please select one or more batches from the list above.")
        return

    if len(selected_bids) == 1:
        bid = selected_bids[0]
        batch_meta_p = os.path.join(sm.base_dir, bid, "batch_results.json")
        if os.path.exists(batch_meta_p):
            try:
                with open(batch_meta_p, 'r') as f:
                    meta = json.load(f)
                
                all_h_r = []
                for g in meta["results"]:
                    for m in g["models"]: all_h_r.extend(m["results"])
                
                if all_h_r:
                    hdf = normalize_metric_columns(pd.DataFrame(all_h_r))
                    hdf["Model_Size"] = hdf["Model"] + " (" + hdf.get("Resolution", "224x224") + ")"
                    h_runtime_col = metric_col(hdf, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
                    h_memory_col = metric_col(hdf, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
                    hdf = hdf.sort_values(by=["Method", "Resolution"])
                    
                    h_total_time = meta.get("total_execution_time", 0)
                    st.markdown('<div class="step-header">Batch Summary (Historical)</div>', unsafe_allow_html=True)
                    if h_total_time:
                        st.markdown(f"**Batch Wall Time:** `{format_time(h_total_time)}`")
                    st.caption(f"Started: {display_timestamp(meta.get('started_at'))} | Completed: {display_timestamp(meta.get('completed_at'))}")
                    render_environment_summary(meta.get("environment"))

                    hx1, hx2, hx3 = st.columns([1, 1, 3])
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

                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.subheader("Configuration Averages")
                        group_cols = ["Model", "Resolution"]
                        if "Original Resolution" in hdf.columns: group_cols.append("Original Resolution")
                        h_summ = hdf.groupby(group_cols).agg({h_runtime_col: "mean", h_memory_col: "mean"}).reset_index()
                        st.table(style_dataframe(h_summ))
                        fig_h, ax_h = plt.subplots(figsize=(12, 7))
                        sns.barplot(data=hdf, x="Method", y=h_runtime_col, hue="Model_Size", palette="colorblind", ax=ax_h, edgecolor="black")
                        plt.xticks(rotation=45); ax_h.legend(loc='upper left', bbox_to_anchor=(1, 1)); plt.tight_layout()
                        st.pyplot(fig_h)
                    with hc2:
                        st.subheader("Method Averages"); st.table(style_dataframe(hdf.groupby("Method").agg({h_runtime_col: "mean", h_memory_col: "mean"}).reset_index()))
                        st.pyplot(plot_method_runtime_log(hdf))
                    st.subheader("Method Detail")
                    st.table(style_dataframe(method_detail_summary(hdf)))
                    h_fs1, h_fs2 = st.columns(2)
                    h_fastest_df, h_slowest_df = fastest_slowest_rows(hdf)
                    with h_fs1:
                        st.subheader("Fastest Runs")
                        st.table(style_dataframe(h_fastest_df))
                    with h_fs2:
                        st.subheader("Slowest Runs")
                        st.table(style_dataframe(h_slowest_df))
                    h_dist1, h_dist2 = st.columns(2)
                    with h_dist1:
                        st.pyplot(plot_runtime_distribution(hdf))
                    with h_dist2:
                        st.pyplot(plot_runtime_memory_scatter(hdf))
                    h_size_summary_df = image_size_summary(hdf)
                    if not h_size_summary_df.empty:
                        st.subheader("Image Size Scaling")
                        st.table(style_dataframe(h_size_summary_df))
                        st.pyplot(plot_image_size_scaling(h_size_summary_df))
                        
                    st.divider()

                # Per-image results
                st.markdown('<div class="step-header">Detailed Per-Image Attribution Heatmaps</div>', unsafe_allow_html=True)
                render_result_view_controls(f"history_results_{bid}")
                for group in meta["results"]:
                    render_result_group(group, meta["methods"])

                st.divider()
                if st.button("🗑️ Delete Batch", key=f"del_{bid}", use_container_width=True):
                    sm.delete_batch(bid)
                    st.success(f"Batch {bid} deleted.")
                    st.rerun()
            except Exception as e:
                st.error(f"Error reading historical data: {str(e)}")
                if st.button("🗑️ Delete Corrupted Batch", key=f"del_corr_{bid}", use_container_width=True):
                    sm.delete_batch(bid)
                    st.success(f"Batch {bid} deleted.")
                    st.rerun()
        else:
            st.info("Loading metadata for this batch...")
    else:
        # Multi-Batch Evaluation Mode!
        st.markdown("### 🎛️ Combined Multi-Batch Evaluation Dashboard")
        
        all_combined_results = []
        env_records = []
        methods_in_batches = set()
        
        for bid in selected_bids:
            batch_meta_p = os.path.join(sm.base_dir, bid, "batch_results.json")
            if os.path.exists(batch_meta_p):
                try:
                    with open(batch_meta_p, 'r') as f:
                        meta = json.load(f)
                    
                    methods_in_batches.update(meta.get("methods", []))
                    
                    env = meta.get("environment", {})
                    gpu_names = ", ".join([d.get("name", "Unknown GPU") for d in env.get("cuda_devices", [])]) or "None"
                    env_records.append({
                        "Batch": bid,
                        "Device/GPU": gpu_names if gpu_names != "None" else env.get("selected_device", "CPU"),
                        "Torch": env.get("torch_version", "unknown"),
                        "CUDA": env.get("torch_cuda_version") or "N/A",
                        "Platform": env.get("platform", "unknown")[:25] + "..." if len(env.get("platform", "unknown")) > 25 else env.get("platform", "unknown"),
                        "Wall Time": format_time(meta.get("total_execution_time", 0))
                    })
                    
                    for group in meta["results"]:
                        for model_entry in group["models"]:
                            for r in model_entry["results"]:
                                r_copy = r.copy()
                                r_copy["Batch"] = bid
                                all_combined_results.append(r_copy)
                except Exception as e:
                    st.warning(f"Failed to load batch {bid}: {str(e)}")
                    
        if all_combined_results:
            combined_df = pd.DataFrame(all_combined_results)
            combined_df = add_input_size_column(normalize_metric_columns(combined_df))
            
            runtime_col = metric_col(combined_df, ATTR_RUNTIME_COL, LEGACY_RUNTIME_COL)
            memory_col = metric_col(combined_df, ATTR_MEMORY_COL, LEGACY_MEMORY_COL)
            
            if env_records:
                st.subheader("💻 Executing Environments")
                st.table(pd.DataFrame(env_records))
                
            st.divider()
            
            st.subheader("📊 Comparative Performance Visualizations")
            chart_tab1, chart_tab2 = st.tabs(["⏱️ Attribution Runtime", "💾 Peak Memory Overhead"])
            
            with chart_tab1:
                st.pyplot(plot_combined_batches_comparison(combined_df))
            with chart_tab2:
                st.pyplot(plot_combined_batches_memory(combined_df))
                
            st.divider()
            
            st.subheader("📈 Combined Grouped Averages")
            group_cols = ["Batch", "Model"]
            if "Resolution" in combined_df.columns:
                group_cols.append("Resolution")
            group_cols.append("Method")
            
            combined_summary_df = combined_df.groupby(group_cols).agg({
                runtime_col: "mean",
                memory_col: "mean",
                "Warmup Runs": "first",
                "Measured Runs": "first"
            }).reset_index()
            
            st.table(style_dataframe(combined_summary_df))
            
            st.divider()
            if st.button("🗑️ Delete All Selected Batches", use_container_width=True):
                for bid in selected_bids:
                    sm.delete_batch(bid)
                st.success("Selected batches deleted successfully!")
                st.rerun()
            
            with st.expander("🔍 View Raw Combined Dataset", expanded=False):
                st.dataframe(style_dataframe(presentation_df(combined_df)))
        else:
            st.error("No valid results found in the selected batches.")

# --- TABS WORKSPACE ---
tab1, tab2, tab3 = st.tabs(["🚀 Benchmark Workspace", "📜 Results History & Evaluation", "📖 Citation"])

with tab1:
    if st.session_state.benchmark_running or st.session_state.benchmark_ready_to_run or st.session_state.is_finished:
        render_active_run_page()
    else:
        render_configure_page()

with tab2:
    render_history_page()

with tab3:
    st.markdown('<div class="step-header">Citation Details</div>', unsafe_allow_html=True)
    st.markdown("If you use this benchmark in your research, papers, or projects, please cite it using the following BibTeX entry:")
    
    st.code("""@software{alimzade2025xai,
  author  = {Anar Alimzade},
  title   = {Efficiency Benchmark for XAI},
  year    = {2025},
  month   = {May},
  version = {1.0.0}
}""", language="bibtex")

# --- ENGINE ---
if st.session_state.benchmark_running and not st.session_state.is_finished:
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
            model_entry = {"model": model_name, "model_label": model_label, "input_size": target_size, "results": [], "session_dir": s_dir, "src_path": fs}
            target_group["models"].append(model_entry)
        
        results = run_benchmark_task({
            "model_name": model_name, 
            "image_source": model_entry["src_path"], 
            "methods": [method_name.lower()], 
            "force_device": "cuda" if "GPU" in st.session_state.selected_device_mode else "cpu", 
            "input_size": target_size,
            "warmup_runs": st.session_state.selected_warmups,
            "repeat_count": st.session_state.selected_repeats,
            "run_order": st.session_state.current_run_order
        }, model_entry["session_dir"])
        
        model_entry["results"].extend(results)
        st.session_state.run_progress_idx += 1
        
        if st.session_state.run_progress_idx >= len(task_queue):
            st.session_state.is_finished = True
            st.session_state.benchmark_running = False
            st.session_state.completed_batch_id = st.session_state.current_batch_id
            st.session_state.total_execution_time = time.time() - st.session_state.batch_start_time
            st.session_state.batch_completed_at = timestamp_now()
            
            clean_results = []
            for g in sorted_result_groups(st.session_state.last_run_results):
                cg = g.copy()
                if hasattr(cg["source"], 'name'): cg["source"] = cg["source"].name
                clean_results.append(cg)
            with open(os.path.join(sm.base_dir, st.session_state.current_batch_id, "batch_results.json"), 'w') as f:
                json.dump({
                    "results": clean_results, 
                    "methods": st.session_state.current_batch_methods,
                    "benchmark_settings": {
                        "warmup_runs": st.session_state.selected_warmups,
                        "repeat_count": st.session_state.selected_repeats,
                        "run_order": st.session_state.current_run_order,
                        "task_count": len(task_queue),
                        "models": st.session_state.current_batch_models,
                        "input_sizes": st.session_state.current_batch_sizes,
                        "methods": st.session_state.current_batch_methods
                    },
                    "environment": collect_environment_metadata("cuda" if "GPU" in st.session_state.selected_device_mode else "cpu"),
                    "started_at": st.session_state.batch_started_at,
                    "completed_at": st.session_state.batch_completed_at,
                    "total_execution_time": st.session_state.total_execution_time
                }, f, indent=4)
        st.rerun()
    else:
        st.session_state.is_finished = True
        st.session_state.benchmark_running = False
        st.session_state.completed_batch_id = st.session_state.current_batch_id
        st.session_state.batch_completed_at = timestamp_now()
        st.rerun()
