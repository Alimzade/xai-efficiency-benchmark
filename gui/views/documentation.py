"""
Documentation Page Module
Renders the in-app documentation and reference guides for the XAI Efficiency Benchmark.
"""
import streamlit as st

from utils.helpers import load_docs_reference

def render_documentation_page():
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    st.markdown("Technical specifications and mathematical formulations for supported attribution algorithms, model architectures, and evaluation metrics.")
    
    docs_data = load_docs_reference()
    
    # Section 1: XAI Methods
    with st.expander("🔬 Explainable AI (XAI) Methods", expanded=False):
        xai_groups = docs_data.get("xai_methods", [])
        for group in xai_groups:
            st.markdown(f"#### {group.get('category', '')}")
            table_md = "| Algorithm | Mathematical Formulation | Complexity | Key Characteristics & Properties |\n| :--- | :--- | :--- | :--- |\n"
            for item in group.get("methods", []):
                table_md += f"| **{item.get('name', '')}** | {item.get('formulation', '')} | {item.get('complexity', '')} | {item.get('characteristics', '')} |\n"
            st.markdown(f'<div class="doc-reference-table">\n\n{table_md}\n\n</div>', unsafe_allow_html=True)

    # Section 2: Model Architectures
    with st.expander("🏗️ Vision Model Architectures", expanded=False):
        model_groups = docs_data.get("models", [])
        for group in model_groups:
            st.markdown(f"#### {group.get('category', '')}")
            table_md = "| Model Backbone | Parameters (M) | Design Paradigm & Key Innovations |\n| :--- | :--- | :--- |\n"
            for item in group.get("models", []):
                table_md += f"| **{item.get('name', '')}** | {item.get('params', '')} | {item.get('characteristics', '')} |\n"
            st.markdown(f'<div class="doc-reference-table">\n\n{table_md}\n\n</div>', unsafe_allow_html=True)

    # Section 3: Benchmark Metrics & Strategies
    with st.expander("📊 Evaluation Metrics & Strategies", expanded=True):
        metric_groups = docs_data.get("metrics", [])
        for group in metric_groups:
            st.markdown(f"#### {group.get('category', '')}")
            table_md = "| Metric Name | Measurement Unit | Definition & Evaluation Logic |\n| :--- | :--- | :--- |\n"
            for item in group.get("metrics", []):
                table_md += f"| **{item.get('name', '')}** | {item.get('unit', '')} | {item.get('definition', '')} |\n"
            st.markdown(f'<div class="doc-reference-table">\n\n{table_md}\n\n</div>', unsafe_allow_html=True)
