import streamlit as st
from z_legacy.data_loader   import load_raw_csv
from src.diagnostics   import show_head, show_info

st.set_page_config(
    page_title="Explroe Data",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📊 Explore Data")

# ── 1) Load raw data (uncleaned) ────────────────────────────────────────
raw_df = load_raw_csv()                           # pure I/O
st.subheader("Raw Data Preview")
st.dataframe(show_head(raw_df))

st.subheader("Raw Data Structure")
st.dataframe(show_info(raw_df))
