import streamlit as st
from src.data_loader   import load_raw_csv
from src.prep          import preprocess_counts
from src.diagnostics   import show_head, show_info, show_description

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Data Preprocessing")

# 1) Load raw
raw_df = load_raw_csv()
st.subheader("Raw Data Preview")
st.dataframe(show_head(raw_df))

# 2) Preprocess
df = preprocess_counts(raw_df, save_path="data/02_interim/clean.parquet")
st.subheader("Cleaned Data Preview")
st.dataframe(show_head(df))

# 3) Diagnostics
st.subheader("Cleaned Data Structure")
st.dataframe(show_info(df))

st.subheader("Cleaned Data Summary Statistics")
st.dataframe(show_description(df))


# Optional: let user download the clean parquet
st.download_button(
    label="Download cleaned data",
    data=open("data/02_interim/clean.parquet","rb"),
    file_name="clean_counts.parquet",
    mime="application/octet-stream"
)
