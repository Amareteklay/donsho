import streamlit as st
from src.data_loader import load_and_clean_counts
#from src.viz import heatmap_counts

st.title("📊 Explore raw counts")
df = load_and_clean_counts()
st.dataframe(df.head())
#st.altair_chart(heatmap_counts(df), use_container_width=True)
