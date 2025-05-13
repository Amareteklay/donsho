import streamlit as st, src

st.set_page_config(page_title="EPPS Shocks", layout="centered")
st.title("🌍 EPPS Shocks & Disease Outbreak Explorer")

st.markdown("""
In this dashboard we:

1. Inspect the raw shock database  
2. Decide preprocessing parameters (rare-to-binary, lag length …)  
3. Fit Poisson / NB / GEE models step-by-step  
4. Compare predictions to observations
""")
