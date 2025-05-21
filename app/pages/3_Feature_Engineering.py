import streamlit as st
from z_legacy.data_loader   import load_raw_csv
from z_legacy.prep        import preprocess_counts
from feature_engineering    import pivot_categories, add_lags, build_panel
from src.diagnostics import show_head
from src.config import MAX_LAG

st.title("⚙️ Feature Engineering")

# 1) Load & preprocess
raw_df = preprocess_counts(load_raw_csv())

st.subheader("Pivot Shock Categories")
panel_wide = pivot_categories(raw_df)
st.dataframe(show_head(panel_wide))

# 2) Add lags
shock_cols = [c for c in panel_wide.columns 
              if c not in ("Country_name","Continent","Year")]
panel_lags = add_lags(panel_wide, cols=shock_cols)
st.subheader(f"With up to {MAX_LAG} Lags")
st.dataframe(show_head(panel_lags))

# 3) Build model-ready panel
model_df = build_panel(raw_df)
st.subheader("Modeling Panel")
st.dataframe(show_head(model_df))
