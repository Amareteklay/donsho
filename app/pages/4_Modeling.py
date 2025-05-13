import streamlit as st
from src.data_loader   import load_raw_csv
from src.prep          import preprocess_counts
from src.models        import (
    train_test_split_ts,
    fit_poisson,
    evaluate_model,
    filter_model_data
)
from src.features      import build_panel

st.title("📈 Modeling")

# 1. Load and preprocess
raw_df = preprocess_counts(load_raw_csv())
df = build_panel(raw_df)  # or load a saved modeling panel

# 2. Predictor + target setup
all_shocks = [c for c in df.columns if c not in ("Country_name", "Continent", "Year", "Infectious_disease")]
predictors = st.multiselect("Choose predictors", options=all_shocks, default=["CLIMATIC"])
outcome = "Infectious_disease"

# 3. Filter the data to remove rows with NaN/Inf and drop zero-variance predictors
df, predictors = filter_model_data(df, outcome, predictors)

# 4. Build formula and fit model
formula = f"{outcome} ~ {' + '.join(predictors)}"
train, test = train_test_split_ts(df, date_col="Year")
model = fit_poisson(train, formula)

# 5. Show model summary
st.subheader("Model Summary")
st.write(model.summary())

# 6. Show evaluation metrics
metrics = evaluate_model(model, test, test[outcome])
st.metric("RMSE", f"{metrics['RMSE']:.2f}")
st.metric("R²", f"{metrics['R2']:.2f}")
