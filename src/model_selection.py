from __future__ import annotations
from typing import List
import itertools
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, log_loss


def generate_model_grid(
    predictors: List[str],
    fixed_effects: List[str],
    scopes: List[str],
    max_predictors_per_model: int = 3
) -> pd.DataFrame:
    combos = []
    for i in range(1, min(max_predictors_per_model, len(predictors)) + 1):
        combos.extend(itertools.combinations(predictors, i))

    grid = list(itertools.product(scopes, fixed_effects, combos))
    return pd.DataFrame(grid, columns=["Scope", "FixedEffects", "Predictors"])


def build_formula(dv: str, predictors: List[str], fixed_effects: str) -> str:
    pred_str = " + ".join(predictors)
    fe_str = f" + {fixed_effects}" if fixed_effects else ""
    return f"{dv} ~ {pred_str}{fe_str}"


def fit_model(df: pd.DataFrame, formula: str):
    model = smf.logit(formula=formula, data=df).fit(disp=False)
    return model


def evaluate_model(model, df: pd.DataFrame, dv: str) -> dict:
    try:
        preds = model.predict(df)
        actual = df[dv]

        # Check for valid predictions
        if np.isnan(preds).any() or np.isnan(actual).any():
            print(f"⚠️ NaN in predictions or actuals for {dv}")
            return {"AUC": np.nan, "LogLoss": np.nan}

        # Check for constant classes
        if actual.nunique() < 2:
            print(f"⚠️ Only one class present in actuals: {actual.unique()}")
            return {"AUC": np.nan, "LogLoss": np.nan}

        auc = roc_auc_score(actual, preds)
        loss = log_loss(actual, preds)
        return {"AUC": auc, "LogLoss": loss}

    except Exception as e:
        print(f"⚠️ Evaluation error for {dv}: {e}")
        print("  Preds sample:", preds[:5].tolist())
        print("  Actual sample:", actual[:5].tolist())
        return {"AUC": np.nan, "LogLoss": np.nan}


def safe_clip(df: pd.DataFrame, cols: List[str], clip_val: float = 10_000) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].clip(-clip_val, clip_val)
    return df


def run_model_selection(
    df: pd.DataFrame,
    grid_df: pd.DataFrame,
    dv: str = "Infectious_disease_binary",
    continent_col: str = "Continent"
) -> pd.DataFrame:
    results = []

    for i, row in grid_df.iterrows():
        scope = row["Scope"]
        predictors = list(row["Predictors"])
        fe = row["FixedEffects"]

        if scope == "global":
            sub_df = df.copy()
        else:
            sub_df = df[df[continent_col] == scope].copy()
            if sub_df.empty:
                continue

        # Drop constant predictors
        predictors = [p for p in predictors if p in sub_df.columns and sub_df[p].nunique() > 1]
        if not predictors:
            continue

        # Drop rows with NaNs in any required columns
        required_cols = predictors + [dv]
        sub_df = sub_df.dropna(subset=required_cols)
        if sub_df.empty or sub_df[dv].nunique() < 2 or sub_df[dv].sum() < 3:
            continue

        # Final clipping
        sub_df = safe_clip(sub_df, predictors)

        # Optional: build required cols for formula handling
        keep_cols = set(predictors + [dv])
        if "C(Year)" in fe:
            keep_cols.add("Year")
        if "C(Country)" in fe:
            keep_cols.add("Country")
        if "C(Continent)" in fe:
            keep_cols.add("Continent")
        sub_df = sub_df[[col for col in keep_cols if col in sub_df.columns]]

        formula = build_formula(dv, predictors, fe)

        try:
            model = fit_model(sub_df, formula)
            metrics = evaluate_model(model, sub_df, dv)
            results.append({
                "ModelID": f"mod_{i}",
                "Scope": scope,
                "FixedEffects": fe,
                "Predictors": predictors,
                "Formula": formula,
                **metrics
            })
        except Exception as e:
            print(f"⚠️ Model {i} failed — {formula}\n  → {e}")
            continue

    print(f"✅ Successfully estimated {len(results)} models.")
    return pd.DataFrame(results)



