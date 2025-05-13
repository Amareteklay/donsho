# src/models.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf


def train_test_split_ts(df: pd.DataFrame, date_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Time-aware train/test split sorted by date_col."""
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def fit_poisson(df: pd.DataFrame, formula: str):
    """Fit a Poisson regression model via GLM."""
    return smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit()


def fit_mixed_effects(df: pd.DataFrame, formula: str, groups: str):
    """Fit a random intercept mixed effects model."""
    model = smf.mixedlm(formula, df, groups=df[groups])
    return model.fit()


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    """Evaluate model predictions against true y."""
    preds = model.predict(X)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y, preds))),
        "R2": float(r2_score(y, preds))
    }


def tune_hyperparameters(estimator, param_grid: dict, X: pd.DataFrame, y: pd.Series, cv=5):
    """Grid search with cross-validation."""
    grid = GridSearchCV(estimator, param_grid, cv=cv)
    grid.fit(X, y)
    return grid.best_estimator_


def filter_model_data(df: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop rows with NaN or Inf in outcome or predictors.
    Drop predictors with zero variance.
    """
    df = df.copy()
    cols = [outcome] + predictors
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)

    zero_var = [col for col in predictors if df[col].std() == 0]
    predictors = [col for col in predictors if col not in zero_var]
    df = df.drop(columns=zero_var, errors="ignore")

    return df, predictors
