"""
train.py — Model training, comparison, and serialization module
House Price Prediction Project | Ames Housing Dataset

Models trained:
  1. Linear Regression  — baseline
  2. Ridge              — L2 regularization
  3. Random Forest      — non-linear ensemble
  4. XGBoost            — gradient boosting (primary candidate)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor

MODEL_PATH = Path(__file__).parent.parent / "models" / "house_price_model.joblib"

# Key features for Streamlit app (top 10 by importance)
APP_FEATURES = [
    "GrLivArea", "OverallQual", "TotalBsmtSF", "GarageCars",
    "YearBuilt", "Neighborhood", "BldgType", "FullBath",
    "BedroomAbvGr", "KitchenQual",
]


def build_preprocessor(X) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales numerics and one-hot encodes categoricals.

    Args:
        X: Feature DataFrame (used to detect column types).

    Returns:
        Unfitted ColumnTransformer.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    return ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
    ], remainder="drop")


def build_pipelines(X) -> dict:
    """
    Build sklearn Pipelines for all four models.

    Each pipeline: ColumnTransformer (scale + encode) → model.

    Args:
        X: Feature DataFrame (to detect column types for preprocessor).

    Returns:
        Dict mapping model name to Pipeline.
    """
    prep = lambda: build_preprocessor(X)
    return {
        "Linear Regression": Pipeline([
            ("prep", prep()),
            ("model", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("prep", prep()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("prep", prep()),
            ("model", XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1, verbosity=0,
            )),
        ]),
    }


def train_all(X_train, y_train_log) -> tuple:
    """
    Train all four model pipelines with 5-fold cross-validation.

    Args:
        X_train:     Training features (preprocessed DataFrame).
        y_train_log: Log-transformed training target (np.log1p applied).

    Returns:
        Tuple of (trained pipelines dict, cv scores dict)
    """
    pipelines = build_pipelines(X_train)
    cv_scores = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        scores = cross_val_score(
            pipeline, X_train, y_train_log,
            cv=5, scoring="neg_root_mean_squared_error"
        )
        cv_rmse = -scores
        cv_scores[name] = {"mean": cv_rmse.mean(), "std": cv_rmse.std()}
        print(f"  CV log-RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
        pipeline.fit(X_train, y_train_log)

    return pipelines, cv_scores


def save_model(pipeline: Pipeline, path: str = None):
    """
    Serialize the best trained pipeline to disk with joblib.

    Args:
        pipeline: Fitted sklearn Pipeline.
        path:     Optional save path override.
    """
    save_path = Path(path) if path else MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"\nModel saved to {save_path}")


def load_model(path: str = None) -> Pipeline:
    """
    Load a serialized pipeline from disk.

    Args:
        path: Optional path override.

    Returns:
        Fitted sklearn Pipeline.

    Raises:
        FileNotFoundError: If no model has been saved yet.
    """
    load_path = Path(path) if path else MODEL_PATH
    if not load_path.exists():
        raise FileNotFoundError(
            f"No model found at {load_path}. Run train.py first to train and save a model."
        )
    return joblib.load(load_path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_train, remove_outliers, split_features_target
    from features import full_preprocess
    from evaluate import evaluate_all, print_comparison

    # Load & clean
    train_df = load_train()
    train_df = remove_outliers(train_df)
    X, y = split_features_target(train_df)

    # Preprocess
    X_proc, medians = full_preprocess(X)
    y_log = np.log1p(y)

    # Train/test split (80/20 stratified on price quartile)
    
    y_q = pd.qcut(y, q=4, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y_log, test_size=0.2, random_state=42, stratify=y_q
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    # Train all models
    trained, cv_scores = train_all(X_train, y_train)

    # Evaluate and compare
    results = evaluate_all(trained, X_test, y_test)
    print_comparison(results, cv_scores)

    # Save best model by test log-RMSE
    best_name = min(results, key=lambda k: results[k]["log_rmse"])
    print(f"\nBest model: {best_name} (log-RMSE = {results[best_name]['log_rmse']:.4f})")
    save_model(trained[best_name])
