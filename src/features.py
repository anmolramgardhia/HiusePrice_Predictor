"""
features.py — Missing value imputation and feature engineering module
House Price Prediction Project | Ames Housing Dataset

Three-strategy imputation (as per De Cock's data dictionary):
  1. Meaningful NaN  → fill with "None"  (absence is informative)
  2. True missing    → fill with median   (data not collected)
  3. Linked numeric  → fill with 0        (feature tied to absent structure)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ── Columns by imputation strategy ──────────────────────────────────────────

# Strategy 1: NaN means "feature doesn't exist for this house"
NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "MasVnrType",
]

# Strategy 2: NaN is genuinely missing — impute with median
MEDIAN_COLS = ["LotFrontage", "MasVnrArea", "GarageYrBlt"]

# Strategy 3: NaN because no garage/basement → numeric should be 0
ZERO_COLS = [
    "GarageArea", "GarageCars",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath",
]

# Ordinal quality/condition mappings (Po → Ex = 1 → 5)
ORDINAL_MAP = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}
ORDINAL_COLS = [
    "ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
    "HeatingQC", "KitchenQual", "FireplaceQu",
    "GarageQual", "GarageCond", "PoolQC",
]


def impute_missing(df: pd.DataFrame, medians: dict = None) -> tuple:
    """
    Apply all three imputation strategies to a DataFrame.

    Args:
        df:      Feature DataFrame (train or test).
        medians: Pre-computed medians from training set. If None, computes from df.
                 Always pass training medians when transforming test data.

    Returns:
        Tuple of (imputed DataFrame, medians dict)
    """
    df = df.copy()

    # Strategy 1 — meaningful NaN → "None"
    for col in NONE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Strategy 2 — true missing → median (fit on train, apply to test)
    if medians is None:
        medians = {col: df[col].median() for col in MEDIAN_COLS if col in df.columns}
    for col, med in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(med)

    # Strategy 3 — linked numeric → 0
    for col in ZERO_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Catch-all for remaining categoricals and numerics
    cat_cols = df.select_dtypes(include="object").columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[cat_cols] = df[cat_cols].fillna("Missing")
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    remaining = df.isnull().sum().sum()
    print(f"Missing values after imputation: {remaining}")
    return df, medians


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert ordinal quality/condition text columns to integers.

    Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0

    Args:
        df: DataFrame after imputation.

    Returns:
        DataFrame with ordinal columns replaced by integers.
    """
    df = df.copy()
    for col in ORDINAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(ORDINAL_MAP).fillna(0).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 6 derived features as specified in the PRD (Section 3).

    Engineered features:
        TotalSF      — total interior square footage
        HouseAge     — age at time of sale
        RemodAge     — years since last remodel at time of sale
        TotalBaths   — weighted bathroom count
        HasPool      — binary pool indicator
        TotalPorchSF — combined porch/outdoor area

    Args:
        df: DataFrame after imputation and ordinal encoding.

    Returns:
        DataFrame with 6 additional columns appended.
    """
    df = df.copy()
    yr_sold = df.get("YrSold", pd.Series([2010] * len(df)))

    df["TotalSF"] = (
        df.get("TotalBsmtSF", 0) +
        df.get("1stFlrSF", 0) +
        df.get("2ndFlrSF", 0)
    )
    df["HouseAge"] = yr_sold - df.get("YearBuilt", yr_sold)
    df["RemodAge"] = yr_sold - df.get("YearRemodAdd", yr_sold)
    df["TotalBaths"] = (
        df.get("FullBath", 0) +
        0.5 * df.get("HalfBath", 0) +
        df.get("BsmtFullBath", 0) +
        0.5 * df.get("BsmtHalfBath", 0)
    )
    df["HasPool"] = (df["Pool Area"] > 0).astype(int)
    df["TotalPorchSF"] = (
        df.get("OpenPorchSF", 0) +
        df.get("EnclosedPorch", 0) +
        df.get("3SsnPorch", 0) +
        df.get("ScreenPorch", 0)
    )

    print(f"Engineered features added: TotalSF, HouseAge, RemodAge, TotalBaths, HasPool, TotalPorchSF")
    return df


def full_preprocess(df: pd.DataFrame, medians: dict = None) -> tuple:
    """
    Run the complete preprocessing pipeline: impute → encode → engineer.

    Args:
        df:      Raw feature DataFrame.
        medians: Training medians for strategy-2 imputation.

    Returns:
        Tuple of (processed DataFrame, medians dict)
    """
    df, medians = impute_missing(df, medians)
    df = encode_ordinals(df)
    df = engineer_features(df)
    return df, medians


if __name__ == "__main__":
    from data import load_train, remove_outliers, split_features_target
    import numpy as np

    train = load_train()
    train = remove_outliers(train)
    X, y = split_features_target(train)
    X_proc, medians = full_preprocess(X)
    print(f"\nProcessed shape: {X_proc.shape}")
    print(f"Engineered cols: {[c for c in X_proc.columns if c in ['TotalSF','HouseAge','RemodAge','TotalBaths','HasPool','TotalPorchSF']]}")
    print(f"log(SalePrice) mean: {np.log1p(y).mean():.4f}")
