"""
data.py — Data loading and outlier removal module
House Price Prediction Project | Ames Housing Dataset
"""

import pandas as pd
from pathlib import Path

TRAIN_PATH = Path(__file__).parent.parent / "data" / "train.csv"
TEST_PATH  = Path(__file__).parent.parent / "data" / "test.csv"


def load_train(path: str = None) -> pd.DataFrame:
    """
    Load and return the training dataset.

    Args:
        path: Optional override path to train.csv.

    Returns:
        DataFrame with 79 feature columns + SalePrice target.

    Raises:
        FileNotFoundError: If train.csv is not found in data/.
    """
    file_path = Path(path) if path else TRAIN_PATH
    if not file_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {file_path}.\n"
            "Download from: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data"
        )
    df = pd.read_csv(file_path)
    print(f"Loaded train.csv — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def load_test(path: str = None) -> pd.DataFrame:
    """
    Load and return the test dataset (no SalePrice column).

    Args:
        path: Optional override path to test.csv.

    Returns:
        DataFrame with 79 feature columns, no target.
    """
    file_path = Path(path) if path else TEST_PATH
    if not file_path.exists():
        raise FileNotFoundError(f"test.csv not found at {file_path}.")
    df = pd.read_csv(file_path)
    print(f"Loaded test.csv  — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes 2 known outliers from the training set, as documented in
    the original paper.
    """
    # Identify and filter out the 2 outliers with Gr Liv Area > 4000 sq ft
    # and SalePrice < 300,000, as they are unusual properties.
    outlier_indices = df[(df['Gr Liv Area'] > 4000) & (df['SalePrice'] < 300000)].index
    if not outlier_indices.empty:
        df = df.drop(outlier_indices)
        print(f"Removed {len(outlier_indices)} outliers based on Gr Liv Area and SalePrice")
    return df 


def split_features_target(df: pd.DataFrame) -> tuple:
    """
    Split a DataFrame into features (X) and target (y).

    Returns:
        Tuple of (X: DataFrame, y: Series)
    """
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = df["SalePrice"]
    return X, y


if __name__ == "__main__":
    train = load_train()
    train = remove_outliers(train)
    X, y = split_features_target(train)
    print(f"Features shape: {X.shape}")
    print(f"Target range: ${y.min():,.0f} — ${y.max():,.0f}")
    print(f"Missing values: {X.isnull().sum().sum()} total across {(X.isnull().sum() > 0).sum()} columns")
