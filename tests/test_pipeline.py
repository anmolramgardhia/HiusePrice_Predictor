"""
test_pipeline.py — Unit and integration tests
House Price Prediction Project | Ames Housing Dataset

Run with:
    pytest tests/ -v
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    impute_missing, encode_ordinals, engineer_features,
    NONE_COLS, ZERO_COLS, ORDINAL_MAP
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking a handful of Ames Housing rows."""
    return pd.DataFrame({
        "GrLivArea":    [1500, 2000, 800],
        "OverallQual":  [7, 8, 5],
        "TotalBsmtSF":  [800.0, np.nan, 0.0],
        "GarageCars":   [2.0, np.nan, 0.0],
        "GarageArea":   [500.0, np.nan, 0.0],
        "YearBuilt":    [1990, 2005, 1960],
        "YrSold":       [2008, 2009, 2007],
        "YearRemodAdd": [2000, 2005, 1960],
        "FullBath":     [2, 2, 1],
        "HalfBath":     [1, 0, 0],
        "BsmtFullBath": [0, 1, 0],
        "BsmtHalfBath": [0, 0, 0],
        "PoolArea":     [0, 200, 0],
        "OpenPorchSF":  [40, 0, 20],
        "EnclosedPorch":[0, 0, 0],
        "3SsnPorch":    [0, 0, 0],
        "ScreenPorch":  [0, 0, 0],
        "LotFrontage":  [70.0, np.nan, 55.0],
        "PoolQC":       [np.nan, "Gd", np.nan],
        "GarageType":   [np.nan, "Attchd", "Detchd"],
        "KitchenQual":  ["Gd", "Ex", "TA"],
        "ExterQual":    ["TA", "Gd", "Fa"],
        "Neighborhood": ["NAmes", "CollgCr", "OldTown"],
        "BldgType":     ["1Fam", "1Fam", "Duplex"],
        "1stFlrSF":     [800, 1000, 800],
        "2ndFlrSF":     [700, 1000, 0],
    })


# ── Imputation tests ──────────────────────────────────────────────────────────

class TestImputation:
    def test_none_cols_filled(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        for col in ["PoolQC", "GarageType"]:
            if col in df.columns:
                assert df[col].isnull().sum() == 0, f"{col} still has NaNs"

    def test_none_cols_value(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        assert df["PoolQC"].iloc[0] == "None"
        assert df["GarageType"].iloc[0] == "None"

    def test_zero_cols_filled(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        for col in ["GarageCars", "GarageArea", "TotalBsmtSF"]:
            if col in df.columns:
                assert df[col].isnull().sum() == 0

    def test_zero_cols_value(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        assert df["GarageCars"].iloc[1] == 0.0
        assert df["GarageArea"].iloc[1] == 0.0

    def test_median_cols_filled(self, sample_df):
        df, medians = impute_missing(sample_df.copy())
        assert df["LotFrontage"].isnull().sum() == 0

    def test_medians_returned(self, sample_df):
        _, medians = impute_missing(sample_df.copy())
        assert "LotFrontage" in medians
        assert isinstance(medians["LotFrontage"], float)

    def test_no_remaining_nulls(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        assert df.isnull().sum().sum() == 0

    def test_test_uses_train_medians(self, sample_df):
        _, train_medians = impute_missing(sample_df.copy())
        test_df = sample_df.copy()
        test_df["LotFrontage"] = np.nan
        df_test, _ = impute_missing(test_df, medians=train_medians)
        assert df_test["LotFrontage"].iloc[1] == train_medians["LotFrontage"]


# ── Ordinal encoding tests ────────────────────────────────────────────────────

class TestOrdinalEncoding:
    def test_kitchen_qual_encoded(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = encode_ordinals(df)
        assert df["KitchenQual"].dtype in [np.int32, np.int64, int]

    def test_correct_mapping(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = encode_ordinals(df)
        assert df["KitchenQual"].iloc[0] == ORDINAL_MAP["Gd"]   # 4
        assert df["KitchenQual"].iloc[1] == ORDINAL_MAP["Ex"]   # 5
        assert df["KitchenQual"].iloc[2] == ORDINAL_MAP["TA"]   # 3

    def test_exterqual_encoded(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = encode_ordinals(df)
        assert df["ExterQual"].iloc[2] == ORDINAL_MAP["Fa"]     # 2


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_total_sf_created(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        assert "TotalSF" in df.columns

    def test_total_sf_value(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        expected = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        pd.testing.assert_series_equal(df["TotalSF"], expected, check_names=False)

    def test_house_age_non_negative(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        assert (df["HouseAge"] >= 0).all()

    def test_has_pool_binary(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        assert set(df["HasPool"].unique()).issubset({0, 1})
        assert df["HasPool"].iloc[1] == 1   # PoolArea=200
        assert df["HasPool"].iloc[0] == 0   # PoolArea=0

    def test_total_baths_formula(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        expected_0 = 2 + 0.5 * 1 + 0 + 0   # row 0
        assert df["TotalBaths"].iloc[0] == pytest.approx(expected_0)

    def test_all_six_features_created(self, sample_df):
        df, _ = impute_missing(sample_df.copy())
        df = engineer_features(df)
        for feat in ["TotalSF", "HouseAge", "RemodAge", "TotalBaths", "HasPool", "TotalPorchSF"]:
            assert feat in df.columns, f"Missing engineered feature: {feat}"


# ── Integration test (requires trained model) ─────────────────────────────────

class TestIntegration:
    @pytest.fixture(scope="class")
    def model(self):
        try:
            from train import load_model
            return load_model()
        except FileNotFoundError:
            pytest.skip("Trained model not found. Run train.py first.")

    def test_model_predicts(self, model, sample_df):
        """Model should predict without errors on a sample row."""
        preds = model.predict(sample_df.head(1))
        assert len(preds) == 1

    def test_prediction_in_log_range(self, model, sample_df):
        """Log-transformed prediction should be in reasonable range (log of ~$50k–$800k)."""
        preds = model.predict(sample_df.head(1))
        assert 10.8 < preds[0] < 13.5, f"Prediction {preds[0]} out of expected log-price range"

    def test_batch_prediction(self, model, sample_df):
        preds = model.predict(sample_df)
        assert len(preds) == len(sample_df)
        assert all(np.isfinite(preds))
