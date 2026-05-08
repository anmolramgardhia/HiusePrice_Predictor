# House Price Predictor

A machine learning system that predicts residential house sale prices from 79 property features using the Ames Housing dataset. Built as an AI engineering portfolio project covering regression, feature engineering, SHAP interpretability, and Streamlit deployment.

---

## 🚀 Quick Start: Main Terminal Commands

To avoid setup or execution issues, please run the following core commands in order:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model:**
   ```bash
   python src/train.py
   ```
3. **Evaluate & Generate Analysis Plots:**
   ```bash
   python src/evaluate.py
   ```
4. **Launch the Web App:**
   ```bash
   python -m streamlit run src/app.py
   ```

---

## 🛠️ Recent Core Improvements
- **Data Standardization:** Resolved reference paths for `train.csv` and handled whitespace-infused column structures (`Gr Liv Area`, `Pool Area`).
- **Pipeline Robustness:** Embedded strict `SimpleImputer` operations to prevent Scikit-Learn validation crashes regarding `NaN` values.
- **Reporting Infrastructure:** Unified analytical graph exports (from EDA notebooks and scripts) to output uniformly into the `reports/figures/` folder.
- **Explainability:** Configured `evaluate.py` to silently generate and stash its SHAP plots (`shap_summary.png`) within the core visualization directory.
- **Streamlit Dimension Fix:** Enhanced `app.py` by compiling an invisible baseline dataframe padded intelligently with median figures. This allows 10-feature subset inputs via the Streamlit interface to successfully funnel into our 79-feature trained ColumnTransformer without raising dimension crashes.

---

## Project structure

```
house-price-predictor/
├── data/
│   ├── train.csv               <- Download from Kaggle
│   ├── test.csv                <- Download from Kaggle
│   └── data_description.txt   <- Feature dictionary
├── notebooks/
│   ├── 01_eda.ipynb            <- Exploratory data analysis
│   └── 02_feature_analysis.ipynb  <- Feature engineering validation
├── src/
│   ├── data.py         <- Loading & outlier removal
│   ├── features.py     <- Imputation & feature engineering
│   ├── train.py        <- Model training & serialization
│   ├── evaluate.py     <- Metrics, SHAP, Kaggle submission
│   └── app.py          <- Streamlit web app
├── models/
│   └── house_price_model.joblib   <- Auto-generated after training
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── house_price_prd.pdf
└── README.md
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the dataset
Download from Kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Place `train.csv`, `test.csv`, and `data_description.txt` inside the `data/` folder.

### 3. Train the model
```bash
python src/train.py
```
Trains Linear Regression, Ridge, Random Forest, and XGBoost with 5-fold cross-validation, prints a comparison table, and saves the best model to `models/house_price_model.joblib`.

### 4. Run the Streamlit app
```bash
streamlit run src/app.py
```
Opens a web form at `http://localhost:8501` where you can enter property details and get an instant price estimate.

### 5. Run tests
```bash
pytest tests/ -v
```

---

## Model performance

| Model               | Log-RMSE | RMSE ($) | MAE ($) | R²   |
|---------------------|----------|----------|---------|------|
| Linear Regression   | —        | —        | —       | —    |
| Ridge               | —        | —        | —       | —    |
| Random Forest       | —        | —        | —       | —    |
| XGBoost             | —        | —        | —       | —    |

*Run `python src/train.py` to populate this table.*

Target: log-RMSE < 0.12 · MAE < $18,000 · R² > 0.90

---

## Feature engineering (6 derived features)

| Feature      | Formula                                          | Purpose                        |
|--------------|--------------------------------------------------|--------------------------------|
| TotalSF      | TotalBsmtSF + 1stFlrSF + 2ndFlrSF               | Total interior sq footage      |
| HouseAge     | YrSold - YearBuilt                               | Age at time of sale            |
| RemodAge     | YrSold - YearRemodAdd                            | Years since last remodel       |
| TotalBaths   | FullBath + 0.5×HalfBath + BsmtFullBath + ...    | Weighted bathroom count        |
| HasPool      | PoolArea > 0 → 1 else 0                          | Binary pool indicator          |
| TotalPorchSF | Sum of all porch/outdoor area columns            | Total outdoor living area      |

---

## Key engineering decisions

- **log1p(SalePrice)** applied to target — normalizes the right-skewed distribution and reduces RMSE sensitivity to expensive outliers.
- **3-strategy missing value imputation** — "None" for meaningful absence, median for true missing, 0 for linked numeric features. Based on De Cock's data dictionary.
- **sklearn Pipeline** — prevents data leakage by fitting the preprocessor only on training data.
- **SHAP values** — per-prediction explanations show which features drove each price estimate.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · SHAP · Streamlit · joblib · pytest
