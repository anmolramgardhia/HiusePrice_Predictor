# House Price Predictor - Product Requirements Document (PRD)

## 🚀 Quick Start: Main Terminal Commands

To ensure you do not face any execution issues, please run the following core commands in order:

1. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the dataset and save the model:**
   ```bash
   python src/train.py
   ```
3. **Evaluate the pipeline & generate SHAP plots:**
   ```bash
   python src/evaluate.py
   ```
4. **Launch the web application:**
   ```bash
   python -m streamlit run src/app.py
   ```

---

## 📌 Project Overview
The House Price Predictor is a machine learning inference system designed to predict residential house sale prices using 79 property features spanning the comprehensive Ames Housing dataset.

### 🛠️ Recent System Improvements
During development, the following critical changes were implemented to stabilize the codebase:
- **Data Referencing & Formatting:** Updated file locators to accurately process `train.csv`. Fixed Pandas mapping errors induced by literal spaces in dataset features (e.g. `Gr Liv Area`, `Pool Area`).
- **Advanced Preprocessing via Pipeline:** Integrated a comprehensive `SimpleImputer` array within the Scikit-Learn `ColumnTransformer` to seamlessly resolve and suppress input `NaN` matrices.
- **Centralized Output Architecture:** Designed a localized `reports/figures/` directory. Exploratory Data Analysis notebooks and standard python evaluators output artifacts homogeneously to this directory.
- **SHAP Integrations:** Improved `src/evaluate.py` to systematically store its predictive explanation grid as `shap_summary.png` without blocking the terminal session.
- **Streamlit Baseline Injection (app.py fix):** Corrected a Streamlit defect where utilizing only the top 10 features crashed the model. The web framework now establishes a background dataset baseline mapping (imputing absent pipeline values with calculated median datasets) ensuring matrix equivalency over 79 parameters.

---

## 🎯 Goals & Requirements
- **Goal:** Develop a robust, interpretable Machine Learning workflow minimizing RMSE.
- **Frameworks:** Python, Scikit-Learn, XGBoost, Pandas, Numpy, Matplotlib, Seaborn, and Streamlit.
- **Architecture Validation:** Implement an explicit source pattern (`src/`) interfacing systematically against Jupyter Notebook exploration streams.
- **Accuracy Targets:** Target test log-RMSE below 0.12 and Mean Absolute Error under $18,000.  

## 📂 Structural Layout Requirements
- Input layer accepts basic inputs cleanly bypassing 70 missing fields.
- Pipeline integrates pre-imputation, normalization scaler operations, and strict ML Regressor serialization routines via `joblib`.
- Deployment interface seamlessly connects Streamlit UX with core pipeline inference logic endpoints.