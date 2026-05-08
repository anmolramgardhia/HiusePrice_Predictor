"""
evaluate.py — Model evaluation, metrics, and SHAP analysis module
House Price Prediction Project | Ames Housing Dataset
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(pipeline, X_test, y_test_log) -> dict:
    """
    Evaluate a trained pipeline on the held-out test set.

    Predictions are made in log-space then inverted with expm1()
    for dollar-scale MAE and R² reporting.

    Args:
        pipeline:    Fitted sklearn Pipeline.
        X_test:      Test features (preprocessed DataFrame).
        y_test_log:  Log-transformed true target values.

    Returns:
        Dict containing log_rmse, rmse_dollars, mae_dollars, r2.
    """
    y_pred_log = pipeline.predict(X_test)

    log_rmse = np.sqrt(mean_squared_error(y_test_log, y_pred_log))

    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test_log)

    rmse_dollars = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
    mae_dollars  = mean_absolute_error(y_test_dollars, y_pred_dollars)
    r2           = r2_score(y_test_dollars, y_pred_dollars)

    return {
        "log_rmse":     round(log_rmse, 5),
        "rmse_dollars": round(rmse_dollars, 2),
        "mae_dollars":  round(mae_dollars, 2),
        "r2":           round(r2, 4),
    }


def evaluate_all(pipelines: dict, X_test, y_test_log) -> dict:
    """
    Evaluate all trained pipelines and return a results dict.

    Args:
        pipelines:   Dict of {model_name: fitted Pipeline}.
        X_test:      Test feature DataFrame.
        y_test_log:  Log-transformed true targets.

    Returns:
        Dict of {model_name: metrics_dict}.
    """
    results = {}
    for name, pipeline in pipelines.items():
        results[name] = evaluate_model(pipeline, X_test, y_test_log)
    return results


def print_comparison(results: dict, cv_scores: dict = None):
    """
    Print a formatted side-by-side comparison of all model metrics.

    Args:
        results:   Dict from evaluate_all().
        cv_scores: Optional dict of cross-validation scores from train_all().
    """
    print(f"\n{'='*75}")
    print(f"{'Model':<22} {'Log-RMSE':>10} {'RMSE ($)':>12} {'MAE ($)':>12} {'R²':>8}")
    print(f"{'-'*75}")
    for name, m in results.items():
        cv_str = ""
        if cv_scores and name in cv_scores:
            cv_str = f"  [CV: {cv_scores[name]['mean']:.4f} ± {cv_scores[name]['std']:.4f}]"
        print(
            f"{name:<22} {m['log_rmse']:>10.5f} {m['rmse_dollars']:>12,.0f}"
            f" {m['mae_dollars']:>12,.0f} {m['r2']:>8.4f}{cv_str}"
        )
    print(f"{'='*75}")


def run_shap_analysis(pipeline, X_sample, feature_names: list = None):
    """
    Compute SHAP values for the best model and generate plots.

    Requires: pip install shap

    Args:
        pipeline:      Fitted sklearn Pipeline (best model).
        X_sample:      Sample of test features to explain (DataFrame).
        feature_names: Optional list of feature names for display.

    Returns:
        SHAP explainer object (for further use in Streamlit app).
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    model_step = pipeline.named_steps["model"]
    prep_step = pipeline.named_steps["prep"]

    X_transformed = prep_step.transform(X_sample)

    model_name = type(model_step).__name__
    if "XGB" in model_name or "Forest" in model_name:
        explainer = shap.TreeExplainer(model_step)
        shap_values = explainer.shap_values(X_transformed)
    else:
        explainer = shap.LinearExplainer(model_step, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    print("\nGenerating SHAP summary plot...")
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    
    fig_path = Path(__file__).parent.parent / "reports" / "figures" / "shap_summary.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"SHAP summary plot saved to {fig_path}")
    plt.show()

    return explainer


def generate_submission(pipeline, X_test_raw, test_ids, medians: dict, output_path: str = "submission.csv"):
    """
    Generate a Kaggle-format submission CSV from test.csv predictions.

    Args:
        pipeline:     Fitted best model pipeline.
        X_test_raw:   Raw test features (before preprocessing).
        test_ids:     Id column from test.csv.
        medians:      Training medians dict from features.full_preprocess().
        output_path:  Path to write submission.csv.
    """
    import pandas as pd
    from features import full_preprocess

    X_proc, _ = full_preprocess(X_test_raw, medians)
    y_pred_log = pipeline.predict(X_proc)
    y_pred = np.expm1(y_pred_log)

    submission = pd.DataFrame({"Id": test_ids, "SalePrice": y_pred})
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} — {len(submission)} rows")
    return submission


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data import load_train, split_features_target, remove_outliers
    from features import full_preprocess
    from train import load_model
    import pandas as pd
    
    # Load data for evaluation
    train_df = load_train()
    train_df = remove_outliers(train_df)
    X, y = split_features_target(train_df)
    
    X_proc, medians = full_preprocess(X)
    y_log = np.log1p(y)

    print("Loading best model...")
    pipeline = load_model()

    print("Generating predictions on full training set...")
    metrics = evaluate_model(pipeline, X_proc, y_log)

    print("\nOverall Model Performance:")
    print(f"  Log-RMSE: {metrics['log_rmse']:.5f}")
    print(f"  RMSE:     ${metrics['rmse_dollars']:,.0f}")
    print(f"  MAE:      ${metrics['mae_dollars']:,.0f}")
    print(f"  R²:       {metrics['r2']:.4f}")

    print("\nRunning SHAP Analysis on a sample...")
    run_shap_analysis(pipeline, X_proc.sample(100, random_state=42))
