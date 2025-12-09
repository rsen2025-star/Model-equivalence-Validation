"""
model_equivalence_test.py
-------------------------
Performance Equivalence - Part 2
Compares Random Forest vs Logistic Regression statistically
using McNemar's test and metric differences.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from statsmodels.stats.contingency_tables import mcnemar

from f1_predictor import F1Predictor
from f1_predictor_logistic import F1Predictor as F1PredictorLogistic


def evaluate_predictions(model1, model2):
    """
    Runs both models on the same test data (2024),
    computes metrics and McNemar's test.
    """
    # Load shared dataset
    X_train, y_train, X_val, y_val, X_test, y_test = model1.data_loader.prepare_features()

    # Predictions & probabilities
    y_pred_rf = model1.model.predict(X_test)
    y_pred_lr = model2.model.predict(X_test)
    y_prob_rf = model1.model.predict_proba(X_test)[:, 1]
    y_prob_lr = model2.model.predict_proba(X_test)[:, 1]

    # --- 1. Metrics side-by-side ---
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"],
        "RandomForest": [
            accuracy_score(y_test, y_pred_rf),
            precision_score(y_test, y_pred_rf),
            recall_score(y_test, y_pred_rf),
            f1_score(y_test, y_pred_rf),
            roc_auc_score(y_test, y_prob_rf)
        ],
        "LogisticRegression": [
            accuracy_score(y_test, y_pred_lr),
            precision_score(y_test, y_pred_lr),
            recall_score(y_test, y_pred_lr),
            f1_score(y_test, y_pred_lr),
            roc_auc_score(y_test, y_prob_lr)
        ]
    })

    # --- 2. McNemar's test ---
    both = pd.DataFrame({
        "RF_correct": (y_pred_rf == y_test).astype(int),
        "LR_correct": (y_pred_lr == y_test).astype(int)
    })

    # b = RF correct, LR wrong | c = LR correct, RF wrong
    b = np.sum((both["RF_correct"] == 1) & (both["LR_correct"] == 0))
    c = np.sum((both["RF_correct"] == 0) & (both["LR_correct"] == 1))
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)

    return metrics, result.pvalue, b, c


def main():
    print("\n🏁 Performance Equivalence Part 2 — Statistical Comparison\n")

    # --- Load both trained models ---
    rf_model = F1Predictor(data_path="f1data")
    rf_model.load_model("f1_model.joblib")

    lr_model = F1PredictorLogistic(data_path="f1data")
    lr_model.load_model("f1_model_logistic.joblib")

    # --- Evaluate ---
    metrics, p_value, b, c = evaluate_predictions(rf_model, lr_model)

    # --- Display results ---
    print("📊 Performance Metrics Comparison (Test 2024)\n")
    print(metrics.to_string(index=False, float_format="%.3f"))

    print("\n🔍 McNemar’s Test Results")
    print(f"b (RF correct / LR wrong): {b}")
    print(f"c (LR correct / RF wrong): {c}")
    print(f"p-value = {p_value:.5f}")

    # --- Interpret equivalence ---
    alpha = 0.05
    if p_value > alpha:
        print("\n✅ No significant difference (p > 0.05) → Performance equivalent.")
    else:
        print("\n❌ Significant difference (p ≤ 0.05) → Not performance-equivalent.")

    # --- Save outputs for report ---
    metrics.to_csv("performance_equivalence_metrics.csv", index=False)
    with open("mcnemar_results.json", "w") as f:
        json.dump({"b": int(b), "c": int(c), "p_value": float(p_value)}, f, indent=4)

    print("\n💾 Saved: performance_equivalence_metrics.csv & mcnemar_results.json\n")


if __name__ == "__main__":
    main()
