
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from f1_predictor import F1Predictor                  # Random Forest model
from f1_predictor_logistic import F1Predictor as F1PredictorLogistic  # Logistic Regression model


def evaluate_model(model):
    """
    Evaluates a trained model on the 2024 test set.
    Returns dictionary of metrics.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = model.data_loader.prepare_features()

    # Predictions and probabilities
    preds = model.model.predict(X_test)
    probas = model.model.predict_proba(X_test)[:, 1]

    metrics = {
        "Test Accuracy": accuracy_score(y_test, preds),
        "Test Precision": precision_score(y_test, preds),
        "Test Recall": recall_score(y_test, preds),
        "Test F1 Score": f1_score(y_test, preds),
        "Test ROC AUC": roc_auc_score(y_test, probas)
    }
    return metrics


def main():
    print("\n🏎️ Starting Performance Equivalence: Random Forest vs Logistic Regression\n")

    # ---------------- Random Forest ----------------
    print("🔹 Training Random Forest Model ...")
    rf_model = F1Predictor(data_path="f1data")
    rf_metrics = rf_model.train_model()

    # ---------------- Logistic Regression ----------------
    print("\n🔸 Training Logistic Regression Model ...")
    lr_model = F1PredictorLogistic(data_path="f1data")
    lr_metrics = lr_model.train_model()

    # ---------------- Display metrics ----------------
    print("\n===== RANDOM FOREST PERFORMANCE =====")
    for k, v in rf_metrics.items():
        print(f"{k:<25}: {v:.3f}")

    print("\n===== LOGISTIC REGRESSION PERFORMANCE =====")
    for k, v in lr_metrics.items():
        print(f"{k:<25}: {v:.3f}")

    # ---------------- Save metrics ----------------
    results = {"RandomForest": rf_metrics, "LogisticRegression": lr_metrics}
    with open("model_performance_comparison.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n💾 Metrics saved to model_performance_comparison.json")
    print("\n✅ Part 1 complete: Both models evaluated on identical data.\n")
    print("Proceed to Part 2 (statistical equivalence testing).")


if __name__ == "__main__":
    main()
