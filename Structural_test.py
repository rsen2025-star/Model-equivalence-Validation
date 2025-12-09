"""
structural_equivalence_zscore.py
--------------------------------
Structural Equivalence — Z-Score Normalized Feature Importance Comparison

Compares Random Forest and Logistic Regression models
based on z-score normalized feature importance values.
"""

import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Model paths
RF_MODEL_PATH = "f1_model.joblib"
LR_MODEL_PATH = "f1_model_logistic.joblib"


def zscore(series):
    """Apply Z-score normalization: (x - mean) / std."""
    s = pd.Series(series).astype(float)
    if s.std() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / s.std()


def extract_feature_importances(model_path, model_type):
    """Extracts feature importance values from saved model files."""
    model_data = joblib.load(model_path)
    feature_importance = model_data["feature_importance"]

    return pd.DataFrame({
        "Feature": feature_importance["feature"],
        f"Importance_{model_type.upper()}": feature_importance["importance"]
    })


def main():
    print("\n🏗️ Structural Equivalence — Z-Score Normalized Feature Importance\n")

    # --- Load feature importances ---
    rf_df = extract_feature_importances(RF_MODEL_PATH, "rf")
    lr_df = extract_feature_importances(LR_MODEL_PATH, "lr")

    # --- Merge ---
    merged = pd.merge(rf_df, lr_df, on="Feature", how="inner")

    # --- Z-score normalization ---
    merged["Importance_RF_Z"] = zscore(merged["Importance_RF"])
    merged["Importance_LR_Z"] = zscore(merged["Importance_LR"])

    # --- Save output CSV ---
    out_path = "structural_feature_importance_zscore.csv"
    merged.to_csv(out_path, index=False)
    print(f"💾 Saved: {out_path}\n")

    print("📊 Preview (Z-score normalized importances):")
    print(merged[["Feature", "Importance_RF_Z", "Importance_LR_Z"]].to_string(index=False))

    # --- Visualization ---
    bar_df = merged.melt(
        id_vars="Feature",
        value_vars=["Importance_RF_Z", "Importance_LR_Z"],
        var_name="Model",
        value_name="Z-Score Importance"
    )

    fig = px.bar(
        bar_df,
        x="Feature",
        y="Z-Score Importance",
        color="Model",
        barmode="group",
        title="Z-Score Normalized Feature Importance — RF vs Logistic Regression"
    )

    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="Z-Score Normalized Importance",
        xaxis_tickangle=-45,
        height=600,
        legend_title_text="Model"
    )

    fig.write_html("structural_equivalence_zscore_bar.html")
    print("💾 Saved: structural_equivalence_zscore_bar.html\n")

    print("✅ Structural equivalence (Z-score) visualization completed successfully!")


if __name__ == "__main__":
    main()
