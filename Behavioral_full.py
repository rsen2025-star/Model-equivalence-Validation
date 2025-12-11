"""
behavioral_equivalence_final.py
--------------------------------
Full Behavioral Equivalence Test (Random Forest vs Logistic Regression vs Real 2025)

This script performs:
1️⃣ Championship simulation (RF, LR) + Real 2025 loading
2️⃣ Rank alignment and Spearman correlation (ρ)
3️⃣ Rank deviation analysis and visualizations

Outputs:
    ranks_aligned.csv
    rank_correlation_summary.csv
    rank_shift_table.csv
    rank_alignment_plot.html
    rank_heatmap.html
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import plotly.express as px
import plotly.figure_factory as ff
import warnings
import plotly.graph_objects as go
warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype")

from f1_predictor import F1Predictor
from f1_predictor_logistic import F1Predictor as F1PredictorLogistic

# -------------------------------
# CONFIGURATION
# -------------------------------
REAL_CSV_PATH = "f1_2025_real_standings.csv"  # your real table
NAME_MAP = {
    "Andrea Kimi Antonelli": "Kimi Antonelli",
    "RB": "Racing Bulls",
}


# -------------------------------
# Utility Functions
# -------------------------------
def normalize_name(name: str) -> str:
    name = str(name).strip()
    return NAME_MAP.get(name, name)

def to_rank_table(df: pd.DataFrame, points_col: str, rank_col: str) -> pd.DataFrame:
    out = df.copy()
    out["Driver"] = out["Driver"].map(normalize_name)
    out[points_col] = pd.to_numeric(out[points_col], errors="coerce").fillna(0.0)
    out[rank_col] = out[points_col].rank(ascending=False, method="min").astype(int)
    return out[["Driver", rank_col]]


# -------------------------------
# PART 1 — Championship Simulation & Rank Alignment
# -------------------------------
def build_rank_tables():
    print("\n🏎️ Part 1 — Simulating Championships and Aligning Ranks\n")

    # --- Load Models ---
    rf = F1Predictor(data_path="f1data")
    rf.load_model("f1_model.joblib")
    lr = F1PredictorLogistic(data_path="f1data")
    lr.load_model("f1_model_logistic.joblib")

    # --- Simulate Championships ---
    print("Simulating Random Forest 2025...")
    rf_df, _, _ = rf.simulate_championship()
    print("Simulating Logistic Regression 2025...")
    lr_df, _, _ = lr.simulate_championship()

    # --- Load Real Standings ---
    real = pd.read_csv(REAL_CSV_PATH)
    for df in [rf_df, lr_df, real]:
        df["Driver"] = df["Driver"].map(normalize_name)

    # --- Convert to Rank Tables ---
    rf_rank = to_rank_table(rf_df, "Points", "Rank_RF")
    lr_rank = to_rank_table(lr_df, "Points", "Rank_LR")
    real_rank = to_rank_table(real, "Points", "Rank_REAL")

    # --- Align Drivers ---
    merged = rf_rank.merge(lr_rank, on="Driver", how="inner").merge(real_rank, on="Driver", how="inner")
    merged = merged.sort_values("Rank_REAL").reset_index(drop=True)
    merged.to_csv("ranks_aligned.csv", index=False)

    print("💾 Saved: ranks_aligned.csv\nPreview:")
    print(merged.head(10).to_string(index=False))
    return merged


# -------------------------------
# PART 2 — Spearman Rank Correlation Analysis
# -------------------------------
def compute_rank_correlations(df: pd.DataFrame):
    print("\n Part 2 — Spearman Rank Correlation Analysis\n")

    pairs = [
        ("RF vs LR", "Rank_RF", "Rank_LR"),
        ("RF vs REAL", "Rank_RF", "Rank_REAL"),
        ("LR vs REAL", "Rank_LR", "Rank_REAL")
    ]

    results = []
    for label, a, b in pairs:
        rho, p = spearmanr(df[a], df[b])
        results.append({"Comparison": label, "Spearman_Rho": rho, "P_Value": p})
        print(f"{label:12s} → ρ = {rho:6.4f},  p = {p:8.5e}")

    summary = pd.DataFrame(results)
    summary.to_csv("rank_correlation_summary.csv", index=False)
    print("\n Saved: rank_correlation_summary.csv")

   

# --- Improved color-coded bar plot ---
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # distinct blue, orange, green

    fig = go.Figure()

    for i, row in enumerate(summary.itertuples()):
        fig.add_trace(go.Bar(
            x=[row.Comparison],
            y=[row.Spearman_Rho],
            text=f"{row.Spearman_Rho:.3f}",
            textposition="outside",
            marker_color=colors[i % len(colors)],
            name=row.Comparison
        ))

    fig.update_layout(
        title="Behavioral Equivalence — Spearman Rank Correlation",
        yaxis=dict(title="Spearman ρ", range=[0, 1.05]),
        xaxis=dict(title="Comparison"),
        showlegend=True,
        height=500
    )

    fig.write_html("rank_correlation_plot.html")
    print(" Saved: rank_correlation_plot.html (with distinct colors)")

    return summary


# -------------------------------
# PART 3 — Rank Deviation & Visualization
# -------------------------------
def visualize_rank_behavior(df: pd.DataFrame):
    print("\n Part 3 — Rank Deviation and Visual Analysis\n")

    df["Δ_RF"] = df["Rank_RF"] - df["Rank_REAL"]
    df["Δ_LR"] = df["Rank_LR"] - df["Rank_REAL"]
    df = df.sort_values("Rank_REAL").reset_index(drop=True)

    df[["Driver", "Rank_REAL", "Rank_RF", "Rank_LR", "Δ_RF", "Δ_LR"]].to_csv("rank_shift_table.csv", index=False)
    print(" Saved: rank_shift_table.csv")

    # --- Scatter Plot RF vs LR ---
    fig1 = px.scatter(
        df, x="Rank_RF", y="Rank_LR", text="Driver",
        title="Random Forest vs Logistic Regression — Rank Comparison",
        labels={"Rank_RF": "Random Forest Rank", "Rank_LR": "Logistic Regression Rank"},
        
    )
    fig1.update_traces(textposition="top center")
    fig1.update_layout(height=600, yaxis_autorange="reversed", xaxis_autorange="reversed")
    fig1.write_html("rank_alignment_plot.html")
    print(" Saved: rank_alignment_plot.html")

    # --- Heatmap (RF/LR/REAL) ---
    z = df[["Rank_REAL", "Rank_RF", "Rank_LR"]].corr(method="spearman").values
    fig2 = ff.create_annotated_heatmap(
        z,
        x=["Real", "RF", "LR"], y=["Real", "RF", "LR"],
        annotation_text=[[f"{v:.2f}" for v in row] for row in z],
        colorscale="Viridis", showscale=True
    )
    fig2.update_layout(title_text="Spearman Rank Correlation Heatmap", height=600)
    fig2.write_html("rank_heatmap.html")
    print(" Saved: rank_heatmap.html")

    # --- Top 5 Alignment Overview ---
    top_real = df.nsmallest(5, "Rank_REAL")[["Driver", "Rank_REAL"]]
    top_rf = df.nsmallest(5, "Rank_RF")[["Driver", "Rank_RF"]]
    top_lr = df.nsmallest(5, "Rank_LR")[["Driver", "Rank_LR"]]

    print("\n Top-5 Drivers Alignment:")
    print("\nReal Top 5:\n", top_real.to_string(index=False))
    print("\nRF Top 5:\n", top_rf.to_string(index=False))
    print("\nLR Top 5:\n", top_lr.to_string(index=False))


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def main():
    aligned_df = build_rank_tables()
    compute_rank_correlations(aligned_df)
    visualize_rank_behavior(aligned_df)
    print("\n Behavioral Equivalence Analysis Completed.\n")

if __name__ == "__main__":
    main()

