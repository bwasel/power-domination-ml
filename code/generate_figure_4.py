import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 4
# SHAP feature importance for Model B
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "model_B_shap_feature_importance.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df)


# ============================================================
# Identify columns automatically
# ============================================================

# The file should contain one feature-name column
# and one mean absolute SHAP-value column.

feature_col = df.columns[0]
importance_col = df.columns[1]


# ============================================================
# Sort by SHAP importance
# ============================================================

plot_df = df[
    [feature_col, importance_col]
].copy()

plot_df = plot_df.sort_values(
    importance_col,
    ascending=True
)


# ============================================================
# Improve feature labels for publication
# ============================================================

label_map = {
    "C0": r"$C_0$",
    "AvgPathLength": "Average path length",
    "AvgClustering": "Average clustering",
    "MeanPairDistanceS": r"Mean pairwise distance in $S$",
    "MeanDegreeS": r"Mean degree in $S$",
    "ActualFraction": r"Actual fraction $|S|/N$",
    "AvgDegree": "Average degree",
    "Density": "Density",
    "Diameter": "Diameter",
    "Topology_WS": "Topology: WS",
    "Strategy_high-degree": "Strategy: high-degree",
    "MinPairDistanceS": r"Minimum pairwise distance in $S$",
    "Topology_ER": "Topology: ER",
    "N": r"$N$",
    "Topology_BA": "Topology: BA",
    "Strategy_distributed": "Strategy: distributed",
    "Strategy_random": "Strategy: random"
}

plot_df["DisplayFeature"] = (
    plot_df[feature_col]
    .astype(str)
    .map(label_map)
    .fillna(plot_df[feature_col].astype(str))
)


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.6, 6.0)
)

bars = ax.barh(
    plot_df["DisplayFeature"],
    plot_df[importance_col]
)

# ============================================================
# Value labels
# ============================================================

for bar, value in zip(
    bars,
    plot_df[importance_col]
):

    ax.text(
        bar.get_width() + 0.025,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=8
    )

# ============================================================
# Formatting
# ============================================================

ax.set_xlabel(
    "Mean absolute SHAP value",
    fontsize=12
)

ax.set_ylabel("")

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.grid(
    axis="x",
    alpha=0.20
)

ax.set_axisbelow(True)

# No internal title.
# The manuscript caption will provide the figure description.

ax.set_xlim(
    0,
    plot_df[importance_col].max() * 1.13
)

fig.tight_layout()


# ============================================================
# Save Figure 4
# ============================================================

fig.savefig(
    "Figure_4_SHAP_feature_importance.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_4_SHAP_feature_importance.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 4")
print("========")

verification = df.sort_values(
    importance_col,
    ascending=False
)

for _, row in verification.iterrows():

    print(
        f"{row[feature_col]}: "
        f"{row[importance_col]:.6f}"
    )

print("\nFiles saved:")
print("Figure_4_SHAP_feature_importance.png")
print("Figure_4_SHAP_feature_importance.pdf")

print("\nFIGURE 4 PASSED.")