import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 8
# Model B ablation study
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "model_B_ablation_results.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df.to_string(index=False))


# ============================================================
# Desired conceptual order
# ============================================================

model_order = [
    "C0_only",
    "Baseline",
    "GraphStructure",
    "SetStructure",
    "Full_without_C0",
    "Full_Structural"
]

plot_df = (
    df.set_index("Model")
      .loc[model_order]
      .reset_index()
)


# ============================================================
# Publication labels
# ============================================================

label_map = {
    "C0_only": r"$C_0$ only",
    "Baseline": "Baseline",
    "GraphStructure": "Graph structure",
    "SetStructure": "Candidate-set focused",
    "Full_without_C0": r"Full without $C_0$",
    "Full_Structural": "Full structural"
}

plot_df["DisplayModel"] = (
    plot_df["Model"]
    .map(label_map)
)


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 5.4)
)

bars = ax.barh(
    plot_df["DisplayModel"],
    plot_df["R2"],
    height=0.62
)


# ============================================================
# Value labels
# ============================================================

for bar, value in zip(
    bars,
    plot_df["R2"]
):

    ax.text(
        value + 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        ha="left",
        fontsize=10
    )


# ============================================================
# Formatting
# ============================================================

# Put C0-only at the top and full model at the bottom
ax.invert_yaxis()

ax.set_xlabel(
    r"Test $R^2$",
    fontsize=12
)

ax.set_ylabel("")

ax.set_xlim(
    0,
    0.95
)

ax.set_xticks(
    [0.0, 0.2, 0.4, 0.6, 0.8]
)

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
# The manuscript caption will describe the ablation settings.

fig.tight_layout()


# ============================================================
# Save Figure 8
# ============================================================

fig.savefig(
    "Figure_8_Model_B_ablation.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_8_Model_B_ablation.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 8")
print("========")

for _, row in plot_df.iterrows():

    print(
        f"{row['Model']}: "
        f"features={int(row['NumFeaturesAfterEncoding'])}, "
        f"MAE={row['MAE']:.4f}, "
        f"RMSE={row['RMSE']:.4f}, "
        f"R2={row['R2']:.4f}, "
        f"gap from full={row['R2GapFromFull']:.4f}"
    )

print("\nFiles saved:")
print("Figure_8_Model_B_ablation.png")
print("Figure_8_Model_B_ablation.pdf")

print("\nFIGURE 8 PASSED.")