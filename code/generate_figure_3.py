import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 3
# Model B predictive performance
# Baseline vs structural model
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "model_B_regression_performance.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df)


# ============================================================
# Automatically identify the two models
# ============================================================

# Expected structure:
# one row for Baseline
# one row for Structural

model_col = "Model"

baseline = df[
    df[model_col].astype(str).str.lower().str.contains("baseline")
].iloc[0]

structural = df[
    df[model_col].astype(str).str.lower().str.contains("structural")
].iloc[0]


# ============================================================
# Values
# ============================================================

metrics = [
    "MAE",
    "RMSE",
    "R2"
]

metric_labels = [
    "MAE (rounds)",
    "RMSE (rounds)",
    r"$R^2$"
]

baseline_values = [
    baseline["MAE"],
    baseline["RMSE"],
    baseline["R2"]
]

structural_values = [
    structural["MAE"],
    structural["RMSE"],
    structural["R2"]
]


# ============================================================
# Create Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(9.6, 4.2)
)

x_labels = [
     "Baseline model",
    "Structural model"
]

panel_labels = [
    "(a)",
    "(b)",
    "(c)"
]


# ============================================================
# Draw panels
# ============================================================

for i, ax in enumerate(axes):

    values = [
        baseline_values[i],
        structural_values[i]
    ]

    bars = ax.bar(
        x_labels,
        values,
        width=0.58
    )

    # Value labels
    for bar, value in zip(bars, values):

        if metrics[i] == "R2":
            label = f"{value:.3f}"
        else:
            label = f"{value:.2f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() +
            max(values) * 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_ylabel(
        metric_labels[i],
        fontsize=12
    )

    ax.text(
        0.03,
        0.96,
        panel_labels[i],
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top"
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    # Give labels enough space
    ax.set_ylim(
        0,
        max(values) * 1.18
    )


# ============================================================
# Special scale for R²
# ============================================================

axes[2].set_ylim(
    0,
    1.0
)

axes[2].set_yticks(
    [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout(
    w_pad=2.0
)


# ============================================================
# Save Figure 3
# ============================================================

fig.savefig(
    "Figure_3_Model_B_performance.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_3_Model_B_performance.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 3")
print("========")

print(
    f"Baseline: "
    f"MAE={baseline['MAE']:.4f}, "
    f"RMSE={baseline['RMSE']:.4f}, "
    f"R2={baseline['R2']:.4f}"
)

print(
    f"Structural: "
    f"MAE={structural['MAE']:.4f}, "
    f"RMSE={structural['RMSE']:.4f}, "
    f"R2={structural['R2']:.4f}"
)

print("\nFiles saved:")
print("Figure_3_Model_B_performance.png")
print("Figure_3_Model_B_performance.pdf")

print("\nFIGURE 3 PASSED.")