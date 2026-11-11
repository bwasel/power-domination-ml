import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FIGURE 9
# Repeated graph-level validation of Model B
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "repeated_validation_model_B.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df.to_string(index=False))


# ============================================================
# Extract R2 values
# ============================================================

baseline = df["BaselineR2"].values
structural = df["StructuralR2"].values

baseline_mean = baseline.mean()
structural_mean = structural.mean()

# Sample standard deviation
baseline_std = baseline.std(ddof=1)
structural_std = structural.std(ddof=1)

improvement = structural - baseline

improvement_mean = improvement.mean()
improvement_std = improvement.std(ddof=1)


# ============================================================
# Create paired validation plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.4)
)

x_baseline = 0
x_structural = 1


# ------------------------------------------------------------
# Connect the same split across the two models
# ------------------------------------------------------------

for b, s in zip(baseline, structural):

    ax.plot(
        [x_baseline, x_structural],
        [b, s],
        color="0.65",
        linewidth=1.0,
        alpha=0.55,
        zorder=1
    )


# ------------------------------------------------------------
# Individual split values
# ------------------------------------------------------------

ax.scatter(
    np.full(len(baseline), x_baseline),
    baseline,
    s=38,
    label="Baseline split",
    zorder=2
)

ax.scatter(
    np.full(len(structural), x_structural),
    structural,
    s=38,
    label="Structural split",
    zorder=2
)


# ------------------------------------------------------------
# Mean ± sample standard deviation
# ------------------------------------------------------------

ax.errorbar(
    [x_baseline, x_structural],
    [baseline_mean, structural_mean],
    yerr=[baseline_std, structural_std],
    fmt="D",
    markersize=7,
    capsize=5,
    linewidth=1.5,
    label="Mean ± SD",
    zorder=4
)


# ============================================================
# Mean-value labels
# ============================================================

ax.text(
    x_baseline,
    baseline_mean - 0.018,
    f"{baseline_mean:.3f}",
    ha="center",
    va="top",
    fontsize=10
)

ax.text(
    x_structural,
    structural_mean + 0.018,
    f"{structural_mean:.3f}",
    ha="center",
    va="bottom",
    fontsize=10
)


# ============================================================
# Formatting
# ============================================================

ax.set_xticks(
    [x_baseline, x_structural]
)

ax.set_xticklabels(
    ["Baseline model", "Structural model"],
    fontsize=11
)

ax.set_ylabel(
    r"Test $R^2$",
    fontsize=12
)

ax.set_xlim(
    -0.45,
    1.45
)

ax.set_ylim(
    0.64,
    0.86
)

ax.tick_params(
    axis="y",
    labelsize=10
)

ax.grid(
    axis="y",
    alpha=0.20
)

ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="lower right"
)

# No internal title.
# The manuscript caption explains the figure.

fig.tight_layout()


# ============================================================
# Save Figure 9
# ============================================================

fig.savefig(
    "Figure_9_repeated_validation.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_9_repeated_validation.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 9")
print("========")

print(
    f"Baseline R2: "
    f"{baseline_mean:.4f} ± {baseline_std:.4f}"
)

print(
    f"Structural R2: "
    f"{structural_mean:.4f} ± {structural_std:.4f}"
)

print(
    f"Mean R2 improvement: "
    f"{improvement_mean:.4f} ± {improvement_std:.4f}"
)

print(
    f"Structural model better in "
    f"{np.sum(structural > baseline)}/{len(df)} splits"
)

print("\nFiles saved:")
print("Figure_9_repeated_validation.png")
print("Figure_9_repeated_validation.pdf")

print("\nFIGURE 9 PASSED.")