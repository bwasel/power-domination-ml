import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 5
# SHAP dependence for the two dominant features
# ============================================================

C0_FILE = Path(__file__).resolve().parent / "shap_dependence_C0.csv"
PATH_FILE = Path(__file__).resolve().parent / "shap_dependence_AvgPathLength.csv"

c0_df = pd.read_csv(C0_FILE)
path_df = pd.read_csv(PATH_FILE)


# ============================================================
# Helper function:
# median SHAP trend within equal-frequency bins
# ============================================================

def binned_median_trend(df, x_col, n_bins=15):

    temp = df[[x_col, "SHAP"]].dropna().copy()

    # Equal-frequency bins
    temp["bin"] = pd.qcut(
        temp[x_col],
        q=n_bins,
        duplicates="drop"
    )

    trend = (
        temp
        .groupby("bin", observed=True)
        .agg(
            x=(x_col, "median"),
            shap=("SHAP", "median")
        )
        .reset_index(drop=True)
    )

    return trend


# ============================================================
# Calculate trends
# ============================================================

c0_trend = binned_median_trend(
    c0_df,
    "C0",
    n_bins=15
)

path_trend = binned_median_trend(
    path_df,
    "AvgPathLength",
    n_bins=15
)


# ============================================================
# Create Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.5, 4.5)
)


# ============================================================
# (a) C0 dependence
# ============================================================

axes[0].scatter(
    c0_df["C0"],
    c0_df["SHAP"],
    s=10,
    alpha=0.22,
    edgecolors="none"
)

axes[0].plot(
    c0_trend["x"],
    c0_trend["shap"],
    linewidth=2.0,
    label="Binned median"
)

axes[0].axhline(
    0,
    linewidth=1.0,
    linestyle="--"
)

axes[0].set_xlabel(
    r"Initial coverage, $C_0$",
    fontsize=12
)

axes[0].set_ylabel(
    r"SHAP value for $C_0$ (rounds)",
    fontsize=12
)

axes[0].text(
    0.03,
    0.96,
    "(a)",
    transform=axes[0].transAxes,
    fontsize=12,
    fontweight="bold",
    va="top"
)

axes[0].legend(
    frameon=False,
    fontsize=9,
    loc="upper right"
)


# ============================================================
# (b) Average path length dependence
# ============================================================

axes[1].scatter(
    path_df["AvgPathLength"],
    path_df["SHAP"],
    s=10,
    alpha=0.22,
    edgecolors="none"
)

axes[1].plot(
    path_trend["x"],
    path_trend["shap"],
    linewidth=2.0,
    label="Binned median"
)

axes[1].axhline(
    0,
    linewidth=1.0,
    linestyle="--"
)

axes[1].set_xlabel(
    "Average path length",
    fontsize=12
)

axes[1].set_ylabel(
    "SHAP value for average path length (rounds)",
    fontsize=12
)

axes[1].text(
    0.03,
    0.96,
    "(b)",
    transform=axes[1].transAxes,
    fontsize=12,
    fontweight="bold",
    va="top"
)

axes[1].legend(
    frameon=False,
    fontsize=9,
    loc="upper right"
)


# ============================================================
# Shared formatting
# ============================================================

for ax in axes:

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.grid(
        alpha=0.18
    )

    ax.set_axisbelow(True)


fig.tight_layout(
    w_pad=2.5
)


# ============================================================
# Save Figure 5
# ============================================================

fig.savefig(
    "Figure_5_SHAP_dependence.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_5_SHAP_dependence.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 5")
print("========")

print(
    f"C0: n={len(c0_df)}, "
    f"range=[{c0_df['C0'].min():.3f}, "
    f"{c0_df['C0'].max():.3f}], "
    f"SHAP range=[{c0_df['SHAP'].min():.3f}, "
    f"{c0_df['SHAP'].max():.3f}]"
)

print(
    f"Average path length: n={len(path_df)}, "
    f"range=[{path_df['AvgPathLength'].min():.3f}, "
    f"{path_df['AvgPathLength'].max():.3f}], "
    f"SHAP range=[{path_df['SHAP'].min():.3f}, "
    f"{path_df['SHAP'].max():.3f}]"
)

print(
    f"C0 trend bins: {len(c0_trend)}"
)

print(
    f"Average path length trend bins: "
    f"{len(path_trend)}"
)

print("\nFiles saved:")
print("Figure_5_SHAP_dependence.png")
print("Figure_5_SHAP_dependence.pdf")

print("\nFIGURE 5 PASSED.")