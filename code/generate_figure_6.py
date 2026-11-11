import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 6
# Cross-topology generalization of Model B
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "cross_topology_model_B_results.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df.to_string(index=False))


# ============================================================
# Arrange test topologies
# ============================================================

test_order = [
    "WS",
    "BA",
    "ER"
]

plot_df = (
    df.set_index("TestTopology")
      .loc[test_order]
      .reset_index()
)

baseline = plot_df["BaselineR2"].values
structural = plot_df["StructuralR2"].values


# ============================================================
# X-axis labels
# ============================================================

labels = [
    "Unseen WS",
    "Unseen BA",
    "Unseen ER"
]

x = np.arange(len(labels))

width = 0.34


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.6, 5.0)
)

bars_baseline = ax.bar(
    x - width / 2,
    baseline,
    width,
    label="Baseline model"
)

bars_structural = ax.bar(
    x + width / 2,
    structural,
    width,
    label="Structural model"
)


# ============================================================
# Value labels
# ============================================================

for bars in [
    bars_baseline,
    bars_structural
]:

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# Formatting
# ============================================================

ax.set_ylabel(
    r"Test $R^2$",
    fontsize=12
)

ax.set_xlabel(
    "Held-out topology",
    fontsize=12
)

ax.set_xticks(x)

ax.set_xticklabels(
    labels,
    fontsize=11
)

ax.set_ylim(
    0,
    0.95
)

ax.set_yticks(
    np.arange(
        0,
        1.0,
        0.2
    )
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
    fontsize=10,
    loc="upper left"
)

# No internal title.
# The manuscript caption will provide the description.

fig.tight_layout()


# ============================================================
# Save Figure 6
# ============================================================

fig.savefig(
    "Figure_6_cross_topology_generalization.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_6_cross_topology_generalization.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 6")
print("========")

for _, row in plot_df.iterrows():

    print(
        f"Test {row['TestTopology']}: "
        f"Baseline R2={row['BaselineR2']:.4f}, "
        f"Structural R2={row['StructuralR2']:.4f}, "
        f"Improvement={row['R2Improvement']:.4f}"
    )

print("\nFiles saved:")
print("Figure_6_cross_topology_generalization.png")
print("Figure_6_cross_topology_generalization.pdf")

print("\nFIGURE 6 PASSED.")