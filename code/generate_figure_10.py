from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 10
# PPT by initial-set fraction and selection strategy
# Successful PDS cases only
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())


# ============================================================
# Keep successful PDS cases only
# ============================================================

plot_df = df[
    (df["IsPDS"] == 1) &
    (df["PPT"].notna())
].copy()

print(f"\nSuccessful PDS observations: {len(plot_df)}")


# ============================================================
# Experimental levels
# ============================================================

fractions = [0.05, 0.10, 0.15]

strategies = [
    "random",
    "high-degree",
    "distributed"
]

strategy_labels = {
    "random": "Random",
    "high-degree": "High-degree",
    "distributed": "Distributed"
}


# ============================================================
# Create figure
# One panel for each initial-set fraction
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(10.8, 4.6),
    sharey=True
)


for i, fraction in enumerate(fractions):

    ax = axes[i]

    fraction_df = plot_df[
        plot_df["RequestedFraction"] == fraction
    ]

    data = []
    labels = []

    for strategy in strategies:

        values = fraction_df.loc[
            fraction_df["Strategy"] == strategy,
            "PPT"
        ].dropna()

        data.append(values)
        labels.append(strategy_labels[strategy])

    # --------------------------------------------------------
    # Boxplot
    # --------------------------------------------------------

    bp = ax.boxplot(
        data,
        labels=labels,
        widths=0.58,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
        meanprops=dict(
            marker="D",
            markerfacecolor="C0",
            markeredgecolor="C0",
            markersize=5
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="none",
            markeredgecolor="0.55",
            markersize=3,
            alpha=0.55
        ),
        medianprops=dict(
            linewidth=1.5
        )
    )

    for box in bp["boxes"]:
        box.set_facecolor("white")

    # --------------------------------------------------------
    # Panel label
    # --------------------------------------------------------

    ax.text(
        0.04,
        0.95,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left"
    )

    # --------------------------------------------------------
    # Fraction label
    # --------------------------------------------------------

    ax.set_xlabel(
        f"Prescribed candidate-set fraction = {int(fraction * 100)}%",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=9
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.set_axisbelow(True)


# ============================================================
# Shared y-axis label
# ============================================================

axes[0].set_ylabel(
    "Power propagation time, PPT (rounds)",
    fontsize=11
)


# ============================================================
# Mean legend
# ============================================================

mean_handle = plt.Line2D(
    [],
    [],
    marker="D",
    linestyle="None",
    markersize=6,
    label="Mean"
)

axes[2].legend(
    handles=[mean_handle],
    loc="upper right",
    frameon=False,
    fontsize=9
)


# ============================================================
# Layout
# ============================================================

fig.tight_layout()


# ============================================================
# Save Figure 10
# ============================================================

fig.savefig(
    "Figure_10_PPT_fraction_strategy.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_10_PPT_fraction_strategy.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 10")
print("=========")

for fraction in fractions:

    print(
        f"\nPrescribed candidate-set fraction = "
        f"{int(fraction * 100)}%"
    )

    for strategy in strategies:

        values = plot_df.loc[
            (plot_df["RequestedFraction"] == fraction) &
            (plot_df["Strategy"] == strategy),
            "PPT"
        ]

        print(
            f"{strategy}: "
            f"n={len(values)}, "
            f"mean={values.mean():.3f}, "
            f"median={values.median():.3f}, "
            f"min={values.min():.0f}, "
            f"max={values.max():.0f}"
        )

print("\nFiles saved:")
print("Figure_10_PPT_fraction_strategy.png")
print("Figure_10_PPT_fraction_strategy.pdf")

print("\nFIGURE 10 PASSED.")