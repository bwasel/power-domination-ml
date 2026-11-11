import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 7
# Structural distribution shift for unseen WS
# Training topologies: ER + BA
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "ws_distribution_shift.csv"

df = pd.read_csv(INPUT_FILE)

print("\nColumns found:")
print(df.columns.tolist())

print("\nData:")
print(df.to_string(index=False))


# ============================================================
# Publication labels
# ============================================================

label_map = {
    "C0": r"$C_0$",
    "AvgPathLength": "Average path length",
    "AvgClustering": "Average clustering",
    "MeanPairDistanceS": r"Mean pairwise distance in $S$",
    "MeanDegreeS": r"Mean degree in $S$"
}

df["DisplayFeature"] = (
    df["Feature"]
    .map(label_map)
    .fillna(df["Feature"])
)


# ============================================================
# Values
# ============================================================

full_range = df["WSOutsideFullRangePercent"].values
central_90 = df["WSOutsideCentral90Percent"].values

labels = df["DisplayFeature"].values

y = np.arange(len(labels))

height = 0.34


# ============================================================
# Create Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.2, 5.2)
)

bars_full = ax.barh(
    y - height / 2,
    full_range,
    height,
    label="Outside full training range"
)

bars_90 = ax.barh(
    y + height / 2,
    central_90,
    height,
    label="Outside central 90% training range"
)


# ============================================================
# Value labels
# ============================================================

def add_labels(bars, values):

    for bar, value in zip(bars, values):

        # Do not label extremely small values,
        # because they would visually overlap the origin.
        if value < 0.1:
            continue

        ax.text(
            value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=9
        )


add_labels(
    bars_full,
    full_range
)

add_labels(
    bars_90,
    central_90
)


# ============================================================
# Formatting
# ============================================================

ax.set_yticks(y)

ax.set_yticklabels(
    labels,
    fontsize=10
)

# Put C0 at the top
ax.invert_yaxis()

ax.set_xlabel(
    "WS observations outside ER+BA training support (%)",
    fontsize=12
)

ax.set_xlim(
    0,
    112
)

ax.set_xticks(
    [0, 20, 40, 60, 80, 100]
)

ax.tick_params(
    axis="x",
    labelsize=10
)

ax.grid(
    axis="x",
    alpha=0.20
)

ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    fontsize=9,
    loc="lower right"
)

# No internal title.
# The manuscript caption will explain that WS is the held-out
# topology and ER+BA constitute the training topologies.

fig.tight_layout()


# ============================================================
# Save Figure 7
# ============================================================

fig.savefig(
    "Figure_7_WS_distribution_shift.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_7_WS_distribution_shift.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 7")
print("========")

for _, row in df.iterrows():

    print(
        f"{row['Feature']}: "
        f"outside full range="
        f"{row['WSOutsideFullRangePercent']:.2f}%, "
        f"outside central 90%="
        f"{row['WSOutsideCentral90Percent']:.2f}%"
    )

print("\nFiles saved:")
print("Figure_7_WS_distribution_shift.png")
print("Figure_7_WS_distribution_shift.pdf")

print("\nFIGURE 7 PASSED.")