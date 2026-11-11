from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 11
# PPT distribution by network size
# Successful PDS cases only
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Successful PDS cases only
# ============================================================

success = df[
    (df["IsPDS"] == 1) &
    (df["PPT"].notna())
].copy()

sizes = [50, 100, 200]

data = [
    success.loc[
        success["N"] == n,
        "PPT"
    ].values
    for n in sizes
]


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)


# ============================================================
# Outlier properties
# ============================================================

flierprops = dict(
    marker="o",
    markersize=2.5,
    markerfacecolor="none",
    markeredgecolor="0.45",
    markeredgewidth=0.6,
    alpha=0.55
)


# ============================================================
# Boxplot
# ============================================================

ax.boxplot(
    data,
    tick_labels=[str(n) for n in sizes],
    widths=0.55,
    showfliers=True,
    patch_artist=False,
    flierprops=flierprops
)


# ============================================================
# Mean markers
# ============================================================

means = [
    success.loc[
        success["N"] == n,
        "PPT"
    ].mean()
    for n in sizes
]

ax.scatter(
    range(1, len(sizes) + 1),
    means,
    marker="D",
    s=28,
    label="Mean",
    zorder=3
)


# ============================================================
# Formatting
# ============================================================

ax.set_xlabel(
    "Network size, $N$",
    fontsize=12
)

ax.set_ylabel(
    "Power propagation time, PPT (rounds)",
    fontsize=12
)

ax.tick_params(
    axis="both",
    labelsize=11
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

fig.tight_layout()


# ============================================================
# Save Figure 11
# ============================================================

fig.savefig(
    "Figure_11_PPT_by_network_size.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_11_PPT_by_network_size.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 11")
print("=========")

for n in sizes:

    values = success.loc[
        success["N"] == n,
        "PPT"
    ]

    print(
        f"N={n}: "
        f"n={len(values)}, "
        f"mean={values.mean():.3f}, "
        f"median={values.median():.1f}, "
        f"std={values.std():.3f}, "
        f"range=[{values.min():.0f}, {values.max():.0f}]"
    )

print("\nFiles saved:")
print("Figure_11_PPT_by_network_size.png")
print("Figure_11_PPT_by_network_size.pdf")

print("\nFIGURE 11 PASSED.")