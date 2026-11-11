from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Publication Figures
# Figure 1: PPT distribution across network topologies
# FINAL VERSION
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)

# PPT is meaningful only for successful PDS cases
success = df[df["IsPDS"] == 1].copy()

topologies = ["ER", "BA", "WS"]

data = [
    success.loc[
        success["Topology"] == topology,
        "PPT"
    ].values
    for topology in topologies
]

# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.2)
)

# Small outlier markers
flierprops = dict(
    marker="o",
    markersize=2.5,
    markerfacecolor="none",
    markeredgecolor="0.45",
    markeredgewidth=0.6,
    alpha=0.55
)

bp = ax.boxplot(
    data,
    tick_labels=topologies,
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
        success["Topology"] == topology,
        "PPT"
    ].mean()
    for topology in topologies
]

ax.scatter(
    range(1, len(topologies) + 1),
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
    "Network topology",
    fontsize=12
)

ax.set_ylabel(
    "Power propagation time, PPT (rounds)",
    fontsize=12
)

# No internal title:
# the figure caption in the manuscript will provide the title.

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.grid(
    axis="y",
    alpha=0.20
)

ax.legend(
    frameon=False,
    fontsize=10,
    loc="upper left"
)

fig.tight_layout()

# ============================================================
# IMPORTANT:
# Same filenames -> overwrite previous Figure 1
# ============================================================

fig.savefig(
    "Figure_1_PPT_by_topology.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_1_PPT_by_topology.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# Verification
# ============================================================

print("\nFIGURE 1 - FINAL VERSION")
print("========================")

for topology in topologies:

    x = success.loc[
        success["Topology"] == topology,
        "PPT"
    ]

    print(
        f"{topology}: "
        f"n={len(x)}, "
        f"mean={x.mean():.3f}, "
        f"median={x.median():.1f}, "
        f"std={x.std():.3f}, "
        f"range=[{x.min():.0f}, {x.max():.0f}]"
    )

print("\nPrevious Figure 1 files overwritten:")
print("Figure_1_PPT_by_topology.png")
print("Figure_1_PPT_by_topology.pdf")

print("\nFIGURE 1 FINAL PASSED.")
