from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FIGURE 2
# PDS success rate by topology, strategy,
# and initial-set fraction
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)


# ============================================================
# Helper functions
# ============================================================

def success_rates(column, order):

    rates = []

    for category in order:

        subset = df[df[column] == category]

        rate = 100.0 * subset["IsPDS"].mean()

        rates.append(rate)

    return rates


def add_labels(ax, bars, values):

    for bar, value in zip(bars, values):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10
        )


# ============================================================
# Create Figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(12.0, 4.2),
    sharey=True
)


# ============================================================
# (a) Network topology
# ============================================================

topology_order = [
    "ER",
    "BA",
    "WS"
]

topology_rates = success_rates(
    "Topology",
    topology_order
)

bars = axes[0].bar(
    topology_order,
    topology_rates,
    width=0.62
)

add_labels(
    axes[0],
    bars,
    topology_rates
)

axes[0].set_xlabel(
    "Network topology",
    fontsize=11
)

axes[0].set_ylabel(
    "PDS success rate (%)",
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


# ============================================================
# (b) Selection strategy
# ============================================================

strategy_order = [
    "random",
    "high-degree",
    "distributed"
]

strategy_labels = [
    "Random",
    "High-degree",
    "Distributed"
]

strategy_rates = success_rates(
    "Strategy",
    strategy_order
)

bars = axes[1].bar(
    strategy_labels,
    strategy_rates,
    width=0.62
)

add_labels(
    axes[1],
    bars,
    strategy_rates
)

axes[1].set_xlabel(
    "Selection strategy",
    fontsize=11
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

axes[1].tick_params(
    axis="x",
    labelrotation=0
)


# ============================================================
# (c) Initial-set fraction
# ============================================================

fraction_order = [
    0.05,
    0.10,
    0.15
]

fraction_labels = [
    "5%",
    "10%",
    "15%"
]

fraction_rates = success_rates(
     "RequestedFraction",
    fraction_order
)

bars = axes[2].bar(
    fraction_labels,
    fraction_rates,
    width=0.62
)

add_labels(
    axes[2],
    bars,
    fraction_rates
)

axes[2].set_xlabel(
    "Prescribed candidate-set fraction",
    fontsize=11
)

axes[2].text(
    0.03,
    0.96,
    "(c)",
    transform=axes[2].transAxes,
    fontsize=12,
    fontweight="bold",
    va="top"
)


# ============================================================
# Shared formatting
# ============================================================

for ax in axes:

    ax.set_ylim(
        0,
        110
    )

    ax.set_yticks(
        [0, 20, 40, 60, 80, 100]
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


fig.tight_layout(
    w_pad=2.0
)


# ============================================================
# Save Figure 2
# ============================================================

fig.savefig(
    "Figure_2_PDS_success_rate.png",
    dpi=600,
    bbox_inches="tight"
)

fig.savefig(
    "Figure_2_PDS_success_rate.pdf",
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Verification
# ============================================================

print("\nFIGURE 2")
print("========")

print("\nTopology:")
for label, value in zip(
    topology_order,
    topology_rates
):
    print(f"{label}: {value:.2f}%")

print("\nStrategy:")
for label, value in zip(
    strategy_labels,
    strategy_rates
):
    print(f"{label}: {value:.2f}%")

print("\nPrescribed candidate-set fraction:")
for label, value in zip(
    fraction_labels,
    fraction_rates
):
    print(f"{label}: {value:.2f}%")

print("\nFiles saved:")
print("Figure_2_PDS_success_rate.png")
print("Figure_2_PDS_success_rate.pdf")

print("\nFIGURE 2 PASSED.")