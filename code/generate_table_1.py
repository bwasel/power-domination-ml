from pathlib import Path
import pandas as pd

# ============================================================
# TABLE 1
# Experimental design and dataset composition
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
OUTPUT_FILE = "Table_1_experimental_design.tex"

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Basic validation
# ============================================================

required_columns = [
    "GraphID",
    "Topology",
    "N",
    "Strategy",
    "RequestedFraction",
    "IsPDS",
    "TargetAvgDegree"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Construct unique graph identifier
# GraphID may repeat across topology and N
# ============================================================

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_"
    + df["N"].astype(str)
    + "_"
    + df["GraphID"].astype(str)
)


# ============================================================
# Extract experimental design
# ============================================================

topologies = ["ER", "BA", "WS"]

network_sizes = sorted(
    df["N"].unique()
)

target_degrees = sorted(
    df["TargetAvgDegree"].unique()
)

fractions = sorted(
    df["RequestedFraction"].unique()
)

strategies = sorted(
    df["Strategy"].unique()
)

total_networks = df["GraphUID"].nunique()

total_configurations = len(df)

successful = int(
    (df["IsPDS"] == 1).sum()
)

failed = int(
    (df["IsPDS"] == 0).sum()
)

success_rate = (
    100.0 * successful / total_configurations
)


# ============================================================
# Networks per topology-size combination
# ============================================================

network_counts = (
    df[
        ["GraphUID", "Topology", "N"]
    ]
    .drop_duplicates()
    .groupby(
        ["Topology", "N"]
    )
    .size()
)

counts_unique = sorted(
    network_counts.unique()
)

if len(counts_unique) == 1:
    networks_per_combination = counts_unique[0]
else:
    networks_per_combination = "Variable"


# ============================================================
# Candidate configurations per network
# ============================================================

configs_per_network = (
    df.groupby("GraphUID")
    .size()
)

config_counts_unique = sorted(
    configs_per_network.unique()
)

if len(config_counts_unique) == 1:
    configs_per_network_value = config_counts_unique[0]
else:
    configs_per_network_value = "Variable"


# ============================================================
# Format values
# ============================================================

topology_text = ", ".join(
    topologies
)

size_text = ", ".join(
    str(int(x))
    for x in network_sizes
)

degree_text = ", ".join(
    f"{x:g}"
    for x in target_degrees
)

fraction_text = ", ".join(
    f"{100*x:.0f}\\%"
    for x in fractions
)

strategy_display = {
    "random": "Random",
    "high-degree": "High-degree",
    "distributed": "Distributed"
}

strategy_text = ", ".join(
    strategy_display.get(
        x,
        str(x)
    )
    for x in [
        "random",
        "high-degree",
        "distributed"
    ]
    if x in strategies
)


# ============================================================
# Verification output
# ============================================================

print("\nTABLE 1")
print("=======")

print(
    f"Topologies: {topology_text}"
)

print(
    f"Network sizes: {size_text}"
)

print(
    f"Target average degree: {degree_text}"
)

print(
    f"Initial-set fractions: "
    f"{', '.join(f'{100*x:.0f}%' for x in fractions)}"
)

print(
    f"Selection strategies: {strategy_text}"
)

print(
    f"Networks per topology-size combination: "
    f"{networks_per_combination}"
)

print(
    f"Total independent networks: "
    f"{total_networks}"
)

print(
    f"Candidate configurations per network: "
    f"{configs_per_network_value}"
)

print(
    f"Total candidate configurations: "
    f"{total_configurations}"
)

print(
    f"Successful PDS configurations: "
    f"{successful}"
)

print(
    f"Failed configurations: "
    f"{failed}"
)

print(
    f"Overall PDS success rate: "
    f"{success_rate:.2f}%"
)


# ============================================================
# Scientific consistency checks
# ============================================================

assert successful + failed == total_configurations

assert total_networks == 900, (
    f"Expected 900 networks, found {total_networks}"
)

assert total_configurations == 18900, (
    f"Expected 18,900 configurations, "
    f"found {total_configurations}"
)

assert successful == 12740, (
    f"Expected 12,740 successful cases, "
    f"found {successful}"
)

assert failed == 6160, (
    f"Expected 6,160 failed cases, "
    f"found {failed}"
)


# ============================================================
# Generate LaTeX table
# ============================================================

latex = rf"""\begin{{table}}[t]
\centering
\caption{{Experimental design and dataset composition.}}
\label{{tab:experimental_design}}
\begin{{tabular}}{{ll}}
\toprule
\textbf{{Parameter}} & \textbf{{Setting}} \\
\midrule
Network topologies
    & {topology_text} \\

Network sizes, $N$
    & {size_text} \\

Target average degree
    & {degree_text} \\

Initial-set fractions
    & {fraction_text} \\

Selection strategies
    & {strategy_text} \\

Networks per topology--size combination
    & {networks_per_combination} \\

Total independent networks
    & {total_networks:,} \\

Candidate configurations per network
    & {configs_per_network_value} \\

Total candidate configurations
    & {total_configurations:,} \\

Successful PDS configurations
    & {successful:,} \\

Failed configurations
    & {failed:,} \\

Overall PDS success rate
    & {success_rate:.2f}\% \\
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ============================================================
# Save .tex file
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write(latex)


print(
    f"\nLaTeX file saved: {OUTPUT_FILE}"
)

print("\nGenerated LaTeX:")
print("================")
print(latex)

print("TABLE 1 PASSED.")