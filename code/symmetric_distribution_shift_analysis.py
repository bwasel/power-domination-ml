"""
Symmetric graph-level distribution-shift analysis for Model B
=============================================================

This script evaluates structural covariate shift symmetrically for all
three held-out-topology directions:

    ER + WS -> BA
    BA + WS -> ER
    ER + BA -> WS

The analysis is restricted to successful PDS configurations, consistent
with Model B. To avoid treating multiple candidate-set configurations
from the same graph as independent observations, each predictor is first
summarized to one value per graph by averaging over that graph's successful
configurations. Graph-level structural predictors (e.g., average path
length and average clustering) are constant within a graph, so this
aggregation leaves them unchanged.

For each target topology, graph-level target values are compared against:
    1. the full source-topology support [min, max]
    2. the central 90% source-topology support [5th, 95th percentile]

Outputs:
    symmetric_graph_level_distribution_shift.csv
    Table_symmetric_graph_level_distribution_shift.tex
"""

import numpy as np
import pandas as pd
from pathlib import Path

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

CSV_OUTPUT = "symmetric_graph_level_distribution_shift.csv"
LATEX_OUTPUT = "Table_symmetric_graph_level_distribution_shift.tex"

FEATURES = [
    "C0",
    "AvgPathLength",
    "AvgClustering",
    "MeanPairDistanceS",
    "MeanDegreeS",
]

FEATURE_LABELS = {
    "C0": r"$C_0$",
    "AvgPathLength": "Average path length",
    "AvgClustering": "Average clustering",
    "MeanPairDistanceS": r"Mean pairwise distance in $S$",
    "MeanDegreeS": r"Mean degree in $S$",
}

TARGET_ORDER = ["BA", "ER", "WS"]


# ============================================================
# LOAD SUCCESSFUL CONFIGURATIONS
# ============================================================

df = pd.read_csv(INPUT_FILE)
df = df[df["IsPDS"] == 1].copy()

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

# ============================================================
# ONE OBSERVATION PER GRAPH
# ============================================================

graph_df = (
    df.groupby(["GraphUID", "Topology"], as_index=False)[FEATURES]
      .mean()
)

print("\nSYMMETRIC GRAPH-LEVEL DISTRIBUTION-SHIFT ANALYSIS")
print("=================================================")
print("\nGraphs per topology:")
print(graph_df.groupby("Topology")["GraphUID"].nunique().to_string())


# ============================================================
# SYMMETRIC SUPPORT ANALYSIS
# ============================================================

rows = []

for target in TARGET_ORDER:

    source = graph_df[graph_df["Topology"] != target].copy()
    target_df = graph_df[graph_df["Topology"] == target].copy()

    for feature in FEATURES:

        source_min = source[feature].min()
        source_max = source[feature].max()

        source_p05 = source[feature].quantile(0.05)
        source_p95 = source[feature].quantile(0.95)

        outside_full = (
            (target_df[feature] < source_min)
            | (target_df[feature] > source_max)
        )

        outside_central90 = (
            (target_df[feature] < source_p05)
            | (target_df[feature] > source_p95)
        )

        rows.append({
            "TargetTopology": target,
            "Feature": feature,
            "TargetGraphs": len(target_df),
            "SourceGraphs": len(source),
            "OutsideFullPct": 100.0 * outside_full.mean(),
            "OutsideCentral90Pct": 100.0 * outside_central90.mean(),
            "SourceMin": source_min,
            "SourceMax": source_max,
            "SourceP05": source_p05,
            "SourceP95": source_p95,
        })

results = pd.DataFrame(rows)

results.to_csv(CSV_OUTPUT, index=False)

print("\nRESULTS")
print("-------")
print(
    results[
        [
            "TargetTopology",
            "Feature",
            "OutsideFullPct",
            "OutsideCentral90Pct",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}",
    )
)


# ============================================================
# LATEX TABLE
# ============================================================

latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\color{blue}",
    r"\caption{Symmetric graph-level distribution-shift analysis for the three held-out-topology experiments. For each target topology, predictor values were first summarized to one value per graph by averaging across successful PDS configurations from that graph. Percentages report the fraction of target graphs whose graph-level predictor value falls outside either the full range or the central 90\% range (5th--95th percentiles) observed across graphs from the two source topologies.}",
    r"\label{tab:symmetric_distribution_shift}",
    r"\setlength{\tabcolsep}{5pt}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{llcc}",
    r"\toprule",
    r"\textbf{Target topology} & \textbf{Predictor} & \textbf{Outside full range (\%)} & \textbf{Outside central 90\% (\%)} \\",
    r"\midrule",
]

for i, target in enumerate(TARGET_ORDER):

    target_rows = results[results["TargetTopology"] == target]

    for j, (_, row) in enumerate(target_rows.iterrows()):
        target_label = target if j == 0 else ""
        latex_lines.append(
            f"{target_label} & {FEATURE_LABELS[row['Feature']]} & "
            f"{row['OutsideFullPct']:.2f} & "
            f"{row['OutsideCentral90Pct']:.2f} \\\\"
        )

    if i < len(TARGET_ORDER) - 1:
        latex_lines.append(r"\midrule")

latex_lines.extend([
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table}",
])

Path(LATEX_OUTPUT).write_text(
    "\n".join(latex_lines),
    encoding="utf-8",
)

print("\nFiles written:")
print(f"  - {CSV_OUTPUT}")
print(f"  - {LATEX_OUTPUT}")
