"""
Success-subset characterization for Table 6
===========================================

This script reproduces the structural characterization used to address
Reviewer 1, Comment 3.

It reports:
1. Overall mean initial coverage (C0) for successful and failed configurations.
2. PDS success rate by:
   - prescribed candidate-set fraction,
   - candidate-set selection strategy,
   - network topology.
3. For each group:
   - mean C0 among successful configurations,
   - mean C0 among failed configurations,
   - mean AvgPathLength among successful configurations,
   - mean MeanDegreeS among successful configurations,
   - mean MeanPairDistanceS among successful configurations.
4. A CSV version of Table 6.
5. A LaTeX table file suitable for direct use in Overleaf.

The calculations are performed on the original sampled dataset; no balancing,
resampling, or model fitting is applied in this script.
"""

from pathlib import Path
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

OUTPUT_CSV = "Table_6_Success_subset_characterization.csv"
OUTPUT_TEX = "Table_6_Success_subset_characterization.tex"


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

required_columns = [
    "Topology",
    "RequestedFraction",
    "Strategy",
    "C0",
    "IsPDS",
    "AvgPathLength",
    "MeanDegreeS",
    "MeanPairDistanceS",
]

input_path = Path(INPUT_FILE)

if not input_path.exists():
    raise FileNotFoundError(
        f"Input file not found: {input_path.resolve()}\n"
        "Place this script in the same folder as the CSV file, or update INPUT_FILE."
    )

df = pd.read_csv(input_path)

missing_columns = [c for c in required_columns if c not in df.columns]
if missing_columns:
    raise ValueError(
        "The input CSV is missing required columns: "
        + ", ".join(missing_columns)
    )

df = df.copy()
df["IsPDS"] = pd.to_numeric(df["IsPDS"], errors="raise").astype(int)

if not df["IsPDS"].isin([0, 1]).all():
    raise ValueError("IsPDS must contain only 0/1 values.")


# ============================================================
# SUMMARY FUNCTION
# ============================================================

def summarize_group(data: pd.DataFrame) -> pd.Series:
    """Compute the quantities reported in Table 6."""
    success = data.loc[data["IsPDS"] == 1]
    failure = data.loc[data["IsPDS"] == 0]

    return pd.Series(
        {
            "PDS success (%)": 100.0 * data["IsPDS"].mean(),
            "C0 (success)": success["C0"].mean(),
            "C0 (failure)": failure["C0"].mean(),
            "Average path length": success["AvgPathLength"].mean(),
            "Mean degree in S": success["MeanDegreeS"].mean(),
            "Mean pairwise distance in S": success["MeanPairDistanceS"].mean(),
        }
    )


# ============================================================
# OVERALL SUCCESS / FAILURE CHARACTERIZATION
# ============================================================

overall_success = df.loc[df["IsPDS"] == 1]
overall_failure = df.loc[df["IsPDS"] == 0]

overall_success_rate = 100.0 * df["IsPDS"].mean()
overall_success_c0 = overall_success["C0"].mean()
overall_failure_c0 = overall_failure["C0"].mean()

print("=" * 78)
print("OVERALL CHARACTERIZATION")
print("=" * 78)
print(f"Total configurations       : {len(df):,}")
print(f"Successful configurations  : {len(overall_success):,}")
print(f"Failed configurations      : {len(overall_failure):,}")
print(f"Overall PDS success rate    : {overall_success_rate:.2f}%")
print(f"Mean C0, successful         : {overall_success_c0:.3f}")
print(f"Mean C0, failed             : {overall_failure_c0:.3f}")


# ============================================================
# 1. PRESCRIBED CANDIDATE-SET FRACTION
# ============================================================

fraction_table = (
    df.groupby("RequestedFraction", sort=True, observed=False)
      .apply(summarize_group)
)

# Convert fractions such as 0.05, 0.10, 0.15 to display labels.
fraction_table.index = [
    f"{int(round(float(x) * 100))}%"
    for x in fraction_table.index
]

fraction_table.index.name = "Group"


# ============================================================
# 2. CANDIDATE-SET SELECTION STRATEGY
# ============================================================

strategy_table = (
    df.groupby("Strategy", sort=False, observed=False)
      .apply(summarize_group)
)

# Preserve the manuscript ordering where present.
strategy_lookup = {
    "random": "Random",
    "high-degree": "High-degree",
    "distributed": "Distributed",
}

available_strategy_keys = [
    key for key in ["random", "high-degree", "distributed"]
    if key in strategy_table.index
]

strategy_table = strategy_table.reindex(available_strategy_keys)
strategy_table.index = [
    strategy_lookup.get(str(x), str(x))
    for x in strategy_table.index
]
strategy_table.index.name = "Group"


# ============================================================
# 3. NETWORK TOPOLOGY
# ============================================================

topology_table = (
    df.groupby("Topology", sort=False, observed=False)
      .apply(summarize_group)
)

available_topologies = [
    key for key in ["ER", "BA", "WS"]
    if key in topology_table.index
]

topology_table = topology_table.reindex(available_topologies)
topology_table.index.name = "Group"


# ============================================================
# PRINT TABLES
# ============================================================

def print_summary(title: str, table: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    out = table.copy()
    out["PDS success (%)"] = out["PDS success (%)"].map(lambda x: f"{x:.2f}")

    for col in [
        "C0 (success)",
        "C0 (failure)",
        "Average path length",
        "Mean degree in S",
        "Mean pairwise distance in S",
    ]:
        out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "NA")

    print(out.to_string())


print_summary(
    "PRESCRIBED CANDIDATE-SET FRACTION",
    fraction_table,
)

print_summary(
    "CANDIDATE-SET SELECTION STRATEGY",
    strategy_table,
)

print_summary(
    "NETWORK TOPOLOGY",
    topology_table,
)


# ============================================================
# COMBINE AND SAVE CSV
# ============================================================

fraction_out = fraction_table.copy()
fraction_out.insert(
    0,
    "Group type",
    "Prescribed candidate-set fraction",
)

strategy_out = strategy_table.copy()
strategy_out.insert(
    0,
    "Group type",
    "Candidate-set selection strategy",
)

topology_out = topology_table.copy()
topology_out.insert(
    0,
    "Group type",
    "Network topology",
)

table6 = pd.concat(
    [fraction_out, strategy_out, topology_out],
    axis=0,
)

table6.index.name = "Group"
table6.to_csv(OUTPUT_CSV, float_format="%.6f")


# ============================================================
# CREATE LATEX TABLE
# ============================================================

def fmt_rate(x):
    return "--" if pd.isna(x) else f"{x:.2f}"


def fmt3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


latex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\color{blue}",
    r"\caption{Characterization of the successful PDS subsets across prescribed candidate-set fraction, selection strategy, and network topology. Success rate is computed over all configurations in the corresponding group. Structural summaries are means computed over successful configurations, except for the column reporting mean $C_0$ among failed configurations.}",
    r"\label{tab:success_subset_characterization}",
    r"\setlength{\tabcolsep}{4pt}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"\textbf{Group} &",
    r"\textbf{PDS success (\%)} &",
    r"\textbf{$C_0$ (success)} &",
    r"\textbf{$C_0$ (failure)} &",
    r"\textbf{Average path length} &",
    r"\textbf{Mean degree in $S$} &",
    r"\textbf{Mean pairwise distance in $S$} \\",
    r"\midrule",
    r"\multicolumn{7}{l}{\textit{Prescribed candidate-set fraction}} \\",
]

for group, row in fraction_table.iterrows():
    latex_lines.append(
        f"{group} & "
        f"{fmt_rate(row['PDS success (%)'])} & "
        f"{fmt3(row['C0 (success)'])} & "
        f"{fmt3(row['C0 (failure)'])} & "
        f"{fmt3(row['Average path length'])} & "
        f"{fmt3(row['Mean degree in S'])} & "
        f"{fmt3(row['Mean pairwise distance in S'])} \\\\"
    )

latex_lines.extend(
    [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Candidate-set selection strategy}} \\",
    ]
)

for group, row in strategy_table.iterrows():
    latex_lines.append(
        f"{group} & "
        f"{fmt_rate(row['PDS success (%)'])} & "
        f"{fmt3(row['C0 (success)'])} & "
        f"{fmt3(row['C0 (failure)'])} & "
        f"{fmt3(row['Average path length'])} & "
        f"{fmt3(row['Mean degree in S'])} & "
        f"{fmt3(row['Mean pairwise distance in S'])} \\\\"
    )

latex_lines.extend(
    [
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Network topology}} \\",
    ]
)

for group, row in topology_table.iterrows():
    latex_lines.append(
        f"{group} & "
        f"{fmt_rate(row['PDS success (%)'])} & "
        f"{fmt3(row['C0 (success)'])} & "
        f"{fmt3(row['C0 (failure)'])} & "
        f"{fmt3(row['Average path length'])} & "
        f"{fmt3(row['Mean degree in S'])} & "
        f"{fmt3(row['Mean pairwise distance in S'])} \\\\"
    )

latex_lines.extend(
    [
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
)

Path(OUTPUT_TEX).write_text(
    "\n".join(latex_lines) + "\n",
    encoding="utf-8",
)


# ============================================================
# FINAL CHECK AGAINST MANUSCRIPT VALUES
# ============================================================

expected = {
    "overall_success_c0": 0.731,
    "overall_failure_c0": 0.430,
}

print("\n" + "=" * 78)
print("MANUSCRIPT CROSS-CHECK")
print("=" * 78)
print(
    f"Overall successful mean C0: {overall_success_c0:.3f} "
    f"(manuscript: {expected['overall_success_c0']:.3f})"
)
print(
    f"Overall failed mean C0    : {overall_failure_c0:.3f} "
    f"(manuscript: {expected['overall_failure_c0']:.3f})"
)

print("\nFiles written:")
print(f"  - {Path(OUTPUT_CSV).resolve()}")
print(f"  - {Path(OUTPUT_TEX).resolve()}")
