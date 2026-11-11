import pandas as pd

# ============================================================
# TABLE 4
# Cross-topology generalization of Model B
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "cross_topology_model_B_results.csv"
OUTPUT_FILE = "Table_4_cross_topology_generalization.tex"

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Required columns
# ============================================================

required_columns = [
    "TrainTopologies",
    "TestTopology",
    "BaselineMAE",
    "StructuralMAE",
    "BaselineRMSE",
    "StructuralRMSE",
    "BaselineR2",
    "StructuralR2"
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
# Desired row order
# ============================================================

test_order = ["WS", "BA", "ER"]

df["TestTopology"] = pd.Categorical(
    df["TestTopology"],
    categories=test_order,
    ordered=True
)

plot_df = (
    df.sort_values("TestTopology")
      .reset_index(drop=True)
)

# ============================================================
# Verification
# ============================================================

print("\nTABLE 4")
print("=======")

for _, row in plot_df.iterrows():

    print(
        f"Train {row['TrainTopologies']} -> "
        f"Test {row['TestTopology']}: "
        f"Baseline "
        f"MAE={row['BaselineMAE']:.4f}, "
        f"RMSE={row['BaselineRMSE']:.4f}, "
        f"R2={row['BaselineR2']:.4f}; "
        f"Structural "
        f"MAE={row['StructuralMAE']:.4f}, "
        f"RMSE={row['StructuralRMSE']:.4f}, "
        f"R2={row['StructuralR2']:.4f}"
    )

# ============================================================
# Scientific consistency check
# ============================================================

for _, row in plot_df.iterrows():

    assert row["StructuralMAE"] < row["BaselineMAE"]
    assert row["StructuralRMSE"] < row["BaselineRMSE"]
    assert row["StructuralR2"] > row["BaselineR2"]

# ============================================================
# Prepare LaTeX rows
# ============================================================

rows = []

for _, row in plot_df.iterrows():

    train = row["TrainTopologies"].replace("+", "+")

    latex_row = (
        f"{train} & "
        f"{row['TestTopology']} & "
        f"{row['BaselineMAE']:.3f} & "
        f"{row['StructuralMAE']:.3f} & "
        f"{row['BaselineRMSE']:.3f} & "
        f"{row['StructuralRMSE']:.3f} & "
        f"{row['BaselineR2']:.3f} & "
        f"{row['StructuralR2']:.3f} \\\\"
    )

    rows.append(latex_row)

rows_text = "\n".join(rows)

# ============================================================
# Generate LaTeX
# ============================================================

latex = rf"""\begin{{table*}}[t]
\centering
\caption{{Cross-topology transfer performance of Model B when evaluated on an unseen network topology.}}
\label{{tab:cross_topology}}
\begin{{tabular}}{{llcccccc}}
\toprule
& &
\multicolumn{{2}}{{c}}{{\textbf{{MAE}}}} &
\multicolumn{{2}}{{c}}{{\textbf{{RMSE}}}} &
\multicolumn{{2}}{{c}}{{\textbf{{$R^2$}}}} \\
\cmidrule(lr){{3-4}}
\cmidrule(lr){{5-6}}
\cmidrule(lr){{7-8}}
\textbf{{Training}} &
\textbf{{Held-out}} &
\textbf{{Base.}} &
\textbf{{Struct.}} &
\textbf{{Base.}} &
\textbf{{Struct.}} &
\textbf{{Base.}} &
\textbf{{Struct.}} \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
\end{{table*}}
"""

# ============================================================
# Save
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    f.write(latex)

# ============================================================
# Output
# ============================================================

print(
    f"\nLaTeX file saved: {OUTPUT_FILE}"
)

print("\nGenerated LaTeX:")
print("================")
print(latex)

print("TABLE 4 PASSED.")