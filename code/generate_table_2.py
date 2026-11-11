import pandas as pd

# ============================================================
# TABLE 2
# Model A classification performance
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "model_A_extended_performance.csv"
OUTPUT_FILE = "Table_2_Model_A_performance.tex"

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Required columns
# ============================================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "BalancedAccuracy",
    "ROC_AUC",
    "PR_AUC"
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
# Model order
# ============================================================

model_order = [
    "Baseline",
    "Structural"
]

plot_df = (
    df.set_index("Model")
      .loc[model_order]
      .reset_index()
)

# ============================================================
# Verification
# ============================================================

print("\nTABLE 2")
print("=======")

for _, row in plot_df.iterrows():

    print(
        f"{row['Model']}: "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Precision={row['Precision']:.4f}, "
        f"Recall={row['Recall']:.4f}, "
        f"Specificity={row['Specificity']:.4f}, "
        f"F1={row['F1']:.4f}, "
        f"BalancedAccuracy={row['BalancedAccuracy']:.4f}, "
        f"ROC-AUC={row['ROC_AUC']:.4f}, "
        f"AP={row['PR_AUC']:.4f}"
    )

# ============================================================
# Prepare LaTeX rows
# ============================================================

rows = []

for _, row in plot_df.iterrows():

    model_name = (
        "Baseline"
        if row["Model"] == "Baseline"
        else "Structural"
    )

    latex_row = (
        f"{model_name} & "
        f"{row['Accuracy']:.3f} & "
        f"{row['Precision']:.3f} & "
        f"{row['Recall']:.3f} & "
        f"{row['Specificity']:.3f} & "
        f"{row['F1']:.3f} & "
        f"{row['BalancedAccuracy']:.3f} & "
        f"{row['ROC_AUC']:.3f} & "
        f"{row['PR_AUC']:.3f} \\\\"
    )

    rows.append(latex_row)

rows_text = "\n\n".join(rows)

# ============================================================
# Generate LaTeX
# ============================================================

latex = rf"""\begin{{table}}[H]
\caption{{Classification performance of Model A on the held-out test networks.}}
\label{{tab:model_a_performance}}

\begin{{adjustwidth}}{{-\extralength}}{{0cm}}
\centering
\small

\begin{{tabularx}}{{\fulllength}}{{lCCCCCCCC}}
\toprule
\textbf{{Model}} &
\textbf{{Accuracy}} &
\textbf{{Precision}} &
\textbf{{Recall}} &
\textbf{{Specificity}} &
\textbf{{F1}} &
\textbf{{Bal. Acc.}} &
\textbf{{ROC-AUC}} &
\textbf{{AP}} \\
\midrule

{rows_text}

\bottomrule
\end{{tabularx}}

\end{{adjustwidth}}
\end{{table}}
"""

# ============================================================
# Save LaTeX file
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

print("TABLE 2 PASSED.")