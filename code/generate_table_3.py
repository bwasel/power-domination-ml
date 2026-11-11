import pandas as pd

# ============================================================
# TABLE 3
# Model B regression performance
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent / "model_B_regression_performance.csv"
OUTPUT_FILE = "Table_3_Model_B_performance.tex"

df = pd.read_csv(INPUT_FILE)

# ============================================================
# Required columns
# ============================================================

required_columns = [
    "Model",
    "MAE",
    "RMSE",
    "R2"
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

print("\nTABLE 3")
print("=======")

for _, row in plot_df.iterrows():

    print(
        f"{row['Model']}: "
        f"MAE={row['MAE']:.4f}, "
        f"RMSE={row['RMSE']:.4f}, "
        f"R2={row['R2']:.4f}"
    )

# ============================================================
# Scientific consistency checks
# ============================================================

baseline = plot_df[
    plot_df["Model"] == "Baseline"
].iloc[0]

structural = plot_df[
    plot_df["Model"] == "Structural"
].iloc[0]

assert structural["MAE"] < baseline["MAE"]
assert structural["RMSE"] < baseline["RMSE"]
assert structural["R2"] > baseline["R2"]

# ============================================================
# Calculate improvements for verification only
# ============================================================

mae_reduction = (
    baseline["MAE"] - structural["MAE"]
)

rmse_reduction = (
    baseline["RMSE"] - structural["RMSE"]
)

r2_gain = (
    structural["R2"] - baseline["R2"]
)

print("\nImprovement:")
print(
    f"MAE reduction = {mae_reduction:.4f}"
)
print(
    f"RMSE reduction = {rmse_reduction:.4f}"
)
print(
    f"R2 gain = {r2_gain:.4f}"
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
        f"{row['MAE']:.3f} & "
        f"{row['RMSE']:.3f} & "
        f"{row['R2']:.3f} \\\\"
    )

    rows.append(latex_row)

rows_text = "\n".join(rows)

# ============================================================
# Generate LaTeX
# ============================================================

latex = rf"""\begin{{table}}[t]
\centering
\caption{{Regression performance of Model B for predicting power propagation time on successful PDS configurations in the held-out test networks.}}
\label{{tab:model_B_performance}}
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Model}} &
\textbf{{MAE}} &
\textbf{{RMSE}} &
\textbf{{$R^2$}} \\
\midrule
{rows_text}
\bottomrule
\end{{tabular}}
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

print("TABLE 3 PASSED.")