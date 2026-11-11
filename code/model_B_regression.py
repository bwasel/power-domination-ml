from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# PDS Propagation Dynamics
# Model B - Regression
#
# Objective:
# Predict PPT among SUCCESSFUL PDS cases only.
#
# IMPORTANT:
# Train/test splitting is performed at GRAPH LEVEL.
# ============================================================

SEED = 2026
np.random.seed(SEED)

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nMODEL B - PPT REGRESSION")
print("========================\n")

print(f"Total observations: {len(df)}")

# Keep successful PDS cases only
df = df[df["IsPDS"] == 1].copy()

print(f"Successful PDS cases used: {len(df)}")
print(f"Mean PPT:   {df['PPT'].mean():.4f}")
print(f"Median PPT: {df['PPT'].median():.4f}")
print(f"Min PPT:    {df['PPT'].min():.0f}")
print(f"Max PPT:    {df['PPT'].max():.0f}\n")

# ============================================================
# 2. Unique graph identifier
# ============================================================

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N"
    + df["N"].astype(str)
    + "_G"
    + df["GraphID"].astype(str)
)

unique_graphs = df["GraphUID"].unique()

print(f"Independent networks represented: {len(unique_graphs)}\n")

# ============================================================
# 3. Graph-level Train/Test split
# ============================================================

rng = np.random.RandomState(SEED)

shuffled_graphs = unique_graphs.copy()
rng.shuffle(shuffled_graphs)

num_train_graphs = round(0.80 * len(shuffled_graphs))

train_graphs = set(shuffled_graphs[:num_train_graphs])
test_graphs = set(shuffled_graphs[num_train_graphs:])

train = df[df["GraphUID"].isin(train_graphs)].copy()
test = df[df["GraphUID"].isin(test_graphs)].copy()

print("GRAPH-LEVEL SPLIT")
print("-----------------")
print(f"Training networks: {train['GraphUID'].nunique()}")
print(f"Testing networks:  {test['GraphUID'].nunique()}")
print(f"Training rows:     {len(train)}")
print(f"Testing rows:      {len(test)}\n")

overlap = set(train["GraphUID"]).intersection(
    set(test["GraphUID"])
)

assert len(overlap) == 0, \
    "DATA LEAKAGE: A graph occurs in both Train and Test."

print("Leakage check: PASSED\n")

# ============================================================
# 4. Response
# ============================================================

y_train = train["PPT"].astype(float)
y_test = test["PPT"].astype(float)

# ============================================================
# 5. BASELINE MODEL
# ============================================================

baseline_features = [
    "N",
    "ActualFraction",
    "C0"
]

X_train_baseline = train[baseline_features]
X_test_baseline = test[baseline_features]

print("Training BASELINE regression model...")

baseline_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

baseline_model.fit(
    X_train_baseline,
    y_train
)

y_pred_baseline = baseline_model.predict(
    X_test_baseline
)

# ============================================================
# 6. STRUCTURAL MODEL
# ============================================================

structural_features = [
    "N",
    "AvgDegree",
    "Density",
    "AvgClustering",
    "Diameter",
    "AvgPathLength",
    "Topology",
    "Strategy",
    "ActualFraction",
    "MeanDegreeS",
    "MeanPairDistanceS",
    "MinPairDistanceS",
    "C0"
]

X_train_structural = train[
    structural_features
].copy()

X_test_structural = test[
    structural_features
].copy()

# Encode categorical predictors using training data
X_train_structural = pd.get_dummies(
    X_train_structural,
    columns=["Topology", "Strategy"],
    dtype=int
)

X_test_structural = pd.get_dummies(
    X_test_structural,
    columns=["Topology", "Strategy"],
    dtype=int
)

X_test_structural = X_test_structural.reindex(
    columns=X_train_structural.columns,
    fill_value=0
)

print("Training STRUCTURAL regression model...")

structural_model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

structural_model.fit(
    X_train_structural,
    y_train
)

y_pred_structural = structural_model.predict(
    X_test_structural
)

# ============================================================
# 7. Evaluation
# ============================================================

def evaluate_regression(y_true, y_pred):

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


baseline_results = evaluate_regression(
    y_test,
    y_pred_baseline
)

structural_results = evaluate_regression(
    y_test,
    y_pred_structural
)

# ============================================================
# 8. Print results
# ============================================================

print("\nMODEL PERFORMANCE")
print("=================")

print("\nBASELINE MODEL")
print("--------------")
print(f"MAE:  {baseline_results['MAE']:.4f}")
print(f"RMSE: {baseline_results['RMSE']:.4f}")
print(f"R2:   {baseline_results['R2']:.4f}")

print("\nSTRUCTURAL MODEL")
print("----------------")
print(f"MAE:  {structural_results['MAE']:.4f}")
print(f"RMSE: {structural_results['RMSE']:.4f}")
print(f"R2:   {structural_results['R2']:.4f}")

# ============================================================
# 9. Improvement
# ============================================================

print("\nSTRUCTURAL IMPROVEMENT OVER BASELINE")
print("------------------------------------")

print(
    f"MAE reduction:  "
    f"{baseline_results['MAE'] - structural_results['MAE']:+.4f}"
)

print(
    f"RMSE reduction: "
    f"{baseline_results['RMSE'] - structural_results['RMSE']:+.4f}"
)

print(
    f"R2 improvement: "
    f"{structural_results['R2'] - baseline_results['R2']:+.4f}"
)

# ============================================================
# 10. Save performance
# ============================================================

performance_table = pd.DataFrame([
    {
        "Model": "Baseline",
        "MAE": baseline_results["MAE"],
        "RMSE": baseline_results["RMSE"],
        "R2": baseline_results["R2"]
    },
    {
        "Model": "Structural",
        "MAE": structural_results["MAE"],
        "RMSE": structural_results["RMSE"],
        "R2": structural_results["R2"]
    }
])

performance_table.to_csv(
    "model_B_regression_performance.csv",
    index=False
)

# ============================================================
# 11. Save predictions
# ============================================================

prediction_table = pd.DataFrame({
    "GraphUID": test["GraphUID"].values,
    "Topology": test["Topology"].values,
    "N": test["N"].values,
    "Strategy": test["Strategy"].values,
    "RequestedFraction":
        test["RequestedFraction"].values,
    "C0": test["C0"].values,
    "ActualPPT": y_test.values,
    "BaselinePrediction":
        y_pred_baseline,
    "StructuralPrediction":
        y_pred_structural
})

prediction_table.to_csv(
    "model_B_test_predictions.csv",
    index=False
)

print("\nResults saved to:")
print("model_B_regression_performance.csv")
print("model_B_test_predictions.csv")

print("\nMODEL B REGRESSION PASSED.")