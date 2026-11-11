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
# Model B - Cross-Topology Generalization
#
# Train on TWO topologies
# Test on the THIRD unseen topology
# ============================================================

SEED = 2026
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)

# Successful PDS cases only
df = df[df["IsPDS"] == 1].copy()

print("\nMODEL B - CROSS-TOPOLOGY GENERALIZATION")
print("========================================\n")

print(f"Successful PDS cases: {len(df)}")

# ============================================================
# Predictors
# ============================================================

baseline_features = [
    "N",
    "ActualFraction",
    "C0"
]

# IMPORTANT:
# Topology label is deliberately EXCLUDED.
#
# Reason:
# The test topology is unseen during training.
# We want to test whether structural measurements themselves
# generalize across network families.
# ============================================================

structural_features = [
    "N",
    "AvgDegree",
    "Density",
    "AvgClustering",
    "Diameter",
    "AvgPathLength",
    "Strategy",
    "ActualFraction",
    "MeanDegreeS",
    "MeanPairDistanceS",
    "MinPairDistanceS",
    "C0"
]

experiments = [
    (["ER", "BA"], "WS"),
    (["ER", "WS"], "BA"),
    (["BA", "WS"], "ER")
]

results = []

# ============================================================
# Evaluation function
# ============================================================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(
        y_true, y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true, y_pred
        )
    )

    r2 = r2_score(
        y_true, y_pred
    )

    return mae, rmse, r2


# ============================================================
# Run experiments
# ============================================================

for train_topologies, test_topology in experiments:

    train = df[
        df["Topology"].isin(train_topologies)
    ].copy()

    test = df[
        df["Topology"] == test_topology
    ].copy()

    y_train = train["PPT"].astype(float)
    y_test = test["PPT"].astype(float)

    print("\n----------------------------------------")
    print(
        f"TRAIN: {' + '.join(train_topologies)}"
        f"  -->  TEST: {test_topology}"
    )
    print("----------------------------------------")

    print(f"Training rows: {len(train)}")
    print(f"Testing rows:  {len(test)}")

    # ========================================================
    # Baseline
    # ========================================================

    baseline_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )

    baseline_model.fit(
        train[baseline_features],
        y_train
    )

    pred_baseline = baseline_model.predict(
        test[baseline_features]
    )

    b_mae, b_rmse, b_r2 = evaluate(
        y_test,
        pred_baseline
    )

    # ========================================================
    # Structural
    # ========================================================

    X_train = train[
        structural_features
    ].copy()

    X_test = test[
        structural_features
    ].copy()

    # Strategy is categorical
    X_train = pd.get_dummies(
        X_train,
        columns=["Strategy"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["Strategy"],
        dtype=int
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    structural_model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )

    structural_model.fit(
        X_train,
        y_train
    )

    pred_structural = structural_model.predict(
        X_test
    )

    s_mae, s_rmse, s_r2 = evaluate(
        y_test,
        pred_structural
    )

    # ========================================================
    # Print
    # ========================================================

    print("\nBASELINE")
    print(f"MAE:  {b_mae:.4f}")
    print(f"RMSE: {b_rmse:.4f}")
    print(f"R2:   {b_r2:.4f}")

    print("\nSTRUCTURAL")
    print(f"MAE:  {s_mae:.4f}")
    print(f"RMSE: {s_rmse:.4f}")
    print(f"R2:   {s_r2:.4f}")

    print("\nSTRUCTURAL vs BASELINE")
    print(
        f"MAE reduction:  "
        f"{b_mae - s_mae:+.4f}"
    )

    print(
        f"RMSE reduction: "
        f"{b_rmse - s_rmse:+.4f}"
    )

    print(
        f"R2 improvement: "
        f"{s_r2 - b_r2:+.4f}"
    )

    results.append({
        "TrainTopologies":
            "+".join(train_topologies),

        "TestTopology":
            test_topology,

        "BaselineMAE": b_mae,
        "StructuralMAE": s_mae,

        "BaselineRMSE": b_rmse,
        "StructuralRMSE": s_rmse,

        "BaselineR2": b_r2,
        "StructuralR2": s_r2,

        "MAEReduction":
            b_mae - s_mae,

        "RMSEReduction":
            b_rmse - s_rmse,

        "R2Improvement":
            s_r2 - b_r2
    })


# ============================================================
# Save results
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "cross_topology_model_B_results.csv",
    index=False
)

print("\n\nSUMMARY")
print("=======")

print(
    results_df[
        [
            "TrainTopologies",
            "TestTopology",
            "BaselineR2",
            "StructuralR2",
            "R2Improvement"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nResults saved to:")
print("cross_topology_model_B_results.csv")

print(
    "\nMODEL B CROSS-TOPOLOGY "
    "GENERALIZATION PASSED."
)