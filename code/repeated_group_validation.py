from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# Repeated Graph-Level Validation
# Models A and B
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

# 10 independent graph-level splits
SEEDS = [101, 202, 303, 404, 505,
         606, 707, 808, 909, 1010]

df = pd.read_csv(INPUT_FILE)

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

baseline_features = [
    "N",
    "ActualFraction",
    "C0"
]

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


def graph_split(data, seed):

    graphs = data["GraphUID"].unique().copy()

    rng = np.random.RandomState(seed)
    rng.shuffle(graphs)

    n_train = round(0.80 * len(graphs))

    train_graphs = set(graphs[:n_train])
    test_graphs = set(graphs[n_train:])

    train = data[
        data["GraphUID"].isin(train_graphs)
    ].copy()

    test = data[
        data["GraphUID"].isin(test_graphs)
    ].copy()

    overlap = set(train["GraphUID"]).intersection(
        set(test["GraphUID"])
    )

    assert len(overlap) == 0

    return train, test


def prepare_structural(train, test):

    X_train = train[structural_features].copy()
    X_test = test[structural_features].copy()

    X_train = pd.get_dummies(
        X_train,
        columns=["Topology", "Strategy"],
        dtype=int
    )

    X_test = pd.get_dummies(
        X_test,
        columns=["Topology", "Strategy"],
        dtype=int
    )

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    return X_train, X_test


classification_results = []
regression_results = []

print("\nREPEATED GRAPH-LEVEL VALIDATION")
print("===============================\n")

for i, seed in enumerate(SEEDS, start=1):

    print(f"Split {i}/10 - Seed {seed}")

    # ========================================================
    # MODEL A - CLASSIFICATION
    # ========================================================

    train, test = graph_split(df, seed)

    y_train = train["IsPDS"].astype(int)
    y_test = test["IsPDS"].astype(int)

    # Baseline
    clf_base = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

    clf_base.fit(
        train[baseline_features],
        y_train
    )

    pred_base = clf_base.predict(
        test[baseline_features]
    )

    # Structural
    X_train_s, X_test_s = prepare_structural(
        train, test
    )

    clf_struct = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

    clf_struct.fit(X_train_s, y_train)
    pred_struct = clf_struct.predict(X_test_s)

    classification_results.append({
        "Seed": seed,

        "BaselineAccuracy":
            accuracy_score(y_test, pred_base),

        "StructuralAccuracy":
            accuracy_score(y_test, pred_struct),

        "BaselineF1":
            f1_score(y_test, pred_base),

        "StructuralF1":
            f1_score(y_test, pred_struct),

        "BaselineBalancedAccuracy":
            balanced_accuracy_score(
                y_test, pred_base
            ),

        "StructuralBalancedAccuracy":
            balanced_accuracy_score(
                y_test, pred_struct
            )
    })

    # ========================================================
    # MODEL B - REGRESSION
    #
    # IMPORTANT:
    # Use the SAME graph partition generated above.
    # Then retain successful PDS rows only.
    # ========================================================

    train_r = train[
        train["IsPDS"] == 1
    ].copy()

    test_r = test[
        test["IsPDS"] == 1
    ].copy()

    y_train_r = train_r["PPT"].astype(float)
    y_test_r = test_r["PPT"].astype(float)

    # Baseline
    reg_base = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

    reg_base.fit(
        train_r[baseline_features],
        y_train_r
    )

    pred_base_r = reg_base.predict(
        test_r[baseline_features]
    )

    # Structural
    X_train_rs, X_test_rs = prepare_structural(
        train_r, test_r
    )

    reg_struct = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )

    reg_struct.fit(
        X_train_rs,
        y_train_r
    )

    pred_struct_r = reg_struct.predict(
        X_test_rs
    )

    regression_results.append({
        "Seed": seed,

        "BaselineMAE":
            mean_absolute_error(
                y_test_r, pred_base_r
            ),

        "StructuralMAE":
            mean_absolute_error(
                y_test_r, pred_struct_r
            ),

        "BaselineRMSE":
            np.sqrt(
                mean_squared_error(
                    y_test_r, pred_base_r
                )
            ),

        "StructuralRMSE":
            np.sqrt(
                mean_squared_error(
                    y_test_r, pred_struct_r
                )
            ),

        "BaselineR2":
            r2_score(
                y_test_r, pred_base_r
            ),

        "StructuralR2":
            r2_score(
                y_test_r, pred_struct_r
            )
    })


# ============================================================
# Convert results to tables
# ============================================================

class_df = pd.DataFrame(
    classification_results
)

reg_df = pd.DataFrame(
    regression_results
)

# ============================================================
# Save individual split results
# ============================================================

class_df.to_csv(
    "repeated_validation_model_A.csv",
    index=False
)

reg_df.to_csv(
    "repeated_validation_model_B.csv",
    index=False
)

# ============================================================
# Print summary
# ============================================================

print("\n\nMODEL A - CLASSIFICATION SUMMARY")
print("================================")

for metric in [
    "Accuracy",
    "F1",
    "BalancedAccuracy"
]:

    b = class_df[
        "Baseline" + metric
    ]

    s = class_df[
        "Structural" + metric
    ]

    improvement = s - b

    print(f"\n{metric}")

    print(
        f"Baseline:   "
        f"{b.mean():.4f} +/- {b.std(ddof=1):.4f}"
    )

    print(
        f"Structural: "
        f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}"
    )

    print(
        f"Improvement:"
        f" {improvement.mean():+.4f}"
        f" +/- {improvement.std(ddof=1):.4f}"
    )


print("\n\nMODEL B - REGRESSION SUMMARY")
print("============================")

for metric in [
    "MAE",
    "RMSE",
    "R2"
]:

    b = reg_df[
        "Baseline" + metric
    ]

    s = reg_df[
        "Structural" + metric
    ]

    if metric in ["MAE", "RMSE"]:
        improvement = b - s
        label = "Reduction"
    else:
        improvement = s - b
        label = "Improvement"

    print(f"\n{metric}")

    print(
        f"Baseline:   "
        f"{b.mean():.4f} +/- {b.std(ddof=1):.4f}"
    )

    print(
        f"Structural: "
        f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}"
    )

    print(
        f"{label}:"
        f" {improvement.mean():+.4f}"
        f" +/- {improvement.std(ddof=1):.4f}"
    )


print("\nResults saved to:")
print("repeated_validation_model_A.csv")
print("repeated_validation_model_B.csv")

print("\nREPEATED VALIDATION PASSED.")