from pathlib import Path
"""
Model A: C0-threshold benchmark and feature ablation
====================================================

Purpose
-------
This script addresses Reviewer 1, Comment 4 by:

1. Fitting a trivial decision rule using only initial coverage C0:
       predict PDS success if C0 >= threshold
   The threshold is selected using TRAINING data only by maximizing
   balanced accuracy (equivalently, Youden's J criterion).

2. Evaluating the fitted C0-threshold rule on the held-out test networks.

3. Reproducing Model A baseline and structural random-forest classifiers
   using the same graph-level 80/20 split, random seed, and RF settings.

4. Performing a Model A feature-ablation analysis using six feature sets:
   - C0 only
   - Baseline
   - Graph structure
   - Candidate-set focused
   - Full structural without C0
   - Full structural

Outputs
-------
- model_A_threshold_benchmark.csv
- model_A_ablation_results.csv

The script prints all key metrics needed for the revised manuscript and
reviewer response.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
SEED = 2026

THRESHOLD_OUTPUT = "model_A_threshold_benchmark.csv"
ABLATION_OUTPUT = "model_A_ablation_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N"
    + df["N"].astype(str)
    + "_G"
    + df["GraphID"].astype(str)
)

print("\nMODEL A - THRESHOLD BENCHMARK AND ABLATION")
print("==========================================")

print(f"Total observations: {len(df)}")
print(f"Successful PDS:     {(df['IsPDS'] == 1).sum()}")
print(f"Failed PDS:         {(df['IsPDS'] == 0).sum()}")
print(f"Independent graphs: {df['GraphUID'].nunique()}")


# ============================================================
# SAME PRIMARY GRAPH-LEVEL SPLIT
# ============================================================

graphs = df["GraphUID"].unique().copy()

rng = np.random.RandomState(SEED)
rng.shuffle(graphs)

n_train = round(0.80 * len(graphs))

train_graphs = set(graphs[:n_train])
test_graphs = set(graphs[n_train:])

train = df[df["GraphUID"].isin(train_graphs)].copy()
test = df[df["GraphUID"].isin(test_graphs)].copy()

assert len(
    set(train["GraphUID"]).intersection(set(test["GraphUID"]))
) == 0

print("\nGRAPH-LEVEL SPLIT")
print("-----------------")
print(f"Training graphs: {train['GraphUID'].nunique()}")
print(f"Testing graphs:  {test['GraphUID'].nunique()}")
print(f"Training rows:   {len(train)}")
print(f"Testing rows:    {len(test)}")
print("Leakage check: PASSED")

y_train = train["IsPDS"].astype(int)
y_test = test["IsPDS"].astype(int)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_classifier(y_true, y_pred, score=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "Specificity": (
            tn / (tn + fp)
            if (tn + fp) > 0
            else np.nan
        ),
        "F1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "BalancedAccuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
    }

    if score is not None:
        result["ROC_AUC"] = roc_auc_score(
            y_true,
            score
        )
        result["AP"] = average_precision_score(
            y_true,
            score
        )

    return result


# ============================================================
# PART 1: TRIVIAL C0 THRESHOLD RULE
# ============================================================

# Candidate thresholds are midpoints between consecutive unique
# C0 values in the TRAINING set, plus two endpoint candidates.

unique_c0 = np.sort(
    train["C0"].dropna().unique()
)

threshold_candidates = np.concatenate(
    [
        [unique_c0[0] - 1e-12],
        (unique_c0[:-1] + unique_c0[1:]) / 2.0,
        [unique_c0[-1] + 1e-12],
    ]
)

best_threshold = None
best_balanced_accuracy = -np.inf

for threshold in threshold_candidates:
    train_pred = (
        train["C0"].to_numpy() >= threshold
    ).astype(int)

    score = balanced_accuracy_score(
        y_train,
        train_pred
    )

    if score > best_balanced_accuracy:
        best_balanced_accuracy = score
        best_threshold = threshold


# Evaluate the fitted threshold on TEST only.
threshold_test_pred = (
    test["C0"].to_numpy() >= best_threshold
).astype(int)

# C0 itself is used as the continuous score for ROC-AUC and AP.
threshold_metrics = evaluate_classifier(
    y_test,
    threshold_test_pred,
    score=test["C0"].to_numpy()
)

threshold_row = {
    "Model": "C0 threshold rule",
    "Threshold": best_threshold,
    "TrainingBalancedAccuracy": best_balanced_accuracy,
    **threshold_metrics,
}

threshold_df = pd.DataFrame(
    [threshold_row]
)

threshold_df.to_csv(
    THRESHOLD_OUTPUT,
    index=False
)


# ============================================================
# RANDOM-FOREST FITTING FUNCTION
# ============================================================

def fit_rf_classifier(features):
    X_train = train[features].copy()
    X_test = test[features].copy()

    categorical = [
        c for c in ["Topology", "Strategy"]
        if c in features
    ]

    if categorical:
        X_train = pd.get_dummies(
            X_train,
            columns=categorical,
            dtype=int
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical,
            dtype=int
        )

        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    result = evaluate_classifier(
        y_test,
        prediction,
        probability
    )

    return result


# ============================================================
# PART 2: MODEL A ABLATION
# ============================================================

feature_sets = {

    "C0 only": [
        "C0"
    ],

    "Baseline": [
        "N",
        "ActualFraction",
        "C0"
    ],

    "Graph structure": [
        "N",
        "AvgDegree",
        "Density",
        "AvgClustering",
        "Diameter",
        "AvgPathLength",
        "Topology"
    ],

    "Candidate-set focused": [
        "N",
        "Strategy",
        "ActualFraction",
        "MeanDegreeS",
        "MeanPairDistanceS",
        "MinPairDistanceS",
        "C0"
    ],

    "Full without C0": [
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
        "MinPairDistanceS"
    ],

    "Full structural": [
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
}


ablation_rows = []

for model_name, features in feature_sets.items():

    print(
        f"\nRunning ablation model: "
        f"{model_name}"
    )

    result = fit_rf_classifier(
        features
    )

    ablation_rows.append(
        {
            "Model": model_name,
            **result,
        }
    )


ablation_df = pd.DataFrame(
    ablation_rows
)

ablation_df.to_csv(
    ABLATION_OUTPUT,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 78)
print("C0 THRESHOLD BENCHMARK")
print("=" * 78)

print(
    f"Fitted training threshold: "
    f"C0 >= {best_threshold:.4f}"
)

print(
    f"Training balanced accuracy: "
    f"{best_balanced_accuracy:.4f}"
)

for key in [
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "BalancedAccuracy",
    "ROC_AUC",
    "AP",
]:
    print(
        f"{key:20s}: "
        f"{threshold_metrics[key]:.4f}"
    )


print("\n")
print("=" * 78)
print("MODEL A ABLATION RESULTS")
print("=" * 78)

display_columns = [
    "Model",
    "Accuracy",
    "F1",
    "BalancedAccuracy",
    "ROC_AUC",
    "AP",
]

print(
    ablation_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# KEY COMPARISONS FOR REVIEWER RESPONSE
# ============================================================

result_by_model = (
    ablation_df
    .set_index("Model")
)

full = result_by_model.loc[
    "Full structural"
]

baseline = result_by_model.loc[
    "Baseline"
]

c0_only = result_by_model.loc[
    "C0 only"
]

without_c0 = result_by_model.loc[
    "Full without C0"
]

print("\n")
print("=" * 78)
print("KEY COMPARISONS")
print("=" * 78)

print(
    "Structural minus threshold rule:"
)
print(
    f"  Accuracy gain = "
    f"{full['Accuracy'] - threshold_metrics['Accuracy']:.4f}"
)
print(
    f"  F1 gain       = "
    f"{full['F1'] - threshold_metrics['F1']:.4f}"
)
print(
    f"  BalAcc gain   = "
    f"{full['BalancedAccuracy'] - threshold_metrics['BalancedAccuracy']:.4f}"
)

print(
    "\nStructural minus C0-only RF:"
)
print(
    f"  Accuracy gain = "
    f"{full['Accuracy'] - c0_only['Accuracy']:.4f}"
)
print(
    f"  F1 gain       = "
    f"{full['F1'] - c0_only['F1']:.4f}"
)
print(
    f"  BalAcc gain   = "
    f"{full['BalancedAccuracy'] - c0_only['BalancedAccuracy']:.4f}"
)

print(
    "\nFull structural without C0:"
)
print(
    f"  Accuracy      = "
    f"{without_c0['Accuracy']:.4f}"
)
print(
    f"  F1            = "
    f"{without_c0['F1']:.4f}"
)
print(
    f"  Balanced Acc. = "
    f"{without_c0['BalancedAccuracy']:.4f}"
)

print("\nFiles written:")
print(f"  - {THRESHOLD_OUTPUT}")
print(f"  - {ABLATION_OUTPUT}")

print(
    "\nMODEL A THRESHOLD + ABLATION ANALYSIS PASSED."
)
