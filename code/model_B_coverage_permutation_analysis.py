from pathlib import Path
"""
Reviewer 1 - Comment 5 analysis
===============================

This script performs two analyses requested by Reviewer 1, Comment 5:

1. Fixed-coverage-bin analysis:
   Among successful PDS configurations, C0 is divided into fixed-width
   bins of 0.10. Within each bin, the script reports:
     - number of configurations,
     - mean and median PPT,
     - Pearson correlation of PPT with:
         AvgPathLength,
         AvgClustering,
         MeanPairDistanceS,
         MeanDegreeS.

2. Grouped permutation importance for structural Model B:
   The same primary graph-level 80/20 split, random seed (2026), feature
   set, and random-forest settings used in the manuscript are retained.
   Importance is quantified as the decrease in held-out test R^2 after
   permuting each original predictor. One-hot columns belonging to
   Topology or Strategy are permuted together as a group.

Outputs:
    coverage_bin_ppt_analysis.csv
    model_B_grouped_permutation_importance.csv
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
SEED = 2026
N_PERMUTATIONS = 30

COVERAGE_OUTPUT = "coverage_bin_ppt_analysis.csv"
PERMUTATION_OUTPUT = "model_B_grouped_permutation_importance.csv"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Model B / propagation-time analyses use successful PDS cases only.
successful = df[df["IsPDS"] == 1].copy()

print("\nREVIEWER 1 - COMMENT 5 ANALYSIS")
print("================================")
print(f"Total configurations:      {len(df):,}")
print(f"Successful configurations: {len(successful):,}")


# ============================================================
# PART 1: FIXED-COVERAGE-BIN ANALYSIS
# ============================================================

# Fixed-width 0.10 bins covering the observed successful C0 range.
# The final edge is slightly above 1.0 so that C0 = 1 is included.
coverage_edges = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.000001,
]

coverage_labels = [
    "0.30-<0.40",
    "0.40-<0.50",
    "0.50-<0.60",
    "0.60-<0.70",
    "0.70-<0.80",
    "0.80-<0.90",
    "0.90-1.00",
]

successful["C0_bin"] = pd.cut(
    successful["C0"],
    bins=coverage_edges,
    labels=coverage_labels,
    include_lowest=True,
    right=False,
)

correlation_features = [
    "AvgPathLength",
    "AvgClustering",
    "MeanPairDistanceS",
    "MeanDegreeS",
]

coverage_rows = []

for bin_name, group in successful.groupby(
    "C0_bin",
    observed=True,
    sort=True,
):
    row = {
        "C0_bin": str(bin_name),
        "N": len(group),
        "Mean_C0": group["C0"].mean(),
        "Mean_PPT": group["PPT"].mean(),
        "Median_PPT": group["PPT"].median(),
    }

    for feature in correlation_features:
        if group[feature].nunique() > 1:
            r = group["PPT"].corr(group[feature])
        else:
            r = np.nan

        row[f"Pearson_r_{feature}"] = r

    coverage_rows.append(row)

coverage_results = pd.DataFrame(coverage_rows)

coverage_results.to_csv(
    COVERAGE_OUTPUT,
    index=False,
)

print("\nFIXED-COVERAGE-BIN ANALYSIS")
print("---------------------------")
print(
    coverage_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


# ============================================================
# PART 2: SAME PRIMARY MODEL B GRAPH-LEVEL SPLIT
# ============================================================

model_df = successful.copy()

model_df["GraphUID"] = (
    model_df["Topology"].astype(str)
    + "_N"
    + model_df["N"].astype(str)
    + "_G"
    + model_df["GraphID"].astype(str)
)

graphs = model_df["GraphUID"].unique().copy()

rng = np.random.RandomState(SEED)
rng.shuffle(graphs)

n_train = round(0.80 * len(graphs))

train_graphs = set(graphs[:n_train])
test_graphs = set(graphs[n_train:])

train = model_df[
    model_df["GraphUID"].isin(train_graphs)
].copy()

test = model_df[
    model_df["GraphUID"].isin(test_graphs)
].copy()

overlap = set(train["GraphUID"]).intersection(
    set(test["GraphUID"])
)

assert len(overlap) == 0

print("\nPRIMARY GRAPH-LEVEL SPLIT")
print("-------------------------")
print(f"Training networks: {train['GraphUID'].nunique()}")
print(f"Testing networks:  {test['GraphUID'].nunique()}")
print(f"Training rows:     {len(train):,}")
print(f"Testing rows:      {len(test):,}")
print("Leakage check: PASSED")


# ============================================================
# STRUCTURAL MODEL B
# ============================================================

features = [
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
    "C0",
]

X_train = train[features].copy()
X_test = test[features].copy()

y_train = train["PPT"].astype(float)
y_test = test["PPT"].astype(float)

X_train = pd.get_dummies(
    X_train,
    columns=["Topology", "Strategy"],
    dtype=int,
)

X_test = pd.get_dummies(
    X_test,
    columns=["Topology", "Strategy"],
    dtype=int,
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0,
)

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1,
)

model.fit(X_train, y_train)

test_prediction = model.predict(X_test)

baseline_r2 = r2_score(
    y_test,
    test_prediction,
)

baseline_mae = mean_absolute_error(
    y_test,
    test_prediction,
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_prediction,
    )
)

print("\nSTRUCTURAL MODEL B CHECK")
print("------------------------")
print(f"Test MAE  = {baseline_mae:.4f}")
print(f"Test RMSE = {baseline_rmse:.4f}")
print(f"Test R2   = {baseline_r2:.4f}")


# ============================================================
# GROUP ORIGINAL FEATURES
# ============================================================

feature_groups = {}

for feature in features:
    if feature in ["Topology", "Strategy"]:
        feature_groups[feature] = [
            col
            for col in X_test.columns
            if col.startswith(feature + "_")
        ]
    else:
        feature_groups[feature] = [feature]


# ============================================================
# GROUPED PERMUTATION IMPORTANCE
# ============================================================

# All one-hot columns belonging to the same original categorical
# feature are permuted together using the same row permutation.
# Importance = baseline test R2 - R2 after permutation.

rng_perm = np.random.RandomState(SEED)

importance_rows = []

for feature, columns in feature_groups.items():

    delta_r2_values = []

    for _ in range(N_PERMUTATIONS):

        permutation = rng_perm.permutation(
            len(X_test)
        )

        X_permuted = X_test.copy()

        X_permuted.loc[:, columns] = (
            X_test.iloc[permutation][columns].to_numpy()
        )

        permuted_prediction = model.predict(
            X_permuted
        )

        permuted_r2 = r2_score(
            y_test,
            permuted_prediction,
        )

        delta_r2_values.append(
            baseline_r2 - permuted_r2
        )

    importance_rows.append(
        {
            "Feature": feature,
            "Mean_Delta_R2": np.mean(delta_r2_values),
            "SD_Delta_R2": np.std(
                delta_r2_values,
                ddof=1,
            ),
            "N_permutations": N_PERMUTATIONS,
        }
    )

permutation_results = (
    pd.DataFrame(importance_rows)
    .sort_values(
        "Mean_Delta_R2",
        ascending=False,
    )
    .reset_index(drop=True)
)

permutation_results.to_csv(
    PERMUTATION_OUTPUT,
    index=False,
)

print("\nGROUPED PERMUTATION IMPORTANCE")
print("------------------------------")
print(
    permutation_results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# KEY REVIEWER-RELEVANT SUMMARY
# ============================================================

print("\nKEY FINDINGS")
print("------------")

populous = coverage_results[
    coverage_results["N"] >= 100
]

apl_correlations = populous[
    "Pearson_r_AvgPathLength"
]

print(
    "Average-path-length/PPT correlation across "
    "coverage bins with N >= 100:"
)
print(
    f"  minimum r = {apl_correlations.min():.3f}"
)
print(
    f"  maximum r = {apl_correlations.max():.3f}"
)

print("\nTop permutation-importance predictors:")
for _, row in permutation_results.head(5).iterrows():
    print(
        f"  {row['Feature']}: "
        f"mean Delta R2 = {row['Mean_Delta_R2']:.4f} "
        f"+/- {row['SD_Delta_R2']:.4f}"
    )

print("\nFiles written:")
print(f"  - {COVERAGE_OUTPUT}")
print(f"  - {PERMUTATION_OUTPUT}")

print("\nCOMMENT 5 ANALYSIS PASSED.")
