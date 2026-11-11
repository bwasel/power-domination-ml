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
# MODEL B - ABLATION STUDY
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
SEED = 2026

df = pd.read_csv(INPUT_FILE)

# PPT is defined only for successful PDS cases
df = df[df["IsPDS"] == 1].copy()

# Unique network identifier
df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N"
    + df["N"].astype(str)
    + "_G"
    + df["GraphID"].astype(str)
)

print("\nMODEL B - ABLATION STUDY")
print("========================")

print(f"Successful PDS cases: {len(df)}")
print(
    f"Independent networks: "
    f"{df['GraphUID'].nunique()}"
)

# ============================================================
# SAME GRAPH-LEVEL SPLIT AS ORIGINAL MODEL B
# ============================================================

graphs = df["GraphUID"].unique()

rng = np.random.RandomState(SEED)
rng.shuffle(graphs)

split_index = round(0.80 * len(graphs))

train_graphs = graphs[:split_index]
test_graphs = graphs[split_index:]

train = df[
    df["GraphUID"].isin(train_graphs)
].copy()

test = df[
    df["GraphUID"].isin(test_graphs)
].copy()

print(f"Training networks: {len(train_graphs)}")
print(f"Testing networks:  {len(test_graphs)}")

print(f"Training rows: {len(train)}")
print(f"Testing rows:  {len(test)}")

leakage = set(train_graphs).intersection(
    set(test_graphs)
)

if len(leakage) == 0:
    print("Graph leakage check: PASSED")
else:
    print("Graph leakage check: FAILED")


# ============================================================
# FEATURE GROUPS
# ============================================================

# Model 1:
# Initial coverage alone
models = {
    "C0_only": [
        "C0"
    ],

    # Model 2:
    # Original baseline
    "Baseline": [
        "N",
        "ActualFraction",
        "C0"
    ],

    # Model 3:
    # Graph structural properties only
    "GraphStructure": [
        "N",
        "AvgDegree",
        "Density",
        "AvgClustering",
        "Diameter",
        "AvgPathLength",
        "Topology"
    ],

    # Model 4:
    # Initial-set structural properties only
    "SetStructure": [
        "N",
        "Strategy",
        "ActualFraction",
        "MeanDegreeS",
        "MeanPairDistanceS",
        "MinPairDistanceS",
        "C0"
    ],

    # Model 5:
    # Full structural model WITHOUT C0
    "Full_without_C0": [
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

    # Model 6:
    # Full structural model
    "Full_Structural": [
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

y_train = train["PPT"].astype(float)
y_test = test["PPT"].astype(float)

results = []

# ============================================================
# RUN ALL ABLATION MODELS
# ============================================================

for model_name, features in models.items():

    print("\n----------------------------------------")
    print(model_name)
    print("----------------------------------------")

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

    model = RandomForestRegressor(
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

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    results.append({
        "Model": model_name,
        "NumFeaturesAfterEncoding":
            X_train.shape[1],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

full_r2 = results_df.loc[
    results_df["Model"] == "Full_Structural",
    "R2"
].iloc[0]

results_df["R2GapFromFull"] = (
    full_r2 - results_df["R2"]
)

results_df.to_csv(
    "model_B_ablation_results.csv",
    index=False
)

print("\n\nABLATION SUMMARY")
print("================")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nResults saved to:")
print("model_B_ablation_results.csv")

print(
    "\nMODEL B ABLATION STUDY PASSED."
)