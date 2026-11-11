from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================
# Strategy-balanced sensitivity analysis
# Reviewer 1, Comment 2
#
# For each graph and requested fraction:
#   1 random configuration + 1 high-degree + 1 distributed
# The analysis is repeated five times so that each random
# replicate (ConfigID = 1,...,5) is used once.
#
# Model A uses all configurations.
# Model B and its ablation analysis use successful PDS cases only.
# Graph-level 80/20 splitting and RF settings match the manuscript.
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
SEED = 2026

df = pd.read_csv(INPUT_FILE)

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

BASELINE_FEATURES = [
    "N",
    "ActualFraction",
    "C0",
]

STRUCTURAL_FEATURES = [
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

ABLATION_MODELS = {
    "C0 only": ["C0"],
    "Baseline": ["N", "ActualFraction", "C0"],
    "Graph structure": [
        "N", "AvgDegree", "Density", "AvgClustering",
        "Diameter", "AvgPathLength", "Topology"
    ],
    "Candidate-set focused": [
        "N", "Strategy", "ActualFraction", "MeanDegreeS",
        "MeanPairDistanceS", "MinPairDistanceS", "C0"
    ],
    "Full without C0": [
        "N", "AvgDegree", "Density", "AvgClustering",
        "Diameter", "AvgPathLength", "Topology", "Strategy",
        "ActualFraction", "MeanDegreeS", "MeanPairDistanceS",
        "MinPairDistanceS"
    ],
    "Full structural": STRUCTURAL_FEATURES,
}


def graph_split(data):
    graphs = data["GraphUID"].unique().copy()
    rng = np.random.RandomState(SEED)
    rng.shuffle(graphs)

    n_train = round(0.80 * len(graphs))
    train_graphs = set(graphs[:n_train])
    test_graphs = set(graphs[n_train:])

    train = data[data["GraphUID"].isin(train_graphs)].copy()
    test = data[data["GraphUID"].isin(test_graphs)].copy()

    assert set(train["GraphUID"]).isdisjoint(set(test["GraphUID"]))
    return train, test


def encode(train, test, features):
    X_train = train[features].copy()
    X_test = test[features].copy()

    categorical = [
        c for c in ["Topology", "Strategy"]
        if c in features
    ]

    if categorical:
        X_train = pd.get_dummies(
            X_train, columns=categorical, dtype=int
        )
        X_test = pd.get_dummies(
            X_test, columns=categorical, dtype=int
        )
        X_test = X_test.reindex(
            columns=X_train.columns, fill_value=0
        )

    return X_train, X_test


model_a_results = []
model_b_results = []
ablation_results = []

for random_rep in range(1, 6):

    # --------------------------------------------------------
    # Balanced dataset: 1 random + 1 high-degree + 1 distributed
    # per graph and requested fraction.
    # --------------------------------------------------------
    balanced = df[
        (df["Strategy"] != "random")
        | (
            (df["Strategy"] == "random")
            & (df["ConfigID"] == random_rep)
        )
    ].copy()

    counts = balanced["Strategy"].value_counts()
    assert counts["random"] == 2700
    assert counts["high-degree"] == 2700
    assert counts["distributed"] == 2700
    assert len(balanced) == 8100

    # ========================================================
    # MODEL A
    # ========================================================
    train, test = graph_split(balanced)

    y_train = train["IsPDS"].astype(int)
    y_test = test["IsPDS"].astype(int)

    # Baseline
    baseline = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    baseline.fit(train[BASELINE_FEATURES], y_train)

    pred_b = baseline.predict(test[BASELINE_FEATURES])
    prob_b = baseline.predict_proba(
        test[BASELINE_FEATURES]
    )[:, 1]

    # Structural
    X_train_s, X_test_s = encode(
        train, test, STRUCTURAL_FEATURES
    )

    structural = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    structural.fit(X_train_s, y_train)

    pred_s = structural.predict(X_test_s)
    prob_s = structural.predict_proba(X_test_s)[:, 1]

    for name, pred, prob in [
        ("Baseline", pred_b, prob_b),
        ("Structural", pred_s, prob_s),
    ]:
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        model_a_results.append({
            "RandomReplicate": random_rep,
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Recall": recall_score(
                y_test, pred, zero_division=0
            ),
            "Specificity": tn / (tn + fp),
            "F1": f1_score(
                y_test, pred, zero_division=0
            ),
            "BalancedAccuracy": balanced_accuracy_score(
                y_test, pred
            ),
            "ROC_AUC": roc_auc_score(y_test, prob),
            "AP": average_precision_score(y_test, prob),
        })

    # ========================================================
    # MODEL B
    # ========================================================
    successful = balanced[
        balanced["IsPDS"] == 1
    ].copy()

    train_b, test_b = graph_split(successful)

    y_train_b = train_b["PPT"].astype(float)
    y_test_b = test_b["PPT"].astype(float)

    # Baseline
    baseline_b = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    baseline_b.fit(
        train_b[BASELINE_FEATURES], y_train_b
    )
    pred_baseline_b = baseline_b.predict(
        test_b[BASELINE_FEATURES]
    )

    # Structural
    X_train_b, X_test_b = encode(
        train_b, test_b, STRUCTURAL_FEATURES
    )

    structural_b = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1,
    )
    structural_b.fit(X_train_b, y_train_b)
    pred_structural_b = structural_b.predict(X_test_b)

    for name, pred in [
        ("Baseline", pred_baseline_b),
        ("Structural", pred_structural_b),
    ]:
        model_b_results.append({
            "RandomReplicate": random_rep,
            "Model": name,
            "MAE": mean_absolute_error(y_test_b, pred),
            "RMSE": np.sqrt(
                mean_squared_error(y_test_b, pred)
            ),
            "R2": r2_score(y_test_b, pred),
        })

    # ========================================================
    # MODEL B ABLATION
    # ========================================================
    for model_name, features in ABLATION_MODELS.items():

        X_train_a, X_test_a = encode(
            train_b, test_b, features
        )

        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X_train_a, y_train_b)
        pred = model.predict(X_test_a)

        ablation_results.append({
            "RandomReplicate": random_rep,
            "FeatureSet": model_name,
            "MAE": mean_absolute_error(y_test_b, pred),
            "RMSE": np.sqrt(
                mean_squared_error(y_test_b, pred)
            ),
            "R2": r2_score(y_test_b, pred),
        })


# ============================================================
# Save replicate-level results
# ============================================================
model_a_df = pd.DataFrame(model_a_results)
model_b_df = pd.DataFrame(model_b_results)
ablation_df = pd.DataFrame(ablation_results)

model_a_df.to_csv(
    "strategy_balanced_model_A_replicates.csv", index=False
)
model_b_df.to_csv(
    "strategy_balanced_model_B_replicates.csv", index=False
)
ablation_df.to_csv(
    "strategy_balanced_ablation_replicates.csv", index=False
)

# ============================================================
# Summaries: mean and SAMPLE SD (ddof=1)
# ============================================================
def summarize(data, group_col, metrics):
    rows = []
    for group, g in data.groupby(group_col):
        row = {group_col: group}
        for metric in metrics:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_SD"] = g[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


model_a_summary = summarize(
    model_a_df,
    "Model",
    [
        "Accuracy", "Precision", "Recall",
        "Specificity", "F1", "BalancedAccuracy",
        "ROC_AUC", "AP"
    ],
)

model_b_summary = summarize(
    model_b_df,
    "Model",
    ["MAE", "RMSE", "R2"],
)

ablation_summary = summarize(
    ablation_df,
    "FeatureSet",
    ["MAE", "RMSE", "R2"],
)

model_a_summary.to_csv(
    "strategy_balanced_model_A_summary.csv", index=False
)
model_b_summary.to_csv(
    "strategy_balanced_model_B_summary.csv", index=False
)
ablation_summary.to_csv(
    "strategy_balanced_ablation_summary.csv", index=False
)

print("\nMODEL A: strategy-balanced sensitivity analysis")
print(model_a_summary.round(4).to_string(index=False))

print("\nMODEL B: strategy-balanced sensitivity analysis")
print(model_b_summary.round(4).to_string(index=False))

print("\nMODEL B: strategy-balanced ablation")
print(ablation_summary.round(4).to_string(index=False))
