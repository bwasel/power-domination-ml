from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# ============================================================
# MODEL A - EXTENDED CLASSIFICATION EVALUATION
# ROC-AUC + PR-AUC
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
SEED = 2026

df = pd.read_csv(INPUT_FILE)

# Unique network identifier
df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N"
    + df["N"].astype(str)
    + "_G"
    + df["GraphID"].astype(str)
)

print("\nMODEL A - EXTENDED CLASSIFICATION EVALUATION")
print("============================================")

print(f"Total observations: {len(df)}")
print(f"Successful PDS: {(df['IsPDS'] == 1).sum()}")
print(f"Failed PDS: {(df['IsPDS'] == 0).sum()}")
print(f"Independent networks: {df['GraphUID'].nunique()}")

# ============================================================
# SAME GRAPH-LEVEL SPLIT
# ============================================================

graphs = df["GraphUID"].unique()

rng = np.random.RandomState(SEED)
rng.shuffle(graphs)

split_index = round(0.80 * len(graphs))

train_graphs = graphs[:split_index]
test_graphs = graphs[split_index:]

train = df[df["GraphUID"].isin(train_graphs)].copy()
test = df[df["GraphUID"].isin(test_graphs)].copy()

print(f"\nTraining networks: {len(train_graphs)}")
print(f"Testing networks:  {len(test_graphs)}")
print(f"Training rows: {len(train)}")
print(f"Testing rows:  {len(test)}")

leakage = set(train_graphs).intersection(set(test_graphs))

if len(leakage) == 0:
    print("Graph leakage check: PASSED")
else:
    print("Graph leakage check: FAILED")

# ============================================================
# FEATURES
# ============================================================

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

y_train = train["IsPDS"].astype(int)
y_test = test["IsPDS"].astype(int)

# ============================================================
# FUNCTION
# ============================================================

def run_model(name, features):

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

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    # Probability that case is a successful PDS
    probability = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        prediction
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction
    ).ravel()

    specificity = tn / (tn + fp)

    roc_auc = roc_auc_score(
        y_test,
        probability
    )

    # Average Precision is the standard summary
    # of the Precision-Recall curve
    pr_auc = average_precision_score(
        y_test,
        probability
    )

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Specificity:       {specificity:.4f}")
    print(f"F1:                {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")

    print(
        f"TP={tp}  FP={fp}  "
        f"FN={fn}  TN={tn}"
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1": f1,
        "BalancedAccuracy": balanced_accuracy,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }


# ============================================================
# RUN MODELS
# ============================================================

baseline_result = run_model(
    "Baseline",
    baseline_features
)

structural_result = run_model(
    "Structural",
    structural_features
)

results = pd.DataFrame([
    baseline_result,
    structural_result
])

results.to_csv(
    "model_A_extended_performance.csv",
    index=False
)

# ============================================================
# IMPROVEMENT
# ============================================================

print("\n\nSTRUCTURAL IMPROVEMENT OVER BASELINE")
print("====================================")

for metric in [
    "Accuracy",
    "F1",
    "BalancedAccuracy",
    "ROC_AUC",
    "PR_AUC"
]:

    improvement = (
        structural_result[metric]
        - baseline_result[metric]
    )

    print(
        f"{metric:18s}: "
        f"{improvement:+.4f}"
    )

print("\nResults saved to:")
print("model_A_extended_performance.csv")

print(
    "\nMODEL A EXTENDED "
    "CLASSIFICATION EVALUATION PASSED."
)