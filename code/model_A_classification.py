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
    confusion_matrix
)

# ============================================================
# PDS Propagation Dynamics
# Model A - Classification
#
# Objective:
# Predict whether a candidate set S is a valid PDS.
#
# IMPORTANT:
# Train/test splitting is performed at GRAPH LEVEL
# to prevent data leakage.
# ============================================================

SEED = 2026
np.random.seed(SEED)

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

# ============================================================
# 1. Load dataset
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nMODEL A - PDS SUCCESS CLASSIFICATION")
print("====================================\n")

print(f"Total observations: {len(df)}")
print(f"Successful PDS:     {int(df['IsPDS'].sum())}")
print(f"Failed PDS:         {int((df['IsPDS'] == 0).sum())}\n")

# ============================================================
# 2. Create unique graph identifier
#
# GraphID alone is not unique because it restarts
# for each topology and network size.
# ============================================================

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N"
    + df["N"].astype(str)
    + "_G"
    + df["GraphID"].astype(str)
)

unique_graphs = df["GraphUID"].unique()

print(f"Independent networks: {len(unique_graphs)}\n")

# ============================================================
# 3. Graph-level Train/Test split
#
# 80% networks -> training
# 20% networks -> testing
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

# Leakage validation

overlap = set(train["GraphUID"]).intersection(set(test["GraphUID"]))

assert len(overlap) == 0, \
    "DATA LEAKAGE: A graph occurs in both Train and Test."

print("Leakage check: PASSED\n")

# ============================================================
# 4. Response variable
# ============================================================

y_train = train["IsPDS"].astype(int)
y_test = test["IsPDS"].astype(int)

# ============================================================
# 5. BASELINE MODEL
#
# Basic information only:
# N + fraction of selected vertices + initial coverage C0
# ============================================================

baseline_features = [
    "N",
    "ActualFraction",
    "C0"
]

X_train_baseline = train[baseline_features]
X_test_baseline = test[baseline_features]

print("Training BASELINE model...")

baseline_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

baseline_model.fit(X_train_baseline, y_train)

y_pred_baseline = baseline_model.predict(X_test_baseline)

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

X_train_structural = train[structural_features].copy()
X_test_structural = test[structural_features].copy()

# One-hot encode categorical predictors.
# Encoding is learned from TRAINING data only.
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

# Force test predictors to have exactly the same columns
# and order as the training predictors.
X_test_structural = X_test_structural.reindex(
    columns=X_train_structural.columns,
    fill_value=0
)

print("Training STRUCTURAL model...")

structural_model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

structural_model.fit(X_train_structural, y_train)

y_pred_structural = structural_model.predict(X_test_structural)

# ============================================================
# 7. Evaluation function
# ============================================================

def evaluate_model(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1Score": f1,
        "BalancedAccuracy": balanced_accuracy
    }


baseline_results = evaluate_model(
    y_test,
    y_pred_baseline
)

structural_results = evaluate_model(
    y_test,
    y_pred_structural
)

# ============================================================
# 8. Print results
# ============================================================

def print_results(name, r):

    print(f"\n{name}")
    print("-" * len(name))

    print(f"Accuracy:           {r['Accuracy']:.4f}")
    print(f"Precision:          {r['Precision']:.4f}")
    print(f"Recall/Sensitivity: {r['Recall']:.4f}")
    print(f"Specificity:        {r['Specificity']:.4f}")
    print(f"F1-score:           {r['F1Score']:.4f}")
    print(f"Balanced Accuracy:  {r['BalancedAccuracy']:.4f}")

    print("\nConfusion Matrix")
    print(f"TP = {r['TP']} | FP = {r['FP']}")
    print(f"FN = {r['FN']} | TN = {r['TN']}")


print("\nMODEL PERFORMANCE")
print("=================")

print_results(
    "BASELINE MODEL",
    baseline_results
)

print_results(
    "STRUCTURAL MODEL",
    structural_results
)

# ============================================================
# 9. Improvement
# ============================================================

print("\nSTRUCTURAL IMPROVEMENT OVER BASELINE")
print("------------------------------------")

print(
    f"Accuracy improvement:          "
    f"{structural_results['Accuracy'] - baseline_results['Accuracy']:+.4f}"
)

print(
    f"F1 improvement:                "
    f"{structural_results['F1Score'] - baseline_results['F1Score']:+.4f}"
)

print(
    f"Balanced Accuracy improvement: "
    f"{structural_results['BalancedAccuracy'] - baseline_results['BalancedAccuracy']:+.4f}"
)

# ============================================================
# 10. Save performance summary
# ============================================================

performance_table = pd.DataFrame([
    {
        "Model": "Baseline",
        "Accuracy": baseline_results["Accuracy"],
        "Precision": baseline_results["Precision"],
        "Recall": baseline_results["Recall"],
        "Specificity": baseline_results["Specificity"],
        "F1Score": baseline_results["F1Score"],
        "BalancedAccuracy": baseline_results["BalancedAccuracy"]
    },
    {
        "Model": "Structural",
        "Accuracy": structural_results["Accuracy"],
        "Precision": structural_results["Precision"],
        "Recall": structural_results["Recall"],
        "Specificity": structural_results["Specificity"],
        "F1Score": structural_results["F1Score"],
        "BalancedAccuracy": structural_results["BalancedAccuracy"]
    }
])

performance_table.to_csv(
    "model_A_classification_performance.csv",
    index=False
)

# ============================================================
# 11. Save test predictions
# ============================================================

prediction_table = pd.DataFrame({
    "GraphUID": test["GraphUID"].values,
    "Topology": test["Topology"].values,
    "N": test["N"].values,
    "Strategy": test["Strategy"].values,
    "RequestedFraction": test["RequestedFraction"].values,
    "C0": test["C0"].values,
    "ActualIsPDS": y_test.values,
    "BaselinePrediction": y_pred_baseline,
    "StructuralPrediction": y_pred_structural
})

prediction_table.to_csv(
    "model_A_test_predictions.csv",
    index=False
)

print("\nResults saved to:")
print("model_A_classification_performance.csv")
print("model_A_test_predictions.csv")

print("\nMODEL A CLASSIFICATION PASSED.")