from pathlib import Path
import pandas as pd
import numpy as np
import shap

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

SEED = 2026
REPEATED_SEEDS = [101,202,303,404,505,606,707,808,909,1010]
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

FULL_FEATURES = [
    "N","AvgDegree","Density","AvgClustering","Diameter","AvgPathLength",
    "Topology","Strategy","ActualFraction","MeanDegreeS",
    "MeanPairDistanceS","MinPairDistanceS","C0"
]

LABEL_ABLATIONS = {
    "Full": FULL_FEATURES,
    "No topology": [x for x in FULL_FEATURES if x != "Topology"],
    "No strategy": [x for x in FULL_FEATURES if x != "Strategy"],
    "No topology or strategy": [
        x for x in FULL_FEATURES if x not in ("Topology","Strategy")
    ],
}

def graph_split(data, seed):
    graphs = data["GraphUID"].unique()
    rng = np.random.RandomState(seed)
    shuffled = graphs.copy()
    rng.shuffle(shuffled)
    n_train = round(0.80 * len(shuffled))
    train_graphs = set(shuffled[:n_train])
    test_graphs = set(shuffled[n_train:])
    train = data[data["GraphUID"].isin(train_graphs)].copy()
    test = data[data["GraphUID"].isin(test_graphs)].copy()
    return train, test

def encode(train, test, features):
    cats = [c for c in ["Topology","Strategy"] if c in features]
    X_train = train[features].copy()
    X_test = test[features].copy()
    if cats:
        X_train = pd.get_dummies(X_train, columns=cats, dtype=int)
        X_test = pd.get_dummies(X_test, columns=cats, dtype=int)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_test

df = pd.read_csv(INPUT_FILE)
df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

# ============================================================
# 1. Matched label ablations: Model A and Model B
# ============================================================

rows = []

train_a, test_a = graph_split(df, SEED)
for name, features in LABEL_ABLATIONS.items():
    Xtr, Xte = encode(train_a, test_a, features)
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(Xtr, train_a["IsPDS"].astype(int))
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:,1]
    rows.append({
        "Model": "A",
        "Variant": name,
        "Accuracy": accuracy_score(test_a["IsPDS"], pred),
        "F1": f1_score(test_a["IsPDS"], pred),
        "BalancedAccuracy": balanced_accuracy_score(test_a["IsPDS"], pred),
        "ROC_AUC": roc_auc_score(test_a["IsPDS"], prob),
        "AP": average_precision_score(test_a["IsPDS"], prob),
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
    })

success = df[df["IsPDS"] == 1].copy()
train_b, test_b = graph_split(success, SEED)

for name, features in LABEL_ABLATIONS.items():
    Xtr, Xte = encode(train_b, test_b, features)
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    model.fit(Xtr, train_b["PPT"].astype(float))
    pred = model.predict(Xte)
    rows.append({
        "Model": "B",
        "Variant": name,
        "Accuracy": np.nan,
        "F1": np.nan,
        "BalancedAccuracy": np.nan,
        "ROC_AUC": np.nan,
        "AP": np.nan,
        "MAE": mean_absolute_error(test_b["PPT"], pred),
        "RMSE": np.sqrt(mean_squared_error(test_b["PPT"], pred)),
        "R2": r2_score(test_b["PPT"], pred),
    })

ablation_df = pd.DataFrame(rows)
ablation_df.to_csv("matched_label_ablations.csv", index=False)

print("\nMATCHED LABEL ABLATIONS")
print("=======================")
print(ablation_df.to_string(index=False))

# ============================================================
# 2. SHAP ranking stability across repeated graph-level splits
#    Model B only, because SHAP interpretation in the manuscript
#    is centered on propagation-time prediction.
# ============================================================

shap_rows = []

for seed in REPEATED_SEEDS:
    print(f"\nSHAP split seed {seed} ...")

    train, test = graph_split(success, seed)
    Xtr, Xte = encode(train, test, FULL_FEATURES)

    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1
    )
    model.fit(Xtr, train["PPT"].astype(float))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        Xte,
        check_additivity=False
    )

    encoded_importance = pd.Series(
        np.abs(shap_values).mean(axis=0),
        index=Xte.columns
    )

    grouped = {}

    continuous = [
        "N","AvgDegree","Density","AvgClustering","Diameter",
        "AvgPathLength","ActualFraction","MeanDegreeS",
        "MeanPairDistanceS","MinPairDistanceS","C0"
    ]

    for feature in continuous:
        grouped[feature] = float(encoded_importance[feature])

    topology_cols = [
        c for c in encoded_importance.index
        if c.startswith("Topology_")
    ]
    strategy_cols = [
        c for c in encoded_importance.index
        if c.startswith("Strategy_")
    ]

    grouped["Topology"] = float(encoded_importance[topology_cols].sum())
    grouped["Strategy"] = float(encoded_importance[strategy_cols].sum())

    ordered = sorted(grouped, key=grouped.get, reverse=True)
    ranks = {feature: ordered.index(feature) + 1 for feature in grouped}

    for feature in grouped:
        shap_rows.append({
            "Seed": seed,
            "Feature": feature,
            "MeanAbsSHAP": grouped[feature],
            "Rank": ranks[feature],
            "TestConfigurations": len(test),
        })

shap_split_df = pd.DataFrame(shap_rows)
shap_split_df.to_csv("shap_stability_by_split.csv", index=False)

summary_rows = []
for feature, g in shap_split_df.groupby("Feature"):
    summary_rows.append({
        "Feature": feature,
        "MeanAbsSHAP_Mean": g["MeanAbsSHAP"].mean(),
        "MeanAbsSHAP_SD": g["MeanAbsSHAP"].std(ddof=1),
        "MedianRank": g["Rank"].median(),
        "MinRank": g["Rank"].min(),
        "MaxRank": g["Rank"].max(),
        "Top1_Count": int((g["Rank"] == 1).sum()),
        "Top2_Count": int((g["Rank"] <= 2).sum()),
    })

shap_summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values(["MedianRank","MeanAbsSHAP_Mean"], ascending=[True,False])
)
shap_summary_df.to_csv("shap_stability_summary.csv", index=False)

print("\nSHAP STABILITY SUMMARY")
print("======================")
print(shap_summary_df.to_string(index=False))

# ============================================================
# 3. Grouped permutation importance on the primary Model B split
#    Jointly permute conceptually related correlated predictors.
# ============================================================

Xtr, Xte = encode(train_b, test_b, FULL_FEATURES)

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)
model.fit(Xtr, train_b["PPT"].astype(float))

baseline_r2 = r2_score(test_b["PPT"], model.predict(Xte))

groups = {
    "Initial state": ["C0","ActualFraction"],
    "Size/connectivity": ["N","AvgDegree","Density"],
    "Global path structure": ["Diameter","AvgPathLength"],
    "Clustering": ["AvgClustering"],
    "Candidate-set structure": [
        "MeanDegreeS","MeanPairDistanceS","MinPairDistanceS"
    ],
    "Topology label": [
        c for c in Xte.columns if c.startswith("Topology_")
    ],
    "Strategy label": [
        c for c in Xte.columns if c.startswith("Strategy_")
    ],
}

rng = np.random.RandomState(SEED)
importance_rows = []

for group_name, group_cols in groups.items():
    drops = []

    for rep in range(30):
        perm = rng.permutation(len(Xte))
        Xperm = Xte.copy()

        # Same row permutation applied jointly to all columns in a group,
        # preserving within-group dependence.
        Xperm.loc[:, group_cols] = Xte.iloc[perm][group_cols].to_numpy()

        perm_r2 = r2_score(
            test_b["PPT"],
            model.predict(Xperm)
        )
        drops.append(baseline_r2 - perm_r2)

    importance_rows.append({
        "Group": group_name,
        "MeanDeltaR2": np.mean(drops),
        "SDDeltaR2": np.std(drops, ddof=1),
    })

grouped_df = (
    pd.DataFrame(importance_rows)
    .sort_values("MeanDeltaR2", ascending=False)
)
grouped_df.to_csv("grouped_permutation_importance.csv", index=False)

print("\nGROUPED PERMUTATION IMPORTANCE")
print("==============================")
print(f"Primary held-out R2 = {baseline_r2:.6f}\n")
print(grouped_df.to_string(index=False))

print("\nSaved:")
print("  matched_label_ablations.csv")
print("  shap_stability_by_split.csv")
print("  shap_stability_summary.csv")
print("  grouped_permutation_importance.csv")
