from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

SEED = 2026
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

FEATURES = [
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

def add_graph_uid(df):
    out = df.copy()
    out["GraphUID"] = (
        out["Topology"].astype(str)
        + "_N" + out["N"].astype(str)
        + "_G" + out["GraphID"].astype(str)
    )
    return out

def split_by_graph(df, seed=SEED):
    graphs = df["GraphUID"].unique()
    rng = np.random.RandomState(seed)
    shuffled = graphs.copy()
    rng.shuffle(shuffled)
    n_train = round(0.80 * len(shuffled))
    train_graphs = set(shuffled[:n_train])
    test_graphs = set(shuffled[n_train:])
    train = df[df["GraphUID"].isin(train_graphs)].copy()
    test = df[df["GraphUID"].isin(test_graphs)].copy()
    return train, test

def encode(train_df, test_df):
    Xtr = train_df[FEATURES].copy()
    Xte = test_df[FEATURES].copy()

    Xtr = pd.get_dummies(
        Xtr, columns=["Topology", "Strategy"], dtype=int
    )
    Xte = pd.get_dummies(
        Xte, columns=["Topology", "Strategy"], dtype=int
    )

    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte

def fit_workflow(train, test):
    # ---------------- Model A ----------------
    Xtr_a, Xte_a = encode(train, test)
    ytr_a = train["IsPDS"].astype(int)

    model_a = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEED,
        n_jobs=-1
    )
    model_a.fit(Xtr_a, ytr_a)

    test = test.copy()
    test["PredSuccess"] = model_a.predict(Xte_a).astype(int)
    test["SuccessProbability"] = model_a.predict_proba(Xte_a)[:, 1]

    # ---------------- Model B ----------------
    # Train only on TRUE successful configurations from training graphs.
    train_b = train[train["IsPDS"] == 1].copy()

    # Apply Model B to every test configuration routed by Model A.
    routed = test[test["PredSuccess"] == 1].copy()

    if len(routed) > 0:
        Xtr_b, Xroute_b = encode(train_b, routed)

        model_b = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=SEED,
            n_jobs=-1
        )
        model_b.fit(Xtr_b, train_b["PPT"].astype(float))
        routed["PredPPT"] = model_b.predict(Xroute_b)

        test["PredPPT"] = np.nan
        test.loc[routed.index, "PredPPT"] = routed["PredPPT"]
    else:
        test["PredPPT"] = np.nan

    return test

def safe_r2(y, pred):
    if len(y) < 2 or np.isclose(np.var(y), 0):
        return np.nan
    return r2_score(y, pred)

def summarize_group(g, label):
    y = g["IsPDS"].astype(int).to_numpy()
    p = g["PredSuccess"].astype(int).to_numpy()

    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()

    precision = precision_score(y, p, zero_division=0)
    recall = recall_score(y, p, zero_division=0)
    bal_acc = balanced_accuracy_score(y, p)
    f1 = f1_score(y, p, zero_division=0)

    routed = g[g["PredSuccess"] == 1]
    tp_rows = routed[routed["IsPDS"] == 1]
    fp_rows = routed[routed["IsPDS"] == 0]

    if len(tp_rows):
        mae = mean_absolute_error(tp_rows["PPT"], tp_rows["PredPPT"])
        rmse = np.sqrt(mean_squared_error(tp_rows["PPT"], tp_rows["PredPPT"]))
        r2 = safe_r2(tp_rows["PPT"].to_numpy(), tp_rows["PredPPT"].to_numpy())
    else:
        mae = rmse = r2 = np.nan

    fp_share = fp / (tp + fp) if (tp + fp) else np.nan
    fn_share_success = fn / (tp + fn) if (tp + fn) else np.nan

    return {
        "Group": label,
        "N": len(g),
        "TrueSuccess": int((g["IsPDS"] == 1).sum()),
        "PredictedSuccess": int((g["PredSuccess"] == 1).sum()),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Precision": precision,
        "Recall": recall,
        "BalancedAccuracy": bal_acc,
        "F1": f1,
        "FP_Share_of_PredictedSuccess": fp_share,
        "FN_Share_of_TrueSuccess": fn_share_success,
        "TP_PPT_MAE": mae,
        "TP_PPT_RMSE": rmse,
        "TP_PPT_R2": r2,
        "FP_PredPPT_Mean": fp_rows["PredPPT"].mean() if len(fp_rows) else np.nan,
        "FP_PredPPT_Median": fp_rows["PredPPT"].median() if len(fp_rows) else np.nan,
    }

def subgroup_table(test, column, ordered_values=None):
    rows = []
    values = ordered_values if ordered_values is not None else list(test[column].drop_duplicates())
    for value in values:
        g = test[test[column] == value].copy()
        if len(g):
            rows.append(summarize_group(g, str(value)))
    return pd.DataFrame(rows)

def run_primary(df):
    train, test = split_by_graph(df, SEED)
    scored = fit_workflow(train, test)

    overall = pd.DataFrame([summarize_group(scored, "Overall")])
    by_topology = subgroup_table(scored, "Topology", ["ER", "BA", "WS"])

    fraction_col = "RequestedFraction" if "RequestedFraction" in scored.columns else "ActualFraction"
    by_fraction = subgroup_table(scored, fraction_col, [0.05, 0.10, 0.15])

    by_strategy = subgroup_table(
        scored, "Strategy", ["random", "high-degree", "distributed"]
    )

    return scored, overall, by_topology, by_fraction, by_strategy

def strategy_balanced_datasets(df):
    # Five equal-strategy datasets:
    # each keeps one of the five random configurations per graph/fraction
    # together with the one high-degree and one distributed configuration.
    out = []
    for random_id in [1, 2, 3, 4, 5]:
        keep = (
            ((df["Strategy"] == "random") & (df["ConfigID"] == random_id))
            | (df["Strategy"] == "high-degree")
            | (df["Strategy"] == "distributed")
        )
        d = df[keep].copy()
        out.append((random_id, d))
    return out

def run_balanced(df):
    rows = []
    for random_id, d in strategy_balanced_datasets(df):
        train, test = split_by_graph(d, SEED)
        scored = fit_workflow(train, test)
        s = summarize_group(scored, f"Balanced_{random_id}")
        s["RandomConfigID"] = random_id
        rows.append(s)

    detail = pd.DataFrame(rows)

    metrics = [
        "Precision", "Recall", "BalancedAccuracy", "F1",
        "FP_Share_of_PredictedSuccess",
        "FN_Share_of_TrueSuccess",
        "TP_PPT_MAE", "TP_PPT_RMSE", "TP_PPT_R2"
    ]

    summary_rows = []
    for m in metrics:
        summary_rows.append({
            "Metric": m,
            "Mean": detail[m].mean(),
            "SD": detail[m].std(ddof=1)
        })
    summary = pd.DataFrame(summary_rows)

    return detail, summary

# ============================================================
# Main
# ============================================================

df = pd.read_csv(INPUT_FILE)
df = add_graph_uid(df)

# Treat non-finite PPT for failures as undefined, not as a finite target.
if "PPT" in df.columns:
    df.loc[df["IsPDS"] == 0, "PPT"] = np.nan

scored, overall, by_topology, by_fraction, by_strategy = run_primary(df)
balanced_detail, balanced_summary = run_balanced(df)

overall.to_csv("two_part_workflow_overall.csv", index=False)
by_topology.to_csv("two_part_workflow_by_topology.csv", index=False)
by_fraction.to_csv("two_part_workflow_by_fraction.csv", index=False)
by_strategy.to_csv("two_part_workflow_by_strategy.csv", index=False)
balanced_detail.to_csv("two_part_workflow_strategy_balanced_detail.csv", index=False)
balanced_summary.to_csv("two_part_workflow_strategy_balanced_summary.csv", index=False)
scored.to_csv("two_part_workflow_primary_scored_test.csv", index=False)

print("\nTWO-PART WORKFLOW: OVERALL")
print("==========================")
print(overall.to_string(index=False))

print("\nBY TOPOLOGY")
print("===========")
print(by_topology.to_string(index=False))

print("\nBY FRACTION")
print("===========")
print(by_fraction.to_string(index=False))

print("\nBY STRATEGY")
print("===========")
print(by_strategy.to_string(index=False))

print("\nEQUAL-STRATEGY SENSITIVITY: SUMMARY")
print("===================================")
print(balanced_summary.to_string(index=False))

print("\nInterpretation:")
print("- Model B regression metrics are computed only for TRUE POSITIVES routed by Model A.")
print("- False positives are counted separately because their true PPT is undefined.")
print("- FP_PredPPT_Mean/Median show the finite PPT that the workflow would incorrectly issue for failed configurations.")
print("- False negatives are successful configurations for which the workflow withholds a PPT prediction.")
print("\nSaved CSV files:")
print("  two_part_workflow_overall.csv")
print("  two_part_workflow_by_topology.csv")
print("  two_part_workflow_by_fraction.csv")
print("  two_part_workflow_by_strategy.csv")
print("  two_part_workflow_strategy_balanced_detail.csv")
print("  two_part_workflow_strategy_balanced_summary.csv")
print("  two_part_workflow_primary_scored_test.csv")
