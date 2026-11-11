import sys
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
import networkx as nx
import matplotlib

try:
    import shap
    SHAP_VERSION = shap.__version__
except Exception:
    SHAP_VERSION = "not available"

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
PRIMARY_SPLIT_SEED = 2026
PRIMARY_RF_SEED = 2026
REPEATED_SEEDS = [101,202,303,404,505,606,707,808,909,1010]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 314159

BASE_FEATURES = ["N", "ActualFraction", "C0"]
STRUCT_FEATURES = [
    "N","AvgDegree","Density","AvgClustering","Diameter","AvgPathLength",
    "Topology","Strategy","ActualFraction","MeanDegreeS",
    "MeanPairDistanceS","MinPairDistanceS","C0"
]
CORR_FEATURES = [
    "C0","AvgPathLength","AvgClustering",
    "MeanPairDistanceS","MeanDegreeS"
]

OUT = Path(".")
rng_boot = np.random.RandomState(BOOTSTRAP_SEED)

def add_graph_uid(df):
    d = df.copy()
    d["GraphUID"] = (
        d["Topology"].astype(str)
        + "_N" + d["N"].astype(str)
        + "_G" + d["GraphID"].astype(str)
    )
    return d

def graph_split(d, seed):
    graphs = d["GraphUID"].unique().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(graphs)
    n_train = round(0.80 * len(graphs))
    tr = set(graphs[:n_train])
    te = set(graphs[n_train:])
    train = d[d["GraphUID"].isin(tr)].copy()
    test = d[d["GraphUID"].isin(te)].copy()
    return train, test

def encode(train, test, features):
    cats = [x for x in ["Topology","Strategy"] if x in features]
    Xtr = train[features].copy()
    Xte = test[features].copy()
    if cats:
        Xtr = pd.get_dummies(Xtr, columns=cats, dtype=int)
        Xte = pd.get_dummies(Xte, columns=cats, dtype=int)
        Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)
    return Xtr, Xte

def fit_A(train, test, features, rf_seed):
    Xtr, Xte = encode(train, test, features)
    model = RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        random_state=rf_seed, n_jobs=-1
    )
    model.fit(Xtr, train["IsPDS"].astype(int))
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:,1]
    return pred, prob

def fit_B(train, test, features, rf_seed):
    Xtr, Xte = encode(train, test, features)
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=5,
        random_state=rf_seed, n_jobs=-1
    )
    model.fit(Xtr, train["PPT"].astype(float))
    pred = model.predict(Xte)
    return pred

def metrics_A(y, pred, prob):
    out = {
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "BalancedAccuracy": balanced_accuracy_score(y, pred),
    }
    try:
        out["ROC_AUC"] = roc_auc_score(y, prob)
    except Exception:
        out["ROC_AUC"] = np.nan
    try:
        out["AP"] = average_precision_score(y, prob)
    except Exception:
        out["AP"] = np.nan
    return out

def metrics_B(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }

def stacked_cluster_sample(df, graph_ids):
    # Preserve multiplicity of resampled graphs.
    parts = [df[df["GraphUID"] == g] for g in graph_ids]
    return pd.concat(parts, ignore_index=True)

def percentile_ci(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    return np.percentile(x, [2.5, 97.5])

# ============================================================
# Load
# ============================================================
df = add_graph_uid(pd.read_csv(INPUT_FILE))
success = df[df["IsPDS"] == 1].copy()

# ============================================================
# A. Reproducible train/test assignments
# ============================================================
assignment_rows = []
for task, d in [("Model A", df), ("Model B", success)]:
    for seed in [PRIMARY_SPLIT_SEED] + REPEATED_SEEDS:
        train, test = graph_split(d, seed)
        tr = set(train["GraphUID"].unique())
        te = set(test["GraphUID"].unique())
        for g in d["GraphUID"].unique():
            assignment_rows.append({
                "Task": task,
                "SplitSeed": seed,
                "GraphUID": g,
                "Partition": "train" if g in tr else "test"
            })
assignments = pd.DataFrame(assignment_rows)
assignments.to_csv(OUT / "train_test_assignments.csv", index=False)

# ============================================================
# B. Primary model predictions
# ============================================================
train_a, test_a = graph_split(df, PRIMARY_SPLIT_SEED)
pred_a_b, prob_a_b = fit_A(train_a, test_a, BASE_FEATURES, PRIMARY_RF_SEED)
pred_a_s, prob_a_s = fit_A(train_a, test_a, STRUCT_FEATURES, PRIMARY_RF_SEED)

predA = test_a[["GraphUID","Topology","N","GraphID","IsPDS"]].copy()
predA["BaselinePred"] = pred_a_b
predA["BaselineProb"] = prob_a_b
predA["StructuralPred"] = pred_a_s
predA["StructuralProb"] = prob_a_s
predA.to_csv(OUT / "primary_model_A_predictions.csv", index=False)

train_b, test_b = graph_split(success, PRIMARY_SPLIT_SEED)
pred_b_b = fit_B(train_b, test_b, BASE_FEATURES, PRIMARY_RF_SEED)
pred_b_s = fit_B(train_b, test_b, STRUCT_FEATURES, PRIMARY_RF_SEED)

predB = test_b[["GraphUID","Topology","N","GraphID","PPT"]].copy()
predB["BaselinePredPPT"] = pred_b_b
predB["StructuralPredPPT"] = pred_b_s
predB.to_csv(OUT / "primary_model_B_predictions.csv", index=False)

# ============================================================
# C. Graph-cluster bootstrap for model comparisons
# ============================================================
comparison_rows = []

# Point estimates
ma_b = metrics_A(test_a["IsPDS"], pred_a_b, prob_a_b)
ma_s = metrics_A(test_a["IsPDS"], pred_a_s, prob_a_s)
for metric in ma_b:
    comparison_rows.append({
        "Task":"Model A","Metric":metric,
        "Baseline":ma_b[metric],"Structural":ma_s[metric],
        "Improvement":ma_s[metric]-ma_b[metric],
        "CI_Low":np.nan,"CI_High":np.nan
    })

mb_b = metrics_B(test_b["PPT"], pred_b_b)
mb_s = metrics_B(test_b["PPT"], pred_b_s)
for metric in mb_b:
    # Positive means structural is better.
    improvement = (
        mb_b[metric]-mb_s[metric] if metric in ["MAE","RMSE"]
        else mb_s[metric]-mb_b[metric]
    )
    comparison_rows.append({
        "Task":"Model B","Metric":metric,
        "Baseline":mb_b[metric],"Structural":mb_s[metric],
        "Improvement":improvement,
        "CI_Low":np.nan,"CI_High":np.nan
    })

bootA = {m: [] for m in ma_b}
testA_aug = test_a.copy()
testA_aug["BaselinePred"] = pred_a_b
testA_aug["BaselineProb"] = prob_a_b
testA_aug["StructuralPred"] = pred_a_s
testA_aug["StructuralProb"] = prob_a_s
graphsA = testA_aug["GraphUID"].unique()

bootB = {m: [] for m in mb_b}
testB_aug = test_b.copy()
testB_aug["BaselinePredPPT"] = pred_b_b
testB_aug["StructuralPredPPT"] = pred_b_s
graphsB = testB_aug["GraphUID"].unique()

for rep in range(BOOTSTRAP_REPS):
    sampleA = rng_boot.choice(graphsA, size=len(graphsA), replace=True)
    a = stacked_cluster_sample(testA_aug, sampleA)
    bmet = metrics_A(a["IsPDS"], a["BaselinePred"], a["BaselineProb"])
    smet = metrics_A(a["IsPDS"], a["StructuralPred"], a["StructuralProb"])
    for m in bootA:
        bootA[m].append(smet[m] - bmet[m])

    sampleB = rng_boot.choice(graphsB, size=len(graphsB), replace=True)
    b = stacked_cluster_sample(testB_aug, sampleB)
    bmet = metrics_B(b["PPT"], b["BaselinePredPPT"])
    smet = metrics_B(b["PPT"], b["StructuralPredPPT"])
    for m in bootB:
        delta = bmet[m]-smet[m] if m in ["MAE","RMSE"] else smet[m]-bmet[m]
        bootB[m].append(delta)

comp = pd.DataFrame(comparison_rows)
for i,row in comp.iterrows():
    vals = bootA[row["Metric"]] if row["Task"]=="Model A" else bootB[row["Metric"]]
    lo,hi = percentile_ci(vals)
    comp.loc[i,"CI_Low"] = lo
    comp.loc[i,"CI_High"] = hi
comp.to_csv(OUT / "cluster_bootstrap_model_comparisons.csv", index=False)

# ============================================================
# D. Graph-cluster bootstrap for descriptive correlations
# ============================================================
corr_rows = []
for topo in ["ER","BA","WS"]:
    dtop = success[success["Topology"] == topo].copy()
    graphs = dtop["GraphUID"].unique()
    for feature in CORR_FEATURES:
        point = dtop[[feature,"PPT"]].corr(method="pearson").iloc[0,1]
        boots = []
        for rep in range(BOOTSTRAP_REPS):
            sampled = rng_boot.choice(graphs, size=len(graphs), replace=True)
            b = stacked_cluster_sample(dtop, sampled)
            r = b[[feature,"PPT"]].corr(method="pearson").iloc[0,1]
            boots.append(r)
        lo,hi = percentile_ci(boots)
        corr_rows.append({
            "Topology":topo,
            "Feature":feature,
            "PearsonR":point,
            "ClusterBootstrap95CI_Low":lo,
            "ClusterBootstrap95CI_High":hi,
            "BootstrapReps":BOOTSTRAP_REPS
        })
corr_df = pd.DataFrame(corr_rows)
corr_df.to_csv(OUT / "cluster_bootstrap_correlations.csv", index=False)

# ============================================================
# E. Separate partition variability from RF-fitting variability
# ============================================================
var_rows = []

def add_var_rows(source, split_seed, rf_seed):
    # Model A
    ta, va = graph_split(df, split_seed)
    for variant,features in [("Baseline",BASE_FEATURES),("Structural",STRUCT_FEATURES)]:
        pred,prob = fit_A(ta,va,features,rf_seed)
        mm = metrics_A(va["IsPDS"],pred,prob)
        for metric,val in mm.items():
            var_rows.append({
                "VariabilitySource":source,"Task":"Model A","Variant":variant,
                "SplitSeed":split_seed,"RFSeed":rf_seed,
                "Metric":metric,"Value":val
            })

    # Model B
    tb, vb = graph_split(success, split_seed)
    for variant,features in [("Baseline",BASE_FEATURES),("Structural",STRUCT_FEATURES)]:
        pred = fit_B(tb,vb,features,rf_seed)
        mm = metrics_B(vb["PPT"],pred)
        for metric,val in mm.items():
            var_rows.append({
                "VariabilitySource":source,"Task":"Model B","Variant":variant,
                "SplitSeed":split_seed,"RFSeed":rf_seed,
                "Metric":metric,"Value":val
            })

# Change partition only; RF seed fixed.
for seed in REPEATED_SEEDS:
    add_var_rows("Partition", seed, PRIMARY_RF_SEED)

# Change RF fitting only; primary partition fixed.
for seed in REPEATED_SEEDS:
    add_var_rows("RF fitting", PRIMARY_SPLIT_SEED, seed)

var_detail = pd.DataFrame(var_rows)
var_detail.to_csv(OUT / "partition_vs_rf_variability_detail.csv", index=False)

var_summary = (
    var_detail.groupby(["VariabilitySource","Task","Variant","Metric"])["Value"]
    .agg(["mean","std"])
    .reset_index()
    .rename(columns={"mean":"Mean","std":"SD"})
)
var_summary.to_csv(OUT / "partition_vs_rf_variability_summary.csv", index=False)

# ============================================================
# F. Python environment report
# ============================================================
env = [
    f"Python: {sys.version}",
    f"Platform: {platform.platform()}",
    f"NumPy: {np.__version__}",
    f"pandas: {pd.__version__}",
    f"scikit-learn: {sklearn.__version__}",
    f"SciPy: {scipy.__version__}",
    f"NetworkX: {nx.__version__}",
    f"Matplotlib: {matplotlib.__version__}",
    f"SHAP: {SHAP_VERSION}",
]
(OUT / "python_environment.txt").write_text("\n".join(env), encoding="utf-8")

print("\nCreated:")
for f in [
    "train_test_assignments.csv",
    "primary_model_A_predictions.csv",
    "primary_model_B_predictions.csv",
    "cluster_bootstrap_model_comparisons.csv",
    "cluster_bootstrap_correlations.csv",
    "partition_vs_rf_variability_detail.csv",
    "partition_vs_rf_variability_summary.csv",
    "python_environment.txt",
]:
    print(" ", f)
