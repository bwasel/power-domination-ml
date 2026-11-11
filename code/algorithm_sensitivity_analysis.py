"""Reviewer 2, Comment 5: algorithm-sensitivity analysis.

Compares baseline vs structural feature sets across representative learning
families using the same graph-level 80/20 held-out split as the manuscript.
Hyperparameters are selected using 3-fold GroupKFold within training graphs
only; held-out test graphs are never used for model selection.

Model A: Logistic Regression, Random Forest, Histogram Gradient Boosting.
Model B: Ridge Regression, Random Forest, Histogram Gradient Boosting.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    roc_auc_score, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

SEED = 2026
INPUT = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
OUTDIR = Path("reviewer2_comment5_outputs")
OUTDIR.mkdir(exist_ok=True)

BASELINE = ["N", "ActualFraction", "C0"]
STRUCTURAL = [
    "N", "AvgDegree", "Density", "AvgClustering", "Diameter",
    "AvgPathLength", "Topology", "Strategy", "ActualFraction",
    "MeanDegreeS", "MeanPairDistanceS", "MinPairDistanceS", "C0",
]
CATEGORICAL = ["Topology", "Strategy"]
CV = GroupKFold(n_splits=3)


def add_uid(df):
    df = df.copy()
    df["GraphUID"] = (
        df["Topology"].astype(str) + "_N" + df["N"].astype(str)
        + "_G" + df["GraphID"].astype(str)
    )
    return df


def graph_split(data):
    graphs = data["GraphUID"].unique().copy()
    rng = np.random.RandomState(SEED)
    rng.shuffle(graphs)
    n_train = round(0.80 * len(graphs))
    tr, te = set(graphs[:n_train]), set(graphs[n_train:])
    train = data[data["GraphUID"].isin(tr)].copy()
    test = data[data["GraphUID"].isin(te)].copy()
    assert set(train["GraphUID"]).isdisjoint(set(test["GraphUID"]))
    return train, test


def prep(features, scale=False):
    cats = [c for c in CATEGORICAL if c in features]
    nums = [c for c in features if c not in cats]
    blocks = [("num", StandardScaler() if scale else "passthrough", nums)]
    if cats:
        blocks.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    return ColumnTransformer(blocks)


def cls_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "F1": f1_score(y, pred),
        "BalancedAccuracy": balanced_accuracy_score(y, pred),
        "ROC_AUC": roc_auc_score(y, prob),
        "AveragePrecision": average_precision_score(y, prob),
    }


def reg_metrics(model, X, y):
    pred = model.predict(X)
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
    }


def run_A(train, test, features, feature_set, alg):
    if alg == "Logistic Regression":
        estimator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=SEED)
        grid = {"model__C": [0.1, 1, 10]}
        scale = True
    elif alg == "Random Forest":
        estimator = RandomForestClassifier(random_state=SEED, n_jobs=-1)
        grid = {"model__n_estimators": [200, 500], "model__min_samples_leaf": [1, 5, 10]}
        scale = False
    elif alg == "Histogram Gradient Boosting":
        estimator = HistGradientBoostingClassifier(random_state=SEED)
        grid = {
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [15, 31],
        }
        scale = False
    else:
        raise ValueError(alg)

    pipe = Pipeline([("prep", prep(features, scale)), ("model", estimator)])
    gs = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=CV, n_jobs=1, refit=True)
    gs.fit(train[features], train["IsPDS"].astype(int), groups=train["GraphUID"])
    return {
        "Algorithm": alg,
        "FeatureSet": feature_set,
        "BestParams": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()}, sort_keys=True),
        "CV": gs.best_score_,
        **cls_metrics(gs.best_estimator_, test[features], test["IsPDS"].astype(int)),
    }


def run_B(train, test, features, feature_set, alg):
    if alg == "Ridge Regression":
        estimator = Ridge()
        grid = {"model__alpha": [0.1, 1, 10]}
        scale = True
    elif alg == "Random Forest":
        estimator = RandomForestRegressor(random_state=SEED, n_jobs=-1)
        grid = {"model__n_estimators": [200, 500], "model__min_samples_leaf": [1, 5, 10]}
        scale = False
    elif alg == "Histogram Gradient Boosting":
        estimator = HistGradientBoostingRegressor(random_state=SEED)
        grid = {
            "model__max_iter": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_leaf_nodes": [15, 31],
        }
        scale = False
    else:
        raise ValueError(alg)

    pipe = Pipeline([("prep", prep(features, scale)), ("model", estimator)])
    gs = GridSearchCV(pipe, grid, scoring="neg_root_mean_squared_error", cv=CV, n_jobs=1, refit=True)
    gs.fit(train[features], train["PPT"].astype(float), groups=train["GraphUID"])
    return {
        "Algorithm": alg,
        "FeatureSet": feature_set,
        "BestParams": json.dumps({k.replace("model__", ""): v for k, v in gs.best_params_.items()}, sort_keys=True),
        "CV": gs.best_score_,
        **reg_metrics(gs.best_estimator_, test[features], test["PPT"].astype(float)),
    }


df = add_uid(pd.read_csv(INPUT))
train_A, test_A = graph_split(df)
success = df[df["IsPDS"] == 1].copy()
train_B, test_B = graph_split(success)

rows_A, rows_B = [], []
for label, features in [("Baseline", BASELINE), ("Structural", STRUCTURAL)]:
    for alg in ["Logistic Regression", "Random Forest", "Histogram Gradient Boosting"]:
        rows_A.append(run_A(train_A, test_A, features, label, alg))
    for alg in ["Ridge Regression", "Random Forest", "Histogram Gradient Boosting"]:
        rows_B.append(run_B(train_B, test_B, features, label, alg))

A = pd.DataFrame(rows_A)
B = pd.DataFrame(rows_B)
A.to_csv(OUTDIR / "model_A_algorithm_sensitivity_final.csv", index=False)
B.to_csv(OUTDIR / "model_B_algorithm_sensitivity_final.csv", index=False)

print("\nMODEL A\n", A.to_string(index=False))
print("\nMODEL B\n", B.to_string(index=False))
