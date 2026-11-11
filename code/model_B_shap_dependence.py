from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# Model B - SHAP Dependence Analysis
# ============================================================

SEED = 2026
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

FEATURES_TO_PLOT = [
    "C0",
    "AvgPathLength",
    "AvgClustering",
    "MeanPairDistanceS",
    "MeanDegreeS"
]

# ============================================================
# 1. Load successful PDS cases
# ============================================================

df = pd.read_csv(INPUT_FILE)
df = df[df["IsPDS"] == 1].copy()

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

# ============================================================
# 2. Same graph-level split used previously
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
    set(train["GraphUID"]).intersection(
        set(test["GraphUID"])
    )
) == 0

print("\nMODEL B - SHAP DEPENDENCE ANALYSIS")
print("==================================\n")

print(f"Training rows: {len(train)}")
print(f"Testing rows:  {len(test)}")
print("Leakage check: PASSED")

# ============================================================
# 3. Features
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
    "C0"
]

X_train = train[features].copy()
X_test = test[features].copy()

y_train = train["PPT"].astype(float)

# One-hot encoding

X_train = pd.get_dummies(
    X_train,
    columns=["Topology", "Strategy"],
    dtype=int
)

X_test = pd.get_dummies(
    X_test,
    columns=["Topology", "Strategy"],
    dtype=int
)

X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# ============================================================
# 4. Train same Random Forest
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# 5. Calculate SHAP values
# ============================================================

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print("SHAP calculation completed.")

# ============================================================
# 6. Create dependence plots
#
# interaction_index=None is intentional:
# we first want the clean marginal relationship.
# ============================================================

for feature in FEATURES_TO_PLOT:

    print(f"Creating dependence plot: {feature}")

    plt.figure(figsize=(7, 5))

    shap.dependence_plot(
        feature,
        shap_values,
        X_test,
        interaction_index=None,
        show=False
    )

    plt.xlabel(feature)
    plt.ylabel("SHAP value for predicted PPT")
    plt.title(f"SHAP Dependence: {feature}")

    plt.tight_layout()

    filename = (
        "shap_dependence_"
        + feature
        + ".png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# 7. Save raw dependence data
# ============================================================

for feature in FEATURES_TO_PLOT:

    idx = list(X_test.columns).index(feature)

    output = pd.DataFrame({
        feature: X_test[feature].values,
        "SHAP": shap_values[:, idx],
        "ActualPPT": test["PPT"].values,
        "Topology": test["Topology"].values,
        "Strategy": test["Strategy"].values
    })

    output.to_csv(
        f"shap_dependence_{feature}.csv",
        index=False
    )

print("\nSaved dependence plots and CSV files for:")

for feature in FEATURES_TO_PLOT:
    print(f"  {feature}")

print("\nMODEL B SHAP DEPENDENCE ANALYSIS PASSED.")