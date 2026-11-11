from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ============================================================
# Explainable AI for Model B
# SHAP analysis of Power Propagation Time (PPT)
# ============================================================

SEED = 2026
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

# ============================================================
# 1. Load data and retain successful PDS cases
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df[df["IsPDS"] == 1].copy()

print("\nMODEL B - SHAP EXPLAINABILITY")
print("=============================\n")

print(f"Successful PDS cases: {len(df)}")

# ============================================================
# 2. Create graph identifier
# ============================================================

df["GraphUID"] = (
    df["Topology"].astype(str)
    + "_N" + df["N"].astype(str)
    + "_G" + df["GraphID"].astype(str)
)

unique_graphs = df["GraphUID"].unique().copy()

rng = np.random.RandomState(SEED)
rng.shuffle(unique_graphs)

n_train = round(0.80 * len(unique_graphs))

train_graphs = set(unique_graphs[:n_train])
test_graphs = set(unique_graphs[n_train:])

train = df[df["GraphUID"].isin(train_graphs)].copy()
test = df[df["GraphUID"].isin(test_graphs)].copy()

overlap = set(train["GraphUID"]).intersection(
    set(test["GraphUID"])
)

assert len(overlap) == 0

print(f"Training networks: {train['GraphUID'].nunique()}")
print(f"Testing networks:  {test['GraphUID'].nunique()}")
print(f"Training rows:     {len(train)}")
print(f"Testing rows:      {len(test)}")
print("Leakage check: PASSED\n")

# ============================================================
# 3. Structural predictors
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
y_test = test["PPT"].astype(float)

# ============================================================
# 4. One-hot encoding
# ============================================================

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

print(f"Encoded predictors: {X_train.shape[1]}")
print("\nPredictor columns:")

for col in X_train.columns:
    print(f"  {col}")

# ============================================================
# 5. Train structural Random Forest
# ============================================================

print("\nTraining structural Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")

# ============================================================
# 6. SHAP values
#
# Explain TEST observations only.
# ============================================================

print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)

print("SHAP calculation completed.")
print(f"SHAP matrix shape: {shap_values.shape}")

# ============================================================
# 7. Global SHAP importance
# Mean absolute SHAP value
# ============================================================

mean_abs_shap = np.abs(shap_values).mean(axis=0)

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "MeanAbsSHAP": mean_abs_shap
})

importance = importance.sort_values(
    "MeanAbsSHAP",
    ascending=False
).reset_index(drop=True)

print("\nGLOBAL SHAP FEATURE IMPORTANCE")
print("==============================")

print(
    importance.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

importance.to_csv(
    "model_B_shap_feature_importance.csv",
    index=False
)

# ============================================================
# 8. SHAP bar plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    plot_type="bar",
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "model_B_shap_bar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 9. SHAP beeswarm summary plot
# ============================================================

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    "model_B_shap_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# 10. Save SHAP values for reproducibility
# ============================================================

shap_table = pd.DataFrame(
    shap_values,
    columns=X_test.columns
)

shap_table.insert(
    0,
    "GraphUID",
    test["GraphUID"].values
)

shap_table.insert(
    1,
    "ActualPPT",
    y_test.values
)

shap_table.to_csv(
    "model_B_shap_values.csv",
    index=False
)

print("\nFiles saved:")
print("model_B_shap_feature_importance.csv")
print("model_B_shap_values.csv")
print("model_B_shap_bar.png")
print("model_B_shap_beeswarm.png")

print("\nMODEL B SHAP ANALYSIS PASSED.")