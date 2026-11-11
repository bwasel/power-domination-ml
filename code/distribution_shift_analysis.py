from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Feature Distribution Shift Analysis
# Focus: Why does ER+BA -> WS generalization weaken?
# ============================================================

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)
df = df[df["IsPDS"] == 1].copy()

features = [
    "C0",
    "AvgPathLength",
    "AvgClustering",
    "MeanPairDistanceS",
    "MeanDegreeS"
]

print("\nFEATURE DISTRIBUTION SHIFT ANALYSIS")
print("===================================\n")

# ============================================================
# 1. Summary by topology
# ============================================================

summary_rows = []

for topology in ["ER", "BA", "WS"]:

    d = df[df["Topology"] == topology]

    print(f"\n{topology}")
    print("=" * len(topology))
    print(f"Successful cases: {len(d)}")

    for feature in features:

        x = d[feature].dropna()

        q05 = x.quantile(0.05)
        q25 = x.quantile(0.25)
        median = x.median()
        q75 = x.quantile(0.75)
        q95 = x.quantile(0.95)

        print(
            f"\n{feature}\n"
            f"  Mean   = {x.mean():.4f}\n"
            f"  Std    = {x.std():.4f}\n"
            f"  Min    = {x.min():.4f}\n"
            f"  Q05    = {q05:.4f}\n"
            f"  Q25    = {q25:.4f}\n"
            f"  Median = {median:.4f}\n"
            f"  Q75    = {q75:.4f}\n"
            f"  Q95    = {q95:.4f}\n"
            f"  Max    = {x.max():.4f}"
        )

        summary_rows.append({
            "Topology": topology,
            "Feature": feature,
            "Mean": x.mean(),
            "Std": x.std(),
            "Min": x.min(),
            "Q05": q05,
            "Q25": q25,
            "Median": median,
            "Q75": q75,
            "Q95": q95,
            "Max": x.max()
        })

# ============================================================
# 2. ER + BA training support versus WS
# ============================================================

train = df[df["Topology"].isin(["ER", "BA"])]
ws = df[df["Topology"] == "WS"]

print("\n\nER + BA TRAINING SUPPORT vs WS")
print("==============================")

overlap_rows = []

for feature in features:

    train_values = train[feature].dropna()
    ws_values = ws[feature].dropna()

    train_min = train_values.min()
    train_max = train_values.max()

    outside = (
        (ws_values < train_min) |
        (ws_values > train_max)
    )

    outside_pct = 100 * outside.mean()

    # More robust central support:
    train_q05 = train_values.quantile(0.05)
    train_q95 = train_values.quantile(0.95)

    outside_central = (
        (ws_values < train_q05) |
        (ws_values > train_q95)
    )

    outside_central_pct = (
        100 * outside_central.mean()
    )

    print(f"\n{feature}")
    print(
        f"ER+BA full range: "
        f"[{train_min:.4f}, {train_max:.4f}]"
    )
    print(
        f"ER+BA 5-95% range: "
        f"[{train_q05:.4f}, {train_q95:.4f}]"
    )
    print(
        f"WS outside full range: "
        f"{outside_pct:.2f}%"
    )
    print(
        f"WS outside ER+BA 5-95% range: "
        f"{outside_central_pct:.2f}%"
    )

    overlap_rows.append({
        "Feature": feature,
        "TrainMin": train_min,
        "TrainMax": train_max,
        "TrainQ05": train_q05,
        "TrainQ95": train_q95,
        "WSOutsideFullRangePercent":
            outside_pct,
        "WSOutsideCentral90Percent":
            outside_central_pct
    })

# ============================================================
# 3. Save
# ============================================================

summary_df = pd.DataFrame(summary_rows)
overlap_df = pd.DataFrame(overlap_rows)

summary_df.to_csv(
    "feature_distribution_by_topology.csv",
    index=False
)

overlap_df.to_csv(
    "ws_distribution_shift.csv",
    index=False
)

print("\n\nFiles saved:")
print("feature_distribution_by_topology.csv")
print("ws_distribution_shift.csv")

print(
    "\nFEATURE DISTRIBUTION SHIFT "
    "ANALYSIS PASSED."
)