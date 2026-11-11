from pathlib import Path
import pandas as pd
import numpy as np

INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"

df = pd.read_csv(INPUT_FILE)
df = df[df["IsPDS"] == 1].copy()

print("\nWITHIN-TOPOLOGY ANALYSIS")
print("========================")

features = [
    "C0",
    "AvgPathLength",
    "AvgClustering",
    "MeanPairDistanceS",
    "MeanDegreeS"
]

results = []

for topology in ["ER", "BA", "WS"]:

    d = df[df["Topology"] == topology]

    print(f"\n{topology}")
    print("-" * len(topology))
    print(f"Successful cases: {len(d)}")

    for feature in features:

        corr = d[feature].corr(d["PPT"])

        results.append({
            "Topology": topology,
            "Feature": feature,
            "CorrelationWithPPT": corr
        })

        print(
            f"{feature:20s}: "
            f"{corr:+.4f}"
        )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "within_topology_correlations.csv",
    index=False
)

# ------------------------------------------------------------
# Also report PPT by topology
# ------------------------------------------------------------

print("\nPPT SUMMARY BY TOPOLOGY")
print("=======================")

summary = (
    df.groupby("Topology")["PPT"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print(summary)

summary.to_csv(
    "ppt_summary_by_topology.csv"
)

print("\nFiles saved:")
print("within_topology_correlations.csv")
print("ppt_summary_by_topology.csv")

print("\nWITHIN-TOPOLOGY ANALYSIS PASSED.")