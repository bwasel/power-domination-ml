from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

SEED = 2026
INPUT_FILE = Path(__file__).resolve().parent.parent / "data" / "pds_propagation_final_results.csv"
REPEATS = 50

df = pd.read_csv(INPUT_FILE)
df['GraphUID'] = (
    df['Topology'].astype(str) + '_N' + df['N'].astype(str)
    + '_G' + df['GraphID'].astype(str)
)

features = [
    'N','AvgDegree','Density','AvgClustering','Diameter','AvgPathLength',
    'Topology','Strategy','ActualFraction','MeanDegreeS',
    'MeanPairDistanceS','MinPairDistanceS','C0'
]
cat_cols = ['Topology','Strategy']
num_cols = [c for c in features if c not in cat_cols]

uids = df['GraphUID'].unique().copy()
rng = np.random.RandomState(SEED)
rng.shuffle(uids)
n_train = round(0.80 * len(uids))
train_graphs = set(uids[:n_train])
train_mask = df['GraphUID'].isin(train_graphs)
test_mask = ~train_mask

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(df.loc[train_mask, cat_cols])

def make_X(frame):
    return np.hstack([
        frame[num_cols].to_numpy(float),
        enc.transform(frame[cat_cols])
    ])

# Model A
X_train_A = make_X(df.loc[train_mask])
X_test_A = make_X(df.loc[test_mask])
y_train_A = df.loc[train_mask, 'IsPDS'].astype(int).to_numpy()

model_A = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)
model_A.fit(X_train_A, y_train_A)

# Model B: successful PDS only
success = df['IsPDS'].eq(1)
train_B = train_mask & success
test_B = test_mask & success
X_train_B = make_X(df.loc[train_B])
X_test_B = make_X(df.loc[test_B])
y_train_B = df.loc[train_B, 'PPT'].astype(float).to_numpy()

model_B = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=SEED,
    n_jobs=-1
)
model_B.fit(X_train_B, y_train_B)

# Warm-up
model_A.predict(X_test_A[:100])
model_B.predict(X_test_B[:100])

def time_prediction(model, X):
    vals = []
    for _ in range(REPEATS):
        t0 = time.perf_counter_ns()
        model.predict(X)
        vals.append((time.perf_counter_ns() - t0) / 1e6)
    vals = np.asarray(vals)
    return {
        'Rows': len(X),
        'MedianBatch_ms': np.median(vals),
        'MeanBatch_ms': np.mean(vals),
        'MedianPerConfiguration_ms': np.median(vals) / len(X),
        'P25Batch_ms': np.percentile(vals,25),
        'P75Batch_ms': np.percentile(vals,75),
    }

results = []
for name, model, X in [
    ('Model A', model_A, X_test_A),
    ('Model B', model_B, X_test_B),
]:
    d = time_prediction(model, X)
    d['Model'] = name
    results.append(d)

out = pd.DataFrame(results)[[
    'Model','Rows','MedianBatch_ms','MeanBatch_ms',
    'MedianPerConfiguration_ms','P25Batch_ms','P75Batch_ms'
]]
out.to_csv('runtime_prediction_summary.csv', index=False)
print(out.to_string(index=False))
