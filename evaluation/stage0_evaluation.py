import pandas as pd
import itertools
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)

from src.config import FEATURES, DATASET_PATH
from src.preprocessing import clean_features


df = pd.read_csv(DATASET_PATH)

y = df["Label"].astype(int)
X = clean_features(df, FEATURES)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


X_train_normal = X_train[y_train == 0]


param_grid = {
    "n_estimators": [200, 400],
    "max_samples": ["auto", 0.8],
    "contamination": [0.05, 0.1],
}

results = []


for values in itertools.product(*param_grid.values()):
    params = dict(zip(param_grid.keys(), values))
    print(f"Training Isolation Forest Stage 0 with parameters: {params}")

    model = IsolationForest(
        **params,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train_normal)

    scores = -model.decision_function(X_test)

    threshold = np.percentile(scores, 100 * (1 - params["contamination"]))
    y_pred = (scores >= threshold).astype(int)

    results.append({
        **params,
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    })


results_df = pd.DataFrame(results)
results_df.sort_values("roc_auc", ascending=False, inplace=True)

results_df.to_csv(
    "results/stage0_hyperparameter_results.csv",
    index=False
)

print("\nStage 0 Isolation Forest hyperparameter search complete.")
print(results_df)