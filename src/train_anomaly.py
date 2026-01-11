import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report

from config import FEATURES, DATASET_PATH, MODELS_DIR
from preprocessing import clean_features


df = pd.read_csv(DATASET_PATH)

y = df["Label"].astype("int8")

X = clean_features(df, FEATURES)

X_benign = X[y == 0]

contamination_rate = y.mean()

model = IsolationForest(
    n_estimators=150,
    contamination=contamination_rate,
    random_state=42,
    n_jobs=-1
)

model.fit(X_benign)

joblib.dump(model, MODELS_DIR / "anomaly_iforest.pkl")

print("Isolation Forest trained and saved")

scores = -model.decision_function(X)

roc = roc_auc_score(y, scores)
precision, recall, thresholds = precision_recall_curve(y, scores)
pr_auc = auc(recall, precision)

print(f"\nAnomaly ROC-AUC: {roc:.4f}")
print(f"Anomaly PR-AUC : {pr_auc:.4f}")

threshold = np.percentile(scores, 100 * (1 - contamination_rate))
y_pred = (scores >= threshold).astype(int)

print(f"Anomaly Threshold used: {threshold:.6f}")

cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4, zero_division=0))

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Isolation Forest — PR Curve (AUC={pr_auc:.4f})")
plt.grid(True)
plt.show()