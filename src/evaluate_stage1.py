import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

from config import FEATURES, SAMPLED_PATH,MODELS_DIR
from preprocessing import clean_features, make_binary_label

df = pd.read_csv(SAMPLED_PATH)

X = clean_features(df, FEATURES)
y = make_binary_label(df)

rf = joblib.load(MODELS_DIR / "stage1_rf.pkl")
xgb = joblib.load(MODELS_DIR / "stage1_xgb.pkl")

print("Stage 1: Binary Detection\n")

print("Random Forest")
print(confusion_matrix(y, rf.predict(X)))
print(classification_report(y, rf.predict(X), digits=4))

print("\nXGBoost")
print(confusion_matrix(y, xgb.predict(X)))
print(classification_report(y, xgb.predict(X), digits=4))