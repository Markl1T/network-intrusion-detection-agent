import pandas as pd
import joblib
from sklearn.metrics import classification_report

from config import FEATURES, SAMPLED_PATH, MODELS_DIR
from preprocessing import clean_features

df = pd.read_csv(SAMPLED_PATH)
df = df[df["Attack"] != "Benign"]

X = clean_features(df, FEATURES)
y_true = df["Attack"]

rf, enc_rf = joblib.load(MODELS_DIR / "stage2_rf.pkl")
xgb, enc_xgb = joblib.load(MODELS_DIR / "stage2_xgb.pkl")

y_rf = enc_rf.inverse_transform(rf.predict(X))
y_xgb = enc_xgb.inverse_transform(xgb.predict(X))

print("Stage 2: Attack Type Classification\n")

print("Random Forest")
print(classification_report(y_true, y_rf, digits=4))

print("\nXGBoost")
print(classification_report(y_true, y_xgb, digits=4))
