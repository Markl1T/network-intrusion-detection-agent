import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import FEATURES, SAMPLED_PATH, MODELS_DIR
from preprocessing import clean_features, encode_multiclass

df = pd.read_csv(SAMPLED_PATH)
df = df[df["Attack"] != "Benign"]  # only attacks

X = clean_features(df, FEATURES)
y, encoder = encode_multiclass(df["Attack"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="mlogloss",
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

joblib.dump((rf, encoder), MODELS_DIR / "stage2_rf.pkl")
joblib.dump((xgb, encoder), MODELS_DIR / "stage2_xgb.pkl")

print("Stage 2 models trained")