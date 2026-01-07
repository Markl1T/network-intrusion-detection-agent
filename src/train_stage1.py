import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from config import FEATURES, SAMPLED_PATH, MODELS_DIR
from preprocessing import clean_features, make_binary_label

df = pd.read_csv(SAMPLED_PATH)

X = clean_features(df, FEATURES)
y = make_binary_label(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

joblib.dump(rf, MODELS_DIR / "stage1_rf.pkl")
joblib.dump(xgb, MODELS_DIR / "stage1_xgb.pkl")

print("Stage 1 models trained")