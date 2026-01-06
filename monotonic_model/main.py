# =====================================================
# MONOTONIC XGBOOST MODEL FOR EXAM SCORE PREDICTION
# =====================================================

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRegressor

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "../enhanced_student_habits_performance_dataset.csv"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Remove ID + leakage
df = df.drop(columns=["student_id", "previous_gpa"])

print("Dataset shape:", df.shape)

# =====================================================
# SAVE DEFAULTS (FOR OPTIONAL INPUTS)
# =====================================================
numeric_defaults = df.select_dtypes(include=np.number).mean().to_dict()
categorical_defaults = {
    col: df[col].mode()[0]
    for col in df.select_dtypes(include="object").columns
}

joblib.dump(numeric_defaults, f"{ARTIFACT_DIR}/numeric_defaults.pkl")
joblib.dump(categorical_defaults, f"{ARTIFACT_DIR}/categorical_defaults.pkl")

# =====================================================
# FEATURE ENGINEERING
# =====================================================
df["study_x_motivation"] = df["study_hours_per_day"] * df["motivation_level"]
df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

# =====================================================
# TARGET / FEATURES
# =====================================================
X = df.drop("exam_score", axis=1)
y = df["exam_score"]

X = pd.get_dummies(X, drop_first=True)

# =====================================================
# MONOTONIC CONSTRAINTS (CORRECT MAPPING)
# =====================================================
# +1 = increasing
# -1 = decreasing
#  0 = unconstrained

monotonic_map = {
    "attendance_percentage": 1,
    "study_hours_per_day": 1,
    "motivation_level": 1,
    "sleep_hours": 1,
    "exercise_frequency": 1,
    "mental_health_rating": 1,
    "time_management_score": 1,
    "extracurricular_participation": 1,
    "social_activity": 1,

    "stress_level": -1,
    "exam_anxiety_score": -1,
    "screen_time": -1,
    "netflix_hours": -1,
    "social_media_hours": -1,
}

constraints = []
for col in X.columns:
    constraint = 0
    for feature, direction in monotonic_map.items():
        if col.startswith(feature):
            constraint = direction
            break
    constraints.append(constraint)

constraints = tuple(constraints)

print("Total features:", len(X.columns))
print("Constrained features:", sum(1 for c in constraints if c != 0))

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# MODEL TRAINING (XGBOOST)
# =====================================================
print("\nTraining XGBoost with monotonic constraints...")
print("Start time:", datetime.now())

model = XGBRegressor(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    monotone_constraints=constraints,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed:", datetime.now())

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n===== MONOTONIC MODEL PERFORMANCE =====")
print(f"R² Score : {r2:.3f}")
print(f"RMSE     : {rmse:.2f}")

# =====================================================
# SAVE ARTIFACTS
# =====================================================
joblib.dump(model, f"{ARTIFACT_DIR}/exam_score_monotonic_xgb.pkl")
joblib.dump(X.columns, f"{ARTIFACT_DIR}/feature_list.pkl")

pd.DataFrame([{
    "model": "XGBoost Monotonic",
    "r2": r2,
    "rmse": rmse,
    "timestamp": datetime.now()
}]).to_csv(f"{ARTIFACT_DIR}/metrics.csv", index=False)

print("\nArtifacts saved in:", ARTIFACT_DIR)

# =====================================================
# SANITY CHECK (MONOTONICITY VERIFICATION)
# =====================================================
print("\n===== SANITY CHECK =====")

sample = X_test.iloc[0].copy()

def test_monotonic(feature, values):
    preds = []
    for v in values:
        row = sample.copy()
        if feature in row:
            row[feature] = v
        preds.append(float(model.predict(pd.DataFrame([row]))[0]))
    return preds

print("Stress ↑ → Score should ↓")
print(test_monotonic("stress_level", [2, 5, 8]))

print("Study hours ↑ → Score should ↑")
print(test_monotonic("study_hours_per_day", [2, 5, 8]))

print("\nDONE ✅")
