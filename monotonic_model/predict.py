# =====================================================
# PREDICT SCRIPT FOR MONOTONIC XGBOOST MODEL
# =====================================================

import joblib
import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================
ARTIFACT_DIR = "artifacts"

MODEL_PATH = f"{ARTIFACT_DIR}/exam_score_monotonic_xgb.pkl"
FEATURES_PATH = f"{ARTIFACT_DIR}/feature_list.pkl"
NUM_DEFAULTS_PATH = f"{ARTIFACT_DIR}/numeric_defaults.pkl"
CAT_DEFAULTS_PATH = f"{ARTIFACT_DIR}/categorical_defaults.pkl"

# =====================================================
# LOAD ARTIFACTS
# =====================================================
print("Loading trained model and artifacts...")

model = joblib.load(MODEL_PATH)
feature_list = joblib.load(FEATURES_PATH)
numeric_defaults = joblib.load(NUM_DEFAULTS_PATH)
categorical_defaults = joblib.load(CAT_DEFAULTS_PATH)

print("Loaded successfully.")

# =====================================================
# EXAMPLE USER INPUT (ALL OPTIONAL)
# =====================================================
new_student = {
    "study_hours_per_day": 2,
    "sleep_hours": 3,
    "motivation_level": 1,
    "stress_level": 3,
    "screen_time": 8,
    "attendance_percentage": 65,
    "learning_style": "Visual"
}

# =====================================================
# PREPROCESS INPUT
# =====================================================
def preprocess_input(user_input: dict):
    # Start from defaults
    data = {}

    # Fill numeric
    for col, default in numeric_defaults.items():
        data[col] = user_input.get(col, default)

    # Fill categorical
    for col, default in categorical_defaults.items():
        data[col] = user_input.get(col, default)

    # Create DataFrame
    df = pd.DataFrame([data])

    # Feature engineering (MUST MATCH TRAINING)
    df["study_x_motivation"] = (
        df["study_hours_per_day"] * df["motivation_level"]
    )
    df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
    df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
    df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

    # One-hot encode
    df = pd.get_dummies(df, drop_first=True)

    # Align columns with training
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_list]

    return df

# =====================================================
# RUN PREDICTION
# =====================================================
X_input = preprocess_input(new_student)
prediction = model.predict(X_input)[0]

print("\n===== PREDICTION RESULT =====")
print("Predicted Exam Score:", round(float(prediction), 2))
