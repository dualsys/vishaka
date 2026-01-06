# =====================================================
# FASTAPI FOR MONOTONIC EXAM SCORE PREDICTION
# =====================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# =====================================================
# APP INITIALIZATION
# =====================================================
app = FastAPI(
    title="Student Exam Score Prediction API (Monotonic)",
    version="1.0"
)

# Allow all CORS (demo / PhD safe)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD ARTIFACTS (FROM monotonic_model/artifacts)
# =====================================================
ARTIFACT_DIR = "monotonic_model/artifacts"

model = joblib.load(f"{ARTIFACT_DIR}/exam_score_monotonic_xgb.pkl")
feature_list = joblib.load(f"{ARTIFACT_DIR}/feature_list.pkl")
numeric_defaults = joblib.load(f"{ARTIFACT_DIR}/numeric_defaults.pkl")
categorical_defaults = joblib.load(f"{ARTIFACT_DIR}/categorical_defaults.pkl")

# =====================================================
# HELPER: PREPROCESS INPUT
# =====================================================
def preprocess_input(user_input: dict):
    data = {}

    # Fill numeric defaults
    for col, default in numeric_defaults.items():
        data[col] = user_input.get(col, default)

    # Fill categorical defaults
    for col, default in categorical_defaults.items():
        data[col] = user_input.get(col, default)

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

    # Align features with training
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_list]
    return df

# =====================================================
# PERFORMANCE CATEGORY
# =====================================================
def performance_category(score: float) -> str:
    if score >= 85:
        return "High"
    elif score >= 70:
        return "Medium"
    else:
        return "Low"

# =====================================================
# ROUTES
# =====================================================
@app.get("/")
def root():
    return {"message": "Monotonic Exam Score Prediction API is running"}

@app.post("/predict")
def predict(payload: dict):
    X = preprocess_input(payload)
    score = float(model.predict(X)[0])

    return {
        "predicted_exam_score": round(score, 2),
        "predicted_performance_category": performance_category(score),
        "inputs_provided": list(payload.keys()),
        "missing_inputs_filled": [
            k for k in numeric_defaults.keys() if k not in payload
        ]
    }
