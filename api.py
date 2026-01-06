# =====================================================
# FASTAPI POST API FOR PREDICTION
# =====================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD ARTIFACTS (ONCE AT STARTUP)
# =====================================================
reg_model = joblib.load("exam_score_regression_model_no_gpa.pkl")
clf_model = joblib.load("performance_classification_model_no_gpa.pkl")

reg_features = joblib.load("regression_features_no_gpa.pkl")
clf_features = joblib.load("classification_features_no_gpa.pkl")

scaler = joblib.load("scaler_no_gpa.pkl")
numeric_defaults = joblib.load("numeric_defaults.pkl")
categorical_defaults = joblib.load("categorical_defaults.pkl")

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(
    title="Student Exam Score Prediction API",
    description="Predict exam score and performance category from partial student input",
    version="1.0"
)

# =====================================================
# INPUT SCHEMA (ALL OPTIONAL)
# =====================================================
class StudentInput(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    major: Optional[str] = None
    study_hours_per_day: Optional[float] = None
    social_media_hours: Optional[float] = None
    netflix_hours: Optional[float] = None
    part_time_job: Optional[str] = None
    attendance_percentage: Optional[float] = None
    sleep_hours: Optional[float] = None
    diet_quality: Optional[str] = None
    exercise_frequency: Optional[int] = None
    parental_education_level: Optional[str] = None
    internet_quality: Optional[str] = None
    mental_health_rating: Optional[float] = None
    extracurricular_participation: Optional[str] = None
    semester: Optional[int] = None
    stress_level: Optional[float] = None
    dropout_risk: Optional[str] = None
    social_activity: Optional[int] = None
    screen_time: Optional[float] = None
    study_environment: Optional[str] = None
    access_to_tutoring: Optional[str] = None
    family_income_range: Optional[str] = None
    parental_support_level: Optional[int] = None
    motivation_level: Optional[int] = None
    exam_anxiety_score: Optional[int] = None
    learning_style: Optional[str] = None
    time_management_score: Optional[float] = None

# =====================================================
# HELPER: MERGE USER INPUT WITH DEFAULTS
# =====================================================
def build_complete_input(user_data: Dict):
    full_input = {}

    for key, default in numeric_defaults.items():
        full_input[key] = user_data.get(key, default)

    for key, default in categorical_defaults.items():
        full_input[key] = user_data.get(key, default)

    return full_input

# =====================================================
# POST ENDPOINT
# =====================================================
@app.post("/predict")
def predict(input_data: StudentInput):
    user_input = input_data.dict(exclude_unset=True)

    # Fill missing fields
    final_input = build_complete_input(user_input)

    # DataFrame
    df = pd.DataFrame([final_input])

    # Feature engineering
    df["study_x_motivation"] = df["study_hours_per_day"] * df["motivation_level"]
    df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
    df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
    df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

    # One-hot encode
    df = pd.get_dummies(df)

    # ---------------- REGRESSION ----------------
    X_reg = df.reindex(columns=reg_features, fill_value=0)

    scaler_cols = scaler.feature_names_in_
    common_cols = [c for c in scaler_cols if c in X_reg.columns]
    X_reg.loc[:, common_cols] = scaler.transform(X_reg[common_cols])

    exam_score = float(reg_model.predict(X_reg)[0])

    # ---------------- CLASSIFICATION ----------------
    X_clf = df.reindex(columns=clf_features, fill_value=0)
    common_cols_clf = [c for c in scaler_cols if c in X_clf.columns]
    X_clf.loc[:, common_cols_clf] = scaler.transform(X_clf[common_cols_clf])

    performance_class = clf_model.predict(X_clf)[0]

    return {
        "predicted_exam_score": round(exam_score, 2),
        "predicted_performance_class": performance_class,
        "inputs_used": list(user_input.keys()),
        "missing_inputs_filled_with_defaults": list(
            set(numeric_defaults.keys()).union(categorical_defaults.keys())
            - set(user_input.keys())
        )
    }
