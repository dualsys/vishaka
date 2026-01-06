# =====================================================
# PREDICT.PY
# Supports partial / optional user input
# =====================================================

import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD TRAINED ARTIFACTS
# =====================================================
print("Loading trained models and preprocessing artifacts...")

reg_model = joblib.load("exam_score_regression_model_no_gpa.pkl")
clf_model = joblib.load("performance_classification_model_no_gpa.pkl")

reg_features = joblib.load("regression_features_no_gpa.pkl")
clf_features = joblib.load("classification_features_no_gpa.pkl")

scaler = joblib.load("scaler_no_gpa.pkl")

numeric_defaults = joblib.load("numeric_defaults.pkl")
categorical_defaults = joblib.load("categorical_defaults.pkl")

print("Artifacts loaded successfully.")

# =====================================================
# USER INPUT (ALL OPTIONAL)
# User can provide ANY subset of fields
# =====================================================
user_input = {
    # Examples (comment out any field you want)
    "study_hours_per_day": 6,
    "sleep_hours": 7,
    "attendance_percentage": 88,
    "motivation_level": 9,
    
    "stress_level": 3
}

# =====================================================
# BUILD FULL INPUT USING DEFAULTS
# =====================================================
def build_complete_input(user_input, num_defaults, cat_defaults):
    full_input = {}

    # Numeric fields
    for key, default in num_defaults.items():
        if key in user_input and user_input[key] is not None:
            full_input[key] = user_input[key]
        else:
            full_input[key] = default

    # Categorical fields
    for key, default in cat_defaults.items():
        if key in user_input and user_input[key] is not None:
            full_input[key] = user_input[key]
        else:
            full_input[key] = default

    return full_input


final_input = build_complete_input(
    user_input,
    numeric_defaults,
    categorical_defaults
)

# =====================================================
# FEATURE ENGINEERING (MUST MATCH TRAINING)
# =====================================================
df = pd.DataFrame([final_input])

df["study_x_motivation"] = df["study_hours_per_day"] * df["motivation_level"]
df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

# =====================================================
# ONE-HOT ENCODING
# =====================================================
df = pd.get_dummies(df)

# =====================================================
# -------- REGRESSION PREDICTION --------
# =====================================================
X_reg = df.reindex(columns=reg_features, fill_value=0)

# Scale ONLY numeric columns seen during training
scaler_features = scaler.feature_names_in_
common_cols = [c for c in scaler_features if c in X_reg.columns]

X_reg.loc[:, common_cols] = scaler.transform(X_reg[common_cols])

predicted_score = reg_model.predict(X_reg)[0]

# =====================================================
# -------- CLASSIFICATION PREDICTION --------
# =====================================================
X_clf = df.reindex(columns=clf_features, fill_value=0)

common_cols_clf = [c for c in scaler_features if c in X_clf.columns]
X_clf.loc[:, common_cols_clf] = scaler.transform(X_clf[common_cols_clf])

predicted_class = clf_model.predict(X_clf)[0]

# =====================================================
# OUTPUT
# =====================================================
print("\n===== PREDICTION RESULT =====")
print("Inputs provided by user:", list(user_input.keys()))
print(f"Predicted Exam Score      : {predicted_score:.2f}")
print(f"Predicted Performance Tier: {predicted_class}")
print("==============================")
