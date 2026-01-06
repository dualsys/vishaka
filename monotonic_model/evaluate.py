# =====================================================
# EVALUATION SCRIPT FOR MONOTONIC XGBOOST MODEL
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================
DATA_PATH = "../enhanced_student_habits_performance_dataset.csv"
ARTIFACT_DIR = "artifacts"

MODEL_PATH = f"{ARTIFACT_DIR}/exam_score_monotonic_xgb.pkl"
FEATURES_PATH = f"{ARTIFACT_DIR}/feature_list.pkl"
NUM_DEFAULTS_PATH = f"{ARTIFACT_DIR}/numeric_defaults.pkl"
CAT_DEFAULTS_PATH = f"{ARTIFACT_DIR}/categorical_defaults.pkl"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL & ARTIFACTS
# =====================================================
print("Loading model and artifacts...")

model = joblib.load(MODEL_PATH)
feature_list = joblib.load(FEATURES_PATH)
numeric_defaults = joblib.load(NUM_DEFAULTS_PATH)
categorical_defaults = joblib.load(CAT_DEFAULTS_PATH)

print("Loaded successfully.")

# =====================================================
# LOAD DATA (SAME PREPROCESSING AS TRAINING)
# =====================================================
print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["student_id", "previous_gpa"])

# Feature engineering (MUST MATCH TRAINING)
df["study_x_motivation"] = df["study_hours_per_day"] * df["motivation_level"]
df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

X = pd.get_dummies(X, drop_first=True)

# Align columns to training features
for col in feature_list:
    if col not in X.columns:
        X[col] = 0

X = X[feature_list]

# =====================================================
# TRAIN / TEST SPLIT (SAME RANDOM STATE)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# PREDICTIONS
# =====================================================
print("\nRunning predictions...")
y_pred = model.predict(X_test)

# =====================================================
# 1️⃣ PERFORMANCE METRICS
# =====================================================
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

acc_5 = np.mean(np.abs(y_test - y_pred) <= 5)
acc_10 = np.mean(np.abs(y_test - y_pred) <= 10)

metrics_df = pd.DataFrame([{
    "Model": "XGBoost Monotonic",
    "R2": r2,
    "RMSE": rmse,
    "MAE": mae,
    "Accuracy_±5": acc_5,
    "Accuracy_±10": acc_10
}])

metrics_path = f"{ARTIFACT_DIR}/performance_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\n===== PERFORMANCE METRICS =====")
print(metrics_df)

# =====================================================
# 2️⃣ PREDICTED vs ACTUAL PLOT
# =====================================================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Predicted vs Actual (Monotonic Model)")
plt.tight_layout()

pred_plot_path = f"{ARTIFACT_DIR}/predicted_vs_actual.png"
plt.savefig(pred_plot_path, dpi=300)
plt.show()

# =====================================================
# 3️⃣ MONOTONIC BEHAVIOR PLOT
# =====================================================
def monotonic_curve(feature, values):
    preds = []
    base = X_test.iloc[0].copy()
    for v in values:
        row = base.copy()
        if feature in row:
            row[feature] = v
        preds.append(float(model.predict(pd.DataFrame([row]))[0]))
    return preds

stress_values = range(1, 10)
stress_preds = monotonic_curve("stress_level", stress_values)

plt.figure(figsize=(6, 4))
plt.plot(stress_values, stress_preds, marker="o")
plt.xlabel("Stress Level")
plt.ylabel("Predicted Exam Score")
plt.title("Monotonic Effect of Stress Level")
plt.grid(True)
plt.tight_layout()

mono_plot_path = f"{ARTIFACT_DIR}/monotonic_stress.png"
plt.savefig(mono_plot_path, dpi=300)
plt.show()

# =====================================================
# DONE
# =====================================================
print("\nEvaluation complete.")
print("Saved files:")
print("-", metrics_path)
print("-", pred_plot_path)
print("-", mono_plot_path)
