# =====================================================
# IMPORTS
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    confusion_matrix,
    f1_score
)

# =====================================================
# LOAD DATA
# =====================================================
print("Loading dataset...")
df = pd.read_csv("enhanced_student_habits_performance_dataset.csv")

# Remove ID + leakage variable
df = df.drop(columns=["student_id", "previous_gpa"])

print("Dataset loaded.")
print("Dataset shape:", df.shape)

# =====================================================
# SAVE DEFAULT VALUES (FOR OPTIONAL INPUTS)
# =====================================================
print("Saving default values for optional inputs...")

numeric_defaults = df.select_dtypes(include=np.number).mean().to_dict()
categorical_defaults = {
    col: df[col].mode()[0]
    for col in df.select_dtypes(include="object").columns
}

joblib.dump(numeric_defaults, "numeric_defaults.pkl")
joblib.dump(categorical_defaults, "categorical_defaults.pkl")

print("Default values saved.")

# =====================================================
# FEATURE ENGINEERING
# =====================================================
print("Creating interaction features...")

df["study_x_motivation"] = df["study_hours_per_day"] * df["motivation_level"]
df["stress_x_sleep"] = df["stress_level"] * df["sleep_hours"]
df["screen_x_study"] = df["screen_time"] * df["study_hours_per_day"]
df["mental_x_stress"] = df["mental_health_rating"] * df["stress_level"]

print("Feature engineering completed.")

# =====================================================
# ================= REGRESSION =========================
# =====================================================
print("\n===== REGRESSION: EXAM SCORE PREDICTION =====")
print("Start time:", datetime.now())

X = df.drop("exam_score", axis=1)
y = df["exam_score"]

X = pd.get_dummies(X, drop_first=True)

# Scale numeric features
num_cols = X.select_dtypes(include=np.number).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

print(f"Data ready → Rows: {X.shape[0]}, Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("▶ Training Gradient Boosting Regressor...")
t0 = time.time()

reg_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

reg_model.fit(X_train, y_train)

print(f"✔ Training completed in {time.time() - t0:.1f} seconds")

y_pred = reg_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.3f}")
print(f"RMSE: {rmse:.2f}")

print("▶ Running 3-fold cross-validation...")
cv_scores = cross_val_score(
    reg_model, X, y, cv=3, scoring="r2"
)

print(f"CV R² Mean: {cv_scores.mean():.3f}")

# Save regression metrics
pd.DataFrame([{
    "Model": "Gradient Boosting Regressor (No GPA)",
    "R2": r2,
    "RMSE": rmse,
    "CV_R2_Mean": cv_scores.mean()
}]).to_csv("regression_metrics_no_gpa.csv", index=False)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================
print("Saving feature importance...")

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": reg_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

feature_importance_df.to_csv(
    "feature_importance_no_gpa.csv",
    index=False
)

plt.figure(figsize=(10, 6))
feature_importance_df.head(15).plot(
    x="Feature", y="Importance", kind="barh", legend=False
)
plt.title("Top Factors Affecting Exam Score (No GPA)")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance_no_gpa.png", dpi=300)
plt.close()

# Save regression model & preprocessing
joblib.dump(reg_model, "exam_score_regression_model_no_gpa.pkl")
joblib.dump(X.columns, "regression_features_no_gpa.pkl")
joblib.dump(scaler, "scaler_no_gpa.pkl")

print("Regression stage completed.")
print("Time now:", datetime.now())

# =====================================================
# ================= CLASSIFICATION =====================
# =====================================================
print("\n===== CLASSIFICATION: PERFORMANCE CATEGORY =====")

df["performance_class"] = pd.cut(
    df["exam_score"],
    bins=[0, 60, 85, 100],
    labels=["Low", "Medium", "High"]
)

X_cls = df.drop(["exam_score", "performance_class"], axis=1)
y_cls = df["performance_class"]

X_cls = pd.get_dummies(X_cls, drop_first=True)

# Scale numeric features
num_cols_cls = X_cls.select_dtypes(include=np.number).columns
X_cls[num_cols_cls] = scaler.fit_transform(X_cls[num_cols_cls])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

print("▶ Training Gradient Boosting Classifier...")
t1 = time.time()

clf = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

clf.fit(Xc_train, yc_train)

print(f"✔ Classifier training completed in {time.time() - t1:.1f} seconds")

yc_pred = clf.predict(Xc_test)

f1 = f1_score(yc_test, yc_pred, average="weighted")
cm = confusion_matrix(yc_test, yc_pred)

print(f"Weighted F1 Score: {f1:.3f}")

# Save classification metrics
pd.DataFrame([{
    "Model": "Gradient Boosting Classifier (No GPA)",
    "F1_Weighted": f1
}]).to_csv("classification_metrics_no_gpa.csv", index=False)

# Confusion matrix plot
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=clf.classes_,
    yticklabels=clf.classes_
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (No GPA)")
plt.tight_layout()
plt.savefig("confusion_matrix_no_gpa.png", dpi=300)
plt.close()

# Save classifier
joblib.dump(clf, "performance_classification_model_no_gpa.pkl")
joblib.dump(X_cls.columns, "classification_features_no_gpa.pkl")

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY")
print("End time:", datetime.now())
