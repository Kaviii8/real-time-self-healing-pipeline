import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
DATASET_PATH = "ml/failure_simulation_dataset.csv"
OUTPUT_DIR = "ml/outputs_100k"
MODEL_PATH = "ml/failure_prediction_model_100k_evaluation.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Load first 100,000 records chronologically
# ---------------------------------------------------------
df = pd.read_csv(DATASET_PATH).iloc[:100000].copy()

print("=" * 65)
print("100K XGBOOST MODEL EVALUATION")
print("=" * 65)

print(f"\nTotal records: {len(df):,}")
print("\nClass distribution:")
print(df["failure"].value_counts())

# ---------------------------------------------------------
# 2. Features and target
# ---------------------------------------------------------
FEATURES = [
    "amount",
    "fraud",
    "processing_time",
    "kafka_lag",
    "tps",
]

TARGET = "failure"

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

# ---------------------------------------------------------
# 3. Chronological 80/20 split
# ---------------------------------------------------------
split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index].copy()
X_test = X.iloc[split_index:].copy()

y_train = y.iloc[:split_index].copy()
y_test = y.iloc[split_index:].copy()

print(f"\nTraining records : {len(X_train):,}")
print(f"Testing records  : {len(X_test):,}")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# ---------------------------------------------------------
# 4. Preprocessing using TRAINING data only
# ---------------------------------------------------------
train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# Keep signed amount and also add magnitude information.
X_train["amount_log"] = np.log1p(np.abs(X_train["amount"]))
X_test["amount_log"] = np.log1p(np.abs(X_test["amount"]))

FINAL_FEATURES = FEATURES + ["amount_log"]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train[FINAL_FEATURES])
X_test_scaled = scaler.transform(X_test[FINAL_FEATURES])

# ---------------------------------------------------------
# 5. Train XGBoost
# ---------------------------------------------------------
negative = int((y_train == 0).sum())
positive = int((y_train == 1).sum())

scale_pos_weight = negative / positive

print(f"\nscale_pos_weight: {scale_pos_weight:.4f}")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 6. Evaluate ONLY on held-out 20,000 records
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n" + "=" * 65)
print("HELD-OUT TEST RESULTS")
print("=" * 65)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print(f"\nTrue Negatives  : {tn:,}")
print(f"False Positives : {fp:,}")
print(f"False Negatives : {fn:,}")
print(f"True Positives  : {tp:,}")

# ---------------------------------------------------------
# 7. Save confusion matrix
# ---------------------------------------------------------
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Failure"]
)

disp.plot(values_format="d")
plt.title("XGBoost Confusion Matrix - 20,000 Held-Out Records")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_100k.png"),
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# 8. Save ROC curve
# ---------------------------------------------------------
RocCurveDisplay.from_predictions(y_test, y_prob)

plt.title("XGBoost ROC Curve - 100K Evaluation")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve_100k.png"),
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# 9. Feature importance
# ---------------------------------------------------------
importance = pd.DataFrame({
    "feature": FINAL_FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
plt.barh(importance["feature"], importance["importance"])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "feature_importance_100k.png"),
    dpi=300
)
plt.close()

# ---------------------------------------------------------
# 10. Save metrics
# ---------------------------------------------------------
metrics = {
    "total_records": int(len(df)),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test)),
    "training_failures": int((y_train == 1).sum()),
    "testing_failures": int((y_test == 1).sum()),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn),
    "true_positives": int(tp),
}

with open(
    os.path.join(OUTPUT_DIR, "model_metrics_100k.json"),
    "w"
) as f:
    json.dump(metrics, f, indent=4)

# ---------------------------------------------------------
# 11. Save SEPARATE evaluation model
# ---------------------------------------------------------
artifact = {
    "model": model,
    "scaler": scaler,
    "features": FINAL_FEATURES,
    "medians": train_medians.to_dict(),
}

joblib.dump(artifact, MODEL_PATH)

print("\nOutputs saved to:")
print(OUTPUT_DIR)

print("\nSeparate model saved as:")
print(MODEL_PATH)

print("\nIMPORTANT:")
print("The original streaming model was NOT overwritten.")