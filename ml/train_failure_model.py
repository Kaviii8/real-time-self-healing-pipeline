from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

ML_DIR = Path(__file__).resolve().parent

DATASET_PATH = ML_DIR / "failure_simulation_dataset.csv"

MODEL_PATH = ML_DIR / "failure_prediction_model.pkl"

OUTPUT_DIR = ML_DIR / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("XGBOOST FAILURE PREDICTION MODEL")
print("=" * 70)

print(f"\nLoading dataset: {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)

print(f"Total records : {len(df):,}")
print(f"Failures      : {int(df['failure'].sum()):,}")
print(
    f"Normal records: "
    f"{int((df['failure'] == 0).sum()):,}"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating engineered features...")


# 1. Log-transformed transaction amount
df["amount_log"] = np.log1p(
    df["amount"]
)


# 2. Squared processing time
df["processing_time_squared"] = (
    df["processing_time"] ** 2
)


# 3. Kafka lag relative to throughput
df["kafka_lag_ratio"] = (
    df["kafka_lag"]
    / (df["tps"] + 1)
)


# Median values calculated from the available data.
processing_median = (
    df["processing_time"].median()
)

kafka_lag_median = (
    df["kafka_lag"].median()
)


# 4. High-processing indicator
df["high_processing"] = (
    df["processing_time"]
    > processing_median
).astype(int)


# 5. High Kafka-lag indicator
df["high_kafka_lag"] = (
    df["kafka_lag"]
    > kafka_lag_median
).astype(int)


# ============================================================
# FEATURES AND TARGET
# ============================================================

feature_names = [

    # Original features
    "amount",
    "fraud",
    "processing_time",
    "kafka_lag",
    "tps",

    # Engineered features
    "amount_log",
    "processing_time_squared",
    "kafka_lag_ratio",
    "high_processing",
    "high_kafka_lag",
]

target_name = "failure"


X = df[feature_names]

y = df[target_name]


print("\nFeatures used:")

for feature in feature_names:
    print(f"  - {feature}")


# ============================================================
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


print("\nDataset split:")
print(
    f"Training records : {len(X_train):,}"
)

print(
    f"Testing records  : {len(X_test):,}"
)

print(
    f"Training failures: {int(y_train.sum()):,}"
)

print(
    f"Testing failures : {int(y_test.sum()):,}"
)


# ============================================================
# STANDARD SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

normal_count = int(
    (y_train == 0).sum()
)

failure_count = int(
    (y_train == 1).sum()
)

if failure_count == 0:
    raise ValueError(
        "Training dataset contains no failure records."
    )

scale_pos_weight = (
    normal_count
    / failure_count
)


print(
    f"\nNormal/failure class weight: "
    f"{scale_pos_weight:.4f}"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

print("\nTraining XGBoost model...")


model = XGBClassifier(

    n_estimators=200,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    eval_metric="logloss",
)


model.fit(
    X_train_scaled,
    y_train
)


print("Training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)

y_probability = model.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# EVALUATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Failure"
    ]
)

display.plot()

plt.title(
    "XGBoost Failure Prediction - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"XGBoost AUC = {roc_auc:.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Failure Prediction ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "roc_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame({

    "feature":
        feature_names,

    "importance":
        model.feature_importances_
})


importance_df = importance_df.sort_values(
    "importance",
    ascending=True
)


plt.figure(
    figsize=(8, 6)
)

plt.barh(
    importance_df["feature"],
    importance_df["importance"]
)

plt.xlabel(
    "Feature Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "XGBoost Failure Prediction Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "dataset_records":
        int(len(df)),

    "training_records":
        int(len(X_train)),

    "testing_records":
        int(len(X_test)),

    "training_failures":
        int(y_train.sum()),

    "testing_failures":
        int(y_test.sum()),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "scale_pos_weight":
        float(scale_pos_weight),

    "processing_time_median":
        float(processing_median),

    "kafka_lag_median":
        float(kafka_lag_median),
}


with open(
    OUTPUT_DIR / "model_metrics.json",
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# SAVE MODEL PACKAGE
# ============================================================

model_package = {

    "model":
        model,

    "scaler":
        scaler,

    "feature_names":
        feature_names,

    "processing_time_median":
        processing_median,

    "kafka_lag_median":
        kafka_lag_median,

    "metrics":
        metrics,
}


joblib.dump(
    model_package,
    MODEL_PATH
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)

print("FILES CREATED")

print("=" * 70)

print(
    f"Model             : "
    f"{MODEL_PATH}"
)

print(
    f"Confusion Matrix  : "
    f"{OUTPUT_DIR / 'confusion_matrix.png'}"
)

print(
    f"ROC Curve         : "
    f"{OUTPUT_DIR / 'roc_curve.png'}"
)

print(
    f"Feature Importance: "
    f"{OUTPUT_DIR / 'feature_importance.png'}"
)

print(
    f"Model Metrics     : "
    f"{OUTPUT_DIR / 'model_metrics.json'}"
)

print("=" * 70)