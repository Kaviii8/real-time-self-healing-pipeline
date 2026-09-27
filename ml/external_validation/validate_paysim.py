from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_THRESHOLD = 0.5


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "ml"
    / "failure_prediction_model.pkl"
)

DATASET_PATH = (
    PROJECT_ROOT
    / "ml"
    / "external_validation"
    / "paysim_aligned_validation_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ml"
    / "external_validation"
    / "outputs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD FROZEN STREAMING MODEL
# ============================================================

print("=" * 72)
print("PAYSIM EXTERNAL VALIDATION - FROZEN STREAMING MODEL")
print("=" * 72)

print(f"\nLoading model:\n{MODEL_PATH}")

bundle = joblib.load(MODEL_PATH)

required_bundle_keys = [
    "model",
    "scaler",
    "feature_names",
    "processing_time_median",
    "kafka_lag_median",
]

missing_bundle_keys = [
    key
    for key in required_bundle_keys
    if key not in bundle
]

if missing_bundle_keys:
    raise KeyError(
        "Model bundle is missing required keys: "
        f"{missing_bundle_keys}"
    )

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

processing_median = float(
    bundle["processing_time_median"]
)

kafka_lag_median = float(
    bundle["kafka_lag_median"]
)

print("\nModel loaded successfully.")
print("The deployed model will NOT be retrained or overwritten.")

print("\nModel-development medians:")
print(
    f"Processing time median : "
    f"{processing_median:.4f}"
)
print(
    f"Kafka lag median       : "
    f"{kafka_lag_median:.4f}"
)

print("\nExpected feature order:")

for i, feature in enumerate(
    feature_names,
    start=1,
):
    print(f"{i:2d}. {feature}")


# ============================================================
# LOAD EXTERNAL VALIDATION DATASET
# ============================================================

print("\n" + "-" * 72)

print(
    f"Loading validation dataset:\n{DATASET_PATH}"
)

df = pd.read_csv(DATASET_PATH)

print(
    f"\nValidation records: {len(df):,}"
)


# ============================================================
# VALIDATE REQUIRED DATASET COLUMNS
# ============================================================

required_columns = [
    "transaction_id",
    "amount",
    "fraud",
    "processing_time",
    "kafka_lag",
    "tps",
    "failure",
    "failure_type",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Validation dataset is missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# CONVERT BASE FEATURES TO NUMERIC
# ============================================================

features = pd.DataFrame(
    index=df.index
)

features["amount"] = pd.to_numeric(
    df["amount"],
    errors="coerce",
)

features["fraud"] = pd.to_numeric(
    df["fraud"],
    errors="coerce",
)

features["processing_time"] = pd.to_numeric(
    df["processing_time"],
    errors="coerce",
)

features["kafka_lag"] = pd.to_numeric(
    df["kafka_lag"],
    errors="coerce",
)

features["tps"] = pd.to_numeric(
    df["tps"],
    errors="coerce",
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# These medians come from the original training model bundle.
#
# They are NOT recalculated using PaySim because doing so would
# allow information from the external validation dataset to
# influence preprocessing.

features["processing_time"] = (
    features["processing_time"]
    .fillna(processing_median)
)

features["kafka_lag"] = (
    features["kafka_lag"]
    .fillna(kafka_lag_median)
)

features["amount"] = (
    features["amount"]
    .fillna(0.0)
)

features["fraud"] = (
    features["fraud"]
    .fillna(0.0)
)

features["tps"] = (
    features["tps"]
    .fillna(0.0)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
#
# IMPORTANT:
# These formulas match train_failure_model.py and
# self_healing_processor.py.
# ============================================================


# ------------------------------------------------------------
# 1. LOG-TRANSFORMED AMOUNT
# ------------------------------------------------------------

features["amount_log"] = np.log1p(
    np.abs(
        features["amount"]
    )
)


# ------------------------------------------------------------
# 2. SQUARED PROCESSING TIME
# ------------------------------------------------------------

features["processing_time_squared"] = (
    features["processing_time"] ** 2
)


# ------------------------------------------------------------
# 3. KAFKA LAG / TPS RATIO
# ------------------------------------------------------------
#
# Must exactly match:
#
# kafka_lag / (tps + 1.0)
#

features["kafka_lag_ratio"] = (
    features["kafka_lag"]
    /
    (
        features["tps"]
        + 1.0
    )
)


# ------------------------------------------------------------
# 4. HIGH PROCESSING INDICATOR
# ------------------------------------------------------------

features["high_processing"] = (
    features["processing_time"]
    > processing_median
).astype(int)


# ------------------------------------------------------------
# 5. HIGH KAFKA LAG INDICATOR
# ------------------------------------------------------------

features["high_kafka_lag"] = (
    features["kafka_lag"]
    > kafka_lag_median
).astype(int)


# ============================================================
# VERIFY EXACT MODEL FEATURES
# ============================================================

missing_model_features = [
    feature
    for feature in feature_names
    if feature not in features.columns
]

if missing_model_features:
    raise ValueError(
        "Could not construct all model features: "
        f"{missing_model_features}"
    )


# Use EXACT feature order stored with deployed model.

X = features[
    feature_names
].copy()


# ============================================================
# TARGET
# ============================================================

y_true = pd.to_numeric(
    df["failure"],
    errors="raise",
).astype(int).to_numpy()


unique_targets = set(
    np.unique(y_true)
)

if not unique_targets.issubset(
    {0, 1}
):
    raise ValueError(
        "The failure target must contain only 0 and 1. "
        f"Found: {sorted(unique_targets)}"
    )


# ============================================================
# DISPLAY VALIDATION CLASS DISTRIBUTION
# ============================================================

actual_failures = int(
    (y_true == 1).sum()
)

actual_normal = int(
    (y_true == 0).sum()
)

print("\nValidation class distribution:")

print(
    f"Normal records  : {actual_normal:,}"
)

print(
    f"Failure records : {actual_failures:,}"
)

print(
    f"Failure rate    : "
    f"{actual_failures / len(y_true) * 100:.2f}%"
)


# ============================================================
# SCALE USING ORIGINAL TRAINING SCALER
# ============================================================

print("\nApplying original training scaler...")

# IMPORTANT:
#
# scaler.fit() is NOT called here.
#
# We only transform using the scaler fitted during the
# original model-development process.

X_scaled = scaler.transform(X)


# ============================================================
# FROZEN MODEL PREDICTION
# ============================================================

print("Running frozen XGBoost predictions...")

# IMPORTANT:
#
# model.fit() is NEVER called.
#
# This is inference only.

y_probability = model.predict_proba(
    X_scaled
)[:, 1]


# Same 0.5 threshold used by self_healing_processor.py.

y_pred = (
    y_probability
    >= PREDICTION_THRESHOLD
).astype(int)


# ============================================================
# OVERALL VALIDATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_true,
    y_probability,
)

pr_auc = average_precision_score(
    y_true,
    y_probability,
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = (
    cm.ravel()
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

print("\n" + "=" * 72)
print("INDEPENDENT DATASET VALIDATION RESULTS")
print("=" * 72)

print(
    f"Dataset records : {len(df):,}"
)

print(
    f"Actual failures : {actual_failures:,}"
)

print(
    f"Actual normal   : {actual_normal:,}"
)

print()

print(
    f"Accuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision       : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall          : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score        : "
    f"{f1 * 100:.2f}%"
)

print(
    f"ROC-AUC         : "
    f"{roc_auc:.4f}"
)

print(
    f"PR-AUC          : "
    f"{pr_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print()

print(
    f"True Negatives  : {tn:,}"
)

print(
    f"False Positives : {fp:,}"
)

print(
    f"False Negatives : {fn:,}"
)

print(
    f"True Positives  : {tp:,}"
)


# ============================================================
# ADD PREDICTIONS TO ANALYSIS DATAFRAME
# ============================================================

analysis_df = df.copy()

analysis_df[
    "predicted_failure"
] = y_pred

analysis_df[
    "failure_probability"
] = y_probability

analysis_df[
    "prediction_correct"
] = (
    analysis_df["failure"]
    == analysis_df["predicted_failure"]
)


# ============================================================
# FAILURE-TYPE DETECTION ANALYSIS
# ============================================================

failure_rows = analysis_df[
    analysis_df["failure"] == 1
].copy()

failure_type_results = []

for (
    failure_type,
    group,
) in failure_rows.groupby(
    "failure_type"
):

    samples = len(group)

    detected = int(
        (
            group["predicted_failure"]
            == 1
        ).sum()
    )

    missed = int(
        (
            group["predicted_failure"]
            == 0
        ).sum()
    )

    detection_recall = (
        detected / samples
        if samples > 0
        else 0.0
    )

    average_probability = float(
        group[
            "failure_probability"
        ].mean()
    )

    failure_type_results.append(
        {
            "failure_type":
                failure_type,

            "samples":
                samples,

            "detected":
                detected,

            "missed":
                missed,

            "recall":
                detection_recall,

            "average_failure_probability":
                average_probability,
        }
    )


failure_type_df = pd.DataFrame(
    failure_type_results
)

if not failure_type_df.empty:

    failure_type_df = (
        failure_type_df
        .sort_values(
            by="samples",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


print("\n" + "=" * 72)
print("FAILURE-TYPE DETECTION")
print("=" * 72)

if failure_type_df.empty:

    print(
        "No failure records were available "
        "for failure-type analysis."
    )

else:

    display_failure_type_df = (
        failure_type_df.copy()
    )

    display_failure_type_df[
        "recall"
    ] = (
        display_failure_type_df[
            "recall"
        ]
        * 100
    )

    display_failure_type_df[
        "recall"
    ] = (
        display_failure_type_df[
            "recall"
        ]
        .map(
            lambda value:
                f"{value:.2f}%"
        )
    )

    display_failure_type_df[
        "average_failure_probability"
    ] = (
        display_failure_type_df[
            "average_failure_probability"
        ]
        .map(
            lambda value:
                f"{value:.4f}"
        )
    )

    print(
        display_failure_type_df.to_string(
            index=False
        )
    )


# ============================================================
# NORMAL-CLASS FALSE POSITIVE ANALYSIS
# ============================================================

normal_rows = analysis_df[
    analysis_df["failure"] == 0
]

normal_false_positives = int(
    (
        normal_rows[
            "predicted_failure"
        ]
        == 1
    ).sum()
)

normal_false_positive_rate = (
    normal_false_positives
    / len(normal_rows)
    if len(normal_rows) > 0
    else 0.0
)


print("\n" + "=" * 72)
print("NORMAL-CLASS ANALYSIS")
print("=" * 72)

print(
    f"Normal transactions       : "
    f"{len(normal_rows):,}"
)

print(
    f"False failure predictions : "
    f"{normal_false_positives:,}"
)

print(
    f"False positive rate       : "
    f"{normal_false_positive_rate * 100:.2f}%"
)


# ============================================================
# SAVE TRANSACTION-LEVEL PREDICTIONS
# ============================================================

prediction_columns = [
    "transaction_id",
]

# Include these when present.

optional_columns = [
    "paysim_step",
    "paysim_type",
    "raw_paysim_amount",
]

for column in optional_columns:

    if column in analysis_df.columns:
        prediction_columns.append(
            column
        )


prediction_columns.extend(
    [
        "amount",
        "fraud",
        "processing_time",
        "kafka_lag",
        "tps",
        "failure",
        "failure_type",
        "predicted_failure",
        "failure_probability",
        "prediction_correct",
    ]
)


prediction_output = analysis_df[
    prediction_columns
].copy()


prediction_output.to_csv(
    OUTPUT_DIR
    / "paysim_external_predictions.csv",
    index=False,
)


# ============================================================
# SAVE FAILURE-TYPE PERFORMANCE
# ============================================================

failure_type_df.to_csv(
    OUTPUT_DIR
    / "failure_type_performance.csv",
    index=False,
)


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(6, 5)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Failure",
    ],
)

display.plot(
    ax=ax,
    values_format=",d",
)

ax.set_title(
    "Frozen XGBoost Model\n"
    "PaySim External Validation"
)

fig.tight_layout()

confusion_matrix_path = (
    OUTPUT_DIR
    / "external_confusion_matrix.png"
)

fig.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, _ = roc_curve(
    y_true,
    y_probability,
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    fpr,
    tpr,
    label=(
        f"ROC-AUC = "
        f"{roc_auc:.4f}"
    ),
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curve - PaySim External Validation"
)

ax.legend()

fig.tight_layout()

roc_path = (
    OUTPUT_DIR
    / "external_roc_curve.png"
)

fig.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PRECISION-RECALL CURVE
# ============================================================

pr_precision, pr_recall, _ = (
    precision_recall_curve(
        y_true,
        y_probability,
    )
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

ax.plot(
    pr_recall,
    pr_precision,
    label=(
        f"PR-AUC = "
        f"{pr_auc:.4f}"
    ),
)

# Prevalence gives a useful no-skill reference for PR analysis.

failure_prevalence = (
    actual_failures
    / len(y_true)
)

ax.axhline(
    y=failure_prevalence,
    linestyle="--",
    label=(
        "Failure prevalence = "
        f"{failure_prevalence:.4f}"
    ),
)

ax.set_xlabel(
    "Recall"
)

ax.set_ylabel(
    "Precision"
)

ax.set_title(
    "Precision-Recall Curve - PaySim External Validation"
)

ax.legend()

fig.tight_layout()

pr_path = (
    OUTPUT_DIR
    / "external_precision_recall_curve.png"
)

fig.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FAILURE-TYPE RECALL PLOT
# ============================================================

failure_type_plot_path = (
    OUTPUT_DIR
    / "failure_type_recall.png"
)

if not failure_type_df.empty:

    plot_df = (
        failure_type_df
        .sort_values(
            by="recall",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        plot_df["failure_type"],
        plot_df["recall"] * 100,
    )

    ax.set_xlabel(
        "Recall (%)"
    )

    ax.set_ylabel(
        "Failure Type"
    )

    ax.set_title(
        "Failure-Type Detection Recall\n"
        "PaySim External Validation"
    )

    ax.set_xlim(
        0,
        100,
    )

    fig.tight_layout()

    fig.savefig(
        failure_type_plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

metrics = {

    "validation": {
        "type":
            "distribution-aligned independent-dataset validation",

        "dataset":
            "PaySim",

        "records":
            int(len(df)),

        "model_retrained":
            False,

        "scaler_refitted":
            False,

        "prediction_threshold":
            PREDICTION_THRESHOLD,
    },

    "class_distribution": {
        "normal":
            actual_normal,

        "failure":
            actual_failures,

        "failure_rate":
            float(
                actual_failures
                / len(df)
            ),
    },

    "metrics": {
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

        "pr_auc":
            float(pr_auc),
    },

    "confusion_matrix": {
        "true_negatives":
            int(tn),

        "false_positives":
            int(fp),

        "false_negatives":
            int(fn),

        "true_positives":
            int(tp),
    },

    "normal_class": {
        "false_positive_rate":
            float(
                normal_false_positive_rate
            ),
    },

    "model": {
        "features":
            list(feature_names),

        "processing_time_median":
            processing_median,

        "kafka_lag_median":
            kafka_lag_median,
    },
}


metrics_path = (
    OUTPUT_DIR
    / "external_validation_metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("VALIDATION COMPLETE")
print("=" * 72)

print(
    f"\nOutputs saved to:\n{OUTPUT_DIR}"
)

print("\nGenerated files:")

print(
    "  external_validation_metrics.json"
)

print(
    "  external_confusion_matrix.png"
)

print(
    "  external_roc_curve.png"
)

print(
    "  external_precision_recall_curve.png"
)

if not failure_type_df.empty:
    print(
        "  failure_type_recall.png"
    )

print(
    "  failure_type_performance.csv"
)

print(
    "  paysim_external_predictions.csv"
)

print("\nIMPORTANT:")

print(
    "  - The deployed XGBoost model was NOT retrained."
)

print(
    "  - The original StandardScaler was NOT refitted."
)

print(
    "  - The original model feature order was preserved."
)

print(
    "  - The original 0.5 prediction threshold was preserved."
)

print(
    "  - No PaySim labels were used to modify the model."
)

print("=" * 72)