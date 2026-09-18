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
# CHRONOLOGICAL 80/20 SPLIT
# ============================================================

# IMPORTANT:
# Split the raw dataframe BEFORE calculating preprocessing
# statistics. This prevents information from the test period
# leaking into the training process.

split_index = int(
    len(df) * 0.80
)

train_df = df.iloc[
    :split_index
].copy()

test_df = df.iloc[
    split_index:
].copy()


print("\nDataset split:")

print(
    f"Training records : "
    f"{len(train_df):,}"
)

print(
    f"Testing records  : "
    f"{len(test_df):,}"
)

print(
    f"Training failures: "
    f"{int(train_df['failure'].sum()):,}"
)

print(
    f"Testing failures : "
    f"{int(test_df['failure'].sum()):,}"
)


# ============================================================
# PREPROCESSING STATISTICS
# ============================================================

# These values MUST come only from the training partition.

processing_time_median = float(
    train_df[
        "processing_time"
    ].median()
)

kafka_lag_median = float(
    train_df[
        "kafka_lag"
    ].median()
)


print("\nTraining-only preprocessing statistics:")

print(
    f"Processing-time median : "
    f"{processing_time_median:.4f}"
)

print(
    f"Kafka-lag median       : "
    f"{kafka_lag_median:.4f}"
)


# ============================================================
# FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(
    source_df,
    processing_median,
    lag_median
):
    """
    Create model features.

    The medians supplied to this function must be calculated
    from the training partition only.
    """

    feature_df = source_df.copy()

    # --------------------------------------------------------
    # 1. LOG-TRANSFORMED AMOUNT
    # --------------------------------------------------------

    # Controlled validation failures can contain negative
    # amounts because the producer injects sign corruption.
    #
    # log1p() cannot safely represent values <= -1.
    # Therefore use transaction magnitude for this engineered
    # feature while retaining the signed raw amount as a
    # separate model feature.
    feature_df[
        "amount_log"
    ] = np.log1p(
        np.abs(
            feature_df["amount"]
        )
    )

    # --------------------------------------------------------
    # 2. SQUARED PROCESSING TIME
    # --------------------------------------------------------

    feature_df[
        "processing_time_squared"
    ] = (
        feature_df[
            "processing_time"
        ] ** 2
    )

    # --------------------------------------------------------
    # 3. KAFKA LAG / TPS RATIO
    # --------------------------------------------------------

    feature_df[
        "kafka_lag_ratio"
    ] = (
        feature_df[
            "kafka_lag"
        ]
        /
        (
            feature_df[
                "tps"
            ]
            + 1.0
        )
    )

    # --------------------------------------------------------
    # 4. HIGH PROCESSING INDICATOR
    # --------------------------------------------------------

    feature_df[
        "high_processing"
    ] = (
        feature_df[
            "processing_time"
        ]
        > processing_median
    ).astype(int)

    # --------------------------------------------------------
    # 5. HIGH KAFKA LAG INDICATOR
    # --------------------------------------------------------

    feature_df[
        "high_kafka_lag"
    ] = (
        feature_df[
            "kafka_lag"
        ]
        > lag_median
    ).astype(int)

    return feature_df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\nCreating engineered features...")

train_features_df = create_features(
    train_df,
    processing_time_median,
    kafka_lag_median
)

test_features_df = create_features(
    test_df,
    processing_time_median,
    kafka_lag_median
)


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


X_train = train_features_df[
    feature_names
].copy()

X_test = test_features_df[
    feature_names
].copy()

y_train = train_df[
    target_name
].copy()

y_test = test_df[
    target_name
].copy()


print("\nFeatures used:")

for feature in feature_names:
    print(
        f"  - {feature}"
    )


# ============================================================
# CHECK DATA
# ============================================================

if X_train.isnull().any().any():
    raise ValueError(
        "NaN values found in training features."
    )

if X_test.isnull().any().any():
    raise ValueError(
        "NaN values found in testing features."
    )

if not np.isfinite(
    X_train.to_numpy()
).all():

    raise ValueError(
        "Infinite values found in training features."
    )

if not np.isfinite(
    X_test.to_numpy()
).all():

    raise ValueError(
        "Infinite values found in testing features."
    )


# ============================================================
# STANDARD SCALER
# ============================================================

scaler = StandardScaler()

# Fit ONLY using training data.
X_train_scaled = scaler.fit_transform(
    X_train
)

# Apply the fitted scaler to testing data.
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


# ROC-AUC requires both classes in the test partition.
if y_test.nunique() < 2:

    roc_auc = None

    print(
        "\nROC-AUC cannot be calculated because "
        "the test partition contains only one class."
    )

else:

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )


# ============================================================
# DISPLAY EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{f1 * 100:.2f}%"
)

if roc_auc is not None:

    print(
        f"ROC-AUC   : "
        f"{roc_auc:.4f}"
    )

else:

    print(
        "ROC-AUC   : N/A"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
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

if roc_auc is not None:

    fpr, tpr, _ = roc_curve(
        y_test,
        y_probability
    )

    plt.figure()

    plt.plot(
        fpr,
        tpr,
        label=(
            f"XGBoost AUC = "
            f"{roc_auc:.3f}"
        )
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
    importance_df[
        "feature"
    ],
    importance_df[
        "importance"
    ]
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
        (
            float(roc_auc)
            if roc_auc is not None
            else None
        ),

    "scale_pos_weight":
        float(scale_pos_weight),

    "processing_time_median":
        float(
            processing_time_median
        ),

    "kafka_lag_median":
        float(
            kafka_lag_median
        ),
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
        processing_time_median,

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

if roc_auc is not None:

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