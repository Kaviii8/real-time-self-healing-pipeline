from pathlib import Path
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Use an independent chronological sample from PaySim.
# You can later increase this to 284807 if desired.
SAMPLE_SIZE = 100_000

BASE_TIMEOUT_MS = 30.0
KAFKA_LAG_THRESHOLD_MS = 100.0
RESOURCE_TPS_THRESHOLD = 110.0

RESOURCE_PRESSURE_RATE = 0.35
TRANSIENT_FAILURE_RATE = 0.03
DATA_VALIDATION_RATE = 0.05


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAYSIM_PATH = (
    PROJECT_ROOT
    / "data"
    / "paysim"
    / "PS_20174392719_1491204439457_log.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "external_validation"
    / "paysim_validation_dataset.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# SYSTEM METRIC GENERATION
# Same controlled logic used in the original experiment
# ============================================================

def generate_system_metrics(amount, fraud):

    # Standard processing time
    processing_time = random.uniform(10, 30)

    # Higher-value transactions require additional processing
    if amount > 1000:
        processing_time += random.uniform(10, 25)

    if amount > 5000:
        processing_time += random.uniform(15, 30)

    # Fraudulent transactions receive additional processing
    if fraud == 1:
        processing_time += random.uniform(15, 35)

    # Kafka consumer lag
    kafka_lag = random.uniform(5, 80)

    # Controlled congestion event
    if random.random() < 0.10:
        kafka_lag += random.uniform(80, 180)

    # System throughput
    tps = random.uniform(70, 120)

    return processing_time, kafka_lag, tps


# ============================================================
# FAILURE SIMULATION
# Same priority used in the original experiment
# ============================================================

def determine_failure(
    transaction_id,
    amount,
    processing_time,
    kafka_lag,
    tps,
):

    # 1. DATA VALIDATION
    if (
        transaction_id is None
        or amount is None
        or amount <= 0
    ):
        return 1, "DATA_VALIDATION"

    # 2. TIMEOUT
    if processing_time > BASE_TIMEOUT_MS:
        return 1, "TIMEOUT"

    # 3. HIGH KAFKA LAG
    if kafka_lag > KAFKA_LAG_THRESHOLD_MS:
        return 1, "HIGH_KAFKA_LAG"

    # 4. RESOURCE PRESSURE
    if (
        tps > RESOURCE_TPS_THRESHOLD
        and random.random() < RESOURCE_PRESSURE_RATE
    ):
        return 1, "RESOURCE_PRESSURE"

    # 5. TRANSIENT FAILURE
    if random.random() < TRANSIENT_FAILURE_RATE:
        return 1, "TRANSIENT_FAILURE"

    return 0, "NONE"


# ============================================================
# LOAD PAYSIM
# ============================================================

print("=" * 70)
print("PAYSIM EXTERNAL VALIDATION DATA PREPARATION")
print("=" * 70)

print(f"\nLoading PaySim from:\n{PAYSIM_PATH}")

df = pd.read_csv(
    PAYSIM_PATH,
    nrows=SAMPLE_SIZE,
)

print(f"\nRecords loaded: {len(df):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = [
    "step",
    "type",
    "amount",
    "isFraud",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing PaySim columns: {missing_columns}"
    )


# ============================================================
# GENERATE VALIDATION RECORDS
# ============================================================

records = []

for index, row in df.iterrows():

    transaction_id = index + 1

    original_amount = float(row["amount"])
    amount = original_amount

    fraud = int(row["isFraud"])

    data_corruption_type = "NONE"

    # --------------------------------------------------------
    # CONTROLLED DATA VALIDATION ERROR
    # Same sign-inversion principle as original producer
    # --------------------------------------------------------

    if random.random() < DATA_VALIDATION_RATE:

        amount = -abs(original_amount)

        data_corruption_type = "NEGATIVE_AMOUNT"

    # --------------------------------------------------------
    # SYSTEM CONDITIONS
    # Use magnitude so corruption does not alter complexity
    # --------------------------------------------------------

    (
        processing_time,
        kafka_lag,
        tps,
    ) = generate_system_metrics(
        abs(amount),
        fraud,
    )

    # --------------------------------------------------------
    # DETERMINE PRIMARY FAILURE
    # --------------------------------------------------------

    failure, failure_type = determine_failure(
        transaction_id,
        amount,
        processing_time,
        kafka_lag,
        tps,
    )

    # --------------------------------------------------------
    # DERIVED MODEL FEATURE
    # --------------------------------------------------------

    amount_log = np.log1p(abs(amount))

    records.append(
        {
            "transaction_id": transaction_id,
            "paysim_step": int(row["step"]),
            "paysim_type": row["type"],
            "amount": round(amount, 4),
            "fraud": fraud,
            "processing_time": round(
                processing_time,
                4,
            ),
            "kafka_lag": round(
                kafka_lag,
                4,
            ),
            "tps": round(
                tps,
                4,
            ),
            "amount_log": round(
                amount_log,
                6,
            ),
            "failure": failure,
            "failure_type": failure_type,
            "original_amount": round(
                original_amount,
                4,
            ),
            "data_corruption_type":
                data_corruption_type,
        }
    )


validation_df = pd.DataFrame(records)


# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

validation_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXTERNAL VALIDATION DATASET SUMMARY")
print("=" * 70)

print(
    f"Total records       : "
    f"{len(validation_df):,}"
)

failure_count = int(
    validation_df["failure"].sum()
)

normal_count = (
    len(validation_df)
    - failure_count
)

print(
    f"Failure records     : "
    f"{failure_count:,}"
)

print(
    f"Normal records      : "
    f"{normal_count:,}"
)

print(
    f"Failure rate        : "
    f"{failure_count / len(validation_df) * 100:.2f}%"
)

print("\nFailure distribution:")

print(
    validation_df[
        "failure_type"
    ].value_counts()
)

print("\nPaySim transaction types:")

print(
    validation_df[
        "paysim_type"
    ].value_counts()
)

print(
    f"\nOutput saved to:\n{OUTPUT_PATH}"
)

print("\nIMPORTANT:")
print(
    "No ML model was trained or modified "
    "during this step."
)

print("=" * 70)