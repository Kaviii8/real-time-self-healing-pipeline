from pathlib import Path
import random

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
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

ULB_EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "failure_simulation_dataset.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "external_validation"
    / "paysim_aligned_validation_dataset.csv"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# QUANTILE MAPPING
# ============================================================

def quantile_map(source_values, reference_values):
    """
    Map the empirical ranks of source_values onto the empirical
    distribution of reference_values.

    This preserves the ordering of PaySim amounts while placing
    their magnitudes within the original model-development domain.
    """

    source = np.asarray(source_values, dtype=float)
    reference = np.sort(
        np.asarray(reference_values, dtype=float)
    )

    # Stable ranking is useful if duplicate amounts occur.
    ranks = (
        pd.Series(source)
        .rank(method="average", pct=True)
        .to_numpy()
    )

    # Convert percentile ranks to positions in reference distribution.
    positions = ranks * (len(reference) - 1)

    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)

    weight = positions - lower

    mapped = (
        reference[lower] * (1.0 - weight)
        + reference[upper] * weight
    )

    return mapped


# ============================================================
# ORIGINAL SYSTEM METRIC GENERATION
# ============================================================

def generate_system_metrics(amount, fraud):

    processing_time = random.uniform(10, 30)

    if amount > 1000:
        processing_time += random.uniform(10, 25)

    if amount > 5000:
        processing_time += random.uniform(15, 30)

    if fraud == 1:
        processing_time += random.uniform(15, 35)

    kafka_lag = random.uniform(5, 80)

    if random.random() < 0.10:
        kafka_lag += random.uniform(80, 180)

    tps = random.uniform(70, 120)

    return processing_time, kafka_lag, tps


# ============================================================
# ORIGINAL FAILURE RULES
# ============================================================

def determine_failure(
    transaction_id,
    amount,
    processing_time,
    kafka_lag,
    tps,
):

    # Priority 1: Data validation
    if (
        transaction_id is None
        or amount is None
        or amount <= 0
    ):
        return 1, "DATA_VALIDATION"

    # Priority 2: Timeout
    if processing_time > BASE_TIMEOUT_MS:
        return 1, "TIMEOUT"

    # Priority 3: Kafka lag
    if kafka_lag > KAFKA_LAG_THRESHOLD_MS:
        return 1, "HIGH_KAFKA_LAG"

    # Priority 4: Resource pressure
    if (
        tps > RESOURCE_TPS_THRESHOLD
        and random.random() < RESOURCE_PRESSURE_RATE
    ):
        return 1, "RESOURCE_PRESSURE"

    # Priority 5: Transient failure
    if random.random() < TRANSIENT_FAILURE_RATE:
        return 1, "TRANSIENT_FAILURE"

    return 0, "NONE"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("PAYSIM DISTRIBUTION-ALIGNED EXTERNAL VALIDATION")
print("=" * 72)

print("\nLoading PaySim...")

paysim = pd.read_csv(
    PAYSIM_PATH,
    nrows=SAMPLE_SIZE,
)

print(
    f"PaySim records loaded : {len(paysim):,}"
)

print("\nLoading original ULB experimental amounts...")

ulb = pd.read_csv(
    ULB_EXPERIMENT_PATH,
    usecols=["original_amount"],
)

ulb_amounts = (
    ulb["original_amount"]
    .dropna()
    .abs()
    .to_numpy()
)

print(
    f"ULB reference records : {len(ulb_amounts):,}"
)


# ============================================================
# MAP AMOUNT DISTRIBUTION
# ============================================================

print("\nPerforming quantile mapping...")

paysim["raw_paysim_amount"] = (
    paysim["amount"].astype(float)
)

paysim["mapped_amount"] = quantile_map(
    paysim["raw_paysim_amount"].to_numpy(),
    ulb_amounts,
)


# ============================================================
# DISTRIBUTION CHECK
# ============================================================

print("\n" + "=" * 72)
print("AMOUNT DISTRIBUTION CHECK")
print("=" * 72)

print("\nRaw PaySim:")

print(
    paysim[
        "raw_paysim_amount"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)

print("\nMapped PaySim:")

print(
    paysim[
        "mapped_amount"
    ].describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


# ============================================================
# CREATE CONTROLLED VALIDATION DATA
# ============================================================

records = []

for index, row in paysim.iterrows():

    transaction_id = index + 1

    # This is the amount used by the model/simulator.
    original_amount = float(
        row["mapped_amount"]
    )

    amount = original_amount

    fraud = int(
        row["isFraud"]
    )

    data_corruption_type = "NONE"

    # --------------------------------------------------------
    # Same controlled validation corruption
    # --------------------------------------------------------

    if random.random() < DATA_VALIDATION_RATE:

        amount = -abs(amount)

        data_corruption_type = (
            "NEGATIVE_AMOUNT"
        )

    # --------------------------------------------------------
    # Same operational simulation
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
    # Same failure hierarchy
    # --------------------------------------------------------

    (
        failure,
        failure_type,
    ) = determine_failure(
        transaction_id,
        amount,
        processing_time,
        kafka_lag,
        tps,
    )

    amount_log = np.log1p(
        abs(amount)
    )

    records.append(
        {
            "transaction_id":
                transaction_id,

            "paysim_step":
                int(row["step"]),

            "paysim_type":
                row["type"],

            # Keep raw amount for traceability.
            "raw_paysim_amount":
                round(
                    float(
                        row[
                            "raw_paysim_amount"
                        ]
                    ),
                    4,
                ),

            # Model-domain amount.
            "amount":
                round(amount, 4),

            "fraud":
                fraud,

            "processing_time":
                round(
                    processing_time,
                    4,
                ),

            "kafka_lag":
                round(
                    kafka_lag,
                    4,
                ),

            "tps":
                round(
                    tps,
                    4,
                ),

            "amount_log":
                round(
                    amount_log,
                    6,
                ),

            "failure":
                failure,

            "failure_type":
                failure_type,

            "original_amount":
                round(
                    original_amount,
                    4,
                ),

            "data_corruption_type":
                data_corruption_type,
        }
    )


validation_df = pd.DataFrame(
    records
)


# ============================================================
# SAVE
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

total = len(
    validation_df
)

failures = int(
    validation_df[
        "failure"
    ].sum()
)

normal = (
    total - failures
)


print("\n" + "=" * 72)
print("ALIGNED EXTERNAL VALIDATION SUMMARY")
print("=" * 72)

print(
    f"Total records       : "
    f"{total:,}"
)

print(
    f"Failure records     : "
    f"{failures:,}"
)

print(
    f"Normal records      : "
    f"{normal:,}"
)

print(
    f"Failure rate        : "
    f"{failures / total * 100:.2f}%"
)

print(
    "\nFailure distribution:"
)

print(
    validation_df[
        "failure_type"
    ].value_counts()
)


print(
    "\nMapped amount thresholds:"
)

print(
    "Above 1000: "
    f"{(validation_df['original_amount'] > 1000).mean() * 100:.3f}%"
)

print(
    "Above 5000: "
    f"{(validation_df['original_amount'] > 5000).mean() * 100:.3f}%"
)


print(
    f"\nOutput saved to:\n"
    f"{OUTPUT_PATH}"
)

print("\nIMPORTANT:")
print(
    "The XGBoost model was NOT "
    "trained or modified."
)

print(
    "Original failure thresholds "
    "and simulation rules were retained."
)

print("=" * 72)