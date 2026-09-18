import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
TOPIC = "transactions"

# Development experiment.
MAX_TRANSACTIONS = 50

# Target producer rate.
TARGET_TPS = 5

# Fixed seed makes controlled failure injection reproducible.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# ============================================================
# EXPERIMENTAL FAILURE PARAMETERS
# ============================================================

# Standard timeout before adaptive timeout healing is applied.
BASE_TIMEOUT_MS = 30.0

# Transactions above this lag are considered affected by
# consumer congestion.
KAFKA_LAG_THRESHOLD_MS = 100.0

# High system throughput may create resource pressure.
RESOURCE_TPS_THRESHOLD = 110.0

# Controlled probability of a resource-pressure event when
# throughput is above the threshold.
RESOURCE_PRESSURE_RATE = 0.35

# Controlled one-shot transient failure probability.
TRANSIENT_FAILURE_RATE = 0.03

# Controlled invalid-data injection rate.
DATA_VALIDATION_RATE = 0.05


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "creditcard.csv"
)

ML_DIR = PROJECT_ROOT / "ml"

FAILURE_DATASET_PATH = (
    ML_DIR
    / "failure_simulation_dataset.csv"
)


# ============================================================
# SYSTEM METRIC SIMULATION
# ============================================================

def generate_system_metrics(amount, fraud):
    """
    Generate controlled system-level conditions.

    The ULB credit-card dataset contains transaction features,
    Amount, Time, and Class, but does not contain processing
    latency, Kafka lag, or system throughput.

    These values therefore represent the controlled
    experimental environment used by this project.
    """

    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    # Standard processing time.
    processing_time = random.uniform(10, 30)

    # Higher-value transactions are assumed to require
    # additional processing.
    if amount > 1000:
        processing_time += random.uniform(10, 25)

    if amount > 5000:
        processing_time += random.uniform(15, 30)

    # Fraudulent transactions receive additional processing.
    if fraud == 1:
        processing_time += random.uniform(15, 35)

    # --------------------------------------------------------
    # KAFKA CONSUMER LAG
    # --------------------------------------------------------

    kafka_lag = random.uniform(5, 80)

    # Occasionally create a congestion event.
    if random.random() < 0.10:
        kafka_lag += random.uniform(80, 180)

    # --------------------------------------------------------
    # SYSTEM THROUGHPUT
    # --------------------------------------------------------

    tps = random.uniform(70, 120)

    return (
        processing_time,
        kafka_lag,
        tps
    )


# ============================================================
# FAILURE SIMULATION
# ============================================================

def determine_failure(
    transaction_id,
    amount,
    processing_time,
    kafka_lag,
    tps
):
    """
    Determine whether the transaction experiences a controlled
    failure.

    Failure priority:

        1. DATA_VALIDATION
        2. TIMEOUT
        3. HIGH_KAFKA_LAG
        4. RESOURCE_PRESSURE
        5. TRANSIENT_FAILURE

    Only one primary failure type is assigned to each
    transaction. This makes evaluation easier and avoids
    counting the same transaction multiple times.
    """

    # --------------------------------------------------------
    # 1. DATA VALIDATION FAILURE
    # --------------------------------------------------------

    if (
        transaction_id is None
        or amount is None
        or amount <= 0
    ):
        return 1, "DATA_VALIDATION"

    # --------------------------------------------------------
    # 2. TIMEOUT FAILURE
    # --------------------------------------------------------

    if processing_time > BASE_TIMEOUT_MS:
        return 1, "TIMEOUT"

    # --------------------------------------------------------
    # 3. HIGH KAFKA LAG
    # --------------------------------------------------------

    if kafka_lag > KAFKA_LAG_THRESHOLD_MS:
        return 1, "HIGH_KAFKA_LAG"

    # --------------------------------------------------------
    # 4. RESOURCE PRESSURE
    # --------------------------------------------------------

    if (
        tps > RESOURCE_TPS_THRESHOLD
        and random.random() < RESOURCE_PRESSURE_RATE
    ):
        return 1, "RESOURCE_PRESSURE"

    # --------------------------------------------------------
    # 5. TRANSIENT FAILURE
    # --------------------------------------------------------

    if random.random() < TRANSIENT_FAILURE_RATE:
        return 1, "TRANSIENT_FAILURE"

    return 0, "NONE"


# ============================================================
# KAFKA PRODUCER
# ============================================================

def create_producer():
    """
    Create the Kafka producer.
    """

    return KafkaProducer(
        bootstrap_servers=KAFKA_SERVER,

        value_serializer=lambda value: (
            json.dumps(value).encode("utf-8")
        ),

        key_serializer=lambda key: (
            str(key).encode("utf-8")
        ),

        acks="all",
        retries=5,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SELF-HEALING PIPELINE - TRANSACTION PRODUCER")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK DATASET
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    ML_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print(
        f"\nLoading dataset: {DATASET_PATH}"
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    transactions_to_process = min(
        MAX_TRANSACTIONS,
        len(df)
    )

    print(
        f"Dataset rows available : {len(df):,}"
    )

    print(
        f"Transactions this run  : "
        f"{transactions_to_process:,}"
    )

    print(
        f"Target TPS             : {TARGET_TPS}"
    )

    print(
        f"Kafka topic            : {TOPIC}"
    )

    print(
        f"Random seed            : {RANDOM_SEED}"
    )

    print(
        f"Validation injection   : "
        f"{DATA_VALIDATION_RATE * 100:.1f}%"
    )

    print(
        f"Base timeout           : "
        f"{BASE_TIMEOUT_MS:.1f} ms"
    )

    print(
        f"Kafka lag threshold    : "
        f"{KAFKA_LAG_THRESHOLD_MS:.1f} ms"
    )

    print(
        f"Resource TPS threshold : "
        f"{RESOURCE_TPS_THRESHOLD:.1f}"
    )

    # --------------------------------------------------------
    # CREATE KAFKA PRODUCER
    # --------------------------------------------------------

    producer = create_producer()

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    successful_sends = 0
    simulated_failures = 0
    injected_validation_errors = 0

    failure_counts = {}

    # --------------------------------------------------------
    # ML TRAINING DATA STORAGE
    # --------------------------------------------------------

    training_records = []

    # --------------------------------------------------------
    # TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    delay = (
        1.0 / TARGET_TPS
        if TARGET_TPS > 0
        else 0
    )

    # ========================================================
    # STREAM TRANSACTIONS
    # ========================================================

    for index, row in df.head(
        transactions_to_process
    ).iterrows():

        # ----------------------------------------------------
        # ORIGINAL TRANSACTION INFORMATION
        # ----------------------------------------------------

        transaction_id = int(
            index + 1
        )

        original_amount = float(
            row["Amount"]
        )

        amount = original_amount

        fraud = int(
            row["Class"]
        )

        # ----------------------------------------------------
        # CONTROLLED DATA CORRUPTION
        # ----------------------------------------------------

        data_corruption_type = "NONE"

        if random.random() < DATA_VALIDATION_RATE:

            # Controlled sign-inversion corruption.
            #
            # Example:
            #     149.62 -> -149.62
            #
            # This allows the cleansing strategy to apply
            # abs(amount) and then revalidate the record.
            amount = -abs(
                original_amount
            )

            data_corruption_type = (
                "NEGATIVE_AMOUNT"
            )

            injected_validation_errors += 1

        # ----------------------------------------------------
        # GENERATE SYSTEM CONDITIONS
        # ----------------------------------------------------

        # Use the magnitude of the amount when generating
        # system conditions. The negative sign represents
        # data corruption and should not change simulated
        # processing complexity.
        (
            processing_time,
            kafka_lag,
            tps
        ) = generate_system_metrics(
            abs(amount),
            fraud
        )

        # ----------------------------------------------------
        # DETERMINE PRIMARY FAILURE
        # ----------------------------------------------------

        (
            failure,
            failure_type
        ) = determine_failure(
            transaction_id,
            amount,
            processing_time,
            kafka_lag,
            tps
        )

        # ----------------------------------------------------
        # COUNT FAILURES
        # ----------------------------------------------------

        if failure:

            simulated_failures += 1

            failure_counts[
                failure_type
            ] = (
                failure_counts.get(
                    failure_type,
                    0
                )
                + 1
            )

        # ----------------------------------------------------
        # CREATE KAFKA MESSAGE
        # ----------------------------------------------------

        transaction = {

            "transaction_id":
                transaction_id,

            "timestamp":
                datetime.now(
                    timezone.utc
                ).isoformat(),

            # Current value presented to the pipeline.
            "amount":
                round(
                    amount,
                    4
                ),

            # Ground-truth value retained for experiment
            # evaluation. Healing logic must NOT simply copy
            # this value to repair a transaction.
            "original_amount":
                round(
                    original_amount,
                    4
                ),

            "data_corruption_type":
                data_corruption_type,

            "fraud":
                fraud,

            "processing_time":
                round(
                    processing_time,
                    4
                ),

            "kafka_lag":
                round(
                    kafka_lag,
                    4
                ),

            "tps":
                round(
                    tps,
                    4
                ),

            "failure":
                failure,

            "failure_type":
                failure_type,
        }

        # ----------------------------------------------------
        # PRESERVE PCA FEATURES V1-V28
        # ----------------------------------------------------

        for i in range(
            1,
            29
        ):

            transaction[
                f"V{i}"
            ] = float(
                row[f"V{i}"]
            )

        # ----------------------------------------------------
        # SAVE RECORD FOR XGBOOST TRAINING
        # ----------------------------------------------------

        training_records.append({

            "transaction_id":
                transaction_id,

            # ML receives the transaction value visible to
            # the processing pipeline.
            "amount":
                round(
                    amount,
                    4
                ),

            "fraud":
                fraud,

            "processing_time":
                round(
                    processing_time,
                    4
                ),

            "kafka_lag":
                round(
                    kafka_lag,
                    4
                ),

            "tps":
                round(
                    tps,
                    4
                ),

            "failure":
                failure,

            "failure_type":
                failure_type,

            # Experimental metadata. These fields are useful
            # for evaluation but are NOT ML input features.
            "original_amount":
                round(
                    original_amount,
                    4
                ),

            "data_corruption_type":
                data_corruption_type,
        })

        # ----------------------------------------------------
        # SEND TO KAFKA
        # ----------------------------------------------------

        producer.send(
            TOPIC,
            key=transaction_id,
            value=transaction
        )

        successful_sends += 1

        # ----------------------------------------------------
        # CONSOLE OUTPUT
        # ----------------------------------------------------

        if (
            successful_sends <= 5
            or successful_sends % 100 == 0
        ):

            print(

                f"Sent "
                f"{transaction_id:5d} | "

                f"Amount="
                f"${amount:9.2f} | "

                f"Fraud="
                f"{fraud} | "

                f"Time="
                f"{processing_time:6.2f}ms | "

                f"Lag="
                f"{kafka_lag:7.2f}ms | "

                f"TPS="
                f"{tps:6.2f} | "

                f"Failure="
                f"{failure} | "

                f"{failure_type}"
            )

        # ----------------------------------------------------
        # CONTROL PRODUCER RATE
        # ----------------------------------------------------

        if delay > 0:
            time.sleep(
                delay
            )

    # ========================================================
    # FINISH KAFKA PRODUCER
    # ========================================================

    producer.flush()
    producer.close()

    # ========================================================
    # SAVE FAILURE SIMULATION DATASET
    # ========================================================

    training_df = pd.DataFrame(
        training_records
    )

    training_df.to_csv(
        FAILURE_DATASET_PATH,
        index=False
    )

    # ========================================================
    # CALCULATE SUMMARY
    # ========================================================

    elapsed = (
        time.perf_counter()
        - start_time
    )

    actual_tps = (
        successful_sends / elapsed
        if elapsed > 0
        else 0
    )

    failure_rate = (
        simulated_failures
        / successful_sends
        * 100
        if successful_sends > 0
        else 0
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PRODUCER SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Transactions sent          : "
        f"{successful_sends:,}"
    )

    print(
        f"Simulated failures         : "
        f"{simulated_failures:,}"
    )

    print(
        f"Failure rate               : "
        f"{failure_rate:.2f}%"
    )

    print(
        f"Injected validation errors : "
        f"{injected_validation_errors:,}"
    )

    print(
        f"Elapsed time               : "
        f"{elapsed:.2f}s"
    )

    print(
        f"Actual producer TPS        : "
        f"{actual_tps:.2f}"
    )

    print(
        "\nFailure distribution:"
    )

    for (
        failure_type,
        count
    ) in sorted(
        failure_counts.items()
    ):

        print(
            f"  "
            f"{failure_type:<20} "
            f"{count}"
        )

    # --------------------------------------------------------
    # ML DATASET INFORMATION
    # --------------------------------------------------------

    print(
        "\nML DATASET"
    )

    print(
        "-" * 70
    )

    print(
        f"Dataset path        : "
        f"{FAILURE_DATASET_PATH}"
    )

    print(
        f"Training records    : "
        f"{len(training_df):,}"
    )

    print(
        f"Failure records     : "
        f"{int(training_df['failure'].sum()):,}"
    )

    print(
        f"Normal records      : "
        f"{int((training_df['failure'] == 0).sum()):,}"
    )

    print(
        "=" * 70
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
