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

# Development test first.
MAX_TRANSACTIONS = 1000

# Target streaming rate.
TARGET_TPS = 100

# Fixed random seed makes the experiment reproducible.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


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
    Generate simulated system-level conditions.

    The original credit-card dataset does not contain
    processing time, Kafka lag, or TPS.

    Therefore these values represent the controlled
    experimental environment.
    """

    # --------------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------------

    # Normal processing time: 10-30 ms
    processing_time = random.uniform(10, 30)

    # Higher-value transactions require more processing.
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

    # Normal Kafka lag.
    kafka_lag = random.uniform(5, 80)

    # Occasionally simulate congestion.
    if random.random() < 0.10:
        kafka_lag += random.uniform(80, 180)

    # --------------------------------------------------------
    # THROUGHPUT
    # --------------------------------------------------------

    # Simulated current system throughput.
    tps = random.uniform(70, 120)

    return processing_time, kafka_lag, tps


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
    Controlled synthetic failure injection.

    Returns:
        failure:
            0 = success
            1 = failure

        failure_type:
            Type of simulated failure
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

    if processing_time > 45:
        return 1, "TIMEOUT"

    # --------------------------------------------------------
    # 3. HIGH KAFKA LAG
    # --------------------------------------------------------

    if kafka_lag > 100:
        return 1, "HIGH_KAFKA_LAG"

    # --------------------------------------------------------
    # 4. RESOURCE PRESSURE
    # --------------------------------------------------------

    if (
        tps > 110
        and random.random() < 0.35
    ):
        return 1, "RESOURCE_PRESSURE"

    # --------------------------------------------------------
    # 5. TRANSIENT FAILURE
    # --------------------------------------------------------

    if random.random() < 0.03:
        return 1, "TRANSIENT_FAILURE"

    return 0, "NONE"


# ============================================================
# KAFKA PRODUCER
# ============================================================

def create_producer():
    """
    Create Kafka producer.
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

    # Make sure ML folder exists.
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

    print(
        f"Dataset rows available : {len(df):,}"
    )

    print(
        f"Transactions this run  : {MAX_TRANSACTIONS:,}"
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

    # --------------------------------------------------------
    # CREATE KAFKA PRODUCER
    # --------------------------------------------------------

    producer = create_producer()

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    successful_sends = 0
    simulated_failures = 0

    failure_counts = {}

    # --------------------------------------------------------
    # TRAINING DATA STORAGE
    # --------------------------------------------------------

    # Every transaction used in the simulation will also be
    # saved here for later XGBoost training.
    training_records = []

    # --------------------------------------------------------
    # TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()

    delay = 1.0 / TARGET_TPS

    # ========================================================
    # STREAM TRANSACTIONS
    # ========================================================

    for index, row in df.head(
        MAX_TRANSACTIONS
    ).iterrows():

        # ----------------------------------------------------
        # TRANSACTION INFORMATION
        # ----------------------------------------------------

        transaction_id = int(
            index + 1
        )

        amount = float(
            row["Amount"]
        )

        fraud = int(
            row["Class"]
        )

        # ----------------------------------------------------
        # GENERATE SYSTEM CONDITIONS
        # ----------------------------------------------------

        (
            processing_time,
            kafka_lag,
            tps
        ) = generate_system_metrics(
            amount,
            fraud
        )

        # ----------------------------------------------------
        # DETERMINE FAILURE
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

            "amount":
                amount,

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
        # SAVE RECORD FOR XGBOOST DATASET
        # ----------------------------------------------------

        training_records.append({

            "transaction_id":
                transaction_id,

            "amount":
                amount,

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
            or
            successful_sends % 100 == 0
        ):

            print(

                f"Sent "
                f"{transaction_id:5d} | "

                f"Amount="
                f"${amount:8.2f} | "

                f"Fraud="
                f"{fraud} | "

                f"Time="
                f"{processing_time:6.2f}ms | "

                f"Lag="
                f"{kafka_lag:6.2f}ms | "

                f"TPS="
                f"{tps:6.2f} | "

                f"Failure="
                f"{failure} | "

                f"{failure_type}"
            )

        # ----------------------------------------------------
        # CONTROL STREAMING RATE
        # ----------------------------------------------------

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
        f"Transactions sent  : "
        f"{successful_sends:,}"
    )

    print(
        f"Simulated failures : "
        f"{simulated_failures:,}"
    )

    print(
        f"Failure rate       : "
        f"{failure_rate:.2f}%"
    )

    print(
        f"Elapsed time       : "
        f"{elapsed:.2f}s"
    )

    print(
        f"Actual producer TPS: "
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
    # DATASET INFORMATION
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