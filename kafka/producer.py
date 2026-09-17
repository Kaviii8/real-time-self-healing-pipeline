import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer


# ============================================================
# Configuration
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "transactions"

# Number of transactions to send per second.
# We can change this later for load/failure experiments.
TRANSACTIONS_PER_SECOND = 20

# Set to None to stream the entire dataset.
# For the first test, keep this small.
MAX_TRANSACTIONS = 100


# ============================================================
# Locate dataset
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


# ============================================================
# Create Kafka producer
# ============================================================

def create_producer():
    print("Connecting to Kafka...")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        key_serializer=lambda key: str(key).encode("utf-8"),
        acks="all",
        retries=5,
    )

    print("Connected to Kafka successfully.")
    return producer


# ============================================================
# Convert dataset row to transaction message
# ============================================================

def create_transaction(transaction_id, row):
    transaction = {
        "transaction_id": int(transaction_id),

        # Original Time value from the Kaggle dataset
        "original_time": float(row["Time"]),

        # Timestamp generated when this transaction enters
        # our simulated real-time pipeline
        "event_time": datetime.now(timezone.utc).isoformat(),

        "Amount": float(row["Amount"]),
        "Class": int(row["Class"]),
    }

    # Add PCA features V1 - V28
    for i in range(1, 29):
        column = f"V{i}"
        transaction[column] = float(row[column])

    return transaction


# ============================================================
# Main streaming function
# ============================================================

def main():

    print("=" * 65)
    print("REAL-TIME SELF-HEALING DATA PIPELINE")
    print("Kafka Transaction Producer")
    print("=" * 65)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():
        print(f"\nERROR: Dataset not found:")
        print(DATASET_PATH)
        return

    print(f"\nDataset: {DATASET_PATH}")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print("Loading credit card transaction dataset...")

    df = pd.read_csv(DATASET_PATH)

    print(f"Dataset loaded successfully.")
    print(f"Total transactions available: {len(df):,}")

    required_columns = (
        ["Time"]
        + [f"V{i}" for i in range(1, 29)]
        + ["Amount", "Class"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\nERROR: Required columns are missing:")
        print(missing_columns)
        return

    # --------------------------------------------------------
    # Limit transactions during development/testing
    # --------------------------------------------------------

    if MAX_TRANSACTIONS is not None:
        df = df.head(MAX_TRANSACTIONS)

    total_to_send = len(df)

    print(f"Transactions for this run: {total_to_send:,}")
    print(f"Target rate: {TRANSACTIONS_PER_SECOND} transactions/sec")
    print(f"Kafka topic: {KAFKA_TOPIC}")

    # --------------------------------------------------------
    # Connect to Kafka
    # --------------------------------------------------------

    try:
        producer = create_producer()

    except Exception as error:
        print("\nCould not connect to Kafka.")
        print(error)
        return

    delay = 1 / TRANSACTIONS_PER_SECOND

    successful = 0
    failed = 0

    start_time = time.perf_counter()

    print("\nStarting transaction stream...")
    print("Press Ctrl+C to stop.\n")

    try:

        for index, row in df.iterrows():

            transaction_id = index + 1

            transaction = create_transaction(
                transaction_id,
                row
            )

            try:

                producer.send(
                    KAFKA_TOPIC,
                    key=transaction_id,
                    value=transaction
                )

                successful += 1

                # Don't print every transaction during large runs.
                # Print first five and then every 20th transaction.
                if successful <= 5 or successful % 20 == 0:

                    print(
                        f"Sent transaction {transaction_id} | "
                        f"Amount=${transaction['Amount']:.2f} | "
                        f"Class={transaction['Class']}"
                    )

            except Exception as error:

                failed += 1

                print(
                    f"Failed transaction {transaction_id}: {error}"
                )

            time.sleep(delay)

    except KeyboardInterrupt:

        print("\nProducer stopped by user.")

    finally:

        # Ensure queued Kafka messages are actually delivered.
        producer.flush()
        producer.close()

    # --------------------------------------------------------
    # Producer statistics
    # --------------------------------------------------------

    elapsed_time = time.perf_counter() - start_time

    actual_rate = (
        successful / elapsed_time
        if elapsed_time > 0
        else 0
    )

    print("\n" + "=" * 65)
    print("PRODUCER SUMMARY")
    print("=" * 65)

    print(f"Successfully sent : {successful}")
    print(f"Failed            : {failed}")
    print(f"Elapsed time      : {elapsed_time:.2f} seconds")
    print(f"Actual rate       : {actual_rate:.2f} transactions/sec")

    print("=" * 65)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()