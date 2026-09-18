import json
import time
from datetime import datetime

import psycopg2

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    LongType,
    DoubleType,
    IntegerType,
    StringType,
)


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
KAFKA_TOPIC = "transactions"

POSTGRES_HOST = "postgres"
POSTGRES_PORT = 5432
POSTGRES_DATABASE = "selfhealing"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"


# ============================================================
# SPARK SESSION
# ============================================================

spark = (
    SparkSession.builder
    .appName("BaselineTransactionPipeline")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("=" * 70)
print("BASELINE TRANSACTION PIPELINE")
print("=" * 70)

print(f"Kafka topic : {KAFKA_TOPIC}")
print(
    f"PostgreSQL  : "
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DATABASE}"
)

print("Healing     : DISABLED")
print("ML model    : DISABLED")

print("=" * 70)


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

transaction_schema = StructType([

    StructField(
        "transaction_id",
        LongType(),
        False
    ),

    StructField(
        "timestamp",
        StringType(),
        True
    ),

    StructField(
        "amount",
        DoubleType(),
        True
    ),

    StructField(
        "fraud",
        IntegerType(),
        True
    ),

    StructField(
        "processing_time",
        DoubleType(),
        True
    ),

    StructField(
        "kafka_lag",
        DoubleType(),
        True
    ),

    StructField(
        "tps",
        DoubleType(),
        True
    ),

    StructField(
        "failure",
        IntegerType(),
        True
    ),

    StructField(
        "failure_type",
        StringType(),
        True
    ),
])


# ============================================================
# READ KAFKA STREAM
# ============================================================

raw_stream = (
    spark.readStream
    .format("kafka")
    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS
    )
    .option(
        "subscribe",
        KAFKA_TOPIC
    )
    .option(
        "startingOffsets",
        "earliest"
    )
    .option(
        "failOnDataLoss",
        "false"
    )
    .load()
)


# ============================================================
# PARSE JSON
# ============================================================

transactions = (
    raw_stream
    .selectExpr(
        "CAST(value AS STRING) AS json_value"
    )
    .select(
        from_json(
            col("json_value"),
            transaction_schema
        ).alias("data")
    )
    .select("data.*")
)


# ============================================================
# POSTGRES CONNECTION
# ============================================================

def get_postgres_connection():

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DATABASE,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ============================================================
# PROCESS MICRO-BATCH
# ============================================================

def process_batch(batch_df, batch_id):

    if batch_df.rdd.isEmpty():
        return

    batch_start = time.perf_counter()

    rows = batch_df.collect()

    total_transactions = len(rows)

    successful_transactions = 0
    failed_transactions = 0

    latencies = []

    print(
        f"\nProcessing baseline batch "
        f"{batch_id} - "
        f"{total_transactions} transactions"
    )

    connection = None
    cursor = None

    try:

        connection = get_postgres_connection()

        cursor = connection.cursor()

        # ====================================================
        # PROCESS TRANSACTIONS
        # ====================================================

        for row in rows:

            transaction_start = (
                time.perf_counter()
            )

            transaction_id = (
                row["transaction_id"]
            )

            amount = row["amount"]

            fraud = row["fraud"]

            failure = (
                row["failure"]
                if row["failure"] is not None
                else 0
            )

            failure_type = (
                row["failure_type"]
                if row["failure_type"]
                else "NONE"
            )

            event_time = row["timestamp"]

            # ------------------------------------------------
            # BASELINE VALIDATION
            # ------------------------------------------------

            validation_failed = False

            if transaction_id is None:
                validation_failed = True

            if amount is None or amount <= 0:
                validation_failed = True

            if fraud not in (0, 1):
                validation_failed = True

            # ------------------------------------------------
            # DETERMINE FINAL BASELINE RESULT
            # ------------------------------------------------

            if validation_failed:

                final_failure = True

                final_failure_type = (
                    "DATA_VALIDATION"
                )

                status = "FAILED"

            elif failure == 1:

                # Failure occurred in the simulated
                # transaction-processing environment.
                #
                # Baseline pipeline does NOT attempt
                # recovery.

                final_failure = True

                final_failure_type = (
                    failure_type
                )

                status = "FAILED"

            else:

                final_failure = False

                final_failure_type = "NONE"

                status = "SUCCESS"

            # ------------------------------------------------
            # LATENCY
            # ------------------------------------------------

            transaction_latency_ms = (
                (
                    time.perf_counter()
                    - transaction_start
                )
                * 1000
            )

            latencies.append(
                transaction_latency_ms
            )

            if final_failure:
                failed_transactions += 1
            else:
                successful_transactions += 1

            # ------------------------------------------------
            # INSERT BASELINE RESULT
            # ------------------------------------------------

            cursor.execute(
                """
                INSERT INTO baseline_results
                (
                    transaction_id,
                    event_time,
                    amount,
                    actual_class,
                    failure_type,
                    failure_occurred,
                    processing_latency_ms,
                    status
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    transaction_id,
                    event_time,
                    amount,
                    fraud,
                    final_failure_type,
                    final_failure,
                    transaction_latency_ms,
                    status,
                )
            )

        # ====================================================
        # BATCH METRICS
        # ====================================================

        batch_elapsed = (
            time.perf_counter()
            - batch_start
        )

        throughput_tps = (
            total_transactions
            / batch_elapsed
            if batch_elapsed > 0
            else 0
        )

        avg_latency_ms = (
            sum(latencies)
            / len(latencies)
            if latencies
            else 0
        )

        # ----------------------------------------------------
        # INSERT PIPELINE METRICS
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO pipeline_metrics
            (
                pipeline_type,
                batch_id,
                batch_size,
                throughput_tps,
                avg_latency_ms,
                total_transactions,
                successful_transactions,
                failed_transactions,
                predicted_failures,
                healing_attempts,
                successful_healings
            )
            VALUES
            (
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s,
                %s
            )
            """,
            (
                "BASELINE",
                int(batch_id),
                total_transactions,
                throughput_tps,
                avg_latency_ms,
                total_transactions,
                successful_transactions,
                failed_transactions,
                0,
                0,
                0,
            )
        )

        connection.commit()

        # ====================================================
        # CONSOLE SUMMARY
        # ====================================================

        success_rate = (
            successful_transactions
            / total_transactions
            * 100
        )

        failure_rate = (
            failed_transactions
            / total_transactions
            * 100
        )

        print("-" * 70)

        print(
            f"Batch ID       : {batch_id}"
        )

        print(
            f"Transactions   : "
            f"{total_transactions}"
        )

        print(
            f"Successful     : "
            f"{successful_transactions}"
        )

        print(
            f"Failed         : "
            f"{failed_transactions}"
        )

        print(
            f"Success rate   : "
            f"{success_rate:.2f}%"
        )

        print(
            f"Failure rate   : "
            f"{failure_rate:.2f}%"
        )

        print(
            f"Average latency: "
            f"{avg_latency_ms:.4f} ms"
        )

        print(
            f"Throughput     : "
            f"{throughput_tps:.2f} TPS"
        )

        print("-" * 70)

    except Exception as error:

        if connection:
            connection.rollback()

        print(
            f"ERROR processing batch "
            f"{batch_id}: {error}"
        )

        raise

    finally:

        if cursor:
            cursor.close()

        if connection:
            connection.close()


# ============================================================
# START STREAM
# ============================================================

query = (
    transactions.writeStream
    .foreachBatch(process_batch)
    .option(
        "checkpointLocation",
        "/tmp/checkpoints/baseline"
    )
    .trigger(
        processingTime="5 seconds"
    )
    .start()
)


print("\nBaseline pipeline is running.")
print("Waiting for Kafka transactions...\n")


query.awaitTermination()