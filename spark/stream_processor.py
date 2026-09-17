from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    LongType,
    DoubleType,
    IntegerType,
)


# ============================================================
# Configuration
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = "kafka:29092"
KAFKA_TOPIC = "transactions"


# ============================================================
# Transaction schema
# ============================================================

def create_transaction_schema():

    fields = [
        StructField("transaction_id", LongType(), False),
        StructField("original_time", DoubleType(), True),
        StructField("event_time", StringType(), True),
        StructField("Amount", DoubleType(), True),
        StructField("Class", IntegerType(), True),
    ]

    # PCA features V1 - V28
    for i in range(1, 29):
        fields.append(
            StructField(f"V{i}", DoubleType(), True)
        )

    return StructType(fields)


# ============================================================
# Create Spark session
# ============================================================

def create_spark_session():

    spark = (
        SparkSession.builder
        .appName("SelfHealingTransactionPipeline")
        .master("spark://spark-master:7077")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    return spark


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("REAL-TIME SELF-HEALING DATA PIPELINE")
    print("Spark Structured Streaming Processor")
    print("=" * 70)

    spark = create_spark_session()

    print("\nSpark session created.")
    print(f"Spark version: {spark.version}")

    schema = create_transaction_schema()

    # --------------------------------------------------------
    # Read streaming transactions from Kafka
    # --------------------------------------------------------

    print("\nConnecting Spark to Kafka...")
    print(f"Kafka server: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")

    kafka_stream = (
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
            "latest"
        )
        .option(
            "failOnDataLoss",
            "false"
        )
        .load()
    )

    # --------------------------------------------------------
    # Kafka value is binary.
    # Convert it into a JSON string.
    # --------------------------------------------------------

    json_stream = kafka_stream.select(
        col("key").cast("string").alias("kafka_key"),
        col("value").cast("string").alias("json_value"),
        col("timestamp").alias("kafka_timestamp")
    )

    # --------------------------------------------------------
    # Parse JSON according to our transaction schema
    # --------------------------------------------------------

    parsed_stream = (
        json_stream
        .select(
            col("kafka_key"),
            from_json(
                col("json_value"),
                schema
            ).alias("transaction"),
            col("kafka_timestamp")
        )
        .select(
            col("kafka_key"),
            col("transaction.*"),
            col("kafka_timestamp")
        )
    )

    # --------------------------------------------------------
    # Select useful fields for our first test
    # --------------------------------------------------------

    output_stream = parsed_stream.select(
        "transaction_id",
        "event_time",
        "Amount",
        "Class",
        "kafka_timestamp"
    )

    # --------------------------------------------------------
    # Print streaming transactions
    # --------------------------------------------------------

    query = (
        output_stream.writeStream
        .format("console")
        .outputMode("append")
        .option("truncate", "false")
        .option("numRows", 20)
        .start()
    )

    print("\nSpark is listening for transactions.")
    print("Run kafka/producer.py in another terminal.")
    print("Press Ctrl+C to stop Spark.\n")

    try:

        query.awaitTermination()

    except KeyboardInterrupt:

        print("\nStopping Spark streaming processor...")

        query.stop()

    finally:

        spark.stop()

        print("Spark stopped.")


if __name__ == "__main__":
    main()