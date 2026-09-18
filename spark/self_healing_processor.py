import time
import math
import json
import joblib
import psycopg2
import numpy as np

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
POSTGRES_DB = "selfhealing"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"

MODEL_PATH = "/opt/ml/failure_prediction_model.pkl"

CHECKPOINT_PATH = "/tmp/checkpoints/self_healing"


# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================

# Must match producer.py.
BASE_TIMEOUT_MS = 30.0

KAFKA_LAG_THRESHOLD_MS = 100.0

RESOURCE_TPS_THRESHOLD = 110.0

# Query-complexity reduction assumption:
# simplified processing requires 65% of the original workload.
QUERY_REDUCTION_FACTOR = 0.65

# Standard batch size used by the experiment.
STANDARD_BATCH_SIZE = 1000


# ============================================================
# LOAD ML MODEL
# ============================================================

print("=" * 70)
print("SELF-HEALING TRANSACTION PIPELINE")
print("=" * 70)

print("\nLoading XGBoost failure prediction model...")

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
scaler = model_bundle["scaler"]
feature_names = model_bundle["feature_names"]

processing_median = model_bundle[
    "processing_time_median"
]

kafka_lag_median = model_bundle[
    "kafka_lag_median"
]

print(
    f"Processing-time median : "
    f"{processing_median:.4f}"
)

print(
    f"Kafka-lag median       : "
    f"{kafka_lag_median:.4f}"
)

print("ML model loaded successfully.")
print(f"Features: {feature_names}")


# ============================================================
# SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("SelfHealingTransactionPipeline")
    .config(
        "spark.sql.shuffle.partitions",
        "4",
    )
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ============================================================
# KAFKA MESSAGE SCHEMA
# ============================================================

schema = StructType(
    [
        StructField(
            "transaction_id",
            LongType(),
            True,
        ),

        StructField(
            "timestamp",
            StringType(),
            True,
        ),

        StructField(
            "amount",
            DoubleType(),
            True,
        ),

        # Experimental ground truth.
        # It is NOT used to perform healing.
        StructField(
            "original_amount",
            DoubleType(),
            True,
        ),

        StructField(
            "data_corruption_type",
            StringType(),
            True,
        ),

        StructField(
            "fraud",
            IntegerType(),
            True,
        ),

        StructField(
            "processing_time",
            DoubleType(),
            True,
        ),

        StructField(
            "kafka_lag",
            DoubleType(),
            True,
        ),

        StructField(
            "tps",
            DoubleType(),
            True,
        ),

        StructField(
            "failure",
            IntegerType(),
            True,
        ),

        StructField(
            "failure_type",
            StringType(),
            True,
        ),
    ]
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


# ============================================================
# HEALING ACTION DATABASE LOGGING
# ============================================================

def insert_healing_actions(
    cursor,
    transaction_id,
    failure_type,
    actions,
):
    """
    Store each healing strategy/action as a separate row.

    This table is intended for:
        - strategy-level analysis
        - Grafana visualizations
        - auditing healing decisions
    """

    if not actions:
        return

    query = """
        INSERT INTO healing_actions (
            transaction_id,
            strategy,
            trigger_reason,
            success,
            details
        )
        VALUES (
            %s,
            %s,
            %s,
            %s,
            %s::jsonb
        )
    """

    for action in actions:

        cursor.execute(
            query,
            (
                transaction_id,
                action["strategy"],
                failure_type,
                bool(
                    action.get(
                        "success",
                        False,
                    )
                ),
                json.dumps(
                    action.get(
                        "details",
                        {},
                    )
                ),
            ),
        )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(row):
    """
    Create the same feature representation used during
    model training.
    """

    amount = float(
        row.amount
        if row.amount is not None
        else 0.0
    )

    processing_time = float(
        row.processing_time
        if row.processing_time is not None
        else 0.0
    )

    kafka_lag = float(
        row.kafka_lag
        if row.kafka_lag is not None
        else 0.0
    )

    tps = float(
        row.tps
        if row.tps is not None
        else 0.0
    )

    fraud = int(
        row.fraud
        if row.fraud is not None
        else 0
    )

    # Must match train_failure_model.py.
    amount_log = math.log1p(
        abs(amount)
    )

    processing_time_squared = (
        processing_time ** 2
    )

    kafka_lag_ratio = (
        kafka_lag
        / (tps + 1.0)
    )

    high_processing = int(
        processing_time
        > processing_median
    )

    high_kafka_lag = int(
        kafka_lag
        > kafka_lag_median
    )

    feature_map = {
        "amount": amount,
        "fraud": fraud,
        "processing_time": processing_time,
        "kafka_lag": kafka_lag,
        "tps": tps,
        "amount_log": amount_log,
        "processing_time_squared":
            processing_time_squared,
        "kafka_lag_ratio":
            kafka_lag_ratio,
        "high_processing":
            high_processing,
        "high_kafka_lag":
            high_kafka_lag,
    }

    return [
        feature_map[name]
        for name in feature_names
    ]


# ============================================================
# ML FAILURE PREDICTION
# ============================================================

def predict_failure(row):

    features = create_features(row)

    feature_array = np.asarray(
        [features],
        dtype=float,
    )

    scaled_features = (
        scaler.transform(
            feature_array
        )
    )

    probability = float(
        model.predict_proba(
            scaled_features
        )[0][1]
    )

    predicted_failure = (
        probability >= 0.5
    )

    return (
        predicted_failure,
        probability,
    )


# ============================================================
# STRATEGY 1 - ADAPTIVE TIMEOUT
# ============================================================

def adaptive_timeout(amount):
    """
    Return timeout multiplier based on transaction value.

    Standard:
        30 ms

    Medium:
        30 * 1.25 = 37.5 ms

    High:
        30 * 1.50 = 45 ms
    """

    absolute_amount = abs(
        float(amount)
    )

    if absolute_amount > 5000:
        return 1.50

    if absolute_amount > 1000:
        return 1.25

    return 1.00


def attempt_timeout_recovery(
    amount,
    processing_time,
):
    """
    A timeout is recovered only when the transaction fits
    inside the adjusted timeout.
    """

    multiplier = adaptive_timeout(
        amount
    )

    adjusted_timeout = (
        BASE_TIMEOUT_MS
        * multiplier
    )

    recovered = (
        processing_time
        <= adjusted_timeout
    )

    return (
        recovered,
        multiplier,
        adjusted_timeout,
    )


# ============================================================
# STRATEGY 2 - BATCH SIZE OPTIMIZATION
# ============================================================

def optimize_batch_size(
    amount,
    predicted_failure,
):
    """
    Thesis batch-size policy.

    Predicted failure -> 100
    High value        -> 200
    Medium value      -> 400
    Standard          -> 1000
    """

    absolute_amount = abs(
        float(amount)
    )

    if predicted_failure:
        return 100

    if absolute_amount > 5000:
        return 200

    if absolute_amount > 1000:
        return 400

    return 1000


def calculate_effective_lag(
    original_lag,
    batch_size,
):
    """
    Controlled experimental workload model.

    Smaller batches reduce effective lag according to:

        effective_lag =
            original_lag
            * sqrt(batch_size / standard_batch_size)

    This is a simulation assumption used by the experiment.
    """

    ratio = (
        batch_size
        / STANDARD_BATCH_SIZE
    )

    effective_lag = (
        original_lag
        * math.sqrt(
            ratio
        )
    )

    return effective_lag


def attempt_lag_recovery(
    kafka_lag,
    batch_size,
):
    """
    Recovery succeeds only when optimized effective lag is
    at or below the experiment's lag threshold.
    """

    effective_lag = (
        calculate_effective_lag(
            kafka_lag,
            batch_size,
        )
    )

    recovered = (
        effective_lag
        <= KAFKA_LAG_THRESHOLD_MS
    )

    return (
        recovered,
        effective_lag,
    )


# ============================================================
# STRATEGY 3 - DATA VALIDATION / CLEANSING
# ============================================================

def validate_and_clean(row):
    """
    Validate and, where justified, clean a transaction.

    The controlled producer creates NEGATIVE_AMOUNT corruption
    by reversing the sign of an otherwise legitimate amount.

    Therefore abs(amount) is an explicitly defined repair for
    that particular injected corruption.

    Missing values and zero amounts are not guessed.
    """

    if row.transaction_id is None:

        return (
            False,
            None,
            "MISSING_TRANSACTION_ID",
        )

    if row.amount is None:

        return (
            False,
            None,
            "MISSING_AMOUNT",
        )

    if (
        row.timestamp is None
        or row.timestamp == ""
    ):

        return (
            False,
            None,
            "INVALID_TIMESTAMP",
        )

    amount = float(
        row.amount
    )

    corruption_type = (
        row.data_corruption_type
        or "NONE"
    )

    # Controlled sign corruption.
    if (
        corruption_type
        == "NEGATIVE_AMOUNT"
        and amount < 0
    ):

        cleaned_amount = abs(
            amount
        )

        if cleaned_amount > 0:

            return (
                True,
                cleaned_amount,
                "NEGATIVE_AMOUNT_REPAIRED",
            )

    # Zero cannot safely be reconstructed.
    if amount == 0:

        return (
            False,
            None,
            "ZERO_AMOUNT_UNRECOVERABLE",
        )

    # Negative amount without the known corruption marker
    # must not be guessed/repaired.
    if amount < 0:

        return (
            False,
            None,
            "UNKNOWN_NEGATIVE_AMOUNT",
        )

    return (
        True,
        amount,
        "VALID",
    )


# ============================================================
# STRATEGY 4 - QUERY COMPLEXITY REDUCTION
# ============================================================

def attempt_query_reduction(
    failure_type,
    processing_time,
    tps,
):
    """
    Simulate switching from the comprehensive processing path
    to a simplified path.

    The simplified path is modeled as requiring 65% of the
    original processing workload.

    Recovery occurs only when the transformed workload falls
    inside the corresponding operating threshold.
    """

    reduced_processing_time = (
        processing_time
        * QUERY_REDUCTION_FACTOR
    )

    reduced_tps_load = (
        tps
        * QUERY_REDUCTION_FACTOR
    )

    if failure_type == "TIMEOUT":

        recovered = (
            reduced_processing_time
            <= BASE_TIMEOUT_MS
        )

        return (
            recovered,
            reduced_processing_time,
            reduced_tps_load,
        )

    if failure_type == "RESOURCE_PRESSURE":

        recovered = (
            reduced_tps_load
            <= RESOURCE_TPS_THRESHOLD
        )

        return (
            recovered,
            reduced_processing_time,
            reduced_tps_load,
        )

    return (
        False,
        reduced_processing_time,
        reduced_tps_load,
    )


# ============================================================
# STRATEGY 5 - DISTRIBUTED RETRY
# ============================================================

def attempt_distributed_retry(
    transaction_id,
    failure_type,
):
    """
    Retry only failures explicitly defined as transient.

    TRANSIENT_FAILURE in producer.py represents a controlled
    one-shot temporary failure.

    Persistent validation/resource/lag/timeout problems are
    NOT automatically repaired by retry.
    """

    if (
        failure_type
        == "TRANSIENT_FAILURE"
    ):

        return True

    return False


# ============================================================
# APPLY SELF-HEALING
# ============================================================

def apply_healing(
    row,
    predicted_failure,
):

    amount = float(
        row.amount
        if row.amount is not None
        else 0.0
    )

    processing_time = float(
        row.processing_time
        if row.processing_time is not None
        else 0.0
    )

    kafka_lag = float(
        row.kafka_lag
        if row.kafka_lag is not None
        else 0.0
    )

    tps = float(
        row.tps
        if row.tps is not None
        else 0.0
    )

    actual_failure = (
        row.failure == 1
    )

    failure_type = (
        row.failure_type
        or "NONE"
    )

    strategies = []

    # Structured records for healing_actions.
    actions = []

    healing_triggered = False
    healing_success = False

    cleaned_amount = amount

    # --------------------------------------------------------
    # PROACTIVE CONFIGURATION
    # --------------------------------------------------------

    timeout_multiplier = (
        adaptive_timeout(
            amount
        )
    )

    adjusted_timeout = (
        BASE_TIMEOUT_MS
        * timeout_multiplier
    )

    batch_size = (
        optimize_batch_size(
            amount,
            predicted_failure,
        )
    )

    # Record proactive strategy names in the existing
    # healing_strategy field exactly as before.

    if timeout_multiplier > 1.0:

        strategies.append(
            f"ADAPTIVE_TIMEOUT_"
            f"{timeout_multiplier:.2f}x"
        )

    if batch_size < STANDARD_BATCH_SIZE:

        strategies.append(
            f"BATCH_OPTIMIZATION_"
            f"{batch_size}"
        )

    # --------------------------------------------------------
    # NO ACTUAL FAILURE
    # --------------------------------------------------------

    if not actual_failure:

        return {
            "healing_triggered":
                False,

            "healing_success":
                False,

            "strategies":
                strategies,

            "actions":
                actions,

            "status":
                "SUCCESS",

            "cleaned_amount":
                cleaned_amount,

            "timeout_multiplier":
                timeout_multiplier,

            "adjusted_timeout":
                adjusted_timeout,

            "batch_size":
                batch_size,

            "effective_lag":
                kafka_lag,

            "recovery_reason":
                "NO_FAILURE",
        }

    healing_triggered = True

    effective_lag = kafka_lag

    recovery_reason = (
        "UNRECOVERED"
    )

    # ========================================================
    # DATA VALIDATION FAILURE
    # ========================================================

    if failure_type == "DATA_VALIDATION":

        strategies.append(
            "DATA_VALIDATION_CLEANSING"
        )

        (
            valid,
            repaired_amount,
            validation_reason,
        ) = validate_and_clean(
            row
        )

        if valid:

            cleaned_amount = (
                repaired_amount
            )

            healing_success = True

            recovery_reason = (
                validation_reason
            )

        else:

            recovery_reason = (
                validation_reason
            )

        actions.append(
            {
                "strategy":
                    "DATA_VALIDATION_CLEANSING",

                "success":
                    bool(valid),

                "details":
                    {
                        "input_amount":
                            amount,

                        "cleaned_amount":
                            repaired_amount,

                        "corruption_type":
                            (
                                row.data_corruption_type
                                or "NONE"
                            ),

                        "validation_reason":
                            validation_reason,

                        "mode":
                            "reactive",
                    },
            }
        )

    # ========================================================
    # TIMEOUT FAILURE
    # ========================================================

    elif failure_type == "TIMEOUT":

        strategies.append(
            "ADAPTIVE_TIMEOUT_RECOVERY"
        )

        (
            timeout_recovered,
            timeout_multiplier,
            adjusted_timeout,
        ) = attempt_timeout_recovery(
            amount,
            processing_time,
        )

        actions.append(
            {
                "strategy":
                    "ADAPTIVE_TIMEOUT_ADJUSTMENT",

                "success":
                    bool(timeout_recovered),

                "details":
                    {
                        "processing_time_ms":
                            processing_time,

                        "base_timeout_ms":
                            BASE_TIMEOUT_MS,

                        "timeout_multiplier":
                            timeout_multiplier,

                        "adjusted_timeout_ms":
                            adjusted_timeout,

                        "mode":
                            "reactive",

                        "purpose":
                            "TIMEOUT_RECOVERY",
                    },
            }
        )

        if timeout_recovered:

            healing_success = True

            recovery_reason = (
                "WITHIN_ADJUSTED_TIMEOUT"
            )

        else:

            # Adaptive timeout was insufficient.
            # Try the simplified query path.

            strategies.append(
                "QUERY_COMPLEXITY_REDUCTION"
            )

            (
                query_recovered,
                reduced_processing_time,
                reduced_tps_load,
            ) = attempt_query_reduction(
                failure_type,
                processing_time,
                tps,
            )

            actions.append(
                {
                    "strategy":
                        "QUERY_COMPLEXITY_REDUCTION",

                    "success":
                        bool(query_recovered),

                    "details":
                        {
                            "original_processing_time_ms":
                                processing_time,

                            "reduced_processing_time_ms":
                                reduced_processing_time,

                            "reduction_factor":
                                QUERY_REDUCTION_FACTOR,

                            "reduced_tps_load":
                                reduced_tps_load,

                            "mode":
                                "reactive",

                            "purpose":
                                "TIMEOUT_RECOVERY",
                        },
                }
            )

            if query_recovered:

                healing_success = True

                recovery_reason = (
                    "SIMPLIFIED_QUERY_WITHIN_TIMEOUT"
                )

            else:

                recovery_reason = (
                    "TIMEOUT_REMAINS_AFTER_HEALING"
                )

    # ========================================================
    # HIGH KAFKA LAG
    # ========================================================

    elif failure_type == "HIGH_KAFKA_LAG":

        # The proactive batch-size selection is now evaluated
        # against the actual lag condition.

        if (
            f"BATCH_OPTIMIZATION_{batch_size}"
            not in strategies
        ):

            strategies.append(
                f"BATCH_OPTIMIZATION_"
                f"{batch_size}"
            )

        (
            lag_recovered,
            effective_lag,
        ) = attempt_lag_recovery(
            kafka_lag,
            batch_size,
        )

        actions.append(
            {
                "strategy":
                    "BATCH_SIZE_OPTIMIZATION",

                "success":
                    bool(lag_recovered),

                "details":
                    {
                        "original_kafka_lag_ms":
                            kafka_lag,

                        "effective_kafka_lag_ms":
                            effective_lag,

                        "lag_threshold_ms":
                            KAFKA_LAG_THRESHOLD_MS,

                        "standard_batch_size":
                            STANDARD_BATCH_SIZE,

                        "optimized_batch_size":
                            batch_size,

                        "mode":
                            "reactive",

                        "purpose":
                            "KAFKA_LAG_RECOVERY",
                    },
            }
        )

        if lag_recovered:

            healing_success = True

            recovery_reason = (
                "EFFECTIVE_LAG_BELOW_THRESHOLD"
            )

        else:

            recovery_reason = (
                "LAG_REMAINS_ABOVE_THRESHOLD"
            )

    # ========================================================
    # RESOURCE PRESSURE
    # ========================================================

    elif failure_type == "RESOURCE_PRESSURE":

        if (
            f"BATCH_OPTIMIZATION_{batch_size}"
            not in strategies
        ):

            strategies.append(
                f"BATCH_OPTIMIZATION_"
                f"{batch_size}"
            )

        batch_ratio = (
            batch_size
            / STANDARD_BATCH_SIZE
        )

        effective_tps_load = (
            tps
            * math.sqrt(
                batch_ratio
            )
        )

        batch_recovered = (
            effective_tps_load
            <= RESOURCE_TPS_THRESHOLD
        )

        actions.append(
            {
                "strategy":
                    "BATCH_SIZE_OPTIMIZATION",

                "success":
                    bool(batch_recovered),

                "details":
                    {
                        "original_tps_load":
                            tps,

                        "effective_tps_load":
                            effective_tps_load,

                        "resource_threshold":
                            RESOURCE_TPS_THRESHOLD,

                        "standard_batch_size":
                            STANDARD_BATCH_SIZE,

                        "optimized_batch_size":
                            batch_size,

                        "mode":
                            "reactive",

                        "purpose":
                            "RESOURCE_PRESSURE_RECOVERY",
                    },
            }
        )

        if batch_recovered:

            healing_success = True

            recovery_reason = (
                "BATCH_OPTIMIZATION_REDUCED_LOAD"
            )

        else:

            strategies.append(
                "QUERY_COMPLEXITY_REDUCTION"
            )

            (
                query_recovered,
                reduced_processing_time,
                reduced_tps_load,
            ) = attempt_query_reduction(
                failure_type,
                processing_time,
                effective_tps_load,
            )

            actions.append(
                {
                    "strategy":
                        "QUERY_COMPLEXITY_REDUCTION",

                    "success":
                        bool(query_recovered),

                    "details":
                        {
                            "effective_tps_before_reduction":
                                effective_tps_load,

                            "reduced_tps_load":
                                reduced_tps_load,

                            "reduced_processing_time_ms":
                                reduced_processing_time,

                            "reduction_factor":
                                QUERY_REDUCTION_FACTOR,

                            "resource_threshold":
                                RESOURCE_TPS_THRESHOLD,

                            "mode":
                                "reactive",

                            "purpose":
                                "RESOURCE_PRESSURE_RECOVERY",
                        },
                }
            )

            if query_recovered:

                healing_success = True

                recovery_reason = (
                    "QUERY_REDUCTION_REDUCED_LOAD"
                )

            else:

                recovery_reason = (
                    "RESOURCE_PRESSURE_REMAINS"
                )

    # ========================================================
    # TRANSIENT FAILURE
    # ========================================================

    elif failure_type == "TRANSIENT_FAILURE":

        strategies.append(
            "DISTRIBUTED_RETRY"
        )

        retry_success = (
            attempt_distributed_retry(
                row.transaction_id,
                failure_type,
            )
        )

        actions.append(
            {
                "strategy":
                    "DISTRIBUTED_RETRY",

                "success":
                    bool(retry_success),

                "details":
                    {
                        "failure_type":
                            failure_type,

                        "transaction_id":
                            row.transaction_id,

                        "mode":
                            "reactive",

                        "purpose":
                            "TRANSIENT_FAILURE_RECOVERY",
                    },
            }
        )

        if retry_success:

            healing_success = True

            recovery_reason = (
                "TRANSIENT_FAILURE_CLEARED_ON_RETRY"
            )

        else:

            recovery_reason = (
                "RETRY_FAILED"
            )

    # ========================================================
    # UNKNOWN FAILURE
    # ========================================================

    else:

        recovery_reason = (
            "UNKNOWN_FAILURE_TYPE"
        )

    # --------------------------------------------------------
    # FINAL STATUS
    # --------------------------------------------------------

    if healing_success:

        status = "RECOVERED"

    else:

        status = "FAILED"

    return {
        "healing_triggered":
            healing_triggered,

        "healing_success":
            healing_success,

        "strategies":
            strategies,

        "actions":
            actions,

        "status":
            status,

        "cleaned_amount":
            cleaned_amount,

        "timeout_multiplier":
            timeout_multiplier,

        "adjusted_timeout":
            adjusted_timeout,

        "batch_size":
            batch_size,

        "effective_lag":
            effective_lag,

        "recovery_reason":
            recovery_reason,
    }


# ============================================================
# PROCESS EACH SPARK MICRO-BATCH
# ============================================================

def process_batch(
    batch_df,
    batch_id,
):

    if batch_df.rdd.isEmpty():
        return

    rows = batch_df.collect()

    print(
        f"\nProcessing self-healing batch "
        f"{batch_id} - "
        f"{len(rows)} transactions"
    )

    connection = (
        get_db_connection()
    )

    cursor = (
        connection.cursor()
    )

    batch_start = (
        time.perf_counter()
    )

    total = 0
    successful = 0
    failed = 0

    predicted_failures = 0

    healing_attempts = 0
    successful_healings = 0

    latency_values = []

    for row in rows:

        transaction_start = (
            time.perf_counter()
        )

        total += 1

        # ----------------------------------------------------
        # ML PREDICTION
        # ----------------------------------------------------

        (
            predicted_failure,
            failure_probability,
        ) = predict_failure(
            row
        )

        if predicted_failure:

            predicted_failures += 1

        # ----------------------------------------------------
        # APPLY HEALING
        # ----------------------------------------------------

        healing_start = (
            time.perf_counter()
        )

        result = apply_healing(
            row,
            predicted_failure,
        )

        recovery_time_ms = (
            (
                time.perf_counter()
                - healing_start
            )
            * 1000
        )

        if result[
            "healing_triggered"
        ]:

            healing_attempts += 1

        if result[
            "healing_success"
        ]:

            successful_healings += 1

        if result["status"] in (
            "SUCCESS",
            "RECOVERED",
        ):

            successful += 1

        else:

            failed += 1

        processing_latency_ms = (
            (
                time.perf_counter()
                - transaction_start
            )
            * 1000
        )

        latency_values.append(
            processing_latency_ms
        )

        # Remove duplicate strategy names while preserving
        # their original order.
        unique_strategies = list(
            dict.fromkeys(
                result["strategies"]
            )
        )

        strategy_text = (
            ",".join(
                unique_strategies
            )
            if unique_strategies
            else "NONE"
        )

        # ----------------------------------------------------
        # SAVE MAIN TRANSACTION RESULT
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO healing_results (
                transaction_id,
                event_time,
                amount,
                actual_class,
                predicted_failure,
                failure_probability,
                failure_type,
                failure_occurred,
                healing_triggered,
                healing_strategy,
                healing_success,
                processing_latency_ms,
                recovery_time_ms,
                status
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                row.transaction_id,
                row.timestamp,

                result[
                    "cleaned_amount"
                ],

                row.fraud,

                predicted_failure,

                failure_probability,

                row.failure_type,

                bool(
                    row.failure
                ),

                result[
                    "healing_triggered"
                ],

                strategy_text,

                result[
                    "healing_success"
                ],

                processing_latency_ms,

                recovery_time_ms,

                result[
                    "status"
                ],
            ),
        )

        # ----------------------------------------------------
        # SAVE INDIVIDUAL HEALING ACTIONS
        # ----------------------------------------------------

        insert_healing_actions(
            cursor=cursor,

            transaction_id=(
                row.transaction_id
            ),

            failure_type=(
                row.failure_type
                or "NONE"
            ),

            actions=(
                result["actions"]
            ),
        )

    # Commit healing_results + healing_actions together.
    connection.commit()

    elapsed = (
        time.perf_counter()
        - batch_start
    )

    throughput = (
        total / elapsed
        if elapsed > 0
        else 0
    )

    avg_latency = (
        sum(latency_values)
        / len(latency_values)
        if latency_values
        else 0
    )

    # --------------------------------------------------------
    # SAVE BATCH METRICS
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO pipeline_metrics (
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
        VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s
        )
        """,
        (
            "SELF_HEALING",
            batch_id,
            total,
            throughput,
            avg_latency,
            total,
            successful,
            failed,
            predicted_failures,
            healing_attempts,
            successful_healings,
        ),
    )

    connection.commit()

    cursor.close()
    connection.close()

    success_rate = (
        successful
        / total
        * 100
        if total
        else 0
    )

    failure_rate = (
        failed
        / total
        * 100
        if total
        else 0
    )

    healing_rate = (
        successful_healings
        / healing_attempts
        * 100
        if healing_attempts
        else 0
    )

    print(
        "-" * 70
    )

    print(
        f"Batch ID            : "
        f"{batch_id}"
    )

    print(
        f"Transactions        : "
        f"{total}"
    )

    print(
        f"Successful          : "
        f"{successful}"
    )

    print(
        f"Failed              : "
        f"{failed}"
    )

    print(
        f"Predicted failures  : "
        f"{predicted_failures}"
    )

    print(
        f"Healing attempts    : "
        f"{healing_attempts}"
    )

    print(
        f"Successful healings : "
        f"{successful_healings}"
    )

    print(
        f"Healing success rate: "
        f"{healing_rate:.2f}%"
    )

    print(
        f"Pipeline success    : "
        f"{success_rate:.2f}%"
    )

    print(
        f"Pipeline failure    : "
        f"{failure_rate:.2f}%"
    )

    print(
        f"Average latency     : "
        f"{avg_latency:.4f} ms"
    )

    print(
        f"Throughput          : "
        f"{throughput:.2f} TPS"
    )

    print(
        "-" * 70
    )


# ============================================================
# READ KAFKA STREAM
# ============================================================

raw_stream = (
    spark.readStream

    .format(
        "kafka"
    )

    .option(
        "kafka.bootstrap.servers",
        KAFKA_BOOTSTRAP_SERVERS,
    )

    .option(
        "subscribe",
        KAFKA_TOPIC,
    )

    .option(
        "startingOffsets",
        "earliest",
    )

    .option(
        "maxOffsetsPerTrigger",
        "1000",
    )

    .option(
        "failOnDataLoss",
        "false",
    )

    .load()
)


parsed_stream = (
    raw_stream

    .selectExpr(
        "CAST(value AS STRING) "
        "AS json_value"
    )

    .select(
        from_json(
            col(
                "json_value"
            ),
            schema,
        ).alias(
            "data"
        )
    )

    .select(
        "data.*"
    )
)


# ============================================================
# START STREAM
# ============================================================

query = (
    parsed_stream.writeStream

    .foreachBatch(
        process_batch
    )

    .option(
        "checkpointLocation",
        CHECKPOINT_PATH,
    )

    .trigger(
        processingTime="5 seconds"
    )

    .start()
)


print(
    "\nSelf-healing pipeline is running."
)

print(
    "Waiting for Kafka transactions...\n"
)

query.awaitTermination()