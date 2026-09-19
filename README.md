# Real-Time Self-Healing Data Pipeline for Reliable Transaction Processing

A real-time transaction-processing research prototype that integrates **Apache Kafka, Apache Spark Structured Streaming, XGBoost, PostgreSQL, Grafana, and Docker Compose** to detect processing failures and automatically apply targeted recovery strategies.

This project investigates how **machine-learning-based failure prediction** and **automated self-healing mechanisms** can be incorporated into a streaming data pipeline to improve reliability and reduce the need for manual intervention.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Problem](#research-problem)
- [Research Aim](#research-aim)
- [Research Objectives](#research-objectives)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End Processing Flow](#end-to-end-processing-flow)
- [Technology Stack](#technology-stack)
- [Dataset](#dataset)
- [Controlled Failure Simulation](#controlled-failure-simulation)
- [Failure Prediction Model](#failure-prediction-model)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Self-Healing Engine](#self-healing-engine)
- [Self-Healing Strategies](#self-healing-strategies)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Starting the Environment](#starting-the-environment)
- [Docker Services](#docker-services)
- [Kafka Configuration](#kafka-configuration)
- [Spark Processing](#spark-processing)
- [Running the Baseline Pipeline](#running-the-baseline-pipeline)
- [Running the Self-Healing Pipeline](#running-the-self-healing-pipeline)
- [PostgreSQL Storage](#postgresql-storage)
- [Grafana Monitoring](#grafana-monitoring)
- [Model Evaluation](#model-evaluation)
- [Stopping the Environment](#stopping-the-environment)
- [Troubleshooting](#troubleshooting)
- [Experimental Scope](#experimental-scope)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Research Ethics and Data Usage](#research-ethics-and-data-usage)
- [Author](#author)
- [Academic Notice](#academic-notice)

---

# Project Overview

Real-time data pipelines are widely used in systems that continuously process high-volume events such as financial transactions, application events, IoT data, operational telemetry, and customer activity.

These pipelines may experience failures caused by conditions such as:

- processing timeouts,
- consumer lag,
- resource pressure,
- malformed or invalid data,
- temporary processing errors,
- downstream service delays,
- unexpected runtime conditions.

Traditional monitoring systems can identify many of these conditions, but recovery often depends on predefined operational procedures or manual intervention.

This project explores a different approach: a **self-healing streaming pipeline** that combines failure prediction, failure detection, targeted recovery actions, persistent auditing, and real-time monitoring.

The system processes transactions using Apache Kafka and Apache Spark Structured Streaming. An XGBoost classifier estimates failure risk using transaction and system-level features. When a failure condition occurs, the healing engine selects and applies one or more recovery strategies.

The resulting transaction status, prediction information, recovery actions, and pipeline metrics are stored in PostgreSQL and visualized using Grafana.

---

# Research Problem

Real-time transaction pipelines are expected to remain available even when abnormal processing conditions occur.

However, several operational problems can interrupt or degrade processing:

- Kafka consumer lag can increase.
- Individual transactions may exceed processing time limits.
- Invalid transaction data can cause processing errors.
- Temporary failures may interrupt processing.
- Resource pressure can reduce processing stability.
- Manual recovery can increase operational response time.

Many existing monitoring approaches primarily focus on **detecting and reporting failures**.

This research investigates whether a streaming pipeline can go beyond monitoring by combining:

1. failure prediction,
2. failure detection,
3. automated recovery,
4. persistent auditing,
5. real-time monitoring.

---

# Research Aim

The main aim of this project is to:

> **Develop a real-time self-healing data pipeline capable of detecting transaction-processing failures and automatically applying targeted recovery strategies while maintaining continuous transaction processing.**

---

# Research Objectives

The project focuses on the following objectives:

1. Design a real-time transaction-processing architecture using Kafka, Spark, PostgreSQL, and Grafana.

2. Develop a controlled environment for introducing multiple types of pipeline-processing failures.

3. Build an XGBoost-based model for predicting transaction-processing failure risk.

4. Integrate the trained machine-learning model with Spark Structured Streaming.

5. Implement multiple self-healing strategies for different failure conditions.

6. Record transaction outcomes, recovery actions, and pipeline metrics in PostgreSQL.

7. Provide operational monitoring through Grafana dashboards.

8. Compare baseline and self-healing pipeline behavior under controlled experimental conditions.

9. Provide a reproducible Docker-based environment for executing the research prototype.

---

# Key Features

The implementation includes:

- Real-time transaction streaming using **Apache Kafka**
- Distributed-style stream processing using **Spark Structured Streaming**
- Machine-learning-based failure prediction using **XGBoost**
- Controlled failure injection
- Automatic failure detection
- Five self-healing strategies
- Strategy chaining for selected failures
- Baseline processing mode
- Self-healing processing mode
- PostgreSQL result persistence
- Healing-action audit logging
- Pipeline metric collection
- Grafana monitoring
- Docker-based service orchestration
- Separate machine-learning evaluation workflow

---

# System Architecture

The high-level architecture is:

```text
                     Transaction Dataset
                            |
                            v
                     Transaction Producer
                            |
                            v
                    +---------------+
                    | Apache Kafka  |
                    | transactions  |
                    +-------+-------+
                            |
                            v
                 +----------------------+
                 | Spark Structured     |
                 | Streaming            |
                 +----------+-----------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Feature Preparation        Failure Detection
              |                           |
              v                           |
       +--------------+                   |
       |   XGBoost    |                   |
       | Prediction   |                   |
       +------+-------+                   |
              |                           |
              +-------------+-------------+
                            |
                            v
                   +------------------+
                   | Self-Healing     |
                   | Decision Engine  |
                   +--------+---------+
                            |
          +-----------------+-------------------+
          |                 |                   |
          v                 v                   v
      Recovery          Transaction          Healing
      Actions            Results             Logs
          |                 |                   |
          +-----------------+-------------------+
                            |
                            v
                    +---------------+
                    | PostgreSQL    |
                    +-------+-------+
                            |
                            v
                    +---------------+
                    | Grafana       |
                    | Monitoring    |
                    +---------------+
```

The services are orchestrated through Docker Compose.

---

# End-to-End Processing Flow

The self-healing pipeline follows the sequence below.

### 1. Transaction Generation

Transactions are read from the source dataset and converted into streaming events.

Additional experimental fields are generated to represent pipeline operating conditions.

### 2. Failure Injection

Controlled failure conditions are introduced for selected transactions.

The purpose of failure injection is to create a reproducible environment in which recovery strategies can be evaluated.

### 3. Kafka Streaming

Transactions are published to the Kafka topic:

```text
transactions
```

Kafka acts as the event-streaming layer between the transaction producer and Spark.

### 4. Spark Structured Streaming

Spark consumes transaction events from Kafka and processes them using micro-batches.

### 5. Feature Preparation

The transaction and system attributes required by the prediction model are prepared.

### 6. Failure Prediction

The trained XGBoost model estimates whether a transaction is at risk of processing failure.

### 7. Failure Detection

The pipeline evaluates actual processing conditions and determines whether a failure has occurred.

### 8. Healing Strategy Selection

The healing engine selects a recovery strategy according to the detected failure condition.

### 9. Recovery

One or more healing strategies may be executed.

### 10. Persistence

Transaction results and healing actions are written to PostgreSQL.

### 11. Monitoring

Grafana queries the stored metrics and provides dashboards for operational monitoring and research analysis.

---

# Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Programming | Python | Producer, ML, processing and experimental logic |
| Event Streaming | Apache Kafka | Transaction event transport |
| Coordination | Zookeeper | Kafka coordination in the experimental environment |
| Stream Processing | Apache Spark 3.4.1 | Structured Streaming processing |
| Machine Learning | XGBoost | Failure-risk classification |
| ML Utilities | scikit-learn | Preprocessing and evaluation |
| Numerical Processing | NumPy | Feature transformation |
| Data Processing | pandas | Dataset preparation and analysis |
| Model Serialization | joblib | Saving/loading ML artifacts |
| Database | PostgreSQL 15 | Results, metrics and audit persistence |
| Database Administration | pgAdmin | Database inspection |
| Monitoring | Grafana 10 | Dashboard visualization |
| Containerization | Docker | Service isolation |
| Orchestration | Docker Compose | Multi-service experimental environment |

Apache Spark supports Kafka as a Structured Streaming source, which is the integration pattern used by the project. :contentReference[oaicite:0]{index=0}

Docker Compose provides the multi-container orchestration model used to define and start the project's services. :contentReference[oaicite:1]{index=1}

---

# Dataset

The transaction source is the **ULB Credit Card Fraud Detection dataset**.

The dataset contains anonymized credit-card transaction records.

The original fields include:

```text
Time
V1
V2
...
V28
Amount
Class
```

`V1` to `V28` are anonymized transformed numerical features.

`Amount` represents the transaction amount.

`Class` represents the fraud label in the original dataset:

```text
0 = non-fraudulent transaction
1 = fraudulent transaction
```

The fraud label is used as one of the transaction-context features in this research.

It is important to distinguish between:

```text
Fraud classification
```

and:

```text
Pipeline failure prediction
```

They represent different problems.

The objective of this project is **not to build a fraud-detection model**.

Instead, the pipeline predicts whether a transaction-processing event may experience a processing failure.

---

# Controlled Failure Simulation

The original credit-card dataset does not contain Kafka lag, processing timeouts, Spark resource pressure, or other infrastructure failures.

Therefore, the research introduces controlled pipeline conditions.

The generated ML/experimental dataset contains fields such as:

```text
transaction_id
amount
fraud
processing_time
kafka_lag
tps
failure
failure_type
original_amount
data_corruption_type
```

The main experimental failure categories are:

```text
DATA_VALIDATION
HIGH_KAFKA_LAG
RESOURCE_PRESSURE
TIMEOUT
TRANSIENT_FAILURE
```

Transactions without an injected/detected failure are represented as:

```text
NONE
```

The controlled simulation allows the same types of failure conditions to be reproduced during experimentation.

---

# Failure Prediction Model

The project uses an **XGBoost binary classifier** for failure-risk prediction.

The prediction target is:

```text
0 = Normal processing
1 = Potential processing failure
```

## Model Input Features

The primary model-development features are:

```text
amount
fraud
processing_time
kafka_lag
tps
```

An additional transformed amount feature is generated:

```text
amount_log = log(1 + abs(amount))
```

Therefore, the final feature representation can include:

```text
amount
fraud
processing_time
kafka_lag
tps
amount_log
```

---

## Why These Features?

### Amount

Provides transaction-level context and preserves the signed amount used by the controlled validation experiment.

### Fraud Indicator

Provides information from the original transaction context.

It is not used to claim that fraud itself causes pipeline failure.

### Processing Time

Represents the amount of time associated with transaction processing.

This is relevant to timeout-related conditions.

### Kafka Lag

Represents delay/backlog conditions in the streaming layer.

Large lag values can indicate that processing is not keeping pace with incoming events.

### Throughput / TPS

Represents transaction-processing rate and contributes information about the current operating state of the pipeline.

### Log-Transformed Amount

Provides an additional magnitude representation for transaction amounts with a wide numerical range.

---

# Machine Learning Workflow

The ML workflow is kept separate from the streaming execution workflow.

```text
Generated Experimental Dataset
            |
            v
Chronological Split
            |
     +------+------+
     |             |
     v             v
 Training       Held-Out
 Dataset        Test Dataset
     |
     v
Training-Only Preprocessing
     |
     v
Feature Transformation
     |
     v
XGBoost Training
     |
     v
Model Serialization
     |
     +--------------------+
                          |
                          v
                  Spark Integration
```

## Chronological Splitting

The dataset is divided chronologically rather than randomly for the main evaluation workflow.

This helps preserve event ordering and avoids mixing later records into earlier training observations.

## Training-Only Preprocessing

Preprocessing statistics are derived from the training portion only.

For example, missing-value medians are calculated using training data and subsequently applied to the held-out data.

This reduces information leakage from the test set into the training process.

## Feature Scaling

The project uses `StandardScaler` in the ML evaluation workflow.

The scaler is fitted using the training set and then applied to the held-out set.

## Class Weighting

Because the normal and failure classes are not equally represented, the XGBoost evaluation workflow calculates:

```text
scale_pos_weight =
number of negative training samples
/
number of positive training samples
```

This allows the model-training process to account for class imbalance.

---

# Model Evaluation

The repository includes a separate model-evaluation workflow.

For example:

```text
ml/train_model_100k.py
```

The purpose of this script is to:

1. load a chronological subset of the generated experimental dataset,
2. separate training and held-out testing records,
3. perform training-only preprocessing,
4. train an XGBoost classifier,
5. generate predictions for the held-out test set,
6. calculate classification metrics,
7. generate diagnostic plots,
8. save the evaluation artifacts separately.

Generated artifacts can include:

```text
confusion_matrix_100k.png
roc_curve_100k.png
feature_importance_100k.png
model_metrics_100k.json
```

The separate evaluation model is stored independently so that the model used by an already-completed streaming experiment is not unintentionally overwritten.

Example:

```text
failure_prediction_model_100k_evaluation.pkl
```

---

# Self-Healing Engine

The self-healing engine is responsible for responding to detected pipeline failures.

A simplified decision flow is:

```text
Transaction
     |
     v
Failure Risk Prediction
     |
     v
Processing
     |
     v
Failure?
   /   \
 No     Yes
 |       |
 v       v
Success  Identify Failure Type
             |
             v
       Select Strategy
             |
             v
       Apply Healing
             |
             v
       Re-evaluate
             |
        +----+----+
        |         |
        v         v
    Recovered    Failed
```

The system can apply different recovery actions depending on the detected failure.

Selected strategies may also be chained.

---

# Self-Healing Strategies

The project implements five healing strategies.

---

## 1. Adaptive Timeout Adjustment

### Purpose

Handle transactions that exceed the standard processing-time threshold.

### Concept

Instead of immediately failing a slow transaction, the system can increase the allowed processing window according to transaction context.

A simplified conceptual rule is:

```text
High-value transaction
    -> larger timeout multiplier

Medium-value transaction
    -> moderate timeout multiplier

Standard transaction
    -> standard timeout
```

The experimental implementation uses controlled timeout adjustment rules.

### Intended Benefit

Reduce unnecessary transaction failures caused by temporary or context-dependent processing delays.

---

## 2. Batch Size Optimization

### Purpose

Respond to processing pressure and Kafka lag.

### Concept

Large micro-batches may increase processing pressure when the system is already experiencing backlog or constrained resources.

The strategy selects a smaller processing batch under selected failure conditions.

Conceptually:

```text
Normal condition
      |
      v
Large/default batch

High lag / pressure
      |
      v
Reduced batch size
```

### Intended Benefit

Allow the processing layer to work with smaller units during stressed conditions.

---

## 3. Data Validation and Cleansing

### Purpose

Handle invalid or corrupted transaction values.

### Validation checks may include:

```text
transaction ID validity
amount validity
timestamp validity
required fields
```

The controlled experiment can introduce invalid values such as a negative amount representation.

The validation strategy identifies the controlled corruption and applies the corresponding cleansing logic when recovery is possible.

### Intended Benefit

Prevent recoverable data-quality problems from unnecessarily terminating transaction processing.

---

## 4. Query Complexity Reduction

### Purpose

Provide a simplified execution path after selected processing failures.

### Concept

If the standard processing path cannot complete successfully, the system can switch to a reduced-complexity path.

Conceptually:

```text
Comprehensive Processing
          |
        Failure
          |
          v
Simplified Processing
```

This behaves similarly to a fallback/circuit-breaker concept within the experimental implementation.

### Intended Benefit

Allow essential processing to continue when the full processing path is temporarily unsuitable.

---

## 5. Distributed Retry

### Purpose

Recover from transient processing failures.

### Concept

Instead of repeatedly retrying the same failed execution context, the transaction can be rescheduled for another processing attempt.

Conceptually:

```text
Executor / processing attempt
          |
        Failure
          |
          v
      Retry Logic
          |
          v
Fresh Processing Attempt
```

### Intended Benefit

Improve recovery from temporary processing conditions.

---

# Strategy Chaining

A transaction may require more than one healing action.

For example:

```text
Initial failure
      |
      v
Batch Optimization
      |
      v
Still failing
      |
      v
Query Complexity Reduction
      |
      v
Recovered
```

Therefore:

> A healing action is not equivalent to a unique recovered transaction.

The healing-action log should be interpreted as an audit trail of recovery attempts rather than a one-to-one count of recovered transactions.

---

# Repository Structure

A typical project structure is:

```text
kafka-spark-self-healing-pipeline/
│
├── data/
│   └── creditcard.csv
│
├── database/
│   └── database initialization / SQL resources
│
├── grafana/
│   └── dashboard / provisioning resources
│
├── kafka/
│   └── Kafka-related configuration
│
├── ml/
│   ├── failure_prediction_model.pkl
│   ├── failure_prediction_model_100k_evaluation.pkl
│   ├── failure_simulation_dataset.csv
│   ├── train_model_100k.py
│   │
│   ├── outputs/
│   │   └── model-development artifacts
│   │
│   └── outputs_100k/
│       ├── confusion_matrix_100k.png
│       ├── roc_curve_100k.png
│       ├── feature_importance_100k.png
│       └── model_metrics_100k.json
│
├── scripts/
│   └── producer / experimental scripts
│
├── spark/
│   ├── baseline_processor.py
│   ├── self_healing_processor.py
│   ├── stream_processor.py
│   └── Dockerfile
│
├── docker-compose-full.yml
│
├── .gitignore
│
└── README.md
```

The exact contents may evolve as the research implementation is refined.

---

# Prerequisites

Before running the project, install:

### Required

- Git
- Docker Desktop
- Docker Compose
- Python 3.x

### Recommended

- Visual Studio Code
- pgAdmin or another PostgreSQL client
- Modern web browser

On Windows, PowerShell can be used to execute the commands in this README.

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
```

Enter the project directory:

```bash
cd kafka-spark-self-healing-pipeline
```

---

## 2. Verify Docker

Check Docker:

```bash
docker --version
```

Check Docker Compose:

```bash
docker compose version
```

Make sure Docker Desktop is running before starting the environment.

---

# Starting the Environment

The multi-service environment is defined in:

```text
docker-compose-full.yml
```

Start the services:

```bash
docker compose -f docker-compose-full.yml up -d
```

Docker Compose manages the multi-container application from the Compose configuration. :contentReference[oaicite:2]{index=2}

Check service status:

```bash
docker ps
```

---

# Docker Services

The environment includes the following major services.

| Service | Purpose | Host Port |
|---|---|---:|
| Zookeeper | Kafka coordination | `2181` |
| Kafka | Transaction streaming | `9092` |
| Spark Master | Spark cluster coordination | `8080` / `7077` |
| Spark Worker | Spark execution worker | `8081` |
| PostgreSQL | Persistent storage | `5432` |
| pgAdmin | Database administration | `5050` |
| Grafana | Monitoring dashboard | `3000` |

---

# Useful Interfaces

## Spark Master

:contentReference[oaicite:3]{index=3}

The Spark Master interface can be used to inspect:

- registered workers,
- available cores,
- available memory,
- running applications,
- completed applications.

---

## Spark Worker

:contentReference[oaicite:4]{index=4}

---

## pgAdmin

:contentReference[oaicite:5]{index=5}

---

## Grafana

:contentReference[oaicite:6]{index=6}

---

# Kafka Configuration

The primary Kafka topic used by the project is:

```text
transactions
```

Kafka receives transaction events from the producer.

Spark Structured Streaming subscribes to the Kafka topic and processes incoming records. Spark's Kafka integration supports using Kafka as a streaming source. :contentReference[oaicite:7]{index=7}

---

## Inspect Kafka Topics

Example:

```bash
docker exec -it selfhealing-kafka \
kafka-topics \
--bootstrap-server localhost:9092 \
--list
```

---

## Inspect the Transactions Topic

```bash
docker exec -it selfhealing-kafka \
kafka-topics \
--bootstrap-server localhost:9092 \
--describe \
--topic transactions
```

---

## Inspect Topic Offsets

Depending on the Kafka image/configuration, Kafka command-line utilities can be used to inspect partition offsets and consumer state.

This is useful when validating:

- whether the producer has written transactions,
- whether Spark has consumed the stream,
- whether a replay begins from the intended offset.

---

# Spark Processing

The repository contains separate processing applications for baseline and self-healing execution.

---

## Baseline Processor

```text
spark/baseline_processor.py
```

The baseline processor:

- consumes transactions from Kafka,
- processes transaction events,
- records successful and failed outcomes,
- does not apply the complete self-healing workflow.

It provides a reference execution mode for comparison with the self-healing system.

---

## Self-Healing Processor

```text
spark/self_healing_processor.py
```

The self-healing processor:

- consumes Kafka transactions,
- prepares ML features,
- loads the trained prediction model,
- performs failure-risk prediction,
- detects processing failures,
- selects healing strategies,
- applies recovery actions,
- logs healing attempts,
- stores final transaction status,
- records pipeline metrics.

---

# Kafka Package for Spark

The Spark job uses the Spark SQL Kafka connector.

For Spark 3.4.1 with Scala 2.12, the package used by the project is:

```text
org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1
```

---

# Running the Baseline Pipeline

Submit the baseline application:

```powershell
docker exec -it selfhealing-spark-master /opt/spark/bin/spark-submit `
  --master spark://selfhealing-spark-master:7077 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 `
  /opt/spark-apps/baseline_processor.py
```

On Bash/Linux/macOS:

```bash
docker exec -it selfhealing-spark-master /opt/spark/bin/spark-submit \
  --master spark://selfhealing-spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
  /opt/spark-apps/baseline_processor.py
```

---

# Running the Self-Healing Pipeline

Submit the self-healing application:

```powershell
docker exec -it selfhealing-spark-master /opt/spark/bin/spark-submit `
  --master spark://selfhealing-spark-master:7077 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 `
  /opt/spark-apps/self_healing_processor.py
```

On Bash/Linux/macOS:

```bash
docker exec -it selfhealing-spark-master /opt/spark/bin/spark-submit \
  --master spark://selfhealing-spark-master:7077 \
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.1 \
  /opt/spark-apps/self_healing_processor.py
```

---

# Spark Checkpoints

Spark Structured Streaming uses checkpoints to maintain streaming progress.

The project may use checkpoint directories such as:

```text
/tmp/checkpoints/baseline
```

and:

```text
/tmp/checkpoints/self_healing
```

Checkpoint state affects Kafka replay behavior.

When intentionally performing a clean experimental replay, checkpoint state should be handled carefully.

> Do not delete checkpoints from an active production streaming application. In this project, checkpoint removal is used only as part of controlled experimental reset procedures.

---

# Kafka Starting Offsets

Spark Kafka streaming sources can be configured using options such as:

```python
.option("startingOffsets", "earliest")
```

or:

```python
.option("startingOffsets", "latest")
```

The correct setting depends on whether the experiment should:

- replay existing Kafka records, or
- process only newly arriving events.

For controlled replay experiments, `earliest` may be used together with a cleared experimental checkpoint.

---

# Micro-Batch Control

The Kafka source can also limit the number of offsets processed per trigger.

Example:

```python
.option("maxOffsetsPerTrigger", "1000")
```

This can be useful when creating repeatable micro-batch conditions.

The chosen micro-batch configuration can influence:

- throughput,
- processing latency,
- memory use,
- execution duration,
- resource pressure.

Therefore, performance comparisons should use equivalent batching conditions.

---

# PostgreSQL Storage

PostgreSQL provides persistent storage for the experimental pipeline.

Important tables include:

```text
baseline_results
healing_results
pipeline_metrics
experiment_runs
healing_actions
```

---

## `baseline_results`

Stores transaction-processing outcomes for baseline execution.

Typical information includes:

- transaction identifier,
- processing status,
- failure information,
- processing metadata.

---

## `healing_results`

Stores final transaction outcomes from the self-healing pipeline.

Typical statuses include:

```text
SUCCESS
RECOVERED
FAILED
```

Conceptually:

```text
SUCCESS
    Transaction completed without requiring recovery.

RECOVERED
    Transaction experienced a failure but completed after healing.

FAILED
    Transaction remained unsuccessful after available recovery actions.
```

---

## `healing_actions`

Stores the recovery actions performed by the healing engine.

Typical information includes:

```text
transaction_id
processed_at
strategy
trigger_reason
success
details
```

The `details` field can store additional structured information about the healing action.

---

## `pipeline_metrics`

Stores batch-level operational measurements.

Typical fields include:

```text
pipeline_type
batch_id
batch_size
throughput_tps
avg_latency_ms
cpu_usage
memory_usage
total_transactions
successful_transactions
failed_transactions
predicted_failures
healing_attempts
successful_healings
```

These metrics support monitoring and experimental analysis.

---

# Accessing PostgreSQL from PowerShell

Open PostgreSQL inside the container:

```powershell
docker exec -it selfhealing-postgres psql `
  -U postgres `
  -d selfhealing
```

Or execute a single query:

```powershell
docker exec -it selfhealing-postgres psql `
  -U postgres `
  -d selfhealing `
  -P pager=off `
  -c "SELECT COUNT(*) FROM healing_results;"
```

---

# Useful Database Queries

## View Available Tables

```sql
\dt
```

## Count Baseline Records

```sql
SELECT COUNT(*)
FROM baseline_results;
```

## Count Self-Healing Records

```sql
SELECT COUNT(*)
FROM healing_results;
```

## Inspect Healing Actions

```sql
SELECT
    strategy,
    COUNT(*) AS attempts
FROM healing_actions
GROUP BY strategy
ORDER BY attempts DESC;
```

## Inspect Pipeline Metric Availability

```sql
SELECT
    pipeline_type,
    COUNT(*) AS batches
FROM pipeline_metrics
GROUP BY pipeline_type;
```

These queries inspect the experiment without embedding specific research-result values in the README.

---

# Grafana Monitoring

Grafana provides the monitoring and visualization layer.

The dashboard is designed to display information such as:

- overall failure recovery,
- recovered transactions,
- baseline and self-healing transaction status,
- failure outcomes,
- failure categories,
- recovery by failure type,
- healing strategy activity,
- throughput,
- processing latency.

---

## Dashboard Purpose

The Grafana dashboard serves two purposes.

### Operational Monitoring

It allows the user to inspect the current or stored state of the pipeline.

### Research Analysis

It provides a visual representation of pipeline behavior for experimental evaluation.

---

# Throughput and Latency Interpretation

The pipeline records throughput and processing-latency measurements.

These measurements should be interpreted according to the conditions under which they were collected.

Factors that can affect them include:

- Kafka replay mode,
- micro-batch size,
- `maxOffsetsPerTrigger`,
- Spark worker resources,
- whether the producer is active during processing,
- number of available executors,
- additional self-healing computations.

Therefore, a direct baseline-vs-self-healing performance comparison should use:

```text
Same Kafka records
Same starting offsets
Same micro-batch configuration
Same Spark resources
Same replay conditions
Same measurement definition
```

---

# Running the ML Evaluation

The separate evaluation script can be executed using:

```powershell
python .\ml\train_model_100k.py
```

The script generates evaluation artifacts under:

```text
ml/outputs_100k/
```

Possible generated files include:

```text
confusion_matrix_100k.png
roc_curve_100k.png
feature_importance_100k.png
model_metrics_100k.json
```

The evaluation model is saved separately from the streaming model.

This prevents the evaluation workflow from unintentionally changing the model associated with an already-completed pipeline experiment.

---

# Generated Data Files

The experimental workflow may generate files such as:

```text
ml/failure_simulation_dataset.csv
```

This file contains the transaction-level experimental features and controlled failure labels used for model development and evaluation.

Because generated datasets can be large, they do not necessarily need to be committed to Git.

---

# Recommended `.gitignore`

A research repository should avoid committing unnecessary generated data, secrets, local environments, and temporary files.

Example:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo

# Virtual environments
venv/
.venv/
env/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Environment variables / secrets
.env
.env.*
*.secret

# Logs
*.log
logs/

# Spark checkpoints
checkpoints/
spark-checkpoints/

# Large/raw datasets
data/creditcard.csv
ml/failure_simulation_dataset.csv

# Temporary files
*.tmp
*.temp

# Jupyter
.ipynb_checkpoints/
```

Whether trained model artifacts should be committed depends on the intended reproducibility workflow.

---

# Stopping the Environment

Stop the Compose environment:

```bash
docker compose -f docker-compose-full.yml down
```

To stop without removing the services:

```bash
docker compose -f docker-compose-full.yml stop
```

To restart:

```bash
docker compose -f docker-compose-full.yml start
```

---

# Viewing Logs

## Kafka

```bash
docker logs selfhealing-kafka
```

## Spark Master

```bash
docker logs selfhealing-spark-master
```

## Spark Worker

```bash
docker logs selfhealing-spark-worker
```

## PostgreSQL

```bash
docker logs selfhealing-postgres
```

Use:

```bash
docker logs -f <container-name>
```

to follow logs continuously.

---

# Troubleshooting

## Docker Containers Are Not Running

Check:

```bash
docker ps -a
```

Inspect a failed container:

```bash
docker logs <container-name>
```

Restart the environment:

```bash
docker compose -f docker-compose-full.yml down
docker compose -f docker-compose-full.yml up -d
```

---

## Kafka Topic Does Not Exist

List topics:

```bash
docker exec -it selfhealing-kafka \
kafka-topics \
--bootstrap-server localhost:9092 \
--list
```

Create the transaction topic if required:

```bash
docker exec -it selfhealing-kafka \
kafka-topics \
--bootstrap-server localhost:9092 \
--create \
--topic transactions \
--partitions 1 \
--replication-factor 1
```

---

## Spark Receives No Transactions

Check:

1. Kafka is running.
2. The `transactions` topic exists.
3. The topic contains records.
4. Spark is connected to the correct Kafka broker.
5. `startingOffsets` is appropriate.
6. Existing checkpoints are not causing Spark to resume after the desired data.
7. The producer is using the correct topic.

A streaming application configured with:

```python
.option("startingOffsets", "latest")
```

will normally begin from new data when no prior checkpoint controls progress.

For a deliberate replay of existing experimental data, use the appropriate replay configuration.

---

## Spark Worker Cannot Import Python Packages

If errors such as:

```text
ModuleNotFoundError: No module named 'numpy'
```

occur, verify that the required Python dependencies are installed inside the Spark worker environment.

Typical dependencies include:

```text
numpy
pandas
joblib
scikit-learn
xgboost
psycopg2-binary
```

Installing a package only on the host computer does not automatically make it available inside a Docker container.

---

## PostgreSQL Connection Failure

Verify the container:

```bash
docker ps
```

Inspect logs:

```bash
docker logs selfhealing-postgres
```

Verify the database port:

```text
5432
```

---

## Grafana Shows No Data

Check:

1. PostgreSQL is running.
2. Grafana datasource configuration is correct.
3. The relevant PostgreSQL tables contain records.
4. The Grafana time filter includes the stored timestamps.
5. Panel SQL queries are pointing to the correct tables.

A restrictive Grafana time range can make stored experiment data appear missing even when it is present in PostgreSQL.

---

## Model Cannot Be Loaded

Verify that:

- the model file exists,
- `joblib` is installed,
- compatible XGBoost/scikit-learn versions are available,
- Spark containers can access the model path.

Model serialization can be sensitive to dependency-version differences, so recording/pinning ML package versions is recommended for reproducibility.

---

# Experimental Scope

This repository represents a **research prototype**.

The project demonstrates:

- transaction streaming,
- controlled failure injection,
- ML-based failure-risk prediction,
- automatic recovery,
- persistent healing logs,
- monitoring.

It does not claim to be a production banking platform.

The controlled environment is designed to make failure and recovery experiments reproducible.

---

# Limitations

The current implementation has several limitations.

## Controlled Failures

The failure conditions are generated experimentally.

They are not production incidents collected from a real financial institution.

## Single-Machine Container Environment

The Kafka, Spark, PostgreSQL, and Grafana services are primarily executed as Docker containers on a single host.

This differs from a production multi-node distributed deployment.

## Synthetic Pipeline Metrics

Some pipeline operating conditions are generated as part of the controlled experiment.

They should not be interpreted as measurements from a live banking infrastructure.

## Machine-Learning Labels

The failure-prediction model learns from controlled experimental failure labels.

Its performance therefore reflects the simulated failure-generation process.

## Performance Comparability

Throughput and latency are sensitive to replay and batching conditions.

A scientifically controlled performance comparison requires identical execution conditions.

## Internal Latency Measurement

The current latency measurement represents processing behavior within the experimental implementation and should not automatically be interpreted as complete producer-to-Kafka-to-Spark-to-database end-to-end latency.

## Limited Infrastructure Failure Coverage

The prototype focuses on selected processing failures and does not currently model every possible infrastructure failure.

Examples that would require further work include:

- broker outages,
- network partitions,
- complete worker loss,
- database unavailability,
- multi-node cluster failure,
- schema-registry failures,
- cloud-service outages.

---

# Future Work

Several extensions could strengthen the system.

## Real Failure Traces

Evaluate the pipeline using real or production-like operational failure traces rather than only controlled simulations.

## Multi-Node Deployment

Deploy Kafka and Spark across multiple physical or cloud nodes.

## Cloud Deployment

Evaluate the architecture using platforms such as:

- AWS,
- Microsoft Azure,
- Google Cloud.

## Additional Failure Scenarios

Introduce failures such as:

- Kafka broker interruption,
- Spark executor termination,
- PostgreSQL connectivity loss,
- network delay,
- schema changes,
- malformed messages,
- service unavailability.

## Model Drift Monitoring

Monitor whether the distribution of incoming pipeline conditions changes over time.

## Automated Retraining

Retrain the failure-prediction model when sufficient new labeled data becomes available.

## Adaptive Strategy Selection

Instead of using only deterministic healing rules, future versions could learn which recovery action is most effective for a given operating condition.

## Stronger Delivery Semantics

Improve idempotency and investigate stronger exactly-once behavior across recovery paths.

## End-to-End Latency

Measure timestamps across:

```text
Producer
   ->
Kafka
   ->
Spark
   ->
Healing
   ->
PostgreSQL
```

to calculate true end-to-end transaction latency.

## Controlled Performance Benchmarking

Execute baseline and self-healing pipelines using identical:

- datasets,
- Kafka offsets,
- batch sizes,
- Spark resources,
- cluster configuration,
- measurement methodology.

## Production Observability

Extend monitoring with:

- Prometheus,
- alerting,
- distributed tracing,
- structured application logs,
- infrastructure metrics.

---

# Research Ethics and Data Usage

The project uses an anonymized public research dataset.

The system is intended for academic experimentation.

No claims should be made that the prototype is suitable for processing real financial transactions without additional:

- security testing,
- privacy analysis,
- compliance review,
- fault-tolerance testing,
- performance testing,
- infrastructure hardening.

---

# Reproducibility Notes

For reproducible experiments, record:

```text
Dataset version
Random seed
Producer configuration
Failure-injection parameters
Model version
Python package versions
Kafka topic configuration
Spark version
Kafka offsets
Starting-offset configuration
Micro-batch configuration
Spark worker resources
Database state
Checkpoint state
```

Changing any of these may change experimental behavior.

---

# Research Project

**Title:**  
Real-Time Self-Healing Data Pipeline for Reliable Transaction Processing

**Module:**  
CSCI 43018 – Research Project

**Degree:**  
BSc (Hons) in Computer Science

**Faculty:**  
Faculty of Computing and Technology

**University:**  
University of Kelaniya

---

# Author

**Kavindi A.L.S.J.**

Student ID:

```text
CS/2020/023
```

BSc (Hons) in Computer Science  
Faculty of Computing and Technology  
University of Kelaniya

---

# Supervisor

**Dr. Carmel Wijegunasekara**

---

# Academic Notice

This repository contains a final-year academic research prototype.

The implementation is intended to demonstrate and evaluate concepts related to:

- real-time stream processing,
- machine-learning-based failure prediction,
- automated self-healing,
- fault recovery,
- observability,
- data-pipeline reliability.

It is **not a production financial system**, and the controlled experimental failure conditions should not be interpreted as guarantees of behavior in a real banking or payment-processing environment.

---

# Acknowledgement

This project was completed as part of the BSc (Hons) in Computer Science research project at the **Faculty of Computing and Technology, University of Kelaniya**.