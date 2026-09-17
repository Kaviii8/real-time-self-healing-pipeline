-- ============================================================
-- Real-Time Self-Healing Data Pipeline
-- Experimental Database Schema
-- ============================================================

-- ------------------------------------------------------------
-- 1. Baseline Pipeline Results
-- Transactions processed WITHOUT self-healing
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS baseline_results (
    id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    event_time TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    amount DOUBLE PRECISION,
    actual_class INTEGER,

    failure_type VARCHAR(100),
    failure_occurred BOOLEAN DEFAULT FALSE,

    processing_latency_ms DOUBLE PRECISION,
    status VARCHAR(50)
);


-- ------------------------------------------------------------
-- 2. Self-Healing Pipeline Results
-- Transactions processed WITH failure prediction + healing
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS healing_results (
    id BIGSERIAL PRIMARY KEY,
    transaction_id BIGINT NOT NULL,
    event_time TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    amount DOUBLE PRECISION,
    actual_class INTEGER,

    predicted_failure BOOLEAN DEFAULT FALSE,
    failure_probability DOUBLE PRECISION,

    failure_type VARCHAR(100),
    failure_occurred BOOLEAN DEFAULT FALSE,

    healing_triggered BOOLEAN DEFAULT FALSE,
    healing_strategy VARCHAR(100),
    healing_success BOOLEAN DEFAULT FALSE,

    processing_latency_ms DOUBLE PRECISION,
    recovery_time_ms DOUBLE PRECISION,

    status VARCHAR(50)
);


-- ------------------------------------------------------------
-- 3. Streaming / System Metrics
-- One row can represent one streaming batch
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id BIGSERIAL PRIMARY KEY,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    pipeline_type VARCHAR(50),

    batch_id BIGINT,
    batch_size INTEGER,

    throughput_tps DOUBLE PRECISION,
    avg_latency_ms DOUBLE PRECISION,

    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,

    total_transactions INTEGER,
    successful_transactions INTEGER,
    failed_transactions INTEGER,

    predicted_failures INTEGER DEFAULT 0,
    healing_attempts INTEGER DEFAULT 0,
    successful_healings INTEGER DEFAULT 0
);


-- ------------------------------------------------------------
-- 4. Experiment Runs
-- Allows us to identify each controlled experiment
-- ------------------------------------------------------------

CREATE TABLE IF NOT EXISTS experiment_runs (
    id BIGSERIAL PRIMARY KEY,
    experiment_name VARCHAR(150) NOT NULL,

    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,

    total_transactions INTEGER DEFAULT 0,

    notes TEXT
);


-- ------------------------------------------------------------
-- Helpful indexes for Grafana
-- ------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_baseline_processed_at
ON baseline_results(processed_at);

CREATE INDEX IF NOT EXISTS idx_healing_processed_at
ON healing_results(processed_at);

CREATE INDEX IF NOT EXISTS idx_metrics_recorded_at
ON pipeline_metrics(recorded_at);

CREATE INDEX IF NOT EXISTS idx_baseline_failure
ON baseline_results(failure_occurred);

CREATE INDEX IF NOT EXISTS idx_healing_failure
ON healing_results(failure_occurred);