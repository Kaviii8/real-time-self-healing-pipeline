FROM apache/spark:3.4.1

USER root

RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --no-cache-dir \
        psycopg2-binary \
        numpy \
        joblib \
        scikit-learn \
        xgboost && \
    rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER 185