FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts

# Default: run full benchmark pipeline
CMD ["bash", "-c", "\
    python src/data/synthetic_generator.py \
        --n_accounts 10000 \
        --daily_events 100000 \
        --simulation_days 3 \
        --output_path artifacts/synthetic_events.parquet \
        --seed 42 && \
    python src/data/feature_engineering.py \
        --events_path artifacts/synthetic_events.parquet \
        --output_path artifacts/feature_store.parquet \
        --decision_date 2024-04-01 \
        --max_accounts 500 && \
    python src/evaluation/benchmark.py \
        --output_path artifacts/benchmark_results.json \
        --seed 42 && \
    echo 'EDMD pipeline complete. Results in artifacts/'"]
