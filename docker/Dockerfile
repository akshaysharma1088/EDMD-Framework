FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p artifacts/simulation

# Default: reproduce Tables 5 and 7 and Figure 4 of the paper
CMD ["bash", "-c", "\
    python src/evaluation/simulation_benchmark.py --output_dir artifacts/simulation && \
    python src/evaluation/plot_figure4.py artifacts/simulation/main_runs.json artifacts/simulation/figure4.png && \
    echo 'Done. Results in artifacts/simulation/'"]
