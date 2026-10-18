# Dockerfile for IoT GNN Zero-Day Benchmark
# Edge-node emulation (1 CPU, 512 MB RAM) is applied at run time:
#   docker run --rm --cpus=1 --memory=512m iot-gnn-edge
FROM python:3.11.9-slim-bookworm

# Set work directory
WORKDIR /app

# No stdout buffering / bytecode files.
# Limit BLAS threads to match --cpus=1: without this, numpy may spawn extra
# threads, making latency measurements non-reproducible.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

# Install dependencies strictly with pinned versions
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository structure
COPY config.yaml .
COPY data/ data/
COPY src/ src/

# Create results directory
RUN mkdir -p results

# Default execution command runs smoke benchmark
CMD ["python", "src/benchmark.py", "--config", "config.yaml"]
