# Dockerfile for IoT GNN Zero-Day Benchmark
# Controlled edge node emulation: 1 CPU, 512MB RAM
FROM python:3.11.9-slim-bookworm

# Set work directory
WORKDIR /app

# Prevent python from buffering stdout/stderr and creating bytecode
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
