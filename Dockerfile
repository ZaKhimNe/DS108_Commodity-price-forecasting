FROM python:3.11-slim

# System deps: git (yfinance), curl (openmeteo requests), build tools (numpy/scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first — leverage Docker layer cache
COPY requirements.txt .

# Install PyTorch CPU-only (~800MB) instead of CUDA variant (~6GB).
# This pipeline does not use GPU — CPU-only is sufficient.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (skip torch line already installed above)
RUN grep -v "^torch" requirements.txt \
    | pip install --no-cache-dir -r /dev/stdin

# Copy source code and config
# data/ and models/ are NOT copied — they are mounted as volumes at runtime
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV PYTHONUNBUFFERED=1

# Default command: generate pipeline report
# Override with docker compose run <service> to run specific stages
CMD ["python", "src/20_pipeline_report.py"]
