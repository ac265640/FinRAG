FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r finrag && useradd -r -m -g finrag finrag

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .
# Ensure we install the local finrag package itself
RUN pip install --no-cache-dir --no-deps -e .

RUN chown -R finrag:finrag /app
USER finrag

EXPOSE 8000
EXPOSE 7860

# Dynamically bind to the port provided by the cloud platform (like HF Spaces) or default to 7860
CMD ["sh", "-c", "uvicorn finrag.api.app:create_app --factory --host 0.0.0.0 --port ${PORT:-7860}"]

