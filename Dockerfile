# ============================================================================
# Dockerfile — Text Classification API (FastAPI + Uvicorn)
#
# This image is optimised for CPU inference. For GPU-enabled deployment on
# Azure (AKS with GPU nodes), you would typically switch to a CUDA base
# image and ensure the appropriate NVIDIA drivers are available.
# ============================================================================

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Default command: run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
