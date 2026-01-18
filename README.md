# Text Classification – LoRA/QLoRA, MLflow, Docker, Kubernetes, Azure

This repository implements an end-to-end, production-ready text classification
pipeline using modern NLP tooling (Hugging Face Transformers, 🤗 Datasets) and
MLOps best practices (LoRA/QLoRA fine-tuning, MLflow tracking, Docker,
Kubernetes, Prometheus/Grafana, and GitHub Actions CI/CD targeting Azure).

## Features

- **Transformer-based text classification** on Hugging Face datasets (e.g. IMDb).
- **Parameter-efficient fine-tuning** with configurable **LoRA / QLoRA**.
- **Experiment tracking** with **MLflow** (local or remote tracking server).
- **Modular pipeline**: data ingestion, validation, preprocessing,
	model training, and evaluation under `src/text_classification`.
- **FastAPI inference service** with `/predict`, `/health`, and `/metrics`
	endpoints (Prometheus-compatible).
- **Dockerised API** ready for deployment to Azure Container Registry.
- **Kubernetes manifests** for deployment + Service (AKS friendly).
- **GitHub Actions CI/CD**: test, build, push Docker image, and deploy to AKS.

## Project Structure (high level)

- `main.py` – runs the full offline training & evaluation pipeline.
- `app.py` – FastAPI app for online inference and metrics.
- `config/` – YAML configuration (`config.yaml`, `params.yaml`).
- `src/text_classification/` – all pipeline components, configs, and utilities.
- `k8s/` – Kubernetes manifests for Deployment and Service.
- `.github/workflows/ci-cd.yaml` – CI/CD pipeline for Azure + AKS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # adjust values as needed
```

## Run Offline Training (LoRA/QLoRA)

Training configuration lives in `config/config.yaml` and `config/params.yaml`.
The `LoRAConfig` section controls whether LoRA/QLoRA is used.

```bash
python main.py
```

Artifacts (tokenised data, model checkpoints, metrics) are written to the
`artifacts/` directory. The final model for inference is stored under
`artifacts/model_trainer/final_model`.

## Run the FastAPI Inference Service Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
		 -H "Content-Type: application/json" \
		 -d '{"text": "This movie was amazing!"}'
```

- Health check: `GET /health`
- Prometheus metrics: `GET /metrics`

## Docker

Build and run the API image locally:

```bash
docker build -t text-classification-api .
docker run -p 8000:8000 text-classification-api
```

## Kubernetes (AKS-ready)

The `k8s/` directory contains:

- `deployment.yaml` – API Deployment with Prometheus scrape annotations.
- `service.yaml` – LoadBalancer Service exposing port 80 -> 8000.

Apply manifests (assuming context already points to your AKS cluster):

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## GitHub Actions CI/CD (Azure)

The workflow in `.github/workflows/ci-cd.yaml`:

- runs tests (`pytest`),
- builds and pushes a Docker image to **Azure Container Registry (ACR)**,
- deploys the updated image to **Azure Kubernetes Service (AKS)**.

You need to configure the following GitHub secrets:

- `AZURE_CREDENTIALS` – JSON output from `az ad sp create-for-rbac`.
- `AZURE_CONTAINER_REGISTRY` – ACR name (without `.azurecr.io`).
- `AKS_CLUSTER_NAME` – AKS cluster name.
- `AKS_RESOURCE_GROUP` – AKS resource group.
- `AKS_NAMESPACE` – Kubernetes namespace for deployment.

## Monitoring with Prometheus & Grafana

- The FastAPI app exposes metrics at `/metrics` using `prometheus-client`.
- The Kubernetes Deployment includes Prometheus scrape annotations.
- You can point a Prometheus server at your AKS cluster and build a
- Grafana dashboard on top of those metrics (request rate & latency).

