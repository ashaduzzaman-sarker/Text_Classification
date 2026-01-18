"""FastAPI inference service for the text classification model.

This service exposes:

- ``GET /health``     – lightweight health check
- ``POST /predict``   – text classification inference
- ``GET /metrics``    – Prometheus metrics endpoint

It is designed to be containerised (Docker) and deployed on Kubernetes.
"""

from __future__ import annotations

import time
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from text_classification.pipeline.prediction_pipeline import PredictionPipeline
from text_classification.logging.logger import logger


app = FastAPI(title="Text Classification API", version="0.1.0")


class PredictRequest(BaseModel):
	"""Request schema for /predict endpoint."""

	text: str


class PredictResponse(BaseModel):
	"""Response schema for /predict endpoint."""

	label: str
	score: float


# ----------------------------------------------------------------------------
# Prometheus Metrics
# ----------------------------------------------------------------------------

REQUEST_COUNT = Counter(
	"text_classification_requests_total",
	"Total number of prediction requests",
	["endpoint", "http_status"],
)

REQUEST_LATENCY = Histogram(
	"text_classification_request_latency_seconds",
	"Latency of prediction requests in seconds",
	["endpoint"],
)


_predictor: PredictionPipeline | None = None


def get_predictor() -> PredictionPipeline:
	"""Lazily initialize and cache the prediction pipeline."""

	global _predictor
	if _predictor is None:
		logger.info("Initialising PredictionPipeline inside FastAPI app...")
		_predictor = PredictionPipeline()
	return _predictor


@app.get("/health")
async def health() -> dict:
	"""Basic health check used by Kubernetes liveness/readiness probes."""

	return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
	"""Run text classification on a single input text."""

	start_time = time.time()
	endpoint = "/predict"

	try:
		predictor = get_predictor()
		results = predictor.predict([payload.text])

		if not results:
			raise HTTPException(status_code=500, detail="Empty prediction result")

		result = results[0]
		REQUEST_COUNT.labels(endpoint=endpoint, http_status="200").inc()

		return PredictResponse(label=result["label"], score=result["score"])

	except HTTPException as exc:  # Bubble up HTTP errors but record metrics
		REQUEST_COUNT.labels(endpoint=endpoint, http_status=str(exc.status_code)).inc()
		raise
	except Exception as exc:  # pragma: no cover - defensive
		logger.exception("Prediction failed: %s", exc)
		REQUEST_COUNT.labels(endpoint=endpoint, http_status="500").inc()
		raise HTTPException(status_code=500, detail="Prediction failed") from exc
	finally:
		elapsed = time.time() - start_time
		REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)


@app.get("/metrics")
async def metrics() -> Response:
	"""Expose Prometheus metrics for scraping by Prometheus server."""

	data = generate_latest()
	return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root() -> dict:
	"""Simple index endpoint for quick manual checks."""

	return {"message": "Text Classification API is running"}


if __name__ == "__main__":  # pragma: no cover
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
