"""Streamlit UI for the Text Classification model.

This app uses the internal PredictionPipeline (not the HTTP API) to provide
an interactive demo suitable for portfolio and stakeholder demos.
"""

from __future__ import annotations

from typing import List

import streamlit as st

from text_classification.pipeline.prediction_pipeline import PredictionPipeline
from text_classification.logging.logger import logger


@st.cache_resource(show_spinner=True)
def load_pipeline() -> PredictionPipeline:
	"""Load and cache the prediction pipeline for the session."""

	logger.info("Initialising PredictionPipeline inside Streamlit app...")
	return PredictionPipeline()


def render_header() -> None:
	"""Render main page header and description."""

	st.title("📚 Text Classification Demo")
	st.caption(
		"Transformer-based text classification with LoRA/QLoRA fine-tuning, "
		"served via a production-ready pipeline."
	)


def render_sidebar() -> None:
	"""Render Streamlit sidebar with project details and links."""

	st.sidebar.title("Project Info")
	st.sidebar.markdown(
		"""This demo is backed by a Hugging Face Transformer model
		fine-tuned using parameter-efficient LoRA/QLoRA techniques.

		**Tech stack**
		- Transformers, Datasets
		- LoRA / QLoRA (peft, bitsandbytes)
		- MLflow tracking
		- FastAPI, Docker, Kubernetes, Azure
		"""
	)

	st.sidebar.markdown("---")
	st.sidebar.markdown("**Repository**")
	st.sidebar.markdown(
		"[GitHub – Text_Classification](https://github.com/ashaduzzaman-sarker/Text_Classification)"
	)


def main() -> None:
	"""Main entry point for the Streamlit UI."""

	render_header()
	render_sidebar()

	pipeline = load_pipeline()

	with st.form("classification_form"):
		text = st.text_area(
			"Enter text to classify",
			height=180,
			placeholder="Type a movie review, news headline, or any short paragraph...",
		)

		submitted = st.form_submit_button("Classify Text")

	if submitted:
		if not text.strip():
			st.warning("Please enter some text to classify.")
			return

		with st.spinner("Running model inference..."):
			try:
				results: List[dict] = pipeline.predict([text])
			except Exception as exc:  # pragma: no cover - defensive
				logger.exception("Streamlit prediction failed: %s", exc)
				st.error("Prediction failed – check logs for details.")
				return

		if not results:
			st.error("No prediction returned from the model.")
			return

		result = results[0]

		st.subheader("Prediction")
		col1, col2 = st.columns([2, 1])
		with col1:
			st.metric("Predicted Label", result.get("label", "N/A"))
		with col2:
			st.metric("Confidence", f"{result.get('score', 0.0):.2%}")

		with st.expander("Raw prediction payload"):
			st.json(result)


if __name__ == "__main__":  # pragma: no cover
	main()
