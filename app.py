from __future__ import annotations

"""FastAPI inference API for fraud probability.

This API is intentionally minimal so it can be smoke-tested in GitHub Actions.

Model loading
-------------
We load a joblib artifact produced by training/registry promotion. The artifact
format is the dict written by src/Train.py:
  {
	"model": sklearn Pipeline,
	"feature_columns": [...],
	...
  }

The workflow promotes a "champion" model and also copies it to ./model.joblib so
the Docker image can include it.
"""

import os
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


FEATURE_NAMES = [
	"distance_from_home",
	"distance_from_last_transaction",
	"ratio_to_median_purchase_price",
	"repeat_retailer",
	"used_chip",
	"used_pin_number",
	"online_order",
]


class FraudFeatures(BaseModel):
	distance_from_home: float
	distance_from_last_transaction: float
	ratio_to_median_purchase_price: float
	repeat_retailer: int = Field(..., ge=0, le=1)
	used_chip: int = Field(..., ge=0, le=1)
	used_pin_number: int = Field(..., ge=0, le=1)
	online_order: int = Field(..., ge=0, le=1)


class PredictRequest(BaseModel):
	features: Optional[FraudFeatures] = None


class PredictResponse(BaseModel):
	fraud_probability: float
	fraud_percent: float


app = FastAPI(title="Fraud Probability API")

_artifact: Optional[dict[str, Any]] = None


def _candidate_model_paths() -> list[Path]:
	# Prefer an explicit env var in deployments.
	env = os.getenv("MODEL_PATH")
	paths: list[Path] = []
	if env:
		paths.append(Path(env))

	# Docker-friendly path.
	paths.append(Path("model.joblib"))

	# Local registry default.
	paths.append(Path("modelinfo/modelregistry/champion/model.joblib"))
	return paths


def _load_artifact() -> dict[str, Any]:
	last_err: Optional[Exception] = None
	for p in _candidate_model_paths():
		try:
			obj = joblib.load(p)
			if not isinstance(obj, dict) or "model" not in obj:
				raise ValueError("Loaded object is not a Train.py artifact dict")
			return obj
		except Exception as e:  # noqa: BLE001
			last_err = e
	raise RuntimeError(f"Failed to load model artifact. Last error: {last_err}")


@app.on_event("startup")
def _startup() -> None:
	global _artifact
	_artifact = _load_artifact()


@app.get("/health")
def health() -> dict[str, str]:
	return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
	if _artifact is None:
		raise HTTPException(status_code=500, detail="Model not loaded")
	if req.features is None:
		raise HTTPException(status_code=400, detail="Missing 'features'")

	model = _artifact["model"]
	feature_columns = _artifact.get("feature_columns", FEATURE_NAMES)
	if list(feature_columns) != FEATURE_NAMES:
		# Keep it strict so we don't silently reorder user inputs.
		raise HTTPException(status_code=500, detail="Model feature schema mismatch")

	x = np.asarray(
		[
			[
				req.features.distance_from_home,
				req.features.distance_from_last_transaction,
				req.features.ratio_to_median_purchase_price,
				req.features.repeat_retailer,
				req.features.used_chip,
				req.features.used_pin_number,
				req.features.online_order,
			]
		],
		dtype=float,
	)

	if not hasattr(model, "predict_proba"):
		raise HTTPException(status_code=500, detail="Model does not support predict_proba")

	proba = model.predict_proba(x)
	if not isinstance(proba, np.ndarray) or proba.ndim != 2 or proba.shape[1] < 2:
		raise HTTPException(status_code=500, detail="Invalid predict_proba output")

	fraud_probability = float(proba[0, 1])
	return PredictResponse(
		fraud_probability=fraud_probability,
		fraud_percent=fraud_probability * 100.0,
	)
