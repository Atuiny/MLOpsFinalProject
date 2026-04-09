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
from fastapi.responses import HTMLResponse
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
	repeat_retailer: bool
	used_chip: bool
	used_pin_number: bool
	online_order: bool


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


@app.get("/", response_class=HTMLResponse)
def index() -> str:
	# Minimal UI for local use; no extra frontend framework.
	# Defaults are all zero / false.
	return """<!doctype html>
<html lang=\"en\">
	<head>
		<meta charset=\"utf-8\" />
		<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
		<title>Fraud Probability</title>
		<style>
			body {
				font-family: system-ui, sans-serif;
				max-width: 720px;
				margin: 2rem auto;
				padding: 0 1rem;
				background: #000;
				color: #8ecbff;
			}
			h1 { margin: 0 0 0.25rem 0; }
			p { margin: 0 0 1rem 0; }
			form { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem 1rem; align-items: end; }
			label { display: grid; gap: 0.25rem; font-size: 0.95rem; }
			input, select, button {
				padding: 0.5rem;
				font-size: 1rem;
				background: #111;
				color: #8ecbff;
				border: 1px solid #2a6ea3;
			}
			button { grid-column: 1 / -1; cursor: pointer; }
			.result { margin-top: 1rem; padding: 0.75rem; border: 1px solid #2a6ea3; }
			.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
		</style>
	</head>
	<body>
		<h1>Fraud Probability</h1>
		<p>Enter feature values and click Predict.</p>

		<form id=\"f\">
			<label>
				Distance from home
				<input id=\"distance_from_home\" type=\"number\" step=\"any\" value=\"0\" required />
			</label>
			<label>
				Distance from last transaction
				<input id=\"distance_from_last_transaction\" type=\"number\" step=\"any\" value=\"0\" required />
			</label>
			<label>
				Purchase ratio vs typical (median)
				<input id=\"ratio_to_median_purchase_price\" type=\"number\" step=\"any\" value=\"0\" required />
			</label>

			<label>
				Same retailer as before?
				<select id=\"repeat_retailer\">
					<option value=\"false\" selected>false</option>
					<option value=\"true\">true</option>
				</select>
			</label>
			<label>
				Used chip?
				<select id=\"used_chip\">
					<option value=\"false\" selected>false</option>
					<option value=\"true\">true</option>
				</select>
			</label>
			<label>
				Used PIN?
				<select id=\"used_pin_number\">
					<option value=\"false\" selected>false</option>
					<option value=\"true\">true</option>
				</select>
			</label>
			<label>
				Online order?
				<select id=\"online_order\">
					<option value=\"false\" selected>false</option>
					<option value=\"true\">true</option>
				</select>
			</label>

			<button type=\"submit\">Predict</button>
		</form>

		<div class=\"result\" id=\"out\">Result: <span class=\"mono\">(none)</span></div>

		<script>
			const byId = (id) => document.getElementById(id);
			const toNum = (id) => Number(byId(id).value);
			const toBool = (id) => byId(id).value === 'true';

			byId('f').addEventListener('submit', async (e) => {
				e.preventDefault();
				const payload = {
					features: {
						distance_from_home: toNum('distance_from_home'),
						distance_from_last_transaction: toNum('distance_from_last_transaction'),
						ratio_to_median_purchase_price: toNum('ratio_to_median_purchase_price'),
						repeat_retailer: toBool('repeat_retailer'),
						used_chip: toBool('used_chip'),
						used_pin_number: toBool('used_pin_number'),
						online_order: toBool('online_order')
					}
				};

				byId('out').innerHTML = 'Result: <span class=\"mono\">loading...</span>';
				try {
					const resp = await fetch('/predict', {
						method: 'POST',
						headers: { 'Content-Type': 'application/json' },
						body: JSON.stringify(payload)
					});
					const data = await resp.json();
					if (!resp.ok) {
						throw new Error(data.detail || 'Request failed');
					}
					byId('out').innerHTML = `Result: <span class=\"mono\">${data.fraud_percent.toFixed(2)}% fraud</span>`;
				} catch (err) {
					byId('out').innerHTML = `Result: <span class=\"mono\">error: ${String(err.message || err)}</span>`;
				}
			});
		</script>
	</body>
</html>
"""


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
				int(req.features.repeat_retailer),
				int(req.features.used_chip),
				int(req.features.used_pin_number),
				int(req.features.online_order),
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
