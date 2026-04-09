"""Train fraud models that output a probability.

Project requirement
-------------------
Train three versions of a model:
  - Logistic Regression (LR)
  - Random Forest (RF)
  - Decision Tree (DT)

All tunable hyperparameters and output paths live in params.yaml (no magic
numbers in code).

Output
------
For each model variant, we write:
  1) A joblib artifact containing a sklearn Pipeline + metadata
  2) A JSON metrics file

These artifacts are intended to be loaded by your API so the frontend can send
feature values and receive a fraud probability ("percent chance").
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	brier_score_loss,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class ModelSpec:
	"""Configuration for one model variant (LR/RF/DT)."""

	name: str
	model_type: str
	model_out: Path
	metrics_out: Path
	use_standard_scaler: bool
	params: dict[str, Any]


@dataclass(frozen=True)
class TrainConfig:
	"""Configuration shared across all model variants."""

	input_csv: Path
	expected_columns: list[str]
	feature_columns: list[str]
	target_column: str
	test_size: float
	random_state: int
	decision_threshold: float
	max_rows: int | None
	models: list[ModelSpec]


def _load_train_config(params_path: Path) -> TrainConfig:
	"""Load training configuration from params.yaml.

	Required shape (recommended):
	  train:
		...shared settings...
		models:
		  lr: {type: logistic_regression, ...}
		  rf: {type: random_forest, ...}
		  dt: {type: decision_tree, ...}

	Backwards compatible:
	If `train.models` is missing, we fall back to the older single-model keys.
	"""

	raw = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
	cfg = raw.get("train", {})

	def req(key: str) -> Any:
		if key not in cfg:
			raise KeyError(f"Missing params.yaml key: train.{key}")
		return cfg[key]

	input_csv = Path(req("input_csv"))
	expected_columns = list(req("expected_columns"))
	feature_columns = list(req("feature_columns"))
	target_column = str(req("target_column"))
	test_size = float(req("test_size"))
	random_state = int(req("random_state"))
	decision_threshold = float(req("decision_threshold"))

	max_rows_raw = cfg.get("max_rows", None)
	max_rows = None if max_rows_raw in (None, "null") else int(max_rows_raw)

	models_cfg = cfg.get("models")
	models: list[ModelSpec] = []

	if isinstance(models_cfg, dict):
		for name, mc in models_cfg.items():
			if not isinstance(mc, dict):
				raise ValueError(f"train.models.{name} must be a mapping")
			enabled = bool(mc.get("enabled", True))
			if not enabled:
				continue
			model_type = str(mc.get("type", "")).strip()
			if not model_type:
				raise KeyError(f"Missing params.yaml key: train.models.{name}.type")
			model_out = Path(mc.get("model_out"))
			metrics_out = Path(mc.get("metrics_out"))
			if not model_out:
				raise KeyError(f"Missing params.yaml key: train.models.{name}.model_out")
			if not metrics_out:
				raise KeyError(f"Missing params.yaml key: train.models.{name}.metrics_out")
			use_standard_scaler = bool(mc.get("use_standard_scaler", False))
			params = dict(mc.get("params", {}) or {})
			models.append(
				ModelSpec(
					name=str(name),
					model_type=model_type,
					model_out=model_out,
					metrics_out=metrics_out,
					use_standard_scaler=use_standard_scaler,
					params=params,
				)
			)
	else:
		# Legacy single-model config (kept so earlier steps don't break).
		model_out = Path(req("model_out"))
		metrics_out = Path(req("metrics_out"))
		use_standard_scaler = bool(req("use_standard_scaler"))
		lr_params = dict(req("logistic_regression"))
		models = [
			ModelSpec(
				name="lr",
				model_type="logistic_regression",
				model_out=model_out,
				metrics_out=metrics_out,
				use_standard_scaler=use_standard_scaler,
				params=lr_params,
			)
		]

	if not models:
		raise ValueError("No enabled models configured under train.models")

	return TrainConfig(
		input_csv=input_csv,
		expected_columns=expected_columns,
		feature_columns=feature_columns,
		target_column=target_column,
		test_size=test_size,
		random_state=random_state,
		decision_threshold=decision_threshold,
		max_rows=max_rows,
		models=models,
	)


def _read_header(csv_path: Path) -> list[str]:
	"""Read only the header row of a CSV file."""

	with csv_path.open("r", encoding="utf-8") as f:
		first_line = f.readline().strip("\n\r")
	return first_line.split(",")


def _load_dataset(
	csv_path: Path,
	expected_columns: list[str],
	feature_columns: list[str],
	target_column: str,
	max_rows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Load the cleaned CSV into numpy arrays.

	This expects an all-numeric CSV (which your cleaning stage produces).
	"""

	if not csv_path.exists():
		raise FileNotFoundError(f"Training input not found: {csv_path}")

	header = _read_header(csv_path)
	if header != expected_columns:
		raise ValueError(f"Unexpected columns/order in training CSV. expected={expected_columns} got={header}")

	# Load the full numeric matrix. For 1,000,000 rows and 8 columns this is manageable.
	# We use float32 to reduce memory usage.
	data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32, max_rows=max_rows)
	if data.ndim == 1:
		# Edge-case: if max_rows=1, loadtxt returns 1D. Normalize to 2D.
		data = data.reshape(1, -1)

	col_to_idx = {c: i for i, c in enumerate(expected_columns)}
	x_idx = [col_to_idx[c] for c in feature_columns]
	y_idx = col_to_idx[target_column]

	X = data[:, x_idx]
	y = data[:, y_idx].astype(np.int32)
	return X, y


def _scrub_params(params: dict[str, Any]) -> dict[str, Any]:
	"""Remove keys with null values so they don't get passed into sklearn constructors."""

	return {k: v for k, v in params.items() if v is not None}


def _build_pipeline(model: ModelSpec, global_random_state: int) -> Pipeline:
	"""Build the sklearn Pipeline for the requested model type."""

	steps: list[tuple[str, Any]] = []
	if model.use_standard_scaler:
		steps.append(("scaler", StandardScaler()))

	params = _scrub_params(model.params)

	if model.model_type == "logistic_regression":
		# If penalty is omitted or set to "l2", let sklearn default handle it.
		# This avoids version-specific warnings while keeping the option to tune.
		penalty = params.get("penalty")
		if penalty in (None, "l2"):
			params.pop("penalty", None)
		estimator = LogisticRegression(**params)
	elif model.model_type == "random_forest":
		# Ensure reproducibility if caller didn't pass it.
		params.setdefault("random_state", global_random_state)
		estimator = RandomForestClassifier(**params)
	elif model.model_type == "decision_tree":
		params.setdefault("random_state", global_random_state)
		estimator = DecisionTreeClassifier(**params)
	else:
		raise ValueError(f"Unsupported model type: {model.model_type}")

	steps.append(("model", estimator))
	return Pipeline(steps)


def _evaluate_probabilistic_classifier(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, Any]:
	"""Compute a consistent metrics set for a probability classifier."""

	y_pred = (proba >= threshold).astype(np.int32)
	return {
		"roc_auc": float(roc_auc_score(y_true, proba)),
		"pr_auc": float(average_precision_score(y_true, proba)),
		"log_loss": float(log_loss(y_true, proba)),
		"brier": float(brier_score_loss(y_true, proba)),
		"threshold": float(threshold),
		"accuracy": float(accuracy_score(y_true, y_pred)),
		"precision": float(precision_score(y_true, y_pred, zero_division=0)),
		"recall": float(recall_score(y_true, y_pred, zero_division=0)),
		"f1": float(f1_score(y_true, y_pred, zero_division=0)),
		"confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
	}


def train_all_models(config: TrainConfig) -> dict[str, Any]:
	"""Train all configured models and write artifacts + metrics for each."""

	X, y = _load_dataset(
		csv_path=config.input_csv,
		expected_columns=config.expected_columns,
		feature_columns=config.feature_columns,
		target_column=config.target_column,
		max_rows=config.max_rows,
	)

	# Single split shared across all models for fair comparison.
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=config.test_size,
		random_state=config.random_state,
		stratify=y,
	)

	results: dict[str, Any] = {}

	for model in config.models:
		pipeline = _build_pipeline(model, global_random_state=config.random_state)
		pipeline.fit(X_train, y_train)

		proba = pipeline.predict_proba(X_test)[:, 1]
		metrics = {
			"model_name": model.name,
			"model_type": model.model_type,
			"n_train": int(X_train.shape[0]),
			"n_test": int(X_test.shape[0]),
			"fraud_rate_train": float(y_train.mean()),
			"fraud_rate_test": float(y_test.mean()),
			**_evaluate_probabilistic_classifier(y_test, proba, threshold=config.decision_threshold),
		}

		artifact: dict[str, Any] = {
			"model": pipeline,
			"feature_columns": config.feature_columns,
			"target_column": config.target_column,
			"decision_threshold": config.decision_threshold,
			"trained_at_utc": datetime.now(timezone.utc).isoformat(),
			"model_name": model.name,
			"model_type": model.model_type,
			"hyperparameters": model.params,
			"metrics": metrics,
		}

		model.model_out.parent.mkdir(parents=True, exist_ok=True)
		model.metrics_out.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(artifact, model.model_out)
		model.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

		results[model.name] = metrics

	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Train fraud probability models")
	parser.add_argument(
		"--params",
		type=Path,
		default=Path("params.yaml"),
		help="Path to params.yaml (default: params.yaml)",
	)
	parser.add_argument(
		"--max-rows",
		type=int,
		default=None,
		help="Optional override to train on only the first N rows (does not edit params.yaml)",
	)
	args = parser.parse_args()

	config = _load_train_config(args.params)
	if args.max_rows is not None:
		config = replace(config, max_rows=args.max_rows)

	results = train_all_models(config)

	# Print a compact summary to stdout for quick feedback.
	summary = {k: {"roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]} for k, v in results.items()}
	print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
