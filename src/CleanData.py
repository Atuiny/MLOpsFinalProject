"""Clean raw fraud dataset into a processed, model-ready CSV.

This script is designed to be used as a DVC pipeline stage.

Key design goals
----------------
1) Reproducible: same raw input + same params -> same processed output.
2) "No magic numbers" in code: anything that might change between runs
	(paths, schema, strictness, thresholds) lives in params.yaml.
3) Safe cleaning (no leakage): we do not do scaling/normalization here.
	Those transforms belong in training, fitted on the training split only.

What "cleaning" means here
--------------------------
- Validate the schema (columns and order).
- Enforce numeric / binary types.
- Enforce basic invariants (finite values; optionally non-negative; optional max thresholds).
- Optionally drop duplicates.
- Write a cleaned CSV + a small JSON report for traceability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CleanConfig:
	"""Typed configuration for the cleaning stage.

	All fields should come from params.yaml so the pipeline is configurable
	without editing code.
	"""

	input_csv: Path
	output_csv: Path
	report_json: Path
	expected_columns: list[str]
	float_columns: list[str]
	binary_columns: list[str]
	target_column: str
	enforce_non_negative: bool
	strict: bool
	drop_duplicates: bool
	max_float_values: dict[str, float | None]


def _load_clean_config(params_path: Path) -> CleanConfig:
	"""Load cleaning configuration from params.yaml.

	We keep the YAML parsing/validation centralized so the rest of the code can
	assume the configuration is present and well-formed.
	"""

	raw = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
	cfg = raw.get("clean_data", {})

	def req(key: str) -> Any:
		if key not in cfg:
			raise KeyError(f"Missing params.yaml key: clean_data.{key}")
		return cfg[key]

	input_csv = Path(req("input_csv"))
	output_csv = Path(req("output_csv"))
	report_json = Path(req("report_json"))

	expected_columns = list(req("expected_columns"))
	float_columns = list(req("float_columns"))
	binary_columns = list(req("binary_columns"))
	target_column = str(req("target_column"))

	enforce_non_negative = bool(req("enforce_non_negative"))
	strict = bool(req("strict"))
	drop_duplicates = bool(req("drop_duplicates"))

	# Optional per-feature maximum thresholds.
	# These are meant to behave like "magic numbers" / constants you can tune
	# from params.yaml (e.g., max_distance_from_home: 20000).
	max_values_raw = cfg.get("max_float_values", {}) or {}
	max_float_values: dict[str, float | None] = {}
	for c in float_columns:
		v = max_values_raw.get(c, None)
		max_float_values[c] = None if v is None else float(v)

	return CleanConfig(
		input_csv=input_csv,
		output_csv=output_csv,
		report_json=report_json,
		expected_columns=expected_columns,
		float_columns=float_columns,
		binary_columns=binary_columns,
		target_column=target_column,
		enforce_non_negative=enforce_non_negative,
		strict=strict,
		drop_duplicates=drop_duplicates,
		max_float_values=max_float_values,
	)


def _as_float(value: str) -> float:
	"""Parse a numeric value and reject NaN/inf.

	Using this helper ensures all float parsing follows the same rules.
	"""

	v = float(value)
	if not math.isfinite(v):
		raise ValueError("non-finite")
	return v


def _as_binary_01(value: str) -> int:
	"""Parse a value that must be exactly 0 or 1.

	We accept numeric strings like "0", "1", "0.0", "1.0".
	"""

	v = _as_float(value)
	if v == 0.0:
		return 0
	if v == 1.0:
		return 1
	raise ValueError("not 0/1")


def clean_data(config: CleanConfig) -> dict[str, Any]:
	"""Run the cleaning pass and return a JSON-serializable report dict."""

	if not config.input_csv.exists():
		raise FileNotFoundError(f"Input CSV not found: {config.input_csv}")

	config.output_csv.parent.mkdir(parents=True, exist_ok=True)
	config.report_json.parent.mkdir(parents=True, exist_ok=True)

	n_in = 0
	n_out = 0
	n_invalid = 0
	n_duplicates = 0
	fraud_count = 0

	mins = {c: float("inf") for c in config.float_columns}
	maxs = {c: float("-inf") for c in config.float_columns}

	# If duplicate dropping is enabled, we store a hashable key for each output row.
	# This is simple and deterministic, but it uses memory proportional to the number
	# of unique rows.
	seen: set[tuple[Any, ...]] | None = set() if config.drop_duplicates else None

	with config.input_csv.open("r", newline="", encoding="utf-8") as f_in, config.output_csv.open(
		"w", newline="", encoding="utf-8"
	) as f_out:
		reader = csv.DictReader(f_in)
		if reader.fieldnames is None:
			raise ValueError("Input CSV has no header")

		incoming_cols = list(reader.fieldnames)

		# Schema guardrail: require the exact columns (and order) we expect.
		# This prevents silent training on the wrong columns.
		if incoming_cols != config.expected_columns:
			missing = [c for c in config.expected_columns if c not in incoming_cols]
			extra = [c for c in incoming_cols if c not in config.expected_columns]
			raise ValueError(
				"Unexpected columns/order. "
				f"expected={config.expected_columns} got={incoming_cols}. "
				f"missing={missing} extra={extra}"
			)

		writer = csv.DictWriter(f_out, fieldnames=config.expected_columns)
		writer.writeheader()

		for row in reader:
			n_in += 1
			try:
				cleaned: dict[str, Any] = {}

				for c in config.float_columns:
					v = _as_float(row[c])

					# Optional invariant: distances/ratios should not be negative.
					if config.enforce_non_negative and v < 0:
						raise ValueError(f"{c} negative")

					# Optional invariant: per-feature maximum threshold.
					max_v = config.max_float_values.get(c)
					if max_v is not None and v > max_v:
						raise ValueError(f"{c} above max")

					if v < mins[c]:
						mins[c] = v
					if v > maxs[c]:
						maxs[c] = v
					cleaned[c] = v

				for c in config.binary_columns:
					cleaned[c] = _as_binary_01(row[c])

				if config.target_column not in config.binary_columns:
					raise ValueError("target_column must be listed in binary_columns")
				fraud_count += int(cleaned[config.target_column] == 1)

				if seen is not None:
					# Dedup key uses the processed values in the expected output order.
					dedup_key = tuple(cleaned[c] for c in config.expected_columns)
					if dedup_key in seen:
						n_duplicates += 1
						continue
					seen.add(dedup_key)

				# Ensure output column order
				out_row = {c: cleaned[c] for c in config.expected_columns}
				writer.writerow(out_row)
				n_out += 1
			except Exception:
				n_invalid += 1

				# Strict mode is useful in CI / grading: fail fast if anything is wrong.
				# Non-strict mode is useful if you want to drop a small number of bad rows.
				if config.strict:
					raise

	report: dict[str, Any] = {
		"input_csv": str(config.input_csv),
		"output_csv": str(config.output_csv),
		"rows_in": n_in,
		"rows_out": n_out,
		"rows_invalid": n_invalid,
		"rows_duplicates_dropped": n_duplicates,
		"fraud_count": fraud_count,
		"fraud_rate": (fraud_count / n_out) if n_out else None,
		"float_column_mins": {c: (None if mins[c] == float("inf") else mins[c]) for c in config.float_columns},
		"float_column_maxs": {c: (None if maxs[c] == float("-inf") else maxs[c]) for c in config.float_columns},
		"config": {
			"expected_columns": config.expected_columns,
			"float_columns": config.float_columns,
			"binary_columns": config.binary_columns,
			"target_column": config.target_column,
			"enforce_non_negative": config.enforce_non_negative,
			"strict": config.strict,
			"drop_duplicates": config.drop_duplicates,
			"max_float_values": config.max_float_values,
		},
	}

	config.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
	return report


def main() -> None:
    """CLI entry point.

    Example:
      python src/CleanData.py --params params.yaml
    """

    parser = argparse.ArgumentParser(description="Clean raw fraud dataset into a processed CSV.")
    parser.add_argument(
        "--params",
        type=Path,
        default=Path("params.yaml"),
        help="Path to params.yaml (default: params.yaml)",
    )
    args = parser.parse_args()

    config = _load_clean_config(args.params)
    report = clean_data(config)
    print(json.dumps({"rows_out": report["rows_out"], "fraud_rate": report["fraud_rate"]}, indent=2))


if __name__ == "__main__":
    main()
