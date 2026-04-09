from __future__ import annotations

"""Promote a trained candidate to a file-based "champion" registry.

This mimics an MLflow model registry using plain files so it works in GitHub
Actions without external services.

Registry layout (under modelinfo/modelregistry/)
------------------------------------------------
- candidates/<timestamp>__<experiment>__<model_name>/
    - model.joblib
    - metrics.json
    - metadata.json
- champion/
    - model.joblib
    - metrics.json
    - metadata.json

How it integrates with this project
-----------------------------------
- src/Train.py trains 3 models (lr/rf/dt) and writes:
    - modelinfo/modelpkl/fraud_model_<name>.joblib
    - data/processed/train_metrics_<name>.json
- This script selects the best candidate by a metric (e.g., pr_auc) and promotes
  it to champion if it beats the existing champion (or if --force is used).
- For Docker compatibility, it also copies the champion model to ./model.joblib
  at the repo root.
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "modelinfo" / "modelregistry"
CANDIDATES_DIR = REGISTRY_DIR / "candidates"
CHAMPION_DIR = REGISTRY_DIR / "champion"


@dataclass(frozen=True)
class Candidate:
    name: str
    model_path: Path
    metrics_path: Path
    metric_value: float
    metrics: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_token(value: Any) -> str:
    s = str(value)
    for ch in [" ", "/", "\\", ":", "=", "@"]:
        s = s.replace(ch, "-")
    return s


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_train_models_from_params(params_path: Path) -> dict[str, dict[str, str]]:
    raw = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    train = raw.get("train", {})
    models = train.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("params.yaml missing train.models")

    out: dict[str, dict[str, str]] = {}
    for name, cfg in models.items():
        if not isinstance(cfg, dict):
            continue
        if not bool(cfg.get("enabled", True)):
            continue
        out[name] = {
            "model_out": str(cfg.get("model_out", "")),
            "metrics_out": str(cfg.get("metrics_out", "")),
            "type": str(cfg.get("type", "")),
        }

    return out


def _get_metric(metrics: dict[str, Any], metric_name: str) -> float:
    v = metrics.get(metric_name)
    if v is None:
        raise KeyError(f"Metrics missing required key '{metric_name}'")
    if not isinstance(v, (int, float)):
        raise TypeError(f"Metric '{metric_name}' must be numeric")
    return float(v)


def _build_candidates(params_path: Path, metric_name: str) -> list[Candidate]:
    model_cfgs = _load_train_models_from_params(params_path)

    candidates: list[Candidate] = []
    for name, cfg in model_cfgs.items():
        model_path = (REPO_ROOT / cfg["model_out"]).resolve() if cfg["model_out"] else None
        metrics_path = (REPO_ROOT / cfg["metrics_out"]).resolve() if cfg["metrics_out"] else None
        if model_path is None or metrics_path is None:
            continue
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found for {name}: {model_path}")
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics not found for {name}: {metrics_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        val = _get_metric(metrics, metric_name)
        candidates.append(Candidate(name=name, model_path=model_path, metrics_path=metrics_path, metric_value=val, metrics=metrics))

    if not candidates:
        raise ValueError("No candidates found (did training run?)")
    return candidates


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Promote best candidate model to champion registry")
    p.add_argument("--params", type=Path, default=REPO_ROOT / "params.yaml")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--metric", type=str, default="pr_auc")
    p.add_argument("--promote", action="store_true", help="If set, updates champion when candidate beats it")
    p.add_argument("--force", action="store_true", help="If set, promote even if not improved")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    metric_name = (args.metric or "").strip()
    if not metric_name:
        metric_name = "pr_auc"

    experiment_name = args.experiment_name
    if not experiment_name:
        # Default experiment name in CI if user didn't choose one.
        experiment_name = os.getenv("EXPERIMENT_NAME") or f"run-{os.getenv('GITHUB_RUN_ID', 'local')}"

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    CHAMPION_DIR.mkdir(parents=True, exist_ok=True)

    candidates = _build_candidates(args.params, metric_name=metric_name)
    # Pick best by metric (higher is better)
    best = max(candidates, key=lambda c: c.metric_value)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = "__".join([ts, _safe_token(experiment_name), _safe_token(best.name)])
    out_dir = CANDIDATES_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_model_path = out_dir / "model.joblib"
    cand_metrics_path = out_dir / "metrics.json"
    cand_meta_path = out_dir / "metadata.json"

    _copy_file(best.model_path, cand_model_path)
    _copy_file(best.metrics_path, cand_metrics_path)

    meta = {
        "created_at": _utc_now_iso(),
        "experiment_name": experiment_name,
        "metric_for_promotion": metric_name,
        "metric_value": best.metric_value,
        "model_name": best.name,
        "git_sha": os.getenv("GITHUB_SHA"),
    }
    _write_json(cand_meta_path, meta)

    print("Candidate prepared")
    print(f"- candidate_dir: {out_dir}")
    print(f"- best {metric_name}: {best.metric_value:.6f} ({best.name})")

    if not args.promote:
        print("Promotion skipped (use --promote to enable).")
        return

    champion_metrics_path = CHAMPION_DIR / "metrics.json"
    champion_model_path = CHAMPION_DIR / "model.joblib"
    champion_meta_path = CHAMPION_DIR / "metadata.json"

    champion_metrics = _read_json(champion_metrics_path) or {}
    champion_metric: Optional[float] = None
    if champion_metrics:
        try:
            champion_metric = _get_metric(champion_metrics, metric_name)
        except Exception:
            champion_metric = None

    should_promote = bool(args.force) or champion_metric is None or (best.metric_value > champion_metric)

    if not should_promote:
        print("Not promoting: candidate did not beat current champion.")
        print(f"- champion {metric_name}: {champion_metric}")
        return

    _copy_file(cand_model_path, champion_model_path)
    _copy_file(cand_metrics_path, champion_metrics_path)

    promote_meta = {
        "promoted_at": _utc_now_iso(),
        "source_candidate_dir": str(out_dir),
        "experiment_name": experiment_name,
        "metric_for_promotion": metric_name,
        "candidate_metric": best.metric_value,
        "previous_champion_metric": champion_metric,
        "model_name": best.name,
        "git_sha": os.getenv("GITHUB_SHA"),
    }
    _write_json(champion_meta_path, promote_meta)

    print("Promoted to champion")
    print(f"- champion_model: {champion_model_path}")


if __name__ == "__main__":
    main()
