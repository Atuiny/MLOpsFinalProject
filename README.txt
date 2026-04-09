
MLOps Final Project — Fraud Probability (Local UI + Docker)
=========================================================

This repo trains a fraud classifier and serves predictions through a FastAPI app.
It includes a simple built-in frontend at `/` where a user can enter feature values
and see the percent chance of fraud.


Project layout (important files)
-------------------------------
- `data/raw/credit_card_data.csv` — raw dataset (committed to git for class scope)
- `src/CleanData.py` — cleaning stage
- `src/Train.py` — trains LR / RF / DT and writes metrics
- `modelinfo/registryhelpers/promote_to_registry.py` — selects best model + writes champion
- `app.py` — FastAPI API + built-in frontend at `/`
- `Dockerfile` — container for FastAPI inference


Quick start (fresh download from GitHub)
--------------------------------------

1) Clone and enter the project

	`git clone https://github.com/Atuiny/MLOpsFinalProject.git`
	`cd MLOpsFinalProject`

2) Create a virtual environment and install dependencies

	Windows PowerShell:
	`python -m venv .venv`
	`./.venv/Scripts/Activate.ps1`
	`python -m pip install --upgrade pip`
	`pip install -r requirements.txt`

3) Run the pipeline end-to-end (creates the trained model)

	`dvc repro`

	What `dvc repro` does:
	- Runs cleaning (creates processed dataset)
	- Trains models (LR/RF/DT) and writes metrics
	- Promotes the best model to "champion" and copies it to repo root for Docker

	Outputs you should see after this:
	- `data/processed/credit_card_clean.csv`
	- `data/processed/train_metrics_lr.json` (and rf/dt)
	- `modelinfo/modelregistry/champion/model.joblib`
	- `model.joblib` (a copy of the champion for Docker)

4) Run the app locally (includes frontend)

	`python -m uvicorn app:app --host 127.0.0.1 --port 8000`

	Open:
	- `http://127.0.0.1:8000/` (frontend)
	- `http://127.0.0.1:8000/docs` (API docs)


Using Docker (after the model exists)
-----------------------------------

Docker builds expect `model.joblib` to exist at the repo root. The easiest path is:

1) Run `dvc repro` (see above). This creates `model.joblib`.
2) Build + run:

	`docker build -t fraud-inference-api:latest .`
	`docker run --rm -p 8000:8000 fraud-inference-api:latest`

Then open `http://127.0.0.1:8000/`.


If you want to use the model trained by GitHub Actions
------------------------------------------------------

GitHub Actions uploads trained artifacts, but they are NOT automatically committed into the repo.

This section is the "no retraining" path: you can clone the repo and then drop in the
already-trained outputs from a GitHub Actions run.

To use an Actions-trained model on your machine:

1) Go to GitHub → Actions → select the workflow run → download the artifact that contains the model.

2) Place the model file in ONE of these supported locations:

	Option A (recommended for Docker):
	- Put it at repo root as: `model.joblib`

	Option B (registry location):
	- Put it at: `modelinfo/modelregistry/champion/model.joblib`
	- (Optional) also copy it to repo root `model.joblib` for Docker builds.

3) (Optional but recommended) Place the champion JSON files (if you downloaded them)

	Put these in the champion folder:
	- `modelinfo/modelregistry/champion/metrics.json`
	- `modelinfo/modelregistry/champion/metadata.json`

3) Run the app or Docker as above.

Notes:
- The app searches in this order: `MODEL_PATH` env var → `./model.joblib` → `modelinfo/modelregistry/champion/model.joblib`.
- If you set `MODEL_PATH`, point it directly to the `.joblib` file.


If you want to use the Docker image tar from GitHub Actions (no build)
---------------------------------------------------------------------

The workflow can export a Docker image to a tar file and upload it as an artifact.

Where to put the tar:
- Anywhere on your machine (it does not need to live inside the repo).

How to use it:
1) Download the Docker-image-tar artifact from the Actions run.
2) Load it into Docker:
	`docker load -i <path-to-downloaded-tar>`
3) Run it:
	`docker run --rm -p 8000:8000 fraud-inference-api:latest`

Important:
- This tar already contains the model that was present in CI at build time.
- You do NOT need to provide `model.joblib` locally when running this image.


Frontend feature inputs
-----------------------

The `/` page asks for 7 features (defaults are 0 / false):
- Distance from home
- Distance from last transaction
- Purchase ratio vs typical (median)
- Same retailer as before? (true/false)
- Used chip? (true/false)
- Used PIN? (true/false)
- Online order? (true/false)

