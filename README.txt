
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
	- Promotes the best model to "champion" in the local registry

	Outputs you should see after this:
	- `data/processed/credit_card_clean.csv`
	- `data/processed/train_metrics_lr.json` (and rf/dt)
	- `modelinfo/modelregistry/champion/model.joblib`

4) Run the app locally (includes frontend)

	`python -m uvicorn app:app --host 127.0.0.1 --port 8000`

	Open:
	- `http://127.0.0.1:8000/` (frontend)
	- `http://127.0.0.1:8000/docs` (API docs)


Using Docker (after the model exists)
-----------------------------------

Docker builds expect the promoted champion to exist at:

	`modelinfo/modelregistry/champion/model.joblib`

The easiest path is:

1) Run `dvc repro` (see above). This creates the champion registry files.
2) Build + run:

	`docker build -t fraud-inference-api:latest .`
	`docker run --rm -p 8000:8000 fraud-inference-api:latest`

Then open `http://127.0.0.1:8000/`.


Kubernetes (Minikube)
---------------------

This project includes Kubernetes manifests:
- [deployment.yaml](deployment.yaml) (Deployment)
- [service.yaml](service.yaml) (NodePort Service)

The manifests expect a local image named `fraud-api:latest`.

Recommended scripts (split into setup vs deploy):
- `KubeSetUp.ps1` — downloads CI artifacts + loads the Docker image into Minikube (no deploy)
- `KubeDeploy.ps1` — applies Kubernetes manifests and shows how to access the service

Option A: Use the CI-built Docker image (no build, no retrain)

1) Setup (downloads artifacts + loads image into Minikube)

	`gh auth login`
	`./KubeSetUp.ps1`

2) Deploy

	`./KubeDeploy.ps1`

Option B: Build locally inside Minikube

1) Ensure the champion model exists in the registry

	`dvc repro`

2) Deploy (and build into Minikube)

	`./KubeDeploy.ps1 -BuildImage`

Access (Windows + Minikube Docker driver)

The most reliable method is port-forward (keep it running):
	`kubectl port-forward svc/fraud-api-service 8000:8000`

Then open:
- `http://127.0.0.1:8000/` (frontend)
- `http://127.0.0.1:8000/health`

You can also use:
	`minikube service fraud-api-service --url`
But on Windows with the Docker driver, you may need to keep that command running in a terminal.


If you want to use the model trained by GitHub Actions
------------------------------------------------------

GitHub Actions uploads trained artifacts, but they are NOT automatically committed into the repo.

This section is the "no retraining" path: you can clone the repo and then drop in the
already-trained outputs from a GitHub Actions run.

To use an Actions-trained model on your machine:

1) Go to GitHub → Actions → select the workflow run → download the artifact that contains the model.

2) Place the model file here (registry location):

	- `modelinfo/modelregistry/champion/model.joblib`

3) (Optional but recommended) Place the champion JSON files (if you downloaded them)

	Put these in the champion folder:
	- `modelinfo/modelregistry/champion/metrics.json`
	- `modelinfo/modelregistry/champion/metadata.json`

3) Run the app or Docker as above.

Notes:
- The app searches in this order: `MODEL_PATH` env var → `modelinfo/modelregistry/champion/model.joblib` → `./model.joblib`.
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

