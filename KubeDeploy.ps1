<#
.SYNOPSIS
  Deploys the Fraud API to a local Minikube Kubernetes cluster.

.DESCRIPTION
  This script is an operator helper for the Kubernetes part of the project.

  It:
    1) Starts Minikube (if needed)
    2) (Optional) Builds the Docker image *inside Minikube* (so the cluster can run it)
    3) Applies `deployment.yaml` and `service.yaml`
    4) Waits for the Deployment rollout
    5) Prints a reliable way to access the service

  Requirements:
    - minikube
    - kubectl
    - docker (when using Minikube's docker driver / building images)

.PARAMETER Image
  Docker image reference used by Kubernetes (must match deployment.yaml).
  Default: fraud-api:latest

.PARAMETER Namespace
  Kubernetes namespace to deploy into.
  Default: default

.PARAMETER DeploymentFile
  Path to deployment manifest.

.PARAMETER ServiceFile
  Path to service manifest.

.PARAMETER ServiceName
  Kubernetes Service name to print URL for.

.PARAMETER RunDvcRepro
  If set, runs `dvc repro` before building the image.
  Use this if you need to (re)create the champion model in the registry.

.PARAMETER BuildImage
  If set, builds the image inside Minikube before deploying.
  If not set, assumes the image is already available in Minikube (e.g. via KubeSetUp.ps1).

.EXAMPLE
  # Most common: build inside minikube and deploy
  .\KubeDeploy.ps1

.EXAMPLE
  # If you already built/loaded the image into minikube
  .\KubeDeploy.ps1 -SkipBuild

.EXAMPLE
  # Ensure model.joblib exists before building
  .\KubeDeploy.ps1 -RunDvcRepro
#>

param(
  [string]$Image = "fraud-api:latest",
  [string]$Namespace = "default",
  [string]$DeploymentFile = "./deployment.yaml",
  [string]$ServiceFile = "./service.yaml",
  [string]$ServiceName = "fraud-api-service",
  [switch]$RunDvcRepro,
  [switch]$BuildImage,

  # On Windows + Docker driver, `minikube service --url` may keep a proxy running
  # and never exit. Keep it opt-in so this script finishes reliably.
  [switch]$PrintServiceUrl
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: '$Name'. Install it and try again."
  }
}

function Resolve-DvcCommand() {
  $venvDvc = Join-Path $PSScriptRoot ".venv\Scripts\dvc.exe"
  if (Test-Path $venvDvc) { return $venvDvc }
  return "dvc"
}

function Resolve-PythonActivate() {
  $activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
  if (Test-Path $activate) { return $activate }
  return $null
}

Require-Command minikube
Require-Command kubectl

Write-Host "Starting/ensuring Minikube is running..." -ForegroundColor Cyan
minikube start | Out-Null

if ($RunDvcRepro) {
  $activate = Resolve-PythonActivate
  if ($activate) {
    Write-Host "Activating venv: $activate" -ForegroundColor Cyan
    . $activate
  } else {
    Write-Host "No .venv found. Running without venv activation." -ForegroundColor Yellow
  }

  $dvc = Resolve-DvcCommand
  Write-Host "Running pipeline: dvc repro" -ForegroundColor Cyan
  & $dvc repro
  if ($LASTEXITCODE -ne 0) { throw "dvc repro failed" }
}

$championModel = Join-Path $PSScriptRoot "modelinfo\modelregistry\champion\model.joblib"
if (-not (Test-Path $championModel)) {
  Write-Host "Warning: champion model not found at: $championModel" -ForegroundColor Yellow
  Write-Host "If you need to generate it locally, run: dvc repro" -ForegroundColor Yellow
  Write-Host "If you want the CI-trained model, run: .\\KubeSetUp.ps1" -ForegroundColor Yellow
}

if ($BuildImage) {
  Require-Command docker

  Write-Host "Configuring shell to use Minikube Docker daemon..." -ForegroundColor Cyan
  # This makes `docker build` build into Minikube's image store.
  minikube -p minikube docker-env | Invoke-Expression

  Write-Host "Building image inside Minikube: $Image" -ForegroundColor Cyan
  docker build -t $Image $PSScriptRoot
  if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
} else {
  Write-Host "Skipping docker build (default)." -ForegroundColor Yellow
  Write-Host "If the image isn't available in Minikube yet, run .\\KubeSetUp.ps1 or re-run with -BuildImage." -ForegroundColor Yellow
}

if (-not (Test-Path $DeploymentFile)) { throw "Deployment file not found: $DeploymentFile" }
if (-not (Test-Path $ServiceFile)) { throw "Service file not found: $ServiceFile" }

Write-Host "Applying Kubernetes manifests..." -ForegroundColor Cyan
kubectl apply -n $Namespace -f $DeploymentFile
kubectl apply -n $Namespace -f $ServiceFile

Write-Host "Waiting for rollout..." -ForegroundColor Cyan
# Deployment name is set in deployment.yaml; we discover it from the file.
# If this parsing fails, user can run kubectl rollout status manually.
try {
  $deployName = (kubectl get -n $Namespace -f $DeploymentFile -o jsonpath='{.items[0].metadata.name}')
  if ($deployName) {
    kubectl rollout status -n $Namespace deployment/$deployName --timeout=120s
  }
} catch {
  Write-Host "Could not auto-detect deployment name for rollout status. Check with: kubectl get deployments" -ForegroundColor Yellow
}

Write-Host "\nService URL:" -ForegroundColor Green
if ($PrintServiceUrl) {
  try {
    $url = (minikube service -n $Namespace $ServiceName --url --wait=1) | Select-Object -First 1
    if ($url) {
      Write-Host $url -ForegroundColor Green
      Write-Host "Note: on Windows with the Docker driver, this URL may only work while 'minikube service' is running." -ForegroundColor Yellow
      Write-Host "If you get ERR_CONNECTION_REFUSED, run this in a separate terminal and keep it open:" -ForegroundColor Yellow
      Write-Host "  minikube service -n $Namespace $ServiceName" -ForegroundColor Yellow
    } else {
      Write-Host "(No URL returned.)" -ForegroundColor Yellow
    }
  } catch {
    Write-Host "Could not fetch minikube service URL." -ForegroundColor Yellow
  }
} else {
  Write-Host "(Skipped. To print a URL, run: minikube service -n $Namespace $ServiceName --url)" -ForegroundColor Yellow
  Write-Host "Tip: run this script with -PrintServiceUrl to try once (non-blocking)." -ForegroundColor Yellow
}

Write-Host "\nReliable access (recommended):" -ForegroundColor Green
Write-Host "  kubectl port-forward -n $Namespace svc/$ServiceName 8000:8000" -ForegroundColor Green
Write-Host "  Then open: http://127.0.0.1:8000/" -ForegroundColor Green

Write-Host "\nTry endpoints:" -ForegroundColor Green
Write-Host "- /        (frontend)" -ForegroundColor Green
Write-Host "- /health  (health check)" -ForegroundColor Green
