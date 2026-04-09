<#
.SYNOPSIS
  Deploys the Fraud API to a local Minikube Kubernetes cluster.

.DESCRIPTION
  This script is an operator helper for the Kubernetes part of the project.

  It:
    1) Starts Minikube (if needed)
    2) (Optional) Runs `dvc repro` to ensure `model.joblib` exists
    3) Builds the Docker image *inside Minikube* (so the cluster can run it)
    4) Applies `deployment.yaml` and `service.yaml`
    5) Waits for the Deployment rollout
    6) Prints the Service URL (NodePort) for access

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
  Use this if you need to regenerate `model.joblib`.

.PARAMETER SkipBuild
  If set, skips docker build (assumes image is already present in Minikube).

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
  [switch]$SkipBuild
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

if (-not (Test-Path (Join-Path $PSScriptRoot "model.joblib"))) {
  Write-Host "Warning: model.joblib not found at repo root." -ForegroundColor Yellow
  Write-Host "Docker/Kubernetes image build will fail if your Dockerfile expects it." -ForegroundColor Yellow
  Write-Host "If needed, run: dvc repro (or re-run this script with -RunDvcRepro)" -ForegroundColor Yellow
}

if (-not $SkipBuild) {
  Require-Command docker

  Write-Host "Configuring shell to use Minikube Docker daemon..." -ForegroundColor Cyan
  # This makes `docker build` build into Minikube's image store.
  minikube -p minikube docker-env | Invoke-Expression

  Write-Host "Building image inside Minikube: $Image" -ForegroundColor Cyan
  docker build -t $Image $PSScriptRoot
  if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
} else {
  Write-Host "Skipping docker build (-SkipBuild set)." -ForegroundColor Yellow
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
minikube service -n $Namespace $ServiceName --url

Write-Host "\nTry endpoints:" -ForegroundColor Green
Write-Host "- /        (frontend)" -ForegroundColor Green
Write-Host "- /health  (health check)" -ForegroundColor Green
