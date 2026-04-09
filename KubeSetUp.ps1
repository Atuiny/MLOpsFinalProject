<#
.SYNOPSIS
  Downloads CI artifacts and deploys the Fraud API to Minikube.

.DESCRIPTION
  This script is the Kubernetes equivalent of StartUp.ps1.

  It is designed for the "no retraining" flow:
    - Downloads GitHub Actions artifacts for inspection:
        - champion-model
        - champion-registry
        - train-metrics
        - fraud-inference-api-tar
    - Syncs the downloaded files into the correct repo folders (for inspection)
    - Loads the Docker image tar into local Docker
    - Tags the image to the name used by Kubernetes (default: fraud-api:latest)
    - Loads that image into Minikube's image cache
    - Applies deployment.yaml + service.yaml
    - Prints the Minikube service URL

  Requirements:
    - gh (GitHub CLI) and authenticated (gh auth login)
    - docker
    - minikube
    - kubectl

.PARAMETER Repo
  GitHub repo (owner/repo).

.PARAMETER WorkflowFile
  Workflow filename (ml_pipeline.yaml).

.PARAMETER Branch
  Branch to query for a successful run (default main).

.PARAMETER RunId
  Optional: specific workflow run databaseId.

.PARAMETER Namespace
  Namespace to deploy to.

.PARAMETER DeploymentFile
  Path to deployment.yaml.

.PARAMETER ServiceFile
  Path to service.yaml.

.PARAMETER ServiceName
  Service name (used to print URL).

.PARAMETER K8sImage
  Image name:tag referenced by deployment.yaml (default fraud-api:latest).
  The script will tag the loaded CI image to this name.

.EXAMPLE
  gh auth login
  .\KubeSetUp.ps1
#>

param(
  [string]$Repo = "Atuiny/MLOpsFinalProject",
  [string]$WorkflowFile = "ml_pipeline.yaml",
  [string]$Branch = "main",
  [string]$RunId = "",

  [string]$ChampionModelArtifactName = "champion-model",
  [string]$ChampionRegistryArtifactName = "champion-registry",
  [string]$TrainMetricsArtifactName = "train-metrics",
  [string]$TarArtifactName = "fraud-inference-api-tar",

  [string]$Namespace = "default",
  [string]$DeploymentFile = "./deployment.yaml",
  [string]$ServiceFile = "./service.yaml",
  [string]$ServiceName = "fraud-api-service",

  [string]$K8sImage = "fraud-api:latest"
)

$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found: '$Name'. Install it and try again."
  }
}

function Ensure-Directory([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Path $Path | Out-Null
  }
}

function Find-FirstFile([string]$root, [string]$fileName, [string]$preferPathContains = "") {
  $files = Get-ChildItem -Path $root -Filter $fileName -Recurse -ErrorAction SilentlyContinue
  if (-not $files) { return $null }
  if ($preferPathContains) {
    $preferred = $files | Where-Object { $_.FullName -like ("*" + $preferPathContains + "*") } | Select-Object -First 1
    if ($preferred) { return $preferred }
  }
  return ($files | Select-Object -First 1)
}

function Download-Artifact([string]$runId, [string]$artifactName, [string]$destDir) {
  Write-Host "Downloading artifact '$artifactName' from run $runId..." -ForegroundColor Cyan
  Ensure-Directory $destDir
  $null = gh run download $runId --repo $Repo --name $artifactName -D $destDir
}

Require-Command gh
Require-Command docker
Require-Command minikube
Require-Command kubectl

Write-Host "Checking GitHub CLI auth..." -ForegroundColor Cyan
try { gh auth status | Out-Null } catch { throw "Not logged into GitHub CLI. Run: gh auth login" }

if (-not $RunId) {
  Write-Host "Finding latest successful workflow run on $Repo ($WorkflowFile @ $Branch)..." -ForegroundColor Cyan
  $RunId = gh run list --repo $Repo --workflow $WorkflowFile --branch $Branch --status success --limit 1 --json databaseId --jq '.[0].databaseId'
}
if (-not $RunId) { throw "No successful runs found for $WorkflowFile on branch $Branch." }

# Scratch directory for downloaded artifacts.
$artifactDir = Join-Path $PSScriptRoot "ci_artifacts"
if (Test-Path $artifactDir) { Remove-Item -Recurse -Force $artifactDir }
New-Item -ItemType Directory -Path $artifactDir | Out-Null

Download-Artifact -runId $RunId -artifactName $ChampionModelArtifactName -destDir (Join-Path $artifactDir $ChampionModelArtifactName)
Download-Artifact -runId $RunId -artifactName $ChampionRegistryArtifactName -destDir (Join-Path $artifactDir $ChampionRegistryArtifactName)
Download-Artifact -runId $RunId -artifactName $TrainMetricsArtifactName -destDir (Join-Path $artifactDir $TrainMetricsArtifactName)
Download-Artifact -runId $RunId -artifactName $TarArtifactName -destDir (Join-Path $artifactDir $TarArtifactName)

# Sync files into repo for inspection.
$repoMetricsDir = Join-Path $PSScriptRoot "data\processed"
Ensure-Directory $repoMetricsDir
Get-ChildItem -Path $artifactDir -Filter "train_metrics_*.json" -Recurse | ForEach-Object {
  Copy-Item -Force $_.FullName (Join-Path $repoMetricsDir $_.Name)
}

$championDir = Join-Path $PSScriptRoot "modelinfo\modelregistry\champion"
Ensure-Directory $championDir

# Prefer copying the exact champion folder layout from the champion-registry artifact.
$registryArtifactRoot = Join-Path $artifactDir $ChampionRegistryArtifactName
$registryChampionPath = Join-Path $registryArtifactRoot "modelinfo\modelregistry\champion"
if (Test-Path $registryChampionPath) {
  Write-Host "Syncing champion registry files into: $championDir" -ForegroundColor Cyan
  # Remove any pre-existing champion files so we don't keep stale leftovers.
  Get-ChildItem -Path $championDir -Force -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse -ErrorAction SilentlyContinue
  Copy-Item -Force -Recurse (Join-Path $registryChampionPath "*") $championDir

  $championModelPath = Join-Path $championDir "model.joblib"
  # Intentionally do NOT copy to repo root. The champion folder is the source of truth.
} else {
  # Fallback: attempt to locate champion files anywhere under the downloaded artifacts.
  $dlChampionModel = Find-FirstFile -root $artifactDir -fileName "model.joblib" -preferPathContains "modelinfo\modelregistry\champion"
  $dlChampionMetrics = Find-FirstFile -root $artifactDir -fileName "metrics.json" -preferPathContains "modelinfo\modelregistry\champion"
  $dlChampionMetadata = Find-FirstFile -root $artifactDir -fileName "metadata.json" -preferPathContains "modelinfo\modelregistry\champion"

  if ($dlChampionModel) {
    Copy-Item -Force $dlChampionModel.FullName (Join-Path $championDir "model.joblib")
  } else {
    Write-Host "Warning: champion model.joblib not found in downloaded artifacts." -ForegroundColor Yellow
  }
  if ($dlChampionMetrics) { Copy-Item -Force $dlChampionMetrics.FullName (Join-Path $championDir "metrics.json") }
  if ($dlChampionMetadata) { Copy-Item -Force $dlChampionMetadata.FullName (Join-Path $championDir "metadata.json") }
}

# Find tar and copy to repo root for convenience.
$tar = Get-ChildItem -Path $artifactDir -Filter "*.tar" -Recurse | Select-Object -First 1
if (-not $tar) { throw "No .tar found under: $artifactDir" }
$repoTarPath = Join-Path $PSScriptRoot "fraud-inference-api.tar"
Copy-Item -Force $tar.FullName $repoTarPath

Write-Host "Loading Docker image from tar: $repoTarPath" -ForegroundColor Cyan
$loadOutput = docker load -i $repoTarPath 2>&1
$loadText = ($loadOutput | Out-String).Trim()
Write-Host $loadText

# Parse loaded image reference.
$imageRef = $null
$match = [regex]::Match($loadText, "Loaded image:\s*(.+)")
if ($match.Success) { $imageRef = $match.Groups[1].Value.Trim() }

if (-not $imageRef) {
  Write-Host "Could not auto-detect loaded image name. Check 'docker images' output." -ForegroundColor Yellow
  docker images
  throw "Image load succeeded but image name could not be parsed."
}

if ($imageRef -ne $K8sImage) {
  Write-Host "Tagging image '$imageRef' -> '$K8sImage'" -ForegroundColor Cyan
  docker tag $imageRef $K8sImage
}

Write-Host "Starting/ensuring Minikube is running..." -ForegroundColor Cyan
minikube start | Out-Null

Write-Host "Loading image into Minikube: $K8sImage" -ForegroundColor Cyan
minikube image load $K8sImage

if (-not (Test-Path $DeploymentFile)) { throw "Deployment file not found: $DeploymentFile" }
if (-not (Test-Path $ServiceFile)) { throw "Service file not found: $ServiceFile" }

Write-Host "Applying Kubernetes manifests..." -ForegroundColor Cyan
kubectl apply -n $Namespace -f $DeploymentFile
kubectl delete -n $Namespace -f $ServiceFile --ignore-not-found
kubectl apply -n $Namespace -f $ServiceFile

Write-Host "Waiting for rollout..." -ForegroundColor Cyan
try {
  $deployName = (kubectl get -n $Namespace -f $DeploymentFile -o jsonpath='{.items[0].metadata.name}')
  if ($deployName) {
    kubectl rollout status -n $Namespace deployment/$deployName --timeout=180s
  }
} catch {
  Write-Host "Could not auto-detect deployment name. Check with: kubectl get deployments" -ForegroundColor Yellow
}

Write-Host "\nService URL:" -ForegroundColor Green
minikube service -n $Namespace $ServiceName --url

Write-Host "\nTry endpoints:" -ForegroundColor Green
Write-Host "- /        (frontend)" -ForegroundColor Green
Write-Host "- /health  (health check)" -ForegroundColor Green
