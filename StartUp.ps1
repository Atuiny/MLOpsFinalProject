<#
.SYNOPSIS
  Starts the Fraud API locally (Docker), optionally using CI artifacts.

.DESCRIPTION
  This script is a small "operator" helper for running your Dockerized FastAPI model.

  It supports THREE modes:

    1) Manual mode (local tar) — NO retraining
      - You already have a Docker image tar file (downloaded from GitHub Actions)
     - The script runs: docker load -i <tar>
     - Then it runs: docker run -p <HostPort>:8000 <loaded-image>

  2) Download mode (convenience) — NO retraining
     - If you pass -DownloadFromGitHub, the script uses GitHub CLI (gh) to download
       the Docker image tar artifact produced by your GitHub Actions workflow.
     - Then it performs the same docker load/run steps.

  3) Local build mode (optional)
     - Runs `dvc repro` to generate `model.joblib` (unless -SkipDvcRepro)
     - Builds a local Docker image from this repo
     - Runs the container on http://127.0.0.1:<HostPort>

  Why a .tar?
    "docker load" is the easiest way to move a prebuilt image from CI to your machine
    without publishing to Docker Hub.

  What you get when it's running:
    - API base URL: http://127.0.0.1:<HostPort>
    - Health check: GET  /health
    - Prediction:   POST /predict
    - Simple UI:    GET  /

.PARAMETER TarPath
  Path to the docker image tar file to load.
  Only used when you are NOT downloading from GitHub.

.PARAMETER DownloadFromGitHub
  If set, downloads the latest successful workflow artifact that contains a .tar file.
  Requires GitHub CLI: gh

.PARAMETER LocalBuild
  If set, builds the Docker image from the current repo and runs it.

.PARAMETER SkipDvcRepro
  Used only with -LocalBuild.
  If set, does NOT run `dvc repro` before building.

.PARAMETER Repo
  GitHub repository in the form "owner/repo".

.PARAMETER WorkflowFile
  The workflow file name used to find runs (e.g. ml_pipeline.yaml).

.PARAMETER Branch
  Branch to search for successful workflow runs.

.PARAMETER ArtifactName
  The artifact name expected to contain the docker .tar (uploaded by Actions).

.PARAMETER ChampionModelArtifactName
  Artifact name that contains the promoted champion model file.

.PARAMETER ChampionRegistryArtifactName
  Artifact name that contains champion registry JSON files (metadata/metrics).

.PARAMETER TrainMetricsArtifactName
  Artifact name that contains training metrics JSON files.


.PARAMETER RunId
  Optional: specify a particular run id to download from.
  If omitted, the script finds the latest successful run.

.PARAMETER Detach
  If set, starts the container in the background (-d) and returns control to your terminal.
  If not set, runs in the foreground and streams container logs until you stop it.

.PARAMETER HostPort
  The port on your machine to expose the API on (defaults to 8000).

.PARAMETER ContainerName
  Name to give the running container (used for stop/remove).

.EXAMPLE
  # Use a local tar downloaded from GitHub Actions
  .\StartUp.ps1 -TarPath .\fraud-inference-api.tar -Detach

.EXAMPLE
  # Auto-download the latest successful CI artifact tar then run it
  gh auth login
  .\StartUp.ps1 -DownloadFromGitHub -Detach

.EXAMPLE
  # Local build (optional): runs dvc repro, then builds and runs
  .\StartUp.ps1 -LocalBuild -Detach
#>

param(
  # If you're using a prebuilt image tar, point this at it.
  [string]$TarPath = "./fraud-inference-api.tar",

  # If set, we will download the docker image tar from GitHub Actions using `gh`.
  [switch]$DownloadFromGitHub,

  # If set, we build the Docker image from the current repo and run it.
  [switch]$LocalBuild,

  # Used only with -LocalBuild.
  # If set, skip `dvc repro` (assumes model.joblib already exists).
  [switch]$SkipDvcRepro,

  # Repo/workflow/run selection for the download mode.
  [string]$Repo = "Atuiny/MLOpsFinalProject",
  [string]$WorkflowFile = "ml_pipeline.yaml",
  [string]$Branch = "main",
  [string]$ArtifactName = "fraud-inference-api-tar",
  [string]$ChampionModelArtifactName = "champion-model",
  [string]$ChampionRegistryArtifactName = "champion-registry",
  [string]$TrainMetricsArtifactName = "train-metrics",
  [string]$RunId = "",

  # If set, run the container in the background (recommended for interactive use).
  [switch]$Detach,

  # Port mapping: http://127.0.0.1:<HostPort> maps to container port 8000.
  [int]$HostPort = 8000,

  # Container name, so we can stop/remove it predictably.
  [string]$ContainerName = "fraud-inference-api",

  # Used only with -LocalBuild.
  # Image tag to build/run.
  [string]$LocalImage = "fraud-api:latest"
)

# Make errors stop the script immediately.
# This avoids confusing "half-success" states.
$ErrorActionPreference = "Stop"

function Require-Command([string]$Name) {
  # Get-Command checks if an executable/cmdlet is available in PATH.
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
  $null = gh run download $runId --repo $Repo --name $artifactName -D $destDir
}

function Resolve-DvcCommand() {
  # Prefer a venv-local DVC if present.
  $venvDvc = Join-Path $PSScriptRoot ".venv\Scripts\dvc.exe"
  if (Test-Path $venvDvc) {
    return $venvDvc
  }
  # Fallback: hope dvc is on PATH.
  return "dvc"
}

function Resolve-PythonActivate() {
  $activate = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
  if (Test-Path $activate) {
    return $activate
  }
  return $null
}

if ($DownloadFromGitHub -and $LocalBuild) {
  throw "Choose ONE mode: -DownloadFromGitHub OR -LocalBuild"
}

if ($DownloadFromGitHub) {
  # In download mode we need GitHub CLI.
  Require-Command gh

  Write-Host "Checking GitHub CLI auth..." -ForegroundColor Cyan
  try {
    gh auth status | Out-Null
  } catch {
    throw "Not logged into GitHub CLI. Run: gh auth login"
  }

  # If the caller didn't provide a specific run id, find the latest successful run.
  if (-not $RunId) {
    Write-Host "Finding latest successful workflow run on $Repo ($WorkflowFile @ $Branch)..." -ForegroundColor Cyan
    $RunId = gh run list --repo $Repo --workflow $WorkflowFile --branch $Branch --status success --limit 1 --json databaseId --jq '.[0].databaseId'
  }

  if (-not $RunId) {
    throw "No successful runs found yet for $WorkflowFile on branch $Branch. Run the workflow once in GitHub Actions first."
  }

  # We'll download/extract artifacts into a local scratch directory.
  # (Recommended to gitignore this folder.)
  $artifactDir = Join-Path $PSScriptRoot "ci_artifacts"
  if (Test-Path $artifactDir) {
    Remove-Item -Recurse -Force $artifactDir
  }
  New-Item -ItemType Directory -Path $artifactDir | Out-Null

  # Download all artifacts we care about.
  Download-Artifact -runId $RunId -artifactName $ChampionModelArtifactName -destDir $artifactDir
  Download-Artifact -runId $RunId -artifactName $ChampionRegistryArtifactName -destDir $artifactDir
  Download-Artifact -runId $RunId -artifactName $TrainMetricsArtifactName -destDir $artifactDir
  Download-Artifact -runId $RunId -artifactName $ArtifactName -destDir $artifactDir

  # Locate downloaded files.
  $downloadedChampionModel = Find-FirstFile -root $artifactDir -fileName "model.joblib" -preferPathContains "modelinfo\modelregistry\champion"
  $downloadedChampionMetrics = Find-FirstFile -root $artifactDir -fileName "metrics.json" -preferPathContains "modelinfo\modelregistry\champion"
  $downloadedChampionMetadata = Find-FirstFile -root $artifactDir -fileName "metadata.json" -preferPathContains "modelinfo\modelregistry\champion"

  # The tar artifact is usually at repo root in CI; find the first *.tar.
  $tar = Get-ChildItem -Path $artifactDir -Filter "*.tar" -Recurse | Select-Object -First 1
  if (-not $tar) {
    throw "Download succeeded but no .tar file was found under: $artifactDir"
  }

  # Copy train metrics into repo (optional but useful for inspection).
  $repoMetricsDir = Join-Path $PSScriptRoot "data\processed"
  Ensure-Directory $repoMetricsDir
  Get-ChildItem -Path $artifactDir -Filter "train_metrics_*.json" -Recurse | ForEach-Object {
    Copy-Item -Force $_.FullName (Join-Path $repoMetricsDir $_.Name)
  }

  # Sync champion files from the selected run for inspection and local use.
  # NOTE: This does not attempt to "pick" a champion. CI already did that.
  $localChampionDir = Join-Path $PSScriptRoot "modelinfo\modelregistry\champion"
  Ensure-Directory $localChampionDir

  if ($downloadedChampionModel) {
    Write-Host "Syncing champion model into repo..." -ForegroundColor Cyan
    Copy-Item -Force $downloadedChampionModel.FullName (Join-Path $localChampionDir "model.joblib")
    Copy-Item -Force (Join-Path $localChampionDir "model.joblib") (Join-Path $PSScriptRoot "model.joblib")
  } else {
    Write-Host "Warning: champion model.joblib not found in downloaded artifacts." -ForegroundColor Yellow
  }

  if ($downloadedChampionMetrics) {
    Copy-Item -Force $downloadedChampionMetrics.FullName (Join-Path $localChampionDir "metrics.json")
  }
  if ($downloadedChampionMetadata) {
    Copy-Item -Force $downloadedChampionMetadata.FullName (Join-Path $localChampionDir "metadata.json")
  }

  # Copy tar to repo root for convenience.
  $repoTarPath = Join-Path $PSScriptRoot "fraud-inference-api.tar"
  Copy-Item -Force $tar.FullName $repoTarPath
  $TarPath = $repoTarPath
  Write-Host "Downloaded tar: $TarPath" -ForegroundColor Green
}

if ($LocalBuild) {
  Require-Command docker

  if (-not $SkipDvcRepro) {
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
    if ($LASTEXITCODE -ne 0) {
      throw "dvc repro failed"
    }
  }

  if (-not (Test-Path (Join-Path $PSScriptRoot "model.joblib"))) {
    throw "model.joblib not found at repo root. Run 'dvc repro' first or provide a model.joblib."
  }

  Write-Host "Building Docker image: $LocalImage" -ForegroundColor Cyan
  docker build -t $LocalImage $PSScriptRoot
  if ($LASTEXITCODE -ne 0) {
    throw "docker build failed"
  }

  Write-Host "Running container '$ContainerName' from image '$LocalImage'..." -ForegroundColor Cyan
  try { docker rm -f $ContainerName | Out-Null } catch {}

  if ($Detach) {
    $null = docker run -d --name $ContainerName -p "${HostPort}:8000" $LocalImage
    Write-Host "Started: http://127.0.0.1:$HostPort" -ForegroundColor Green
    Write-Host "UI:      http://127.0.0.1:$HostPort/" -ForegroundColor Green
    Write-Host "Swagger: http://127.0.0.1:$HostPort/docs" -ForegroundColor Green
    Write-Host "Stop:    docker rm -f $ContainerName" -ForegroundColor Yellow
    return
  }

  docker run --rm --name $ContainerName -p "${HostPort}:8000" $LocalImage
  return
}

# Docker is required for both modes.
Require-Command docker

# Safety check: we should have a tar path by this point.
if (-not (Test-Path $TarPath)) {
  throw "Tar file not found: $TarPath"
}

# Load the image tar into your local Docker image cache.
# Example:
#   docker load -i champion-inference-api.tar
Write-Host "Loading Docker image from tar: $TarPath" -ForegroundColor Cyan
$loadOutput = docker load -i $TarPath 2>&1
$loadText = ($loadOutput | Out-String).Trim()
Write-Host $loadText

# Try to parse the loaded image reference from the docker output.
# Typical output includes: "Loaded image: champion-inference-api:latest"
# If Docker changes its wording, this auto-detection could fail.
$imageRef = $null
$match = [regex]::Match($loadText, "Loaded image:\s*(.+)")
if ($match.Success) {
  $imageRef = $match.Groups[1].Value.Trim()
}

Write-Host "\nCheck available images:" -ForegroundColor Cyan
Write-Host "docker images" -ForegroundColor Gray
docker images

if (-not $imageRef) {
  # Fallback behavior: we don't know the image name/tag.
  # The user can read `docker images` and run docker manually.
  Write-Host "\nCould not auto-detect image name from 'docker load' output." -ForegroundColor Yellow
  Write-Host "Re-run with the image ref you see in 'docker images', e.g.:" -ForegroundColor Yellow
  Write-Host "docker run --rm -p ${HostPort}:8000 <IMAGE_NAME:TAG>" -ForegroundColor Yellow
  return
}

Write-Host "\nRunning container '$ContainerName' from image '$imageRef'..." -ForegroundColor Cyan

# If a container with this name already exists, remove it so we can restart cleanly.
try { docker rm -f $ContainerName | Out-Null } catch {}

if ($Detach) {
  # Detached mode: docker returns immediately.
  # Good when you want to keep using your terminal for API calls.
  $null = docker run -d --name $ContainerName -p "${HostPort}:8000" $imageRef
  Write-Host "Started: http://127.0.0.1:$HostPort" -ForegroundColor Green
  Write-Host "UI:      http://127.0.0.1:$HostPort/" -ForegroundColor Green
  # FastAPI generates Swagger automatically at /docs.
  # If you don't want to use it, you can ignore this.
  Write-Host "Swagger: http://127.0.0.1:$HostPort/docs" -ForegroundColor Green
  Write-Host "Stop:    docker rm -f $ContainerName" -ForegroundColor Yellow
} else {
  # Foreground mode: this will stream container logs until you stop it (Ctrl+C).
  # The container is removed automatically when it exits (--rm).
  docker run --rm --name $ContainerName -p "${HostPort}:8000" $imageRef
}
