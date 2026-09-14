# AgriVision 1-Click Cloud Deployment Script
# Automatically syncs latest Model/ changes to Hugging Face Spaces

$ErrorActionPreference = "Stop"

$MODEL_DIR = $PSScriptRoot
$DEPLOY_DIR = Join-Path (Split-Path -Parent $MODEL_DIR) "deploy_hf"

if (-not (Test-Path $DEPLOY_DIR)) {
    Write-Error "Deployment directory '$DEPLOY_DIR' not found. Please ensure deploy_hf exists."
    exit 1
}

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " [AgriVision] Syncing latest Model updates to Cloud...    " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green

# 1. Sync code modules
Write-Host "`n[1/4] Syncing API, Core, Inference, and Model architectures..." -ForegroundColor Cyan
Copy-Item -Path (Join-Path $MODEL_DIR "api\*") -Destination (Join-Path $DEPLOY_DIR "api\") -Recurse -Force
Copy-Item -Path (Join-Path $MODEL_DIR "core\*") -Destination (Join-Path $DEPLOY_DIR "core\") -Recurse -Force
Copy-Item -Path (Join-Path $MODEL_DIR "inference\*") -Destination (Join-Path $DEPLOY_DIR "inference\") -Recurse -Force
Copy-Item -Path (Join-Path $MODEL_DIR "models\*") -Destination (Join-Path $DEPLOY_DIR "models\") -Recurse -Force

# Clean any pycache in deploy folder
Get-ChildItem -Path $DEPLOY_DIR -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 2. Sync configs & metadata
Write-Host "[2/4] Syncing taxonomy, classes, and calibration thresholds..." -ForegroundColor Cyan
$configs = @(
    "class_names.txt",
    "crop_names.txt",
    "calibration_thresholds.json",
    "canonical_class_map.json",
    "canonical_disease_to_crop.json"
)
foreach ($cfg in $configs) {
    $src = Join-Path $MODEL_DIR "weights\$cfg"
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $DEPLOY_DIR "weights\$cfg") -Force
    }
}

# 3. Sync model weights if modified
Write-Host "[3/4] Checking model weights..." -ForegroundColor Cyan
$weights = @(
    "efficientnet_b5_cbam_best.pt",
    "crop_expert_candidate.pt",
    "unet_resnet34_best.pt"
)
foreach ($w in $weights) {
    $src = Join-Path $MODEL_DIR "weights\$w"
    $dst = Join-Path $DEPLOY_DIR "weights\$w"
    if ((Test-Path $src) -and ((-not (Test-Path $dst)) -or ((Get-Item $src).LastWriteTime -gt (Get-Item $dst).LastWriteTime))) {
        Write-Host "  -> Updating $w ($( [math]::Round((Get-Item $src).Length / 1MB, 1) ) MB)..." -ForegroundColor Yellow
        Copy-Item -Path $src -Destination $dst -Force
    }
}

# Sync YOLOv8 if modified
$yolo_src = Join-Path $MODEL_DIR "yolov8n.pt"
$yolo_dst = Join-Path $DEPLOY_DIR "weights\yolov8n_pest.pt"
if ((Test-Path $yolo_src) -and ((-not (Test-Path $yolo_dst)) -or ((Get-Item $yolo_src).LastWriteTime -gt (Get-Item $yolo_dst).LastWriteTime))) {
    Copy-Item -Path $yolo_src -Destination $yolo_dst -Force
}

# 4. Commit and Push to Hugging Face
Write-Host "`n[4/4] Pushing to Hugging Face Space (24/7 Cloud)..." -ForegroundColor Cyan
Push-Location $DEPLOY_DIR
try {
    git add -A
    $status = git status --porcelain
    if ($status) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        git commit -m "Update AgriVision deployment [$timestamp]"
        git push origin main
        Write-Host "`n Deployment successfully pushed to Hugging Face!" -ForegroundColor Green
        Write-Host "Your cloud server will rebuild and go live in ~1-2 minutes." -ForegroundColor Green
    } else {
        Write-Host "`n No changes detected between Model/ and cloud deployment. Already up to date!" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}
