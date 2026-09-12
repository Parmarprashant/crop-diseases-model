Write-Host "======================================================================" -ForegroundColor Green
Write-Host "   AgriVision AI - Cloud Model Inference Engine" -ForegroundColor Cyan
Write-Host "   FastAPI + PyTorch CUDA RTX 5060 + Cloudflare Public HTTPS Tunnel" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""

# 1. Check if model server is running on port 8000
$portCheck = netstat -ano | findstr :8000
if (-not $portCheck) {
    Write-Host "[1/2] Launching FastAPI model server on port 8000 with CUDA GPU..." -ForegroundColor Yellow
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn api.server:app --host 127.0.0.1 --port 8000"
    Start-Sleep -Seconds 6
} else {
    Write-Host "[1/2] Model server is already running on port 8000." -ForegroundColor Green
}

# 2. Launch Cloudflare Public Tunnel
Write-Host "[2/2] Connecting to Cloudflare global network for public HTTPS URL..." -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Your live public HTTPS URL will appear below (look for .trycloudflare.com):" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Green
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
