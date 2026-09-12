@echo off
title "AgriVision Cloud Inference Server and Tunnel"
echo ======================================================================
echo    AgriVision AI - Cloud Model Inference Engine
echo    FastAPI + PyTorch CUDA RTX 5060 + Cloudflare Public HTTPS Tunnel
echo ======================================================================
echo.

:: 1. Check if model server is running on port 8000
netstat -ano | findstr :8000 >nul
if %errorlevel% neq 0 (
    echo [1/2] Launching FastAPI model server on port 8000 with CUDA GPU...
    start "AgriVision Model Server" cmd /k "python -m uvicorn api.server:app --host 127.0.0.1 --port 8000"
    timeout /t 6 /nobreak >nul
) else (
    echo [1/2] Model server is already running on port 8000.
)

:: 2. Launch Cloudflare Public Tunnel
echo [2/2] Connecting to Cloudflare global network for public HTTPS URL...
echo.
echo ======================================================================
echo Your live public HTTPS URL will appear below:
echo ======================================================================
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause


