@echo off
title "AgriVision Cloud Inference Server and Tunnel"
echo ======================================================================
echo    AgriVision AI - Cloud Model Inference Engine
echo    FastAPI + PyTorch CUDA RTX 5060 + Cloudflare Public HTTPS Tunnel
echo ======================================================================
echo.

:: 1. Terminate any stale process on port 8000 to ensure fresh code is loaded
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo [1/2] Terminating previous server process (PID %%a) to load fresh code...
    taskkill /F /PID %%a >nul 2>&1
)

echo [1/2] Launching FastAPI model server on port 8000 with CUDA GPU...
start "AgriVision Model Server" cmd /k "python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload"
timeout /t 5 /nobreak >nul

:: 2. Launch Cloudflare Public Tunnel
echo [2/2] Connecting to Cloudflare global network for public HTTPS URL...
echo.
echo ======================================================================
echo Your live public HTTPS URL will appear below:
echo ======================================================================
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause


