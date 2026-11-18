@echo off
title SentinelCV - Lanzador de Servidores
echo ========================================================
echo        INICIANDO SISTEMA DE SEGURIDAD SENTINELCV
echo ========================================================
echo.

set REPO_DIR=%~dp0
set BUN_EXE=C:\Users\Carlos\.bun\bin\bun.exe

echo [1/2] Iniciando Backend de IA (Python + Keras 3 + CUDA) en puerto 8000...
start "SentinelCV - Backend IA (:8000)" powershell -NoExit -Command ^
  "cd '%REPO_DIR%';" ^
  "$env:KERAS_HOME = '%REPO_DIR%.keras';" ^
  "$env:KERAS_BACKEND = 'torch';" ^
  "$env:TORCH_HOME = '%REPO_DIR%.torch';" ^
  "Write-Host '==================================================' -ForegroundColor Cyan;" ^
  "Write-Host '   BACKEND IA SENTINELCV - CARGANDO MODELOS EN GPU' -ForegroundColor Cyan;" ^
  "Write-Host '==================================================' -ForegroundColor Cyan;" ^
  "python run_server.py"

timeout /t 3 /nobreak >nul

echo [2/2] Iniciando Consola Web (Astro) en puerto 4321...
start "SentinelCV - Consola Web Astro (:4321)" powershell -NoExit -Command ^
  "cd '%REPO_DIR%web_security';" ^
  "Write-Host '==================================================' -ForegroundColor Green;" ^
  "Write-Host '   CONSOLA WEB ASTRO - MATRIZ DE 6 CAMARAS' -ForegroundColor Green;" ^
  "Write-Host '==================================================' -ForegroundColor Green;" ^
  "& '%BUN_EXE%' run dev"

echo.
echo ========================================================
echo  Servidores iniciados en ventanas independientes:
echo  - Backend IA:    http://localhost:8000
echo  - Consola Web:   http://localhost:4321
echo ========================================================
echo.
pause
