# Lanzador rapido de SentinelCV (Backend IA + Frontend Web Astro)
$repoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $repoDir) { $repoDir = Get-Location }

$bunExe = "C:\Users\Carlos\.bun\bin\bun.exe"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       INICIANDO SISTEMA DE SEGURIDAD SENTINELCV       " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Iniciar Backend en ventana separada
Write-Host "[1/2] Levantando Backend IA en puerto 8000..." -ForegroundColor Yellow
$backendCmd = @"
cd '$repoDir'
`$env:KERAS_HOME = '$repoDir\.keras'
`$env:KERAS_BACKEND = 'torch'
`$env:TORCH_HOME = '$repoDir\.torch'
Write-Host '==================================================' -ForegroundColor Cyan
Write-Host '   BACKEND IA SENTINELCV - CARGANDO MODELOS EN GPU' -ForegroundColor Cyan
Write-Host '==================================================' -ForegroundColor Cyan
python run_server.py
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Start-Sleep -Seconds 3

# 2. Iniciar Frontend Web Astro en ventana separada
Write-Host "[2/2] Levantando Consola Web Astro en puerto 4321..." -ForegroundColor Yellow
$frontendCmd = @"
cd '$repoDir\web_security'
Write-Host '==================================================' -ForegroundColor Green
Write-Host '   CONSOLA WEB ASTRO - MATRIZ DE 6 CAMARAS' -ForegroundColor Green
Write-Host '==================================================' -ForegroundColor Green
& '$bunExe' run dev
"@
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Servidores activos en ventanas independientes:" -ForegroundColor Green
Write-Host "  - Backend IA:    http://localhost:8000" -ForegroundColor White
Write-Host "  - Consola Web:   http://localhost:4321" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Green
