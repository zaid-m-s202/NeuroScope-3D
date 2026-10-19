@echo off
title NeuroScope 3D

echo ================================================
echo         NeuroScope 3D - Starting App
echo ================================================
echo.

:: Start Backend
echo [1/2] Starting Backend (FastAPI)...
start "NeuroScope Backend" cmd /k "cd /d %~dp0 && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000"

:: Small delay to let backend initialize
timeout /t 2 /nobreak >nul

:: Start Frontend
echo [2/2] Starting Frontend (Vite)...
start "NeuroScope Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ================================================
echo  Backend  : http://localhost:8000
echo  Frontend : http://localhost:5173
echo  API Docs : http://localhost:8000/docs
echo ================================================
echo.
echo Both servers are running in separate windows.
echo Close those windows to stop the servers.
echo.
pause
