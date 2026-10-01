@echo off
echo ===================================================
echo Universal Dataset AI Platform - Full Stack Launcher
echo ===================================================

cd /d "%~dp0"

echo [1/3] Setting up database tables and verifying RAG index...
python scripts\setup_database.py
python scripts\validate_all.py

echo [2/3] Launching Backend Server in separate terminal...
start "Universal AI - Backend" cmd /k "run_backend.bat"

echo [3/3] Launching Frontend UI in separate terminal...
start "Universal AI - Frontend" cmd /k "run_frontend.bat"

echo.
echo ===================================================
echo Platform is starting up!
echo   - Backend API: http://localhost:8000
echo   - API Docs:    http://localhost:8000/docs
echo   - Frontend UI: http://localhost:3000
echo ===================================================
timeout /t 5
