@echo off
echo ===================================================
echo Starting Universal Dataset AI - Backend (FastAPI)
echo ===================================================

cd /d "%~dp0"

REM Activate virtualenv if present
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

echo Initializing database and verifying catalog...
python scripts\setup_database.py

echo Starting Uvicorn API server on http://localhost:8000...
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
