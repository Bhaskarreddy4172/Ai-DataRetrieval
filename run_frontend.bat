@echo off
echo ===================================================
echo Starting Universal Dataset AI - Frontend (React/Vite)
echo ===================================================

cd /d "%~dp0frontend"

if not exist "node_modules\" (
    echo Installing npm dependencies...
    npm install
)

echo Starting Vite development server on http://localhost:3000...
npm run dev
pause
