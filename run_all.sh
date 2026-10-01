#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "==================================================="
echo "Universal Dataset AI Platform - Full Stack Launcher"
echo "==================================================="

echo "[1/3] Setting up database and running diagnostics..."
python3 scripts/setup_database.py
python3 scripts/validate_all.py

echo "[2/3] Starting Backend API in background..."
./run_backend.sh &
BACKEND_PID=$!

echo "[3/3] Starting Frontend UI in background..."
./run_frontend.sh &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo ""
echo "==================================================="
echo "Platform is running!"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs:    http://localhost:8000/docs"
echo "  - Frontend UI: http://localhost:3000"
echo "Press Ctrl+C to terminate both servers."
echo "==================================================="

wait
