#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

echo "==================================================="
echo "Starting Universal Dataset AI - Backend (FastAPI)"
echo "==================================================="

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "Initializing database..."
python3 scripts/setup_database.py

echo "Starting Uvicorn API server on http://localhost:8000..."
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
