#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR/frontend"

echo "==================================================="
echo "Starting Universal Dataset AI - Frontend (React/Vite)"
echo "==================================================="

if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

echo "Starting Vite development server on http://localhost:3000..."
exec npm run dev
