# Production Deployment Guide

## Overview

This guide details running the Universal AI Dataset Chatbot in local production, Docker containerized environments, and cloud deployments.

---

## 1. Local Environment Prerequisites

- **Python**: `3.10` to `3.14`
- **Node.js**: `v18` or later (`npm v9+`)
- **Ollama**: Local instance running at `http://127.0.0.1:11434`

---

## 2. Step-by-Step Local Setup

### Step 1: Install Python Dependencies
```bash
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r backend/requirements.txt
```

### Step 2: Install & Pull Ollama Model
```bash
ollama serve
ollama pull llama3.1:8b
```

### Step 3: Launch FastAPI Backend
```bash
python run.py
```
Backend runs on `http://127.0.0.1:8000`.

### Step 4: Launch / Build React Frontend
```bash
cd frontend
npm install
npm run build
npm run preview
```
Frontend runs on `http://localhost:4173` (or `http://localhost:5173` in dev mode).

---

## 3. Docker Deployment

### Multi-Container Deployment (`docker-compose.yml`)
```bash
docker-compose up -d --build
```

### Access Points
- **Web App**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

