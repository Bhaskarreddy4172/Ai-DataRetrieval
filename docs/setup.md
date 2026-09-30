# Setup and Installation Guide

## Requirements
- Python 3.10+ (Recommended Python 3.11 or 3.12)
- Local Ollama installation (optional but recommended for natural language conversational explanations)

## Quickstart Installation

### 1. Clone & Setup Virtual Environment
```bash
git clone https://github.com/example/AI-Dataset-Retrieval.git
cd AI-Dataset-Retrieval

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 3. Run Application
```bash
# Start backend on http://localhost:8000
python run.py
```
Visit `http://localhost:8000` to interact with the ChatGPT-like conversational chat interface.

### 4. Running Benchmark Suite
```bash
# Run 10,000+ benchmark tests
python benchmark/run_10k_benchmark.py
```

