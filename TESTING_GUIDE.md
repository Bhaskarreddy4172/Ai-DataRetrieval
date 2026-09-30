# Quality Assurance & Testing Guide

## Overview

The Universal AI Dataset System includes an extensive suite of unit, integration, regression, and 10,000+ benchmark automated tests.

---

## 1. Running Backend Unit & Regression Tests

Execute all 270+ backend test cases using `pytest`:

```bash
# Quiet summary execution
python -m pytest tests/ -q

# Verbose output with full test details
python -m pytest tests/ -v
```

### Specific Test Modules
- `tests/test_agent_orchestrator.py`: Master state machine tests.
- `tests/test_agent_tools.py`: Tool registry and execution tests.
- `tests/test_ollama_fallback.py`: Health enums, retries, and fallback tests.
- `tests/test_boolean_questions.py`: Fact-check assertion tests.
- `tests/test_hallucination.py`: Zero hallucination firewall tests.

---

## 2. Running 10,000+ Benchmark Evaluation

To execute the automated synthetic 10,000-question benchmark suite:

```bash
# Generate 10k synthetic test benchmark suite
python benchmark/generate_10k_benchmark.py

# Execute full 10k benchmark suite and record accuracy metrics
python benchmark/run_10k_benchmark.py
```

### Output Reports
- `benchmark/benchmark_10k_report.json`: Summary stats (accuracy %, latency, recall).
- `benchmark/benchmark_10k_report.csv`: Detailed line-by-line test execution logs.
- `benchmark/failure_analysis_10k.json`: Categorized root cause failure classification.

---

## 3. Frontend Production Build Verification

```bash
cd frontend
npm run build
```
Ensures zero TypeScript, JSX, or bundling errors exist.

