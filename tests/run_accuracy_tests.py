"""Accuracy benchmark test runner located inside tests/ directory."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_accuracy_tests import run_benchmark

if __name__ == "__main__":
    run_benchmark()
