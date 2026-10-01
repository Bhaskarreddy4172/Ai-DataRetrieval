"""Alias script for validate_all.py."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.validate_all import validate_all

if __name__ == "__main__":
    sys.exit(validate_all())
