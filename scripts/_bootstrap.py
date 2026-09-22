"""Put `src/` on sys.path so scripts can be run directly from anywhere.

Importing this module has no side effect other than the path change.
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
