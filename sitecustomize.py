"""Ensure project root is on sys.path early (imported by site at startup).

This makes `import src` work during test collection and other runs
without requiring installing the package.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
