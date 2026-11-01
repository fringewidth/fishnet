"""Puts circuit/ and machine/ on sys.path, so the tests run as plain scripts from anywhere."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("machine", "circuit"):
    p = os.path.join(ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)
