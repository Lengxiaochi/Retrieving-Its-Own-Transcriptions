#!/usr/bin/env python3
"""Check syntax and list the public scripts; this does not run GPU experiments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted((ROOT / "scripts").rglob("*.py"))
for path in files:
    compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")
print(f"PASS: Python syntax for {len(files)} scripts.")
print("This check does not reproduce the GPU experiments.")
