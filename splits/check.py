#!/usr/bin/env python3
"""Check the prepared splits under dataset/ against splits/manifest.json (SHA-256 per file)."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
manifest = json.loads((REPO / "splits/manifest.json").read_text())
bad = 0
for name, info in manifest["sets"].items():
    for rel, expected in info["sha256"].items():
        path = REPO / info["path"] / rel
        if not path.is_file():
            print(f"MISSING   {name}/{rel}")
            bad += 1
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            print(f"DIFFERENT {name}/{rel}")
            bad += 1
        else:
            print(f"ok        {name}/{rel}")
print("all splits match" if not bad else f"{bad} file(s) do not match")
sys.exit(1 if bad else 0)
