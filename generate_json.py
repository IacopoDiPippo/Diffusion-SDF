#!/usr/bin/env python3
import sys
import json
from pathlib import Path

def main():
    # Uso: python3 make_mug_json.py [cartella_mug] [output_json]
    mug_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/acronym/mug")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("mug_augmented.json")

    if not mug_dir.is_dir():
        sys.exit(f"Directory non trovata: {mug_dir}")

    # Prende TUTTI i nomi delle sottodirectory immediate (no file)
    names = [p.name for p in sorted(mug_dir.iterdir()) if p.is_dir()]

    data = {"acronym": {"mug": names}}
    out_path.write_text(json.dumps(data, indent=4), encoding="utf-8")
    print(f"Scritte {len(names)} voci in: {out_path}")

if __name__ == "__main__":
    main()
