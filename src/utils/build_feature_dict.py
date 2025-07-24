# scripts/build_feature_dict.py
"""
Generate docs/feature_dictionary.md
–––––––––––––––––––––––––––––––––––
• Merges the machine audit JSON + optional YAML notes
• Fails fast with human-readable errors if either file is missing
  or malformed.
• Can be called stand-alone *or* is auto-invoked by the Phase-5 CLI
  at the very end of feature engineering.

CLI:
    python scripts/build_feature_dict.py        # default locations
    python scripts/build_feature_dict.py --audit path/to/audit.json --notes notes.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# ────────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"❌  Required audit file not found: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        sys.exit(f"❌  Audit JSON malformed ({path}):\n{e}")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as e:
        sys.exit(f"❌  YAML notes malformed ({path}):\n{e}")


# ────────────────────────────────────────────────────────────────


def build(audit_path: Path, notes_path: Path, out_path: Path) -> None:
    audit = load_json(audit_path)
    notes = load_yaml(notes_path)

    required_keys = {"columns", "n_features_after_clean"}
    if not required_keys.issubset(audit):
        sys.exit(f"❌  Audit JSON missing keys: {required_keys-audit.keys()}")

    rows = []
    for col in audit["columns"]:
        rows.append(
            {
                "Feature": f'`{col["name"]}`',
                "Origin": col.get("origin", "—"),
                "Transform": col.get("transform", "—"),
                "Kept": "✅" if col.get("kept", True) else "❌",
                "Notes": notes.get(col["name"], ""),
            }
        )

    df = pd.DataFrame(rows)

    md = "# 📖 Feature Dictionary\n\n"
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    md += f"*Auto-generated {ts} – {audit['n_features_after_clean']} columns.*\n\n"
    md += df.to_markdown(index=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"✅  Feature dictionary written → {out_path.relative_to(Path.cwd())}")


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--audit", default="reports/feature/feature_audit.json", type=Path)
    p.add_argument("--notes", default="docs/feature_notes.yaml", type=Path)
    p.add_argument("--out", default="docs/feature_dictionary.md", type=Path)
    args = p.parse_args()

    build(args.audit, args.notes, args.out)
