"""Generate ``web_config.js`` from ``config.yaml`` so the HTML stays in sync.

Run after editing config.yaml:
    python gen_web_config.py
"""
from __future__ import annotations

import json
from pathlib import Path

from swarm_config import load_config_dict

OUT = Path(__file__).parent / "web_config.js"


def main():
    cfg = load_config_dict()
    js = (
        "// AUTO-GENERATED from config.yaml by gen_web_config.py — DO NOT EDIT.\n"
        "// Re-run `python gen_web_config.py` after changing config.yaml.\n"
        "const CONFIG = " + json.dumps(cfg, indent=2) + ";\n"
    )
    OUT.write_text(js)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
