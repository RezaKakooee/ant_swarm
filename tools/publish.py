"""Export the public part of this project into a clean staging folder.

    python tools/publish.py            # build staging + report (never pushes)

Why an export and not a public mirror: this repo's git history contains
cluster scripts and personal notes. Publishing the history would expose them
even if the files were deleted first. So the public repo gets a fresh, clean
copy instead.

Rules:
  * ALLOW is an allowlist. Anything not listed is NOT published.
  * After copying, every text file is scanned for FORBIDDEN patterns. If one
    is found, the export FAILS and prints file:line — nothing is published.

Then, to publish:
    cd storage_local/public_export
    git init && git add -A && git commit -m "..."
    git remote add origin git@github.com:RezaKakooee/ant-piano-movers-rl.git
    git push -u origin main            # --force on later publishes
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "storage_local" / "public_export"

# --------------------------------------------------------------- allowlist
ALLOW = [
    "ant_swarm",                       # the environment package
    "scripts/rl", "scripts/il", "scripts/heuristic", "scripts/README.md",
    "scripts/train.sh",                # cluster-free launcher for everyone else
    "configs",                         # rl / il / heuristic configs
    "interactive",                     # browser sandbox
    "requirements.txt",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
]

# never copied, even inside an allowed folder
SKIP_NAMES = {"__pycache__", ".ipynb_checkpoints", ".DS_Store"}
SKIP_SUFFIX = {".pyc", ".pyo"}

# ------------------------------------------------------------- the scrubber
FORBIDDEN = [
    r"scicore",
    r"kakooe0000",
    r"graber0001",
    r"azureuser",
    r"\bsbatch\b",
    r"\bsqueue\b",
    r"\bscancel\b",
    r"/home/[a-z]",
    r"#SBATCH",
]
# harmless matches (generic env vars any SLURM user would set)
ALLOWED_EXCEPTIONS = [r"SLURM_JOB_ID", r"slurm/cluster-agnostic"]
TEXT_SUFFIX = {".py", ".md", ".yaml", ".yml", ".sh", ".html", ".css", ".js", ".txt", ".cff"}

# small edits applied to the COPY only, so the working repo keeps its cluster docs
REWRITES = {
    "scripts/README.md": [
        ("or via the env var / SLURM wrapper:", "or via the env var:"),
        ("    sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml [key=value ...]\n", ""),
    ],
    "configs/README.md": [
        ("    sbatch ops/sb_train.sh train_sac configs/rl/pnas_kin_geo.yaml sac.timesteps=5e6\n", ""),
    ],
}

# example run directories -> a generic placeholder
RUN_DIR_RE = re.compile(r"storage_local/ant__[0-9A-Za-z_\-]+")


def copy_allowed() -> list[Path]:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    copied = []
    for rel in ALLOW:
        src = ROOT / rel
        if not src.exists():
            print(f"  !! listed but missing: {rel}")
            continue
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(
                src, dst,
                ignore=shutil.ignore_patterns(*SKIP_NAMES, "*.pyc", "*.pyo"))
            copied += [p for p in dst.rglob("*") if p.is_file()]
        else:
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def scan(files: list[Path]) -> list[str]:
    pat = re.compile("|".join(FORBIDDEN), re.I)
    ok = re.compile("|".join(ALLOWED_EXCEPTIONS))
    hits = []
    for f in files:
        if f.suffix.lower() not in TEXT_SUFFIX:
            continue
        try:
            lines = f.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if pat.search(line) and not ok.search(line):
                hits.append(f"{f.relative_to(OUT)}:{i}: {line.strip()[:110]}")
    return hits


def main() -> int:
    print(f"exporting to {OUT}\n")
    files = copy_allowed()

    for rel, subs in REWRITES.items():
        f = OUT / rel
        if not f.exists():
            continue
        s = f.read_text()
        for a, b in subs:
            s = s.replace(a, b)
        f.write_text(s)
    # genericise example run-dir names in every published text file
    for f in OUT.rglob("*"):
        if f.is_file() and f.suffix.lower() in TEXT_SUFFIX:
            s = f.read_text(errors="ignore")
            if "storage_local/ant__" in s:
                f.write_text(RUN_DIR_RE.sub("storage_local/<run>", s))
    print(f"  applied export rewrites to {len(REWRITES)} file(s)\n")

    # public README replaces the working one
    readme = ROOT / "tools" / "README_public.md"
    if readme.exists():
        shutil.copy2(readme, OUT / "README.md")
        files.append(OUT / "README.md")
        print("  public README.md installed\n")

    tree = sorted({str(p.relative_to(OUT)).split("/")[0] for p in files})
    print("published top level:", ", ".join(tree))
    print(f"files: {len(files)}   size: {sum(f.stat().st_size for f in files)/1e6:.1f} MB\n")

    hits = scan(files)
    if hits:
        print(f"BLOCKED — {len(hits)} forbidden reference(s) found:\n")
        for h in hits[:40]:
            print("  ", h)
        print("\nNothing is published. Fix these (or add to ALLOWED_EXCEPTIONS) and re-run.")
        return 1
    print("scrubber: clean — no cluster paths, usernames or SLURM commands found.")
    print(f"\nReview {OUT}, then commit and push it to the public repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
