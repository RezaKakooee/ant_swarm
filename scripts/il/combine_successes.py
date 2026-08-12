"""Combine many per-episode success_*.json files into a few pretty-printed files.

Each successful episode is saved as its own small JSON during training; that
makes huge numbers of tiny files. This consolidates them — per ``successes/``
directory — into chunked, indented ``combined_*.json`` files and removes the
originals.

Usage:
    python combine_successes.py <path> [--chunk N] [--indent N] [--keep]

  <path>     a run dir, a successes/ dir, or any parent (searched recursively
             for `successes/` dirs). Default: storage_local
  --chunk N  trajectories per combined file (default 1000; bounds memory)
  --indent N JSON indent for readability (default 2)
  --keep     do NOT delete the original per-episode files (default: delete)

Each combined file is a JSON list of trajectory objects (the same dicts the
training callback wrote). Streaming: only `--chunk` trajectories are held in
memory at a time, and originals are deleted only after their chunk is written.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _flush(out_dir: Path, idx: int, buf: list, indent: int) -> Path:
    out = out_dir / f"combined_{idx:04d}.json"
    with open(out, "w") as f:
        json.dump(buf, f, indent=indent)
    return out


def combine_dir(succ_dir: Path, chunk: int, indent: int, delete: bool) -> tuple[int, int]:
    files = sorted(succ_dir.glob("success_*.json"))
    if not files:
        return 0, 0

    # continue numbering after any pre-existing combined files
    existing = sorted(succ_dir.glob("combined_*.json"))
    idx = (int(existing[-1].stem.split("_")[-1]) + 1) if existing else 0

    buf, buf_files, n_in, n_out = [], [], 0, 0
    for f in files:
        try:
            buf.append(json.load(open(f)))
            buf_files.append(f)
        except Exception as e:
            print(f"  SKIP (unreadable): {f.name}  ({e})")
            continue
        if len(buf) >= chunk:
            out = _flush(succ_dir, idx, buf, indent)
            n_in += len(buf); n_out += 1
            if delete:
                for bf in buf_files:
                    bf.unlink()
            print(f"  wrote {out.name}  ({len(buf)} trajectories)", flush=True)
            buf, buf_files = [], []
            idx += 1
    if buf:
        out = _flush(succ_dir, idx, buf, indent)
        n_in += len(buf); n_out += 1
        if delete:
            for bf in buf_files:
                bf.unlink()
        print(f"  wrote {out.name}  ({len(buf)} trajectories)", flush=True)
    return n_in, n_out


def main():
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    root = Path(pos[0]) if pos else Path("storage_local")
    chunk = int(_opt(argv, "--chunk", 1000))
    indent = int(_opt(argv, "--indent", 2))
    delete = "--keep" not in argv

    if root.name == "successes":
        succ_dirs = [root]
    else:
        succ_dirs = sorted(p for p in root.rglob("successes") if p.is_dir())
    if not succ_dirs:
        print(f"No successes/ directories found under {root}")
        return

    total_in = total_out = 0
    for d in succ_dirs:
        print(f"[{d}]")
        n_in, n_out = combine_dir(d, chunk, indent, delete)
        total_in += n_in; total_out += n_out
    print(f"\nCombined {total_in} trajectories → {total_out} file(s)"
          f"{' (originals deleted)' if delete else ' (originals kept)'}")


def _opt(argv, name, default):
    if name in argv:
        return argv[argv.index(name) + 1]
    return default


if __name__ == "__main__":
    main()
