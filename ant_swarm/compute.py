"""Device selection with a loud failure instead of a silent CPU fallback.

The cluster's GPU nodes do not all have the same driver. This env's torch is
built for CUDA 13.0, which needs driver >= 610. Nodes on 575 report
``torch.cuda.is_available() == False``, and code that asks for ``"auto"`` then
runs on CPU without saying so -- two fine-tuning runs were lost that way before
anyone noticed.

Working nodes (driver 610.57.04):  calc-g-002 (RTX 2080 Ti), calc-g-003 (TITAN RTX)
Broken nodes  (driver 575.57.08):  calc-g-004, calc-g-008, calc-g-010

    sbatch -M cluster -p performance -w calc-g-002 --gres=gpu:1 ...

Default is ``cpu``: the environment is single-threaded NumPy, so for RL the
env stepping dominates and a GPU buys little. Use ``cuda`` for supervised
training on the cached datasets, where it is worth roughly 10x.
"""
from __future__ import annotations

GPU_NODES = ("calc-g-002", "calc-g-003")


def resolve_device(name: str = "cpu", *, allow_fallback: bool = False) -> str:
    """Return 'cpu' or 'cuda'.

    ``cpu``    always CPU (the default).
    ``cuda``   CUDA, or raise if it is unavailable -- never a silent fallback,
               unless ``allow_fallback=True``.
    ``auto``   CUDA when available, else CPU.
    """
    import torch

    want = str(name).strip().lower()
    if want in ("gpu",):
        want = "cuda"
    if want == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if want == "auto":
        return "cuda" if available else "cpu"
    if want != "cuda":
        raise ValueError(f"device must be cpu, cuda or auto, got {name!r}")
    if available:
        return "cuda"
    if allow_fallback:
        return "cpu"
    raise RuntimeError(
        "device: cuda was requested but torch.cuda.is_available() is False. "
        f"This node's driver is probably too old for torch {torch.__version__}. "
        f"Submit to a working GPU node, e.g. -w {GPU_NODES[0]}, "
        "or set the device to cpu.")
