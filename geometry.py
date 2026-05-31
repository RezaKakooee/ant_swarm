"""Geometry utilities shared by the T-shape, layout, and env.

Pure functions + the ``LocalRect`` primitive.  No simulator/config knowledge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class LocalRect:
    """An axis-aligned rectangle in some local frame: centre + half-extents."""
    center: np.ndarray
    half_size: np.ndarray


def rotation_matrix(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def obb_aabb_overlap(obb_corners: np.ndarray, aabb: tuple, angle: float) -> bool:
    """True if an oriented rectangle overlaps an axis-aligned box (SAT).

    Args:
        obb_corners: (4, 2) world corners of the oriented rectangle.
        aabb: ``(xmin, xmax, ymin, ymax)`` of the axis-aligned box.
        angle: rotation of the oriented rectangle (its two edge axes).

    Uses the 2 world axes + the OBB's 2 oriented axes as separating-axis
    candidates — exact for rectangle-vs-rectangle.
    """
    xmin, xmax, ymin, ymax = aabb
    wc = np.array([[xmin, ymin], [xmax, ymin],
                   [xmax, ymax], [xmin, ymax]], dtype=np.float32)
    ca, sa = math.cos(angle), math.sin(angle)
    axes = ((1.0, 0.0), (0.0, 1.0), (ca, sa), (-sa, ca))
    for ax, ay in axes:
        rp = obb_corners[:, 0] * ax + obb_corners[:, 1] * ay
        wp = wc[:, 0] * ax + wc[:, 1] * ay
        if rp.max() < wp.min() or wp.max() < rp.min():
            return False  # found a separating axis
    return True


def aabb_of(corners: np.ndarray) -> tuple:
    """Axis-aligned bounding box ``(xmin, xmax, ymin, ymax)`` of a corner set."""
    return (float(corners[:, 0].min()), float(corners[:, 0].max()),
            float(corners[:, 1].min()), float(corners[:, 1].max()))
