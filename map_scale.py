"""
map_scale.py - map scale (meters per pixel) and pose <-> pixel conversion in one place.

Scale belongs to the DATASET, not to the code. Two datasets are in use and they differ:

  ucf_test   3.52 m/px  the 900 m UCF scene in 256 px (data/gnn_inference_test, UCF-50K).
                        Checked against the scene meshes; 3.52 is the team's rounded value
                        (900 / 256 = 3.5156) and matches benchmark_reconstruction.py and the ROS node.
  diverse    1.0 m/px   dataset_256_256_diverse, the set the first decoder was trained on.
                        From a free-space fit (0.86-1.23 m/px, 2.4 GHz assumed), not from geometry.

An unknown dataset raises instead of falling back to a default, so meter-based metrics can
never be silently computed at the wrong scale.

Grid convention (same as ros2/gnn_reconstruction INTERFACE.md and nav_msgs/OccupancyGrid):
  pixel (row, col) covers x in [origin_x + col*res, +res) and y in [origin_y + row*res, +res),
  so row 0 is the LOWEST y.
The dataset parquet files are stored the other way up (row 0 = HIGHEST y, the generator applies
np.flipud). Use dataset_to_grid() on a dataset array before indexing it with grid rows.
"""
import math
import os

import numpy as np

H = W = 256
HIT_PX = 3.0  # team definition of an emitter hit: within 3 px (10.56 m at 3.52 m/px)

DATASETS = {
    "ucf_test": {"m_per_px": 3.52, "path_hints": ("gnn_inference_test", "Spectrum-Cartography-256x256-UCF-50K")},
    "diverse": {"m_per_px": 1.0, "path_hints": ("dataset_256_256_diverse",)},
}


def _check(m_per_px):
    m = float(m_per_px)
    if not math.isfinite(m) or m <= 0:
        raise ValueError(f"m_per_px must be a finite number > 0, got {m_per_px!r}")
    return m


def dataset_from_path(path):
    """Dataset name from a folder name in the path. Raises if no folder, or more than one, matches."""
    parts = set(os.path.normpath(str(path)).split(os.sep))
    found = [name for name, d in DATASETS.items() if parts & set(d["path_hints"])]
    if len(found) != 1:
        why = "matches no known dataset" if not found else f"matches several datasets {found}"
        raise ValueError(f"path {path!r} {why}; pass the dataset name ({', '.join(DATASETS)}) or m_per_px")
    return found[0]


def resolve_scale(dataset=None, path=None, m_per_px=None):
    """Meters per pixel. Priority: explicit m_per_px, then dataset name, then a folder name in path."""
    if m_per_px is not None:
        return _check(m_per_px)
    if dataset is None:
        if path is None:
            raise ValueError("no scale given: pass dataset, path or m_per_px")
        dataset = dataset_from_path(path)
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
    return _check(DATASETS[dataset]["m_per_px"])


def hit_radius_m(m_per_px, hit_px=HIT_PX):
    """Emitter hit radius in meters for a radius given in pixels."""
    return float(hit_px) * _check(m_per_px)


def pose_to_pixel(x, y, m_per_px, origin_x=0.0, origin_y=0.0, shape=(H, W)):
    """Map-frame position (m) -> (row, col) in the grid convention, or None outside the grid."""
    res = _check(m_per_px)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f"pose must be finite, got ({x!r}, {y!r})")
    col = int(math.floor((x - origin_x) / res))
    row = int(math.floor((y - origin_y) / res))
    if 0 <= row < shape[0] and 0 <= col < shape[1]:
        return row, col
    return None


def pixel_to_pose(row, col, m_per_px, origin_x=0.0, origin_y=0.0):
    """Center of pixel (row, col) in the map frame (m)."""
    res = _check(m_per_px)
    return origin_x + (col + 0.5) * res, origin_y + (row + 0.5) * res


def dataset_to_grid(arr):
    """Flip a dataset-format array (row 0 = highest y) into the grid convention (row 0 = lowest y).
    The same call converts back. Works on (..., H, W) arrays."""
    return np.flip(np.asarray(arr), axis=-2).copy()
