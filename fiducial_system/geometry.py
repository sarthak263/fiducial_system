"""Small geometry helpers for 4x4 transforms.

All transforms are homogeneous matrices in the same coordinate system unless
otherwise noted. Distances are in meters.
"""

import numpy as np
import math


def transform_between(A_in_world: np.ndarray, B_in_world: np.ndarray) -> np.ndarray:
    """Transform from frame A to frame B, given both in world coordinates.

    Returns T_A_B such that p_B = T_A_B * p_A.
    """
    A_w = A_in_world
    B_w = B_in_world
    A_w_inv = np.eye(4)
    R = A_w[:3, :3]
    t = A_w[:3, 3]
    A_w_inv[:3, :3] = R.T
    A_w_inv[:3, 3] = -R.T @ t
    return A_w_inv @ B_w


def pose_delta(T_ref: np.ndarray, T_cur: np.ndarray) -> tuple:
    """Compute translation (mm) and rotation (deg) difference between poses."""
    d = np.linalg.inv(T_ref) @ T_cur
    t_mm = np.linalg.norm(d[:3, 3]) * 1000.0
    R = d[:3, :3]
    angle_rad = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    deg = math.degrees(angle_rad)
    return t_mm, deg

