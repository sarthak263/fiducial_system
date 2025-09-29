"""AprilTag detection and pose estimation utilities.

Responsibilities:
- Wrap the underlying AprilTag detector (pupil_apriltags)
- Estimate each tag's 6-DoF pose using OpenCV's PnP with a square model
- Provide helpers for 4x4 transformation math and averaging

All math uses meters for distances, 4x4 homogeneous transforms, and OpenCV's
Rodrigues vectors for intermediate conversions.
"""

from typing import Dict, List, Tuple
import numpy as np
import cv2
from pupil_apriltags import Detector


class AprilTagDetector:
    def __init__(self, family: str, nthreads: int, refine_edges: bool, decode_sharpening: float):
        self.detector = Detector(
            families=family,
            nthreads=nthreads,
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )

    def detect(self, image_bgr) -> List[dict]:
        """Run AprilTag detection on a BGR image and return detections.

        Each detection contains ``tag_id`` and ``corners`` among other fields.
        Pose is not estimated here to keep CPU use minimal when not needed.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return self.detector.detect(gray, estimate_tag_pose=False)


def estimate_tag_poses(detections: List[dict], tag_size_m: float, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Estimate pose for each detected tag via IPPE_SQUARE.

    Returns a mapping ``tag_id -> (rvec, tvec)`` in the camera frame.
    ``rvec`` is a 3x1 Rodrigues vector; ``tvec`` is a 3x1 translation (meters).
    """
    id_to_pose: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    objp = np.array([
        [-tag_size_m / 2, -tag_size_m / 2, 0],
        [ tag_size_m / 2, -tag_size_m / 2, 0],
        [ tag_size_m / 2,  tag_size_m / 2, 0],
        [-tag_size_m / 2,  tag_size_m / 2, 0],
    ], dtype=np.float32)

    for det in detections:
        # det.corners is 4x2 in order [A, B, C, D]
        imgp = np.array(det.corners, dtype=np.float32)
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if ok:
            id_to_pose[det.tag_id] = (rvec.reshape(3, 1), tvec.reshape(3, 1))
    return id_to_pose


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = tvec
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def average_transforms(transforms: List[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("No transforms to average")
    translations = np.array([T[:3, 3] for T in transforms])
    mean_t = translations.mean(axis=0)
    # Average rotation via quaternion
    quats = []
    for T in transforms:
        R = T[:3, :3]
        q = rotation_matrix_to_quaternion(R)
        quats.append(q)
    mean_q = average_quaternions(np.array(quats))
    R_mean = quaternion_to_rotation_matrix(mean_q)
    T_mean = np.eye(4)
    T_mean[:3, :3] = R_mean
    T_mean[:3, 3] = mean_t
    return T_mean


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = np.argmax([R[0, 0], R[1, 1], R[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ])


def average_quaternions(Q: np.ndarray) -> np.ndarray:
    # Markley method
    A = np.zeros((4, 4))
    for q in Q:
        q = q.reshape(4, 1)
        A += q @ q.T
    eigvals, eigvecs = np.linalg.eigh(A)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)

