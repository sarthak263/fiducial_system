"""Camera calibration tool using a chessboard pattern.

Collect several images of a printed chessboard at different positions/angles,
then run this script to compute and save intrinsics (K) and distortion.
"""

import argparse
import glob
import os
import yaml
import cv2
import numpy as np


def save_intrinsics(path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, image_size):
    data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=str, default="calib/*.jpg", help="glob for chessboard images")
    ap.add_argument("--board_cols", type=int, default=9)
    ap.add_argument("--board_rows", type=int, default=6)
    ap.add_argument("--square_mm", type=float, default=25.0)
    ap.add_argument("--output", type=str, default="config/camera_intrinsics.yaml")
    args = ap.parse_args()

    images = sorted(glob.glob(args.images))
    if not images:
        print("No images found for calibration.")
        return 1

    objp = np.zeros((args.board_rows * args.board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.board_cols, 0:args.board_rows].T.reshape(-1, 2)
    objp *= args.square_mm / 1000.0

    objpoints = []
    imgpoints = []
    img_size = None

    for path in images:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (args.board_cols, args.board_rows), None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_size = (gray.shape[1], gray.shape[0])

    if not objpoints:
        print("No valid chessboard detections.")
        return 1

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    print(f"RMS reprojection error: {ret:.4f}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_intrinsics(args.output, K, dist, img_size)
    print(f"Saved intrinsics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

