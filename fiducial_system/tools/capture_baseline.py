"""Establish the baseline (reference) poses for each object group.

This script averages multiple frames so the stored baseline is stable.
Baseline entries are 4x4 transforms in the world frame for each group name.
"""

import argparse
import os
import yaml
import numpy as np
import cv2

from fiducial_system.config import AppConfig
from fiducial_system.camera import grab_frame
from fiducial_system.detector import AprilTagDetector, estimate_tag_poses, rvec_tvec_to_matrix, average_transforms


def load_intrinsics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    out_path = args.output or cfg.output.baseline_file

    K, dist = load_intrinsics(cfg.camera_intrinsics)
    det = AprilTagDetector(cfg.apriltag.family, cfg.apriltag.nthreads, cfg.apriltag.refine_edges, cfg.apriltag.decode_sharpening)

    groups = cfg.objects
    accum = {k: [] for k in groups.keys()}

    for _ in range(args.frames):
        got = grab_frame(cfg.rtsp_url, cfg.frame_grab.attempts, cfg.frame_grab.delay_ms)
        if not got:
            continue
        _, frame = got
        dets = det.detect(frame)
        id_to_pose = estimate_tag_poses(dets, cfg.apriltag.tag_size_m, K, dist)

        for name, info in groups.items():
            transforms = []
            for tag_id in info.get("tag_ids", []):
                if tag_id in id_to_pose:
                    rvec, tvec = id_to_pose[tag_id]
                    T = rvec_tvec_to_matrix(rvec, tvec)
                    transforms.append(T)
            if transforms:
                accum[name].append(average_transforms(transforms))

    baseline = {}
    for name, Ts in accum.items():
        if not Ts:
            continue
        baseline[name] = average_transforms(Ts).tolist()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(baseline, f)
    print(f"Saved baseline to {out_path}")


if __name__ == "__main__":
    main()

