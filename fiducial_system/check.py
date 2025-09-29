"""Single-run checker used for the hourly monitoring job.

Workflow per run:
1) Grab one frame from RTSP
2) Detect AprilTags and estimate group transforms in the camera frame
3) Convert to world frame via the world group's pose
4) Compare current relative poses to baseline thresholds and log/notify
"""

import argparse
import os
import yaml
import time
import cv2
import numpy as np

from fiducial_system.config import AppConfig
from fiducial_system.camera import grab_frame
from fiducial_system.detector import AprilTagDetector, estimate_tag_poses, rvec_tvec_to_matrix, average_transforms
from fiducial_system.geometry import transform_between, pose_delta


def load_intrinsics(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    K = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return K, dist


def load_baseline(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return {k: np.array(v, dtype=np.float64) for k, v in raw.items()}


def log_event(path: str, message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def notify(cfg: AppConfig, message: str):
    if cfg.notify.console:
        print(message)
    # Placeholders for email/slack; can be implemented later


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    K, dist = load_intrinsics(cfg.camera_intrinsics)
    baseline = load_baseline(cfg.output.baseline_file)

    det = AprilTagDetector(cfg.apriltag.family, cfg.apriltag.nthreads, cfg.apriltag.refine_edges, cfg.apriltag.decode_sharpening)

    got = grab_frame(cfg.rtsp_url, cfg.frame_grab.attempts, cfg.frame_grab.delay_ms)
    if not got:
        msg = "Failed to read frame from RTSP"
        log_event(cfg.output.events_log, msg)
        notify(cfg, msg)
        return 1
    _, frame = got

    dets = det.detect(frame)
    id_to_pose = estimate_tag_poses(dets, cfg.apriltag.tag_size_m, K, dist)

    # Estimate group transforms in camera frame
    groups = cfg.objects
    group_T_cam = {}
    for name, info in groups.items():
        transforms = []
        for tag_id in info.get("tag_ids", []):
            if tag_id in id_to_pose:
                rvec, tvec = id_to_pose[tag_id]
                transforms.append(rvec_tvec_to_matrix(rvec, tvec))
        if transforms:
            group_T_cam[name] = average_transforms(transforms)

    # Require minimum world tags
    if "world" not in group_T_cam or len(group_T_cam) == 0:
        msg = "Insufficient world tags visible"
        log_event(cfg.output.events_log, msg)
        notify(cfg, msg)
        return 2

    world_in_cam = group_T_cam["world"]
    cam_in_world = np.linalg.inv(world_in_cam)

    # Compute current poses in world frame
    current_world_poses = {}
    for name, T_cam in group_T_cam.items():
        current_world_poses[name] = cam_in_world @ T_cam

    # Compare against baseline
    alerts = []
    def compare(a_name: str, b_name: str, thr_key: str):
        if a_name not in current_world_poses or b_name not in current_world_poses:
            return
        if a_name not in baseline or b_name not in baseline:
            return
        T_ref = transform_between(baseline[a_name], baseline[b_name])
        T_cur = transform_between(current_world_poses[a_name], current_world_poses[b_name])
        t_mm, r_deg = pose_delta(T_ref, T_cur)
        thr = cfg.thresholds[thr_key]
        if t_mm > thr.translation_mm or r_deg > thr.rotation_deg:
            alerts.append((thr_key, t_mm, r_deg))

    compare("hotel", "table", "hotel_vs_table")
    compare("table", "world", "table_vs_world")
    compare("camera", "world", "camera_vs_world")

    if alerts:
        for name, t_mm, r_deg in alerts:
            msg = f"ALERT {name}: translation {t_mm:.2f} mm, rotation {r_deg:.3f} deg"
            log_event(cfg.output.events_log, msg)
            notify(cfg, msg)
        return 3
    else:
        msg = "OK: no movement beyond thresholds"
        log_event(cfg.output.events_log, msg)
        print(msg)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

