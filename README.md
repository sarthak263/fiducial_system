# Fiducial System (AprilTag-based monitoring)

This project monitors whether lab components (camera, table, hotel/base) have moved using AprilTags detected from an RTSP camera.

## Components
- Calibration script to save camera intrinsics
- Tag generator to print tag36h11 markers
- Baseline capture to establish reference poses
- Hourly checker to compare current poses vs baseline and alert

## Repository layout
- `requirements.txt`: Python dependencies.
- `config/config.example.yaml`: Sample config with RTSP URL, tag groups, thresholds, and outputs. Copy to `config/config.yaml` and edit.
- `fiducial_system/config.py`: Loads YAML into typed dataclasses. Change here to add new settings.
- `fiducial_system/camera.py`: Minimal RTSP frame grabber with retries.
- `fiducial_system/detector.py`: AprilTag detection and pose estimation utilities, transform math and averaging.
- `fiducial_system/geometry.py`: Small helpers for composing/inverting transforms and computing deltas.
- `fiducial_system/tools/calibrate.py`: Chessboard-based camera calibration (produces `camera_intrinsics.yaml`).
- `fiducial_system/tools/generate_tags.py`: Produces a PDF of printable tags.
- `fiducial_system/tools/capture_baseline.py`: Captures averaged baseline poses for each group.
- `fiducial_system/check.py`: Single-run checker used by the hourly job.

## Quick start
1. Create and edit `config/config.yaml` (a sample is provided in `config/config.example.yaml`).
2. Install dependencies:
```bash
python -m venv .venv
.venv\\Scripts\\pip install -r requirements.txt
```
3. Calibrate camera (one-time):
```bash
python -m fiducial_system.tools.calibrate --output config/camera_intrinsics.yaml
```
4. Print tags for testing:
```bash
 python -m fiducial_system.tools.generate_tags --family 36h11 --ids 0-5 --size 1772 --label --outdir .\apriltags_out --sheet --cols 3 --rows 2 --dpi 300 --sheet_width_mm 210 --sheet_height_mm 297
```
5. Capture baseline poses:
```bash
python -m fiducial_system.tools.capture_baseline --config config/config.yaml --output config/baseline.yaml
```
6. Run a check (or schedule hourly via Task Scheduler):
```bash
python -m fiducial_system.check --config config/config.yaml
```

See comments in the config for how to map tag IDs to `world`, `table`, `hotel`, and `camera` groups and set thresholds.

## Typical workflow
1. Print 60–100 mm `tag36h11` markers. Affix to world board/rail, table edges, and black base (not the green body). Note IDs in the config.
2. Calibrate the camera once using chessboard images. Save to `config/camera_intrinsics.yaml`.
3. With everything correctly positioned, run baseline capture to save stable poses.
4. Test: run the checker while moving only the tags to verify thresholds and logic. Then secure the markers permanently.
5. Schedule hourly on Windows Task Scheduler to execute `python -m fiducial_system.check --config C:\path\config.yaml`.

## Tuning thresholds
- Hotel vs table: start at 3 mm / 0.5°
- Table vs world: start at 8 mm / 1.5°
- Camera vs world: start at 2 mm / 0.3°
Adjust based on your mechanical rigidity and noise; increase `--frames` during baseline for higher stability.

