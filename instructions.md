# Lab Movement Monitoring - Testing Instructions

## 1) Setup & Calibration
```powershell
# Install deps (if not done)
pip install -r requirements.txt

# Calibrate camera (one-time, with chessboard images)
python -m fiducial_system.tools.calibrate --images calib\*.jpg --output config\camera_intrinsics.yaml
```

## 2) Capture Baseline (with everything in correct position)
```powershell
python -m fiducial_system.tools.capture_baseline --config config\config.yaml --frames 15
```
**What this does:** Takes 15 frames, detects all tags, averages poses, saves to `config/baseline.yaml`

## 3) Run Checks
```powershell
python -m fiducial_system.check --config config\config.yaml
```

## 4) What Results You'll Get

**Console output:**
```
OK: no movement beyond thresholds
```
or
```
ALERT hotel_vs_table: translation 5.2 mm, rotation 0.8 deg
ALERT table_vs_world: translation 12.1 mm, rotation 2.3 deg
```

**Log file (`logs/events.log`):**
```
[2024-01-15 14:30:15] OK: no movement beyond thresholds
[2024-01-15 14:31:15] ALERT hotel_vs_table: translation 5.2 mm, rotation 0.8 deg
```

## 5) Testing Movement
- **Hotel moved:** Nudge only hotel base tags → expect `hotel_vs_table` alert
- **Table moved:** Nudge only table tags → expect `table_vs_world` alert  
- **Camera moved:** Nudge camera/mount → expect `camera_vs_world` alert

## 6) Tune Thresholds
Edit `config/config.yaml`:
```yaml
thresholds:
  hotel_vs_table:
    translation_mm: 3.0    # Start here, adjust based on testing
    rotation_deg: 0.5
  table_vs_world:
    translation_mm: 8.0
    rotation_deg: 1.5
```

## 7) Expected Results
- **Translation:** Millimeters of movement
- **Rotation:** Degrees of rotation
- **Exit codes:** 0=OK, 1=RTSP failed, 2=insufficient tags, 3=movement detected

The system compares current poses to your baseline and alerts if thresholds are exceeded.

## 8) Tag Placement
- **World tags:** 3 tags on back board/rail, wide triangle, facing camera
- **Table tags:** 3 tags spread along table edges, facing camera
- **Hotel tags:** 2 tags on different faces of black base (not green body)
- **Camera tags:** 1-2 tags on camera housing/mount (optional)

## 9) Troubleshooting
- **"Insufficient world tags visible":** Spread world tags wider, check lighting
- **No detection for a group:** Verify tag IDs in config, ensure at least one tag visible
- **False alarms:** Increase thresholds in config
- **No alerts when moving:** Decrease thresholds in config
