"""Configuration loading and schema types.

This module converts the YAML config file into strongly-typed dataclasses so
the rest of the code can rely on clear, validated structures. If you add new
settings, extend the dataclasses here and adjust the loader accordingly.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import os


@dataclass
class FrameGrabConfig:
    attempts: int = 3
    delay_ms: int = 400


@dataclass
class AprilTagConfig:
    family: str = "tag36h11"
    tag_size_m: float = 0.08
    nthreads: int = 4
    refine_edges: bool = True
    decode_sharpening: float = 0.25


@dataclass
class Thresholds:
    translation_mm: float
    rotation_deg: float


@dataclass
class PolicyConfig:
    consecutive_violations: int = 2
    min_visible_tags_per_group: int = 1
    min_visible_world_tags: int = 2


@dataclass
class OutputConfig:
    baseline_file: str
    events_log: str
    snapshots_dir: str


@dataclass
class NotifyEmail:
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_addr: str = ""
    to_addrs: List[str] = field(default_factory=list)


@dataclass
class NotifySlack:
    enabled: bool = False
    webhook_url: str = ""


@dataclass
class NotifyConfig:
    console: bool = True
    email: NotifyEmail = field(default_factory=NotifyEmail)
    slack: NotifySlack = field(default_factory=NotifySlack)


@dataclass
class AppConfig:
    rtsp_url: str
    frame_grab: FrameGrabConfig
    camera_intrinsics: str
    apriltag: AprilTagConfig
    objects: Dict[str, Dict[str, List[int]]]
    thresholds: Dict[str, Thresholds]
    policy: PolicyConfig
    output: OutputConfig
    notify: NotifyConfig

    @staticmethod
    def load(path: str) -> "AppConfig":
        """Load YAML config from ``path`` into an ``AppConfig`` instance.

        Side-effects: Ensures output directories exist as configured.
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        def map_thresholds(d: Dict[str, Dict]) -> Dict[str, Thresholds]:
            return {k: Thresholds(**v) for k, v in d.items()}

        cfg = AppConfig(
            rtsp_url=raw["rtsp_url"],
            frame_grab=FrameGrabConfig(**raw.get("frame_grab", {})),
            camera_intrinsics=raw["camera_intrinsics"],
            apriltag=AprilTagConfig(**raw.get("apriltag", {})),
            objects=raw["objects"],
            thresholds=map_thresholds(raw["thresholds"]),
            policy=PolicyConfig(**raw.get("policy", {})),
            output=OutputConfig(**raw["output"]),
            notify=NotifyConfig(
                console=raw.get("notify", {}).get("console", True),
                email=NotifyEmail(**raw.get("notify", {}).get("email", {})),
                slack=NotifySlack(**raw.get("notify", {}).get("slack", {})),
            ),
        )

        # Ensure output directories exist
        os.makedirs(os.path.dirname(cfg.output.events_log), exist_ok=True)
        os.makedirs(cfg.output.snapshots_dir, exist_ok=True)
        return cfg

