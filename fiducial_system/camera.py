"""Camera capture utilities.

This module intentionally stays small and focused on reading a single frame
from an RTSP stream, with a simple retry policy. Keeping it isolated makes it
easy to swap in a different capture backend later (e.g., GStreamer).
"""

import time
import cv2
from typing import Optional, Tuple


def grab_frame(rtsp_url: str, attempts: int = 3, delay_ms: int = 400) -> Optional[Tuple[bool, any]]:
    """Read one frame from an RTSP URL with basic retry.

    Returns a tuple ``(True, frame)`` on success, or ``None`` on failure.
    We return a tuple instead of just the frame to allow future metadata.

    Parameters
    - rtsp_url: Full RTSP URL to the camera stream.
    - attempts: Number of times to retry opening/reading the stream.
    - delay_ms: Delay between retries, to give the camera/RTSP server time.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        # retry loop creating new capture each time
        for i in range(1, attempts + 1):
            time.sleep(delay_ms / 1000.0)
            cap = cv2.VideoCapture(rtsp_url)
            if cap.isOpened():
                break
        else:
            return None

    ok, frame = cap.read()
    cap.release()
    if not ok:
        # Retry reads without recreating cap (simple approach)
        for _ in range(attempts - 1):
            cap = cv2.VideoCapture(rtsp_url)
            ok, frame = cap.read()
            cap.release()
            if ok:
                break
        if not ok:
            return None
    return True, frame

