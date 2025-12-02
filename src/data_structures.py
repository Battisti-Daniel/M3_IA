from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]


def random_color() -> Color:
    return tuple(int(v) for v in np.random.choice(range(50, 255), size=3))  # type: ignore[return-value]


@dataclass
class PlateReading:
    text: Optional[str] = None
    ocr_confidence: float = 0.0
    detection_confidence: float = 0.0
    plate_type: Optional[str] = None
    crop: Optional[np.ndarray] = None
    last_updated: float = field(default_factory=time.time)


@dataclass
class VehicleTrack:
    track_id: int
    bbox: BBox
    label: str
    score: float
    color: Color = field(default_factory=random_color)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    model_vehicle: str = ""
    model_plate: str = ""
    average_fps: float = 0.0
    reading: PlateReading = field(default_factory=PlateReading)
    trail: List[Tuple[int, int]] = field(default_factory=list)
    hits: int = 0
    miss_count: int = 0

    def update_bbox(self, bbox: BBox) -> None:
        self.bbox = bbox
        self.last_seen = time.time()

    def update_trail(self) -> None:
        x1, y1, x2, y2 = self.bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        self.trail.append((cx, cy))

    def should_timeout(self, timeout_seconds: float) -> bool:
        return (time.time() - self.last_seen) > timeout_seconds


class VehicleRegistry:
    def __init__(self) -> None:
        self.tracks: Dict[int, VehicleTrack] = {}

    def upsert(self, track: VehicleTrack) -> VehicleTrack:
        self.tracks[track.track_id] = track
        return track

    def get(self, track_id: int) -> Optional[VehicleTrack]:
        return self.tracks.get(track_id)

    def remove(self, track_id: int) -> Optional[VehicleTrack]:
        return self.tracks.pop(track_id, None)

    def active(self) -> List[VehicleTrack]:
        return list(self.tracks.values())

    def expired(self, timeout_seconds: float) -> List[VehicleTrack]:
        return [track for track in self.tracks.values() if track.should_timeout(timeout_seconds)]
