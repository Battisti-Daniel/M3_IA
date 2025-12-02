from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .data_structures import BBox, VehicleTrack


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.2) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = itertools.count(1)
        self.active: List[VehicleTrack] = []

    @staticmethod
    def _iou(box_a: BBox, box_b: BBox) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _match(self, detections: List[VehicleTrack]) -> List[VehicleTrack]:
        updated: List[VehicleTrack] = []
        for det in detections:
            match: Optional[VehicleTrack] = None
            best_iou = self.iou_threshold
            for track in self.active:
                score = self._iou(det.bbox, track.bbox)
                if score >= best_iou:
                    best_iou = score
                    match = track
            if match:
                match.update_bbox(det.bbox)
                match.score = det.score
                match.label = det.label
                updated.append(match)
            else:
                det.track_id = next(self._next_id)
                updated.append(det)
        self.active = updated
        return self.active

    def update(self, detections: List[VehicleTrack]) -> List[VehicleTrack]:
        return self._match(detections)


class ByteTrack:
    """Implementação simplificada do ByteTrack focada em robustez e performance."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        high_confidence: float = 0.6,
        low_confidence: float = 0.1,
        max_age: int = 30,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.high_confidence = high_confidence
        self.low_confidence = low_confidence
        self.max_age = max_age
        self._next_id = itertools.count(1)
        self.tracked: List[VehicleTrack] = []
        self.lost: List[VehicleTrack] = []

    @staticmethod
    def _iou(box_a: BBox, box_b: BBox) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _match(self, tracks: Sequence[VehicleTrack], detections: Sequence[VehicleTrack]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, det.bbox)

        matches: List[Tuple[int, int]] = []
        while True:
            max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_idx]
            if max_iou < self.iou_threshold:
                break
            track_idx, det_idx = int(max_idx[0]), int(max_idx[1])
            matches.append((track_idx, det_idx))
            iou_matrix[track_idx, :] = -1
            iou_matrix[:, det_idx] = -1

        matched_tracks = {m[0] for m in matches}
        matched_dets = {m[1] for m in matches}
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]
        return matches, unmatched_tracks, unmatched_dets

    def _activate(self, detection: VehicleTrack) -> VehicleTrack:
        detection.track_id = next(self._next_id)
        detection.hits = 1
        detection.miss_count = 0
        return detection

    def _update_track(self, track: VehicleTrack, detection: VehicleTrack) -> VehicleTrack:
        track.update_bbox(detection.bbox)
        track.score = detection.score
        track.label = detection.label
        track.hits += 1
        track.miss_count = 0
        return track

    def update(self, detections: List[VehicleTrack]) -> List[VehicleTrack]:
        high_conf_dets = [d for d in detections if d.score >= self.high_confidence]
        low_conf_dets = [d for d in detections if self.low_confidence <= d.score < self.high_confidence]

        matches, unmatched_tracked_idx, unmatched_high_idx = self._match(self.tracked, high_conf_dets)

        updated_tracks: List[VehicleTrack] = []
        for track_idx, det_idx in matches:
            track = self._update_track(self.tracked[track_idx], high_conf_dets[det_idx])
            updated_tracks.append(track)

        newly_lost: List[VehicleTrack] = []
        for idx in unmatched_tracked_idx:
            track = self.tracked[idx]
            track.miss_count += 1
            if track.miss_count <= self.max_age:
                newly_lost.append(track)

        new_tracks = [self._activate(high_conf_dets[i]) for i in unmatched_high_idx]

        matches_low, _, _ = self._match(self.lost + newly_lost, low_conf_dets)
        recovered_tracks: List[VehicleTrack] = []
        combined_lost = self.lost + newly_lost
        for track_idx, det_idx in matches_low:
            track = self._update_track(combined_lost[track_idx], low_conf_dets[det_idx])
            recovered_tracks.append(track)

        remaining_lost: List[VehicleTrack] = []
        for idx, track in enumerate(combined_lost):
            if idx in {m[0] for m in matches_low}:
                continue
            track.miss_count += 1
            if track.miss_count <= self.max_age:
                remaining_lost.append(track)

        self.tracked = updated_tracks + new_tracks + recovered_tracks
        self.lost = remaining_lost
        return self.tracked


__all__ = ["IoUTracker", "ByteTrack"]
