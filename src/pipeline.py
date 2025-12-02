from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import (
    DEFAULT_LOG_FILE,
    DEFAULT_TIMEOUT_SECONDS,
    MIN_PLATE_CONFIDENCE,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_SKIP_FRAMES,
    PLATE_SKIP_IF_CONFIDENT,
    YOLO_IMGSZ,
)
from .data_structures import BBox, PlateReading, VehicleRegistry, VehicleTrack
from .device import get_device, use_half_precision
from .model_manager import ModelManager
from .ocr import OcrService
from .preprocessing import correct_perspective, crop_from_bbox, enhance_for_ocr
from .tracker import ByteTrack


class Logger:
    def __init__(self, csv_path: Path = DEFAULT_LOG_FILE) -> None:
        self.csv_path = csv_path
        if not self.csv_path.exists():
            self.csv_path.write_text(
                "track_id,plate_text,ocr_confidence,plate_confidence,plate_type,"
                "vehicle_type,vehicle_model,plate_model,average_fps,first_seen,last_seen\n"
            )

    def log_track(self, track: VehicleTrack) -> None:
        with self.csv_path.open("a", encoding="utf-8") as fp:
            fp.write(
                f"{track.track_id},{track.reading.text or ''},{track.reading.ocr_confidence:.3f},"
                f"{track.reading.detection_confidence:.3f},{track.reading.plate_type or ''},"
                f"{track.label},{track.model_vehicle},{track.model_plate},"
                f"{track.average_fps:.2f},{track.first_seen:.3f},{track.last_seen:.3f}\n"
            )


class Pipeline:
    def __init__(self, model_manager: ModelManager, logger: Optional[Logger] = None) -> None:
        self.model_manager = model_manager
        self.device = get_device()
        self.half_precision = use_half_precision(self.device)
        self.tracker = ByteTrack(iou_threshold=0.25, high_confidence=0.55, low_confidence=0.1, max_age=15)
        self.registry = VehicleRegistry()
        self.logger = logger or Logger()
        self.ocr = OcrService()
        self.frame_counter = 0
        self.target_fps = 20
        self.frame_skip = 0
        self._fps_history: List[float] = []

    def set_target_fps(self, target: int) -> None:
        self.target_fps = max(1, target)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[VehicleTrack]]:
        start = time.time()
        self.frame_counter += 1
        vehicles = self._detect_vehicles(frame)
        tracks = self.tracker.update(vehicles)
        for track in tracks:
            track.model_vehicle = self.model_manager.vehicle_model_path or ""
            track.model_plate = self.model_manager.plate_model_path or ""
            self.registry.upsert(track)
        self._process_plates(frame, tracks)
        elapsed = time.time() - start
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self._fps_history.append(fps)
        for track in tracks:
            track.average_fps = self._average_fps()
        self._flush_expired_tracks()
        return frame, tracks

    def _average_fps(self) -> float:
        if not self._fps_history:
            return 0.0
        return sum(self._fps_history[-120:]) / min(len(self._fps_history), 120)

    def _detect_vehicles(self, frame: np.ndarray) -> List[VehicleTrack]:
        if not self.model_manager.vehicle_model:
            return []
        model = self.model_manager.vehicle_model
        results = model(frame, imgsz=YOLO_IMGSZ, device=self.device, half=self.half_precision, verbose=False)
        vehicles: List[VehicleTrack] = []
        names: Dict[int, str] = model.names  # type: ignore[attr-defined]
        for det in results[0].boxes:
            cls_id = int(det.cls)
            label = names.get(cls_id, str(cls_id))
            if label not in {"car", "motorbike", "motorcycle", "moto"}:
                continue
            x1, y1, x2, y2 = det.xyxy[0].tolist()
            vehicles.append(
                VehicleTrack(
                    track_id=-1,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    label="car" if label != "motorbike" and label != "motorcycle" and label != "moto" else "moto",
                    score=float(det.conf),
                )
            )
        return vehicles

    def _process_plates(self, frame: np.ndarray, tracks: List[VehicleTrack]) -> None:
        if not self.model_manager.plate_model:
            return
        model = self.model_manager.plate_model
        for track in tracks:
            # Otimização: pular se já tem leitura confiável
            if PLATE_SKIP_IF_CONFIDENT and track.reading.ocr_confidence >= OCR_CONFIDENCE_THRESHOLD:
                continue
            vehicle_crop = crop_from_bbox(frame, track.bbox)
            if vehicle_crop.size == 0:
                continue
            results = model(
                vehicle_crop,
                imgsz=YOLO_IMGSZ,
                device=self.device,
                half=self.half_precision,
                verbose=False,
            )
            for det in results[0].boxes:
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                plate_conf = float(det.conf)
                plate_bbox_local: BBox = (float(x1), float(y1), float(x2), float(y2))
                plate_bbox_global = self._local_to_global_bbox(plate_bbox_local, track.bbox)
                reading = self._process_plate_detection(frame, plate_bbox_global, plate_conf)
                if reading:
                    self._update_track_with_plate(track, reading)

    def _local_to_global_bbox(self, plate_bbox: BBox, vehicle_bbox: BBox) -> BBox:
        vx1, vy1, vx2, vy2 = vehicle_bbox
        px1, py1, px2, py2 = plate_bbox
        return vx1 + px1, vy1 + py1, vx1 + px2, vy1 + py2

    def _process_plate_detection(self, frame: np.ndarray, bbox: BBox, conf: float) -> Optional[PlateReading]:
        if conf < MIN_PLATE_CONFIDENCE:
            return None
        # Otimização: rodar OCR apenas a cada N frames
        if OCR_SKIP_FRAMES > 1 and self.frame_counter % OCR_SKIP_FRAMES != 0:
            return None
        x1, y1, x2, y2 = bbox
        if (x2 - x1) < 20 or (y2 - y1) < 12:
            return None
        crop = crop_from_bbox(frame, bbox)
        if crop.size == 0:
            return None
        rectified = correct_perspective(crop)
        enhanced = enhance_for_ocr(rectified)
        text, ocr_conf, plate_type = self.ocr.infer(enhanced)
        if text is None:
            return None
        return PlateReading(text=text, ocr_confidence=ocr_conf, detection_confidence=conf, plate_type=plate_type, crop=enhanced)

    def _update_track_with_plate(self, track: VehicleTrack, new_reading: PlateReading) -> None:
        current = track.reading
        if new_reading.ocr_confidence >= current.ocr_confidence:
            track.reading = new_reading
            track.reading.last_updated = time.time()

    def _flush_expired_tracks(self) -> None:
        expired = self.registry.expired(DEFAULT_TIMEOUT_SECONDS)
        for track in expired:
            self.logger.log_track(track)
            self.registry.remove(track.track_id)

    def reset_tracking(self) -> None:
        """Reseta o tracker e registry para permitir novas detecções."""
        self.tracker = ByteTrack(iou_threshold=0.25, high_confidence=0.55, low_confidence=0.1, max_age=15)
        self.registry = VehicleRegistry()
        self._fps_history.clear()

    def close(self) -> None:
        for track in list(self.registry.active()):
            self.logger.log_track(track)
            self.registry.remove(track.track_id)
