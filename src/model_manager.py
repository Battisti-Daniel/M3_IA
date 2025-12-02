from __future__ import annotations

from pathlib import Path
from typing import List

from ultralytics import YOLO

from .device import get_device, use_half_precision


class ModelManager:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.vehicle_model_path = None
        self.plate_model_path = None
        self.device = get_device()
        self.vehicle_model = None
        self.plate_model = None

    def list_models(self) -> List[str]:
        return [f.name for f in self.models_dir.glob("*.pt")]

    def _load_model(self, model_name: str):
        path = self.models_dir / model_name
        model = YOLO(path)
        model.to(self.device)
        model.fuse()
        model.half() if use_half_precision(self.device) else None
        return model

    def set_vehicle_model(self, model_name: str) -> None:
        self.vehicle_model = self._load_model(model_name)
        self.vehicle_model_path = model_name

    def set_plate_model(self, model_name: str) -> None:
        self.plate_model = self._load_model(model_name)
        self.plate_model_path = model_name

    def ready(self) -> bool:
        return self.vehicle_model is not None and self.plate_model is not None


__all__ = ["ModelManager"]
