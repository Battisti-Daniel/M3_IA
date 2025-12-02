from __future__ import annotations

import re
from typing import Optional, Tuple

import easyocr
import numpy as np
import torch

from .device import get_device

PLATE_PATTERN_OLD = re.compile(r"^[A-Z]{3}\d{4}$")
PLATE_PATTERN_MERCOSUL = re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$")


class OcrService:
    def __init__(self) -> None:
        device = get_device()
        self.reader = easyocr.Reader(["pt", "en"], gpu=device.type == "cuda")

    def infer(self, image: np.ndarray) -> Tuple[Optional[str], float, Optional[str]]:
        results = self.reader.readtext(image)
        if not results:
            return None, 0.0, None
        best = max(results, key=lambda x: x[2])
        text_raw = best[1].replace(" ", "").upper()
        confidence = float(best[2])
        plate_type = self._classify_plate(text_raw)
        return text_raw, confidence, plate_type

    def _classify_plate(self, text: str) -> Optional[str]:
        if PLATE_PATTERN_MERCOSUL.match(text):
            return "mercosul"
        if PLATE_PATTERN_OLD.match(text):
            return "old"
        return None


__all__ = ["OcrService"]
