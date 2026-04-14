"""
image_processing.processor

Simple image processing pipeline utilities for light level detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.image_processing.light_level import day_capture_conf, night_capture_conf


@dataclass(frozen=True)
class LightLevelResult:
    level: str
    mean_brightness: float


class Processor:
    """
    Lightweight processing helper for determining scene brightness and selecting
    camera capture presets.
    """

    LEVELS = ("night", "morning", "day", "afternoon")

    def __init__(
        self,
        night_threshold: float = 50.0,
        morning_threshold: float = 100.0,
        day_threshold: float = 170.0,
    ):
        self.night_threshold = night_threshold
        self.morning_threshold = morning_threshold
        self.day_threshold = day_threshold

    def estimate_light_level(self, frame: np.ndarray) -> LightLevelResult:
        """
        Estimate scene lighting from BGR image brightness.
        Returns one of: night, morning, day, afternoon.
        """
        if frame is None or not hasattr(frame, "size") or frame.size == 0:
            raise ValueError("Frame is empty or invalid.")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))
        level = self._classify_brightness(mean_brightness)
        return LightLevelResult(level=level, mean_brightness=mean_brightness)

    def process_image_path(self, image_path: str) -> LightLevelResult:
        """Load an image from disk and estimate its light level."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        return self.estimate_light_level(frame)

    def select_capture_config(
        self,
        picam2: Any,
        image_size: tuple[int, int],
        display_size: Optional[tuple[int, int]],
        light_level: str,
    ) -> Dict[str, Any]:
        """
        Select camera settings based on detected light level.
        Night uses the night profile; all other states use day profile.
        """
        if light_level == "night":
            return night_capture_conf(picam2, image_size, display_size)
        return day_capture_conf(picam2, image_size, display_size)

    def _classify_brightness(self, brightness: float) -> str:
        if brightness < self.night_threshold:
            return "night"
        if brightness < self.morning_threshold:
            return "morning"
        if brightness < self.day_threshold:
            return "day"
        return "afternoon"