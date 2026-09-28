"""
YOLO Person Detector module.
Dedicated exclusively to extracting person bounding boxes, bottom-center tracking points,
and confidence metrics without any premature Re-ID extraction.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from ..config import DetectorConfig


@dataclass
class PersonDetection:
    """Standardized person detection payload."""
    bbox: np.ndarray             # [x1, y1, x2, y2]
    confidence: float            # Detection score [0.0 - 1.0]
    center: Tuple[float, float]  # (cx, cy)
    bottom_center: Tuple[float, float]  # (bc_x, bc_y) -> ((x1+x2)/2, y2)
    frame_id: int
    crop: Optional[np.ndarray] = None  # RGB image crop (lazily evaluated or provided)

    @property
    def x1(self) -> float:
        return float(self.bbox[0])

    @property
    def y1(self) -> float:
        return float(self.bbox[1])

    @property
    def x2(self) -> float:
        return float(self.bbox[2])

    @property
    def y2(self) -> float:
        return float(self.bbox[3])

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.height / max(1.0, self.width)

    def to_tlbr(self) -> np.ndarray:
        return self.bbox.copy()

    def to_tlwh(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.width, self.height], dtype=np.float32)

    def extract_crop(self, frame: np.ndarray) -> np.ndarray:
        """Extract a bounding box crop safely from the frame."""
        h, w = frame.shape[:2]
        x1 = max(0, int(round(self.x1)))
        y1 = max(0, int(round(self.y1)))
        x2 = min(w, int(round(self.x2)))
        y2 = min(h, int(round(self.y2)))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros((32, 16, 3), dtype=np.uint8)
            
        crop = frame[y1:y2, x1:x2].copy()
        self.crop = crop
        return crop


class YoloDetector:
    """
    Person detector wrapping YOLOv8 with GPU/CPU support.
    Strictly filters for class 0 (person).
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.device = self._select_device(self.config.device)
        
        # Load YOLO model
        self.model = YOLO(self.config.model_name)
        # Move model to target device if supported
        if self.device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.model.to("cpu")

    @staticmethod
    def _select_device(requested: str) -> str:
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def detect(self, frame: np.ndarray, frame_id: int, extract_crops: bool = False) -> List[PersonDetection]:
        """
        Run inference on a single frame.
        
        Args:
            frame: BGR numpy image array.
            frame_id: Monotonically increasing frame counter.
            extract_crops: If True, crops the person bounding box (used only during Re-ID).
            
        Returns:
            List of PersonDetection objects.
        """
        if frame is None or frame.size == 0:
            return []

        # Run YOLO with person class filter (classes=[0])
        results = self.model.predict(
            source=frame,
            classes=[self.config.person_class_id],
            conf=self.config.conf_thresh,
            iou=self.config.iou_thresh,
            imgsz=self.config.img_size,
            device=self.device,
            verbose=False
        )

        detections: List[PersonDetection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        for bbox, conf in zip(boxes_xyxy, confs):
            x1, y1, x2, y2 = bbox
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            
            # Bottom-center tracking point: ( (x1 + x2)/2, y2 )
            bc_x = cx
            bc_y = float(y2)

            det = PersonDetection(
                bbox=bbox.astype(np.float32),
                confidence=float(conf),
                center=(cx, cy),
                bottom_center=(bc_x, bc_y),
                frame_id=frame_id
            )

            if extract_crops:
                det.extract_crop(frame)

            detections.append(det)

        return detections
