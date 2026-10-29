"""
Frame analysis for hazard monitoring.
=====================================
`VisionDetector.analyse(frame)` runs the stove model and the stock COCO
model (for people) on one frame and returns a `FrameAnalysis`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import stove_detector

SCRIPT_DIR = Path(__file__).resolve().parent
PERSON_MODEL_PATH = SCRIPT_DIR / "YoloModel" / "defaultModel.pt"   # stock YOLOv8n, 80 COCO classes
PERSON_CLASS = "person"


@dataclass
class FrameAnalysis:
    stove: str | None = None          # "STOVE-ON", "STOVE-OFF" or None (no stove visible)
    stove_confidence: float = 0.0
    burners_on: int = 0               # STOVE-ON boxes above the threshold
    person: bool = False
    person_confidence: float = 0.0
    labels: list = field(default_factory=list)   # every (label, confidence) seen, for logging

    def describe(self) -> str:
        if self.stove == stove_detector.STOVE_ON:
            stove = f"stove ON ({self.burners_on} burner{'s' if self.burners_on != 1 else ''})"
        elif self.stove == stove_detector.STOVE_OFF:
            stove = "stove off"
        else:
            stove = "no stove visible"
        return f"{stove}, {'person present' if self.person else 'nobody in view'}"


class VisionDetector:
    def __init__(self, min_confidence: float = stove_detector.DEFAULT_CONFIDENCE, detect_people: bool = True):
        self.min_confidence = min_confidence
        self.stove_model = stove_detector.get_model()
        self.person_model = None
        if detect_people:
            from ultralytics import YOLO
            self.person_model = YOLO(PERSON_MODEL_PATH)

    def analyse(self, frame) -> FrameAnalysis:
        result = FrameAnalysis()

        on_conf, off_conf = [], []
        for r in self.stove_model(frame, conf=self.min_confidence, verbose=False):
            for box in r.boxes:
                label = self.stove_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                result.labels.append((label, round(conf, 2)))
                if label == stove_detector.STOVE_ON:
                    on_conf.append(conf)
                elif label == stove_detector.STOVE_OFF:
                    off_conf.append(conf)

        # A single lit burner means the stove is on
        if on_conf:
            result.stove, result.stove_confidence, result.burners_on = stove_detector.STOVE_ON, max(on_conf), len(on_conf)
        elif off_conf:
            result.stove, result.stove_confidence = stove_detector.STOVE_OFF, max(off_conf)

        if self.person_model is not None:
            person_id = next(i for i, n in self.person_model.names.items() if n == PERSON_CLASS)
            for r in self.person_model(frame, conf=self.min_confidence, classes=[person_id], verbose=False):
                for box in r.boxes:
                    conf = float(box.conf[0])
                    result.labels.append((PERSON_CLASS, round(conf, 2)))
                    result.person = True
                    result.person_confidence = max(result.person_confidence, conf)
        return result
