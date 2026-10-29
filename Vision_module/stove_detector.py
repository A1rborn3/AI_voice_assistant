import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
modelPath = SCRIPT_DIR / "YoloModel" / "Stove.pt"

testPicsPath = SCRIPT_DIR / "TestPic" / "stoveOn2.JPG"

STOVE_ON = "STOVE-ON"
STOVE_OFF = "STOVE-OFF"
STOVE_LABELS = (STOVE_ON, STOVE_OFF)

# Detections below this confidence are ignored
DEFAULT_CONFIDENCE = 0.5

_model = None


def get_model():
    """Load the YOLO model on first use, so importing this module stays cheap."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(modelPath)
    return _model


def check_stove(frame, min_confidence=DEFAULT_CONFIDENCE):
    """
    Classify the stove state in a frame (image path, numpy array, PIL image...).

    Returns "STOVE-ON", "STOVE-OFF", or None if no stove was detected above
    `min_confidence`. If any burner is detected as on, the result is
    "STOVE-ON" — for a safety check, a single lit burner wins.
    """
    model = get_model()
    results = model(frame, conf=min_confidence, verbose=False)

    best = {}  # label -> highest confidence seen
    for result in results:
        for detection in result.boxes:
            label = model.names[int(detection.cls[0])]
            conf = float(detection.conf[0])
            if label in STOVE_LABELS and conf >= min_confidence:
                best[label] = max(conf, best.get(label, 0.0))

    if STOVE_ON in best:
        return STOVE_ON
    if STOVE_OFF in best:
        return STOVE_OFF
    return None


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else testPicsPath
    print(check_stove(target))
