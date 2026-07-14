import torch
from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
modelPath = SCRIPT_DIR / "YoloModel" / "Stove.pt"
model = YOLO(modelPath)

testPicsPath = SCRIPT_DIR / "TestPic" / "stoveOn2.JPG"

def check_stove(frame):
    results = model(frame)
    
    # Check if any detected object is a stove
    for result in results:
        for detection in result.boxes:
            cls = int(detection.cls[0])
            label = model.names[cls]
            
            if label == "STOVE-OFF" or label == "STOVE-ON":
                return label
    
    return label

if __name__ == '__main__':
    result = check_stove(testPicsPath)
    print(result)
