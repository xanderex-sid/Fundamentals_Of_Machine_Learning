import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
model = YOLO("yolov8s.pt")
results = model.predict(source="0", show = True)
print(results)
