import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Download weights
weights_path = hf_hub_download(repo_id="mudasir13cs/floorcad-yolov8n-seg", filename="best.pt")
model = YOLO(weights_path)

# Create a sample synthetic floor plan image with a rectangular room, door and window
img = np.ones((600, 800, 3), dtype=np.uint8) * 255
# Outer walls (thick black lines)
cv2.rectangle(img, (100, 100), (700, 500), (0, 0, 0), thickness=15)
# Door opening gap
cv2.line(img, (350, 500), (450, 500), (255, 255, 255), thickness=16)
# Door swing arc
cv2.ellipse(img, (350, 500), (100, 100), 0, 270, 360, (0, 0, 0), 2)
# Window gap
cv2.line(img, (350, 100), (450, 100), (200, 200, 200), thickness=8)

res = model.predict(img, conf=0.1, imgsz=640)
print(f"Detected {len(res[0])} objects in synthetic test.")
for box, cls, conf in zip(res[0].boxes.xyxy, res[0].boxes.cls, res[0].boxes.conf):
    name = model.names[int(cls)]
    print(f" - {name} ({conf:.2f}): box {box.tolist()}")
