import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

weights_path = hf_hub_download(repo_id="mudasir13cs/floorcad-yolov8n-seg", filename="best.pt")
model = YOLO(weights_path)

img = np.ones((600, 800, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (700, 500), (0, 0, 0), thickness=15)
cv2.line(img, (350, 500), (450, 500), (255, 255, 255), thickness=16)

res = model.predict(img, conf=0.1, imgsz=640)
print("has masks:", res[0].masks is not None)
if res[0].masks is not None:
    print("num mask polygons:", len(res[0].masks.xy))
    for i, p in enumerate(res[0].masks.xy):
        cls_id = int(res[0].boxes.cls[i])
        print(f"Mask {i} ({model.names[cls_id]}): {len(p)} points, e.g. {p[:3]}")
