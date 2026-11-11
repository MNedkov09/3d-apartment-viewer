import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

p = hf_hub_download('mudasir13cs/floorcad-yolov8n-seg', 'best.pt')
m = YOLO(p)

img_path = 'floorplan-detector/public/samples/residential_plan.png'
img = cv2.imread(img_path)
h, w = img.shape[:2]

res = m.predict(img, conf=0.1)

# Mask for non-wall objects (furniture, doors, etc.)
object_mask = np.zeros((h, w), dtype=np.uint8)

for i in range(len(res[0])):
    cls_id = int(res[0].boxes.cls[i])
    name = m.names[cls_id]
    if name not in ['wall', 'class_31', 'class_32']:
        # If there's a polygon mask, fill it
        if res[0].masks is not None and i < len(res[0].masks.xy):
            poly = np.array(res[0].masks.xy[i], dtype=np.int32)
            cv2.fillPoly(object_mask, [poly], 255)
        else:
            # Bounding box fallback
            box = res[0].boxes.xyxy[i].cpu().numpy().astype(int)
            cv2.rectangle(object_mask, (box[0], box[1]), (box[2], box[3]), 255, -1)

# Gray and threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)

# Subtract detected objects!
clean_walls = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(object_mask))

print("Original white pixels:", np.count_nonzero(thresh))
print("Clean walls white pixels (furniture & clutter removed!):", np.count_nonzero(clean_walls))
