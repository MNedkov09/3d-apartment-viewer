import cv2
import numpy as np
import math
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Mapping of FloorCAD 35 classes to simplified categories
CATEGORY_MAP = {
    'wall': 'wall',
    
    'window': 'window',
    'bay_window': 'window',
    'blind_window': 'window',
    
    'single_door': 'door',
    'double_door': 'door',
    'sliding_door': 'door',
    'opening_symbol': 'door',
    
    'bed': 'furniture',
    'bedside_cupboard': 'furniture',
    'sofa': 'furniture',
    'table': 'furniture',
    'chair': 'furniture',
    'tv_cabinet': 'furniture',
    'wardrobe': 'furniture',
    'high_cabinet': 'furniture',
    'half_height_cabinet': 'furniture',
    
    'bath': 'fixture',
    'bath_tub': 'fixture',
    'sink': 'fixture',
    'toilet': 'fixture',
    'squat_toilet': 'fixture',
    'urinal': 'fixture',
    'gas_stove': 'fixture',
    'refrigerator': 'fixture',
    'washing_machine': 'fixture',
    
    'stair': 'stairs',
    'elevator': 'stairs',
    'escalator': 'stairs',
    'railing': 'stairs',
    'parking': 'other',
}

def extract_oriented_props(polygon_pts):
    """Compute center, width, depth, and rotation angle using minAreaRect."""
    if len(polygon_pts) < 3:
        return None
    pts = np.array(polygon_pts, dtype=np.float32)
    rect = cv2.minAreaRect(pts)
    (cx, cy), (w, h), angle = rect
    # Ensure w is longer side (length of window/door) and h is thickness
    if h > w:
        w, h = h, w
        angle += 90.0
    return {
        "center": [round(float(cx), 2), round(float(cy), 2)],
        "width": round(float(w), 2),
        "depth": round(float(h), 2),
        "angle_deg": round(float(angle % 180), 2),
        "angle_rad": round(float(math.radians(angle % 180)), 4)
    }

print("Extractor helper defined.")
