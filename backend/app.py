import time
import math
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from wall_extractor import (
    get_category,
    extract_oriented_box,
    extract_walls_from_mask
)

app = FastAPI(title="Floorplan AI Detector", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3D Model mapping for auto-furnishing
MODEL_MAP = {
    "bed": "Bed1.glb",
    "sofa": "Sofa1.glb",
    "table": "TableB1.glb",
    "chair": "Chair1.glb",
    "tv_cabinet": "Storage1.glb",
    "wardrobe": "Storage2.glb",
    "high_cabinet": "Storage3.glb",
    "half_height_cabinet": "Storage4.glb",
    "desk": "Desk1.glb",
}

model = None

def get_model():
    global model
    if model is None:
        print("Downloading / Loading FloorCAD YOLO model...")
        weights_path = hf_hub_download(repo_id="mudasir13cs/floorcad-yolov8n-seg", filename="best.pt")
        model = YOLO(weights_path)
        print("FloorCAD Model ready!")
    return model

@app.on_event("startup")
def startup_event():
    get_model()

@app.get("/api/health")
def health_check():
    m = get_model()
    return {
        "status": "healthy",
        "model": "FloorCAD YOLOv8n-seg",
        "classes": len(m.names) if m else 0
    }

@app.post("/api/detect")
async def detect_floorplan(
    file: UploadFile = File(...),
    conf: float = Form(0.12),
    use_hybrid: bool = Form(True),
    canvas_w: float = Form(800.0),
    canvas_h: float = Form(600.0)
):
    start_time = time.time()
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    orig_h, orig_w = img.shape[:2]
    
    # Compute canvas scaling and centering offsets
    scale = min(canvas_w / orig_w, canvas_h / orig_h)
    rendered_w = orig_w * scale
    rendered_h = orig_h * scale
    ox = (canvas_w - rendered_w) / 2.0
    oy = (canvas_h - rendered_h) / 2.0
    
    def to_canvas_pt(pt):
        return [round(ox + pt[0] * scale, 2), round(oy + pt[1] * scale, 2)]

    def to_canvas_pts(pts):
        return [to_canvas_pt(p) for p in pts]

    m = get_model()
    results = m.predict(img, conf=conf, imgsz=640)
    res = results[0]
    
    boxes = res.boxes
    has_masks = res.masks is not None
    
    wall_polygons = []
    doors = []
    windows = []
    furniture = []
    fixtures = []
    stairs = []
    other = []
    
    clutter_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    ai_wall_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    
    num_detections = len(boxes) if boxes is not None else 0
    
    for i in range(num_detections):
        cls_id = int(boxes.cls[i])
        raw_name = m.names[cls_id]
        category = get_category(raw_name)
        confidence = float(boxes.conf[i])
        box_coords = [round(float(v), 2) for v in boxes.xyxy[i].tolist()]
        
        polygon = []
        if has_masks and i < len(res.masks.xy):
            poly_pts = res.masks.xy[i]
            if len(poly_pts) > 0:
                polygon = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in poly_pts]
                
        oriented = extract_oriented_box(polygon if len(polygon) >= 3 else box_coords)
        
        # Calculate line segment (start and end) along the length of window/door
        cx, cy = oriented["center"]
        half_len = oriented["width"] / 2.0
        angle_rad = oriented["angle_rad"]
        hx = math.cos(angle_rad) * half_len
        hy = math.sin(angle_rad) * half_len
        
        start_pt = [round(cx - hx, 2), round(cy - hy, 2)]
        end_pt = [round(cx + hx, 2), round(cy + hy, 2)]
        
        item = {
            "id": f"{category}_{i}",
            "label": raw_name,
            "category": category,
            "confidence": round(confidence, 3),
            "box": box_coords,
            "polygon": polygon,
            "center": oriented["center"],
            "width": oriented["width"],
            "depth": oriented["depth"],
            "angle_deg": oriented["angle_deg"],
            "angle_rad": oriented["angle_rad"],
            "start": start_pt,
            "end": end_pt,
            # Canvas space projections
            "canvas_center": to_canvas_pt(oriented["center"]),
            "canvas_start": to_canvas_pt(start_pt),
            "canvas_end": to_canvas_pt(end_pt),
            "canvas_box": [
                ox + box_coords[0] * scale,
                oy + box_coords[1] * scale,
                ox + box_coords[2] * scale,
                oy + box_coords[3] * scale
            ],
            "canvas_polygon": to_canvas_pts(polygon) if polygon else []
        }
        
        if category == "furniture":
            item["model_file"] = MODEL_MAP.get(raw_name, None)
            
        if category == "wall":
            wall_polygons.append(item)
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(ai_wall_mask, [pts], 255)
            else:
                bx = [int(v) for v in box_coords]
                cv2.rectangle(ai_wall_mask, (bx[0], bx[1]), (bx[2], bx[3]), 255, -1)
        else:
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(clutter_mask, [pts], 255)
            else:
                bx = [int(v) for v in box_coords]
                cv2.rectangle(clutter_mask, (bx[0], bx[1]), (bx[2], bx[3]), 255, -1)
                
            if category == "door":
                doors.append(item)
            elif category == "window":
                windows.append(item)
            elif category == "furniture":
                furniture.append(item)
            elif category == "fixture":
                fixtures.append(item)
            elif category == "stairs":
                stairs.append(item)
            else:
                other.append(item)
                
    kernel_clutter = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clutter_mask = cv2.dilate(clutter_mask, kernel_clutter, iterations=1)
    
    final_wall_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if np.count_nonzero(ai_wall_mask) > 100:
        final_wall_mask = cv2.bitwise_or(final_wall_mask, ai_wall_mask)
        
    if use_hybrid or np.count_nonzero(final_wall_mask) < 200:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8
        )
        cleaned_thresh = cv2.bitwise_and(thresh, thresh, mask=cv2.bitwise_not(clutter_mask))
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_thresh, 8)
        min_area = 40
        clean_comp = np.zeros_like(cleaned_thresh)
        for label_idx in range(1, num_labels):
            if stats[label_idx, cv2.CC_STAT_AREA] >= min_area:
                clean_comp[labels == label_idx] = 255
                
        final_wall_mask = cv2.bitwise_or(final_wall_mask, clean_comp)
        
    raw_wall_edges, _ = extract_walls_from_mask(final_wall_mask, min_edge_len=14.0)
    
    # Project wall edges into canvas space
    canvas_wall_edges = []
    for edge in raw_wall_edges:
        canvas_pts = to_canvas_pts(edge["points"])
        canvas_wall_edges.append({
            "points": canvas_pts,
            "closed": edge.get("closed", False)
        })
        
    elapsed_ms = round((time.time() - start_time) * 1000, 1)
    
    # Format windows for Viewer3D CSG cutting
    viewer3d_windows = [
        {
            "start": w_item["canvas_start"],
            "end": w_item["canvas_end"],
            "width": max(w_item["depth"] * scale, 8.0)
        }
        for w_item in windows
    ]
    
    # Format doors for Viewer3D CSG cutting
    viewer3d_doors = [
        {
            "start": d_item["canvas_start"],
            "end": d_item["canvas_end"],
            "width": max(d_item["depth"] * scale, 10.0),
            "label": d_item["label"]
        }
        for d_item in doors
    ]
    
    return {
        "success": True,
        "image_size": {"width": orig_w, "height": orig_h},
        "canvas_size": {"width": canvas_w, "height": canvas_h},
        "scale_info": {"scale": scale, "ox": ox, "oy": oy},
        "stats": {
            "wall_edges": len(canvas_wall_edges),
            "walls_detected": len(wall_polygons),
            "doors": len(doors),
            "windows": len(windows),
            "furniture": len(furniture),
            "fixtures": len(fixtures),
            "stairs": len(stairs),
            "inference_time_ms": elapsed_ms
        },
        "wall_edges": canvas_wall_edges,
        "windows": viewer3d_windows,
        "doors": viewer3d_doors,
        "furniture": furniture,
        "fixtures": fixtures,
        "raw_detections": {
            "walls": wall_polygons,
            "doors": doors,
            "windows": windows,
            "furniture": furniture,
            "fixtures": fixtures,
            "stairs": stairs,
            "other": other
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
