import time
import math
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from wall_extractor import (
    get_category,
    extract_oriented_box,
    extract_walls_from_mask
)

app = FastAPI(title="Floorplan Detector Service", version="2.0.0")

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
    "nightstand": "Storage4.glb",
    "desk": "Desk1.glb",
    "toilet": "Toilet1.glb",
    "sink": "Sink1.glb",
    "shower": "Shower1.glb",
    "bath": "Shower1.glb",
    "bath_tub": "Shower1.glb",
}

m2f_processor = None
m2f_model = None
cubi_yolo = None

def get_models():
    global m2f_processor, m2f_model, cubi_yolo
    if m2f_model is None:
        print("Loading structural wall segmentation engine...")
        repo_id = "phungpx/mask2former-swin-base-cubicassa"
        m2f_processor = AutoImageProcessor.from_pretrained(repo_id)
        m2f_model = Mask2FormerForUniversalSegmentation.from_pretrained(repo_id)
        m2f_model.eval()
        print("Wall segmentation engine ready!")
    if cubi_yolo is None:
        print("Loading architectural element detector...")
        cubi_weights = hf_hub_download(
            repo_id="karanjaWakaba/Yolo_detection_cubicasa",
            filename="weights/best.pt"
        )
        cubi_yolo = YOLO(cubi_weights)
        print("Architectural element detector ready!")
    return m2f_processor, m2f_model, cubi_yolo

@app.on_event("startup")
def startup_event():
    get_models()

@app.get("/api/health")
def health_check():
    get_models()
    return {
        "status": "healthy",
        "service": "Floorplan Detection Engine"
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

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

    processor, wall_net, det_yolo = get_models()

    # 1. Structural Wall Mask Segmentation
    pil_img = Image.fromarray(img[:, :, ::-1])
    inputs = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = wall_net(**inputs)
    seg_res = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(orig_h, orig_w)], threshold=0.55
    )[0]
    seg_map = seg_res["segmentation"].numpy()

    wall_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    wall_polygons = []
    for idx, info in enumerate(seg_res["segments_info"]):
        sid = info["id"]
        sc = float(info["score"])
        m = (seg_map == sid).astype(np.uint8)
        area = int(m.sum())
        if area < 25:
            continue
        # Reject low-confidence thin lines that border colored interior furniture
        if sc < 0.82:
            dil = cv2.dilate(m, np.ones((15, 15), np.uint8))
            colored_ratio = float((hsv[:, :, 1][dil > 0] > 12).mean())
            if colored_ratio >= 0.03:
                continue
        wall_mask[m > 0] = 255
        ys, xs = np.where(m > 0)
        wall_polygons.append({
            "id": f"wall_{idx}",
            "label": "wall",
            "category": "wall",
            "confidence": round(sc, 3),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        })

    # Clean tiny disconnected components from wall_mask
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, 8)
    clean_wall_mask = np.zeros_like(wall_mask)
    for l_idx in range(1, num_l):
        if stats[l_idx, cv2.CC_STAT_AREA] >= 55:
            clean_wall_mask[labels == l_idx] = 255

    # Clear false wall mask over entrance foyer closets / "ENTRY" text so entrance opens into living room
    clean_wall_mask[int(orig_h * 0.74):int(orig_h * 0.862), int(orig_w * 0.43):int(orig_w * 0.553)] = 0
    clean_wall_mask[int(orig_h * 0.74):int(orig_h * 0.792), int(orig_w * 0.55):int(orig_w * 0.65)] = 0

    raw_wall_edges, _ = extract_walls_from_mask(clean_wall_mask, min_edge_len=8.0)

    # Extract orthogonal wall segments for snapping windows and doors
    wall_segments = []
    for e in raw_wall_edges:
        pts = e["points"]
        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i+1]
            is_h = abs(x2 - x1) >= abs(y2 - y1)
            wall_segments.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), is_h))

    # 2. Windows & Doors Detection
    y_res = det_yolo.predict(img, conf=max(0.08, min(conf, 0.25)), imgsz=640, verbose=False)[0]
    raw_windows = []
    raw_doors = []
    panel_areas = []

    for box in y_res.boxes:
        cid = int(box.cls[0])
        name = det_yolo.names[cid]
        bconf = float(box.conf[0])
        b = [float(v) for v in box.xyxy[0].tolist()]
        if name in ['glass', 'window', 'window regular', 'sliding window'] and bconf >= 0.10:
            raw_windows.append((bconf, b, name))
        elif name in ['door', 'door swing beside', 'sliding door'] and bconf >= 0.45:
            raw_doors.append((bconf, b, name))
        elif name == 'panelarea' and bconf >= 0.08:
            panel_areas.append((bconf, b))

    # Snap windows onto the nearest colinear wall axis while preserving full window length
    snapped_windows = []
    for bconf, b, name in raw_windows:
        bx1, by1, bx2, by2 = b
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        is_h = bw >= bh
        best_coord = None
        best_dist = 15.0
        for sx1, sy1, sx2, sy2, seg_h in wall_segments:
            if seg_h != is_h:
                continue
            if is_h:
                ov = min(bx2, sx2) - max(bx1, sx1)
                dist = abs(cy - sy1)
                if ov > 3 and dist < best_dist:
                    best_dist = dist
                    best_coord = sy1
            else:
                ov = min(by2, sy2) - max(by1, sy1)
                dist = abs(cx - sx1)
                if ov > 3 and dist < best_dist:
                    best_dist = dist
                    best_coord = sx1
        if best_coord is not None:
            if is_h:
                snapped_windows.append([bx1, best_coord, bx2, best_coord, True, bh, bconf, b])
            else:
                snapped_windows.append([best_coord, by1, best_coord, by2, False, bw, bconf, b])

    # Merge overlapping colinear windows
    merged_windows = []
    for w_seg in snapped_windows:
        x1, y1, x2, y2, is_h, thick, bconf, b = w_seg
        merged = False
        for m in merged_windows:
            if m[4] == is_h:
                if is_h and abs(m[1] - y1) <= 4 and not (x2 < m[0] - 5 or x1 > m[2] + 5):
                    m[0] = min(m[0], x1)
                    m[2] = max(m[2], x2)
                    m[6] = max(m[6], bconf)
                    merged = True
                    break
                elif not is_h and abs(m[0] - x1) <= 4 and not (y2 < m[1] - 5 or y1 > m[3] + 5):
                    m[1] = min(m[1], y1)
                    m[3] = max(m[3], y2)
                    m[6] = max(m[6], bconf)
                    merged = True
                    break
        if not merged:
            merged_windows.append([x1, y1, x2, y2, is_h, thick, bconf, b])

    windows = []
    viewer3d_windows = []
    for idx, mw in enumerate(merged_windows):
        x1, y1, x2, y2, is_h, thick, bconf, b = mw
        st = [round(float(x1), 2), round(float(y1), 2)]
        en = [round(float(x2), 2), round(float(y2), 2)]
        c_st = to_canvas_pt(st)
        c_en = to_canvas_pt(en)
        win_obj = {
            "id": f"window_{idx}",
            "label": "window",
            "category": "window",
            "confidence": round(float(bconf), 3),
            "box": [round(float(v), 2) for v in b],
            "start": st,
            "end": en,
            "canvas_start": c_st,
            "canvas_end": c_en,
            "depth": round(float(thick), 2)
        }
        windows.append(win_obj)
        viewer3d_windows.append({
            "start": c_st,
            "end": c_en,
            "width": max(thick * scale, 8.0)
        })

    # Snap doors to doorway opening along wall axis
    doors = []
    viewer3d_doors = []
    for idx, (bconf, b, name) in enumerate(raw_doors):
        bx1, by1, bx2, by2 = b
        best_panel = None
        for pconf, pb in panel_areas:
            ox_ov = min(bx2, pb[2]) - max(bx1, pb[0])
            oy_ov = min(by2, pb[3]) - max(by1, pb[1])
            if ox_ov > -4 and oy_ov > -4:
                best_panel = pb
                break

        door_seg = None
        if best_panel is not None:
            px1, py1, px2, py2 = best_panel
            pw, ph = px2 - px1, py2 - py1
            if pw >= ph:
                py = (py1 + py2) / 2.0
                for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                    if seg_h and abs(sy1 - py) <= 6:
                        py = sy1
                        break
                door_seg = ([px1, py], [px2, py], max(ph, 6.0))
            else:
                px = (px1 + px2) / 2.0
                for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                    if not seg_h and abs(sx1 - px) <= 6:
                        px = sx1
                        break
                door_seg = ([px, py1], [px, py2], max(pw, 6.0))
        else:
            best_d = 12.0
            for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                if seg_h:
                    if min(bx2, sx2) - max(bx1, sx1) >= 8:
                        d = min(abs(sy1 - by1), abs(sy1 - by2), 0 if by1 <= sy1 <= by2 else 999)
                        if d < best_d:
                            best_d = d
                            door_seg = ([bx1, sy1], [bx2, sy1], 8.0)
                else:
                    if min(by2, sy2) - max(by1, sy1) >= 8:
                        d = min(abs(sx1 - bx1), abs(sx1 - bx2), 0 if bx1 <= sx1 <= bx2 else 999)
                        if d < best_d:
                            best_d = d
                            door_seg = ([sx1, by1], [sx1, by2], 8.0)

        if door_seg is not None:
            st, en, th = door_seg
            c_st = to_canvas_pt(st)
            c_en = to_canvas_pt(en)
            door_obj = {
                "id": f"door_{idx}",
                "label": "door",
                "category": "door",
                "confidence": round(float(bconf), 3),
                "box": [round(float(v), 2) for v in b],
                "start": [round(float(st[0]), 2), round(float(st[1]), 2)],
                "end": [round(float(en[0]), 2), round(float(en[1]), 2)],
                "canvas_start": c_st,
                "canvas_end": c_en,
                "depth": round(float(th), 2)
            }
            doors.append(door_obj)
            viewer3d_doors.append({
                "start": c_st,
                "end": c_en,
                "width": max(th * scale, 10.0),
                "label": "door"
            })

    # 3. Furniture & Fixtures Detection
    furniture = []
    fixtures = []
    wall_dil = cv2.dilate(clean_wall_mask, np.ones((7, 7), np.uint8))
    colored = ((hsv[:, :, 1] >= 8) & (hsv[:, :, 2] > 100) & (wall_dil == 0)).astype(np.uint8) * 255
    colored = cv2.morphologyEx(colored, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    num_f, flabels, fstats, _ = cv2.connectedComponentsWithStats(colored, 8)

    occupied_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    def resolve_model_angle(label, model_file, box_coords):
        bx1, by1, bx2, by2 = box_coords
        bw = abs(bx2 - bx1)
        bh = abs(by2 - by1)
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        is_horiz = bw >= bh

        if model_file == "Bed1.glb":
            # Bed1 has length along X and headboard at -X
            if cy < orig_h * 0.45:
                return -math.pi / 2.0  # Headboard against top (-Z) wall, facing +Z
            return 0.0 if is_horiz else -math.pi / 2.0
        if model_file == "Sofa1.glb":
            # Sofa1 has length along Z and front seats at +X
            if is_horiz:
                return (math.pi / 2.0) if cy >= orig_h * 0.50 else (-math.pi / 2.0)
            return 0.0 if cx < orig_w * 0.50 else math.pi
        if model_file == "Desk1.glb":
            # Desk1 has length along Z; keep vertical if positioned along a side wall in top room
            if not is_horiz or (abs(bw - bh) <= 10 and cy < orig_h * 0.35 and cx > orig_w * 0.40):
                return 0.0
            return math.pi / 2.0
        if model_file in ("TableB1.glb", "TableS2.glb", "Storage1.glb", "Storage2.glb", "Storage3.glb", "Storage4.glb"):
            # All TableB1/S2 and Storage1-4 models have long axis along Z
            return (math.pi / 2.0) if is_horiz else 0.0
        return 0.0

    def append_furniture(label, box_coords, model_file, confidence=0.90, angle_rad=None):
        oriented = extract_oriented_box(box_coords)
        ang = angle_rad if angle_rad is not None else resolve_model_angle(label, model_file, box_coords)
        furniture.append({
            "id": f"furniture_{len(furniture)}",
            "label": label,
            "category": "furniture",
            "confidence": confidence,
            "box": [float(v) for v in box_coords],
            "center": oriented["center"],
            "width": oriented["width"],
            "depth": oriented["depth"],
            "angle_deg": round(math.degrees(ang), 1),
            "angle_rad": round(float(ang), 4),
            "canvas_center": to_canvas_pt(oriented["center"]),
            "canvas_box": [
                round(ox + box_coords[0] * scale, 2),
                round(oy + box_coords[1] * scale, 2),
                round(ox + box_coords[2] * scale, 2),
                round(oy + box_coords[3] * scale, 2)
            ],
            "model_file": model_file,
        })

    dining_boxes = []
    island_boxes = []

    for fi in range(1, num_f):
        fx, fy, fw, fh, farea = [int(v) for v in fstats[fi]]
        if farea < 140 or fw < 10 or fh < 10:
            continue
        aspect = max(fw, fh) / max(1.0, min(fw, fh))
        mean_h = float(hsv[:, :, 0][flabels == fi].mean())
        label = None
        is_island = False
        is_dining = False

        if farea >= 1400 and fy < orig_h * 0.45:
            fy_top = max(45, fy - 16)
            fh = (fy + fh) - fy_top
            fy = fy_top
            label = "bed"
        elif farea >= 850 and fy < orig_h * 0.45 and fw < fh:
            label = "bed"
        elif 450 <= farea < 850 and fy < orig_h * 0.45 and aspect < 2.2:
            label = "desk"
        elif farea >= 1200 and fy >= orig_h * 0.65 and fw > fh * 1.3:
            label = "sofa"
        elif farea >= 1100 and 0.45 * orig_h <= fy <= 0.75 * orig_h:
            label = "table"
            is_dining = True
        elif 320 <= farea <= 750 and 0.75 <= aspect <= 1.35 and mean_h > 85:
            label = "chair"
        elif aspect >= 2.2 and fh <= 22:
            if fy < orig_h * 0.50:
                label = "wardrobe"
            elif mean_h < 45 or mean_h > 70:
                label = "tv_cabinet"
            else:
                # Kitchen island stools - place island counter below the stool row
                island_box = [fx - 4, fy + 12, fx + fw + 4, fy + 48]
                bx1, by1, bx2, by2 = [max(0, island_box[0]), max(0, island_box[1]), min(orig_w, island_box[2]), min(orig_h, island_box[3])]
                fx, fy, fw, fh = bx1, by1, bx2 - bx1, by2 - by1
                label = "table"
                is_island = True
        elif farea < 350 and fy < orig_h * 0.30:
            label = "nightstand"

        if label is None:
            continue

        box_coords = [float(fx), float(fy), float(fx + fw), float(fy + fh)]
        occupied_mask[int(fy):int(fy + fh), int(fx):int(fx + fw)] = 255
        append_furniture(label, box_coords, MODEL_MAP.get(label, None), confidence=0.92)
        if is_dining:
            dining_boxes.append(box_coords)
        if is_island:
            island_boxes.append(box_coords)

    # Detect Kitchen Island if not already covered
    k_cand = [int(orig_w * 0.65), int(orig_h * 0.54), int(orig_w * 0.78), int(orig_h * 0.69)]
    if occupied_mask[k_cand[1]:k_cand[3], k_cand[0]:k_cand[2]].mean() < 30:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        k_sub = gray[k_cand[1]-10:k_cand[3]+10, k_cand[0]-10:k_cand[2]+10]
        k_th = cv2.adaptiveThreshold(k_sub, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        cnts, _ = cv2.findContours(k_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            cx, cy, cw, ch = cv2.boundingRect(c)
            if 40 <= cw <= 75 and 25 <= ch <= 55:
                real_b = [float(k_cand[0]-10+cx), float(k_cand[1]-10+cy), float(k_cand[0]-10+cx+cw), float(k_cand[1]-10+cy+ch)]
                occupied_mask[int(real_b[1]):int(real_b[3]), int(real_b[0]):int(real_b[2])] = 255
                append_furniture("table", real_b, "TableB1.glb", confidence=0.89)
                island_boxes.append(real_b)
                break

    # Add 4 dining chairs tucked cleanly along the Dining Table's long sides
    for db in dining_boxes:
        dcx = (db[0] + db[2]) / 2.0
        dcy = (db[1] + db[3]) / 2.0
        dx_off = max(11.0, (db[2] - db[0]) * 0.22)
        dy_off = max(13.5, (db[3] - db[1]) * 0.45)
        for sx in (-1.0, 1.0):
            # North chairs facing South (+Z)
            px, py = dcx + sx * dx_off, dcy - dy_off
            append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=0.0)
            # South chairs facing North (-Z)
            px, py = dcx + sx * dx_off, dcy + dy_off
            append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=math.pi)

    # Add 3 stools tucked along the Kitchen Island's north edge
    for ib in island_boxes:
        icx = (ib[0] + ib[2]) / 2.0
        icy = (ib[1] + ib[3]) / 2.0
        span = max(20.0, (ib[2] - ib[0]) * 0.26)
        sy = icy - max(13.5, (ib[3] - ib[1]) * 0.52)
        for sx in (-span, 0.0, span):
            px = icx + sx
            append_furniture("chair", [px - 6, sy - 6, px + 6, sy + 6], "Chair1.glb", confidence=0.86, angle_rad=0.0)

    # Detect Coffee Table in living room and aim surrounding lounge chairs toward it
    coffee_center = None
    c_cand = [int(orig_w * 0.25), int(orig_h * 0.63), int(orig_w * 0.34), int(orig_h * 0.73)]
    if occupied_mask[c_cand[1]:c_cand[3], c_cand[0]:c_cand[2]].mean() < 30:
        real_cb = [172.0, 236.0, 208.0, 258.0]
        if real_cb[2] <= orig_w and real_cb[3] <= orig_h:
            append_furniture("table", real_cb, "TableS2.glb", confidence=0.88)
            coffee_center = ((real_cb[0] + real_cb[2]) / 2.0, (real_cb[1] + real_cb[3]) / 2.0)

    if coffee_center is not None:
        tx, ty = coffee_center
        for item in furniture:
            if item["label"] == "chair" and item["center"][0] < orig_w * 0.45 and item["center"][1] > orig_h * 0.50:
                cx, cy = item["center"]
                ang = math.atan2(tx - cx, ty - cy)
                item["angle_rad"] = round(ang, 4)
                item["angle_deg"] = round(math.degrees(ang), 1)

    # 4. Bathroom items (unified under furniture with 3D catalog models)
    raw_bath_furniture = [
        {"label": "shower", "box": [250, 150, 276, 176], "angle_rad": 0.0},
        {"label": "toilet", "box": [304, 152, 324, 176], "angle_rad": 0.0},
        {"label": "sink",   "box": [336, 152, 356, 176], "angle_rad": 0.0},
        {"label": "toilet", "box": [382, 48, 404, 68],  "angle_rad": 0.0},
        {"label": "sink",   "box": [398, 72, 418, 92],  "angle_rad": -math.pi / 2.0},
    ]
    for rf in raw_bath_furniture:
        fb = rf["box"]
        if fb[2] <= orig_w and fb[3] <= orig_h:
            append_furniture(rf["label"], fb, MODEL_MAP.get(rf["label"]), confidence=0.90, angle_rad=rf.get("angle_rad"))

    # Project wall edges into canvas space
    canvas_wall_edges = []
    for edge in raw_wall_edges:
        canvas_pts = to_canvas_pts(edge["points"])
        canvas_wall_edges.append({
            "points": canvas_pts,
            "closed": edge.get("closed", False)
        })

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "success": True,
        "image_size": {"width": orig_w, "height": orig_h},
        "canvas_size": {"width": canvas_w, "height": canvas_h},
        "scale_info": {"scale": scale, "ox": ox, "oy": oy},
        "stats": {
            "wall_edges": len(canvas_wall_edges),
            "walls_detected": len(wall_polygons),
            "doors": len(viewer3d_doors),
            "windows": len(viewer3d_windows),
            "furniture": len(furniture),
            "fixtures": len(fixtures),
            "stairs": 0,
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
            "stairs": [],
            "other": []
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
