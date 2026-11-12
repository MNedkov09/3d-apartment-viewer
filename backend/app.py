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

def _nearest_wall_dir(box, wall_mask, orig_w, orig_h):
    """Returns 'N', 'S', 'W', or 'E' indicating which wall edge is closest to box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pad = max(14, int(min(orig_w, orig_h) * 0.035))
    scores = {
        "N": float((wall_mask[max(0, y1 - pad):max(0, y1), max(0, x1):min(orig_w, x2)] > 0).mean()) if y1 > 0 else 0.0,
        "S": float((wall_mask[min(orig_h, y2):min(orig_h, y2 + pad), max(0, x1):min(orig_w, x2)] > 0).mean()) if y2 < orig_h else 0.0,
        "W": float((wall_mask[max(0, y1):min(orig_h, y2), max(0, x1 - pad):max(0, x1)] > 0).mean()) if x1 > 0 else 0.0,
        "E": float((wall_mask[max(0, y1):min(orig_h, y2), min(orig_w, x2):min(orig_w, x2 + pad)] > 0).mean()) if x2 < orig_w else 0.0,
    }
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _wall_facing_yaw(model_file, box_coords, wall_dir):
    """Computes Babylon Y-rotation so furniture backs the nearest wall and faces into the room."""
    bx1, by1, bx2, by2 = box_coords
    is_horiz = (bx2 - bx1) >= (by2 - by1)

    if model_file == "Bed1.glb":
        # Bed1 headboard sits at local -X
        return {"N": -math.pi / 2.0, "S": math.pi / 2.0, "W": 0.0, "E": math.pi}.get(
            wall_dir, 0.0 if is_horiz else -math.pi / 2.0
        )
    if model_file == "Sofa1.glb":
        # Sofa1 backrest sits at local -X, seats face +X
        if is_horiz:
            return (math.pi / 2.0) if wall_dir == "S" else (-math.pi / 2.0)
        return math.pi if wall_dir == "E" else 0.0
    if model_file in ("Toilet1.glb", "Sink1.glb", "Shower1.glb"):
        # Bathroom GLBs face +Z at yaw=0
        return {"N": 0.0, "S": math.pi, "W": math.pi / 2.0, "E": -math.pi / 2.0}.get(wall_dir, 0.0)
    if model_file in ("Desk1.glb", "TableB1.glb", "TableS2.glb", "Storage1.glb", "Storage2.glb", "Storage3.glb", "Storage4.glb"):
        # Long axis aligns along Z at yaw=0 and X at yaw=pi/2
        return (math.pi / 2.0) if is_horiz else 0.0
    return 0.0


def _boxes_overlap_ratio(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(area_a, area_b)


def _map_yolo_furniture_label(raw_name):
    """Maps CubiCasa fixed-furniture class names into catalog furniture labels."""
    n = raw_name.lower()
    if "toilet" in n:
        return "toilet"
    if "sink" in n or "basin" in n:
        return "sink"
    if "shower" in n or "bathtub" in n or "tub" in n or "jacuzzi" in n:
        return "shower"
    if "closet" in n or "cabinet" in n:
        return "wardrobe"
    if "counter" in n or "table" in n:
        return "table"
    return None


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
        if area < 40:
            continue
        ys, xs = np.where(m > 0)
        bw_m = int(xs.max() - xs.min() + 1)
        bh_m = int(ys.max() - ys.min() + 1)
        # Reject only low-confidence ultra-thin decorative/furniture strokes near colored areas
        if sc < 0.78 and min(bw_m, bh_m) <= 4:
            dil = cv2.dilate(m, np.ones((11, 11), np.uint8))
            colored_ratio = float((hsv[:, :, 1][dil > 0] > 18).mean())
            if colored_ratio >= 0.08:
                continue
        # Reject green leafy landscaping blobs misclassified as wall segments
        b_m = float(img[:, :, 0][m > 0].mean())
        g_m = float(img[:, :, 1][m > 0].mean())
        r_m = float(img[:, :, 2][m > 0].mean())
        if g_m > max(r_m, b_m) + 10 and area > 200:
            continue
        wall_mask[m > 0] = 255
        wall_polygons.append({
            "id": f"wall_{idx}",
            "label": "wall",
            "category": "wall",
            "confidence": round(sc, 3),
            "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        })

    # Clean tiny disconnected fragments from wall_mask
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, 8)
    clean_wall_mask = np.zeros_like(wall_mask)
    for l_idx in range(1, num_l):
        if stats[l_idx, cv2.CC_STAT_AREA] >= 55:
            clean_wall_mask[labels == l_idx] = 255

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

    # 2. Windows, Doors & Architectural Metadata Detection
    y_res = det_yolo.predict(img, conf=max(0.08, min(conf, 0.25)), imgsz=640, verbose=False)[0]
    raw_windows = []
    raw_doors = []
    panel_areas = []
    space_boxes = []
    yolo_furniture_boxes = []

    for box in y_res.boxes:
        cid = int(box.cls[0])
        name = det_yolo.names[cid]
        bconf = float(box.conf[0])
        b = [float(v) for v in box.xyxy[0].tolist()]
        if name in ['glass', 'window', 'window regular', 'window full', 'window sauna', 'sliding window'] and bconf >= 0.10:
            raw_windows.append((bconf, b, name))
        elif name.startswith('door') and bconf >= 0.32:
            raw_doors.append((bconf, b, name))
        elif name == 'panelarea' and bconf >= 0.08:
            panel_areas.append((bconf, b))
        elif name.startswith('space ') and bconf >= 0.08:
            space_boxes.append((name, bconf, b))
        elif bconf >= 0.14:
            mapped_lbl = _map_yolo_furniture_label(name)
            if mapped_lbl is not None:
                yolo_furniture_boxes.append((mapped_lbl, bconf, b))

    # Snap windows onto the nearest colinear wall axis while preserving full window length
    snapped_windows = []
    for bconf, b, name in raw_windows:
        bx1, by1, bx2, by2 = b
        bw, bh = bx2 - bx1, by2 - by1
        cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
        is_h = bw >= bh
        best_coord = None
        best_dist = 18.0
        for sx1, sy1, sx2, sy2, seg_h in wall_segments:
            if seg_h != is_h:
                continue
            if is_h:
                ov = min(bx2, sx2) - max(bx1, sx1)
                dist = abs(cy - sy1)
                if ov > 2 and dist < best_dist:
                    best_dist = dist
                    best_coord = sy1
            else:
                ov = min(by2, sy2) - max(by1, sy1)
                dist = abs(cx - sx1)
                if ov > 2 and dist < best_dist:
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

    # Snap doors onto doorway panel area or adjacent colinear wall segments (even across gaps)
    doors = []
    viewer3d_doors = []
    accepted_door_boxes = []
    for idx, (bconf, b, name) in enumerate(sorted(raw_doors, key=lambda d: -d[0])):
        if any(_boxes_overlap_ratio(b, prev_b) > 0.55 for prev_b in accepted_door_boxes):
            continue
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
            best_score = 1e9
            for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                if seg_h:
                    gap_x = max(0.0, max(bx1 - sx2, sx1 - bx2))
                    if gap_x <= 32.0:
                        d = min(abs(sy1 - by1), abs(sy1 - by2), abs(sy1 - (by1 + by2) / 2.0))
                        score = d + gap_x * 0.35
                        if d <= 16.0 and score < best_score:
                            best_score = score
                            door_seg = ([bx1, sy1], [bx2, sy1], 8.0)
                else:
                    gap_y = max(0.0, max(by1 - sy2, sy1 - by2))
                    if gap_y <= 32.0:
                        d = min(abs(sx1 - bx1), abs(sx1 - bx2), abs(sx1 - (bx1 + bx2) / 2.0))
                        score = d + gap_y * 0.35
                        if d <= 16.0 and score < best_score:
                            best_score = score
                            door_seg = ([sx1, by1], [sx1, by2], 8.0)

        if door_seg is not None:
            accepted_door_boxes.append(b)
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

    # 3. General Interior Room & Furniture/Fixture Detection
    furniture = []
    fixtures = []
    occupied_boxes = []

    def append_furniture(label, box_coords, model_file, confidence=0.90, angle_rad=None):
        oriented = extract_oriented_box(box_coords)
        wall_dir = _nearest_wall_dir(box_coords, clean_wall_mask, orig_w, orig_h)
        ang = angle_rad if angle_rad is not None else _wall_facing_yaw(model_file, box_coords, wall_dir)
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
        occupied_boxes.append([float(v) for v in box_coords])

    # Build enclosed building interior mask by bridging doorways onto walls
    k_bridge = max(25, int(min(orig_w, orig_h) * 0.06) | 1)
    bridge_mask = clean_wall_mask.copy()
    for _, b, _ in raw_doors:
        cv2.rectangle(bridge_mask, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), 255, -1)
    closed_walls = cv2.morphologyEx(bridge_mask, cv2.MORPH_CLOSE, np.ones((k_bridge, k_bridge), np.uint8))
    pad_inv = cv2.copyMakeBorder((closed_walls == 0).astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    ff_mask = np.zeros((orig_h + 4, orig_w + 4), dtype=np.uint8)
    cv2.floodFill(pad_inv, ff_mask, (0, 0), 128)
    interior_mask = (pad_inv[1:-1, 1:-1] == 255) & (clean_wall_mask == 0)

    # Fallback if walls do not form any closed perimeter (rare partial crop)
    if int(interior_mask.sum()) < int(orig_w * orig_h * 0.04) and wall_polygons:
        wx1 = min(w["box"][0] for w in wall_polygons)
        wy1 = min(w["box"][1] for w in wall_polygons)
        wx2 = max(w["box"][2] for w in wall_polygons)
        wy2 = max(w["box"][3] for w in wall_polygons)
        interior_mask[wy1:wy2, wx1:wx2] = True
        interior_mask[clean_wall_mask > 0] = False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wall_dil = cv2.dilate(clean_wall_mask, np.ones((7, 7), np.uint8))
    inv_rooms = (interior_mask & (wall_dil == 0)).astype(np.uint8) * 255
    nr, rlab, rstats, _ = cv2.connectedComponentsWithStats(inv_rooms, 8)
    unit_scale = (800.0 * 600.0) / max(1.0, float(orig_w * orig_h))
    len_scale = math.sqrt(unit_scale)

    valid_rooms = []
    for ri in range(1, nr):
        rx, ry, rw, rh, rarea = [int(v) for v in rstats[ri]]
        norm_room_area = rarea * unit_scale
        if norm_room_area < 500:
            continue
        rmask = rlab == ri
        b_m = float(img[:, :, 0][rmask].mean())
        g_m = float(img[:, :, 1][rmask].mean())
        r_m = float(img[:, :, 2][rmask].mean())
        s_m = float(hsv[:, :, 1][rmask].mean())
        v_p30 = float(np.percentile(gray[rmask], 30))

        # Exclude outdoor grass yards / wood decks
        is_grass = g_m > max(r_m, b_m) + 6 and s_m >= 18
        is_deck = r_m > b_m + 35 and s_m >= 50
        overlaps_garage = False
        overlaps_bath = False
        overlaps_bed = False
        for sp_nm, sp_cf, sb in space_boxes:
            ix = max(0.0, min(rx + rw, sb[2]) - max(rx, sb[0]))
            iy = max(0.0, min(ry + rh, sb[3]) - max(ry, sb[1]))
            ov_r = (ix * iy) / max(1.0, rw * rh)
            ov_s = (ix * iy) / max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
            if max(ov_r, ov_s) >= 0.45:
                if "garage" in sp_nm and norm_room_area > 18000:
                    overlaps_garage = True
                if "bath" in sp_nm and norm_room_area < 15000:
                    overlaps_bath = True
                if "bedroom" in sp_nm and 6000 <= norm_room_area <= 36000:
                    overlaps_bed = True
        if is_grass or is_deck or overlaps_garage:
            continue

        room_kind = "living_kitchen"
        if overlaps_bath or (b_m > r_m + 14 and s_m >= 18 and norm_room_area < 16000) or norm_room_area < 6000:
            room_kind = "bath"
        elif overlaps_bed or (6000 <= norm_room_area <= 35000):
            room_kind = "bedroom"

        valid_rooms.append({
            "id": ri,
            "mask": rmask,
            "box": [rx, ry, rx + rw, ry + rh],
            "norm_area": norm_room_area,
            "mean_s": s_m,
            "v_p30": v_p30,
            "kind": room_kind,
        })

    # Collect per-room candidate boxes from Track C (YOLO), Track B (bright contrast), Track A (colored)
    raw_candidates = []

    # Track C: Explicit YOLO fixed-furniture inside interior rooms
    interior_union = np.zeros((orig_h, orig_w), dtype=bool)
    for rm in valid_rooms:
        interior_union |= rm["mask"]
    for lbl, bconf, b in yolo_furniture_boxes:
        cx_i = min(orig_w - 1, max(0, int((b[0] + b[2]) / 2.0)))
        cy_i = min(orig_h - 1, max(0, int((b[1] + b[3]) / 2.0)))
        if not interior_union[cy_i, cx_i]:
            continue
        room_kind = "living_kitchen"
        for rm in valid_rooms:
            if rm["mask"][cy_i, cx_i]:
                room_kind = rm["kind"]
                break
        bw_b = max(1.0, b[2] - b[0])
        bh_b = max(1.0, b[3] - b[1])
        raw_candidates.append(("yolo", b, (bw_b * bh_b) * unit_scale, lbl, room_kind, bconf))

    for rm in valid_rooms:
        rmask = rm["mask"]
        room_kind = rm["kind"]
        v_p30 = rm["v_p30"]
        s_m = rm["mean_s"]
        room_cands = []
        is_vehicle_garage = False

        # Track B: White / high-contrast furniture shapes inside shaded/tinted rooms
        rm_w = max(1, rm["box"][2] - rm["box"][0])
        rm_h = max(1, rm["box"][3] - rm["box"][1])
        if v_p30 <= 243:
            thr = max(242, int(v_p30) + 10)
            br_fg = ((gray >= thr) & (hsv[:, :, 1] < 26) & rmask).astype(np.uint8) * 255
            dark_edge = (gray < 140) & rmask
            br_fg[dark_edge] = 0
            br_fg = cv2.morphologyEx(br_fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            br_fg[dark_edge] = 0
            nb, blab, bstats, _ = cv2.connectedComponentsWithStats(br_fg, 8)
            for bi in range(1, nb):
                fx, fy, fw, fh, farea = [int(v) for v in bstats[bi]]
                na = farea * unit_scale
                nw, nh = fw * len_scale, fh * len_scale
                extent = na / max(1.0, nw * nh)
                fills_whole_room = fw >= rm_w * 0.88 and fh >= rm_h * 0.88
                if fills_whole_room:
                    continue
                if rm["norm_area"] >= 24000 and 8500 <= na <= 28000 and extent >= 0.45:
                    is_vehicle_garage = True
                    break
                if 150 <= na <= 7500 and nw >= 10 and nh >= 10 and extent >= 0.42:
                    box_b = [float(fx), float(fy), float(fx + fw), float(fy + fh)]
                    room_cands.append(("bright", box_b, na, None, room_kind, 0.90))

        if is_vehicle_garage:
            continue

        # Track A: Saturated colored furniture inside neutral/light floors
        if s_m < 16:
            col_fg = ((hsv[:, :, 1] >= 10) & (hsv[:, :, 2] > 95) & rmask).astype(np.uint8) * 255
            col_fg = cv2.morphologyEx(col_fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            nc, clab, cstats, _ = cv2.connectedComponentsWithStats(col_fg, 8)
            for ci in range(1, nc):
                fx, fy, fw, fh, farea = [int(v) for v in cstats[ci]]
                na = farea * unit_scale
                nw, nh = fw * len_scale, fh * len_scale
                if not (140 <= na <= 7500 and nw >= 8 and nh >= 8):
                    continue
                comp_mask = clab == ci
                b_c = float(img[:, :, 0][comp_mask].mean())
                g_c = float(img[:, :, 1][comp_mask].mean())
                r_c = float(img[:, :, 2][comp_mask].mean())
                # Skip small green decorative houseplants
                if g_c > max(r_c, b_c) + 12 and na < 650:
                    continue
                box_c = [float(fx), float(fy), float(fx + fw), float(fy + fh)]
                room_cands.append(("color", box_c, na, None, room_kind, 0.88))

        raw_candidates.extend(room_cands)

    # Deduplicate overlapping candidates (YOLO > bright on shaded rooms > color on neutral rooms)
    priority_order = {"yolo": 0, "bright": 1, "color": 2}
    raw_candidates.sort(key=lambda c: (priority_order.get(c[0], 9), -c[2]))

    filtered_candidates = []
    for src, box, na, hint_lbl, room_kind, conf_val in raw_candidates:
        if any(_boxes_overlap_ratio(box, pb[1]) > 0.35 for pb in filtered_candidates):
            continue
        filtered_candidates.append((src, box, na, hint_lbl, room_kind, conf_val))

    dining_boxes = []
    sofa_boxes = []
    preliminary_tables = []

    def _wall_contact_ratio(box_c):
        x1, y1, x2, y2 = [int(round(v)) for v in box_c]
        pad = 6
        edge_vals = [
            float((clean_wall_mask[max(0, y1 - pad):max(0, y1), max(0, x1):min(orig_w, x2)] > 0).mean()) if y1 > 0 else 0.0,
            float((clean_wall_mask[min(orig_h, y2):min(orig_h, y2 + pad), max(0, x1):min(orig_w, x2)] > 0).mean()) if y2 < orig_h else 0.0,
            float((clean_wall_mask[max(0, y1):min(orig_h, y2), max(0, x1 - pad):max(0, x1)] > 0).mean()) if x1 > 0 else 0.0,
            float((clean_wall_mask[max(0, y1):min(orig_h, y2), min(orig_w, x2):min(orig_w, x2 + pad)] > 0).mean()) if x2 < orig_w else 0.0,
        ]
        return max(edge_vals)

    for src, box, na, hint_lbl, room_kind, conf_val in filtered_candidates:
        bx1, by1, bx2, by2 = box
        nw = max(1.0, (bx2 - bx1) * len_scale)
        nh = max(1.0, (by2 - by1) * len_scale)
        aspect = max(nw, nh) / max(1.0, min(nw, nh))
        label = hint_lbl
        model_override = None

        if label is None:
            if room_kind == "bath":
                if na >= 700 and aspect <= 1.65:
                    label = "shower"
                elif na < 700 and aspect >= 1.35:
                    label = "toilet"
                elif na < 700:
                    label = "sink"
            elif room_kind == "bedroom":
                if na >= 1650 and aspect <= 1.95 and min(nw, nh) >= 28:
                    label = "bed"
                elif na < 450 and aspect < 1.85:
                    label = "nightstand"
                elif aspect >= 2.1 and min(nw, nh) <= 26 and na >= 450:
                    label = "wardrobe"
                elif 450 <= na < 1650 and aspect < 2.2:
                    label = "desk"
            else:
                # Living / Dining / Kitchen / Hallway
                if aspect >= 2.2 and min(nw, nh) <= 25 and na >= 420:
                    label = "tv_cabinet" if na >= 550 else "wardrobe"
                elif na >= 1350 and aspect >= 1.45 and min(nw, nh) <= 44:
                    label = "sofa"
                elif na >= 1100 and min(nw, nh) >= 26:
                    if aspect >= 1.6 and _wall_contact_ratio(box) >= 0.30:
                        label = "high_cabinet"
                    else:
                        label = "table"
                        preliminary_tables.append((box, na, aspect))
                elif 240 <= na <= 1150 and aspect <= 1.55:
                    label = "chair"

        if label is None:
            continue

        model_file = model_override or MODEL_MAP.get(label)
        if not model_file:
            continue

        if label == "sofa":
            sofa_boxes.append(box)
        if label == "table":
            # Coffee table vs Dining table refinement
            if na <= 1900 and aspect <= 1.6:
                model_file = "TableS2.glb"
            else:
                model_file = "TableB1.glb"
                dining_boxes.append(box)

        append_furniture(label, box, model_file, confidence=round(conf_val, 2))

    # Add 4 dining chairs cleanly along Dining Table long sides
    for db in dining_boxes:
        dcx = (db[0] + db[2]) / 2.0
        dcy = (db[1] + db[3]) / 2.0
        dw = db[2] - db[0]
        dh = db[3] - db[1]
        if dw >= dh:
            dx_off = max(11.0, dw * 0.22)
            dy_off = max(13.5, dh * 0.52)
            for sx in (-1.0, 1.0):
                px, py = dcx + sx * dx_off, dcy - dy_off
                append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=0.0)
                px, py = dcx + sx * dx_off, dcy + dy_off
                append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=math.pi)
        else:
            dx_off = max(13.5, dw * 0.52)
            dy_off = max(11.0, dh * 0.22)
            for sy in (-1.0, 1.0):
                px, py = dcx - dx_off, dcy + sy * dy_off
                append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=math.pi / 2.0)
                px, py = dcx + dx_off, dcy + sy * dy_off
                append_furniture("chair", [px - 6, py - 6, px + 6, py + 6], "Chair1.glb", confidence=0.86, angle_rad=-math.pi / 2.0)

    # Aim any free-standing lounge chairs toward the nearest coffee/living table
    coffee_tables = [f for f in furniture if f.get("model_file") == "TableS2.glb"]
    if coffee_tables:
        for item in furniture:
            if item["label"] == "chair" and not any(_boxes_overlap_ratio(item["box"], db) > 0 for db in dining_boxes):
                cx, cy = item["center"]
                nearest_t = min(coffee_tables, key=lambda t: math.hypot(t["center"][0] - cx, t["center"][1] - cy))
                tx, ty = nearest_t["center"]
                if math.hypot(tx - cx, ty - cy) <= max(orig_w, orig_h) * 0.22:
                    ang = math.atan2(tx - cx, ty - cy)
                    item["angle_rad"] = round(ang, 4)
                    item["angle_deg"] = round(math.degrees(ang), 1)

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
