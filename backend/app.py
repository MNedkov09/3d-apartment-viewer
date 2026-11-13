import re
import time
import math
from collections import deque
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    CLIPProcessor,
    CLIPModel,
)
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from rapidocr_onnxruntime import RapidOCR
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
    "coffee_table": "TableS2.glb",
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

CLIP_PROMPTS = [
    ("bed", "top-down floorplan symbol of a bedroom bed with two pillows and duvet"),
    ("sofa", "top-down floorplan symbol of an upholstered sofa couch with back cushions and armrests"),
    ("dining_table", "top-down floorplan symbol of a rectangular dining table surrounded by dining chairs"),
    ("coffee_table", "top-down floorplan symbol of a plain rectangular or round center coffee table"),
    ("kitchen_counter", "top-down floorplan symbol of built-in U-shaped or L-shaped kitchen cabinets with stove burners and sink"),
    ("desk", "top-down floorplan symbol of a study office desk and chair against a wall"),
    ("chair", "top-down floorplan symbol of a single armchair or single dining chair"),
    ("cabinet", "top-down floorplan symbol of a narrow rectangular wardrobe closet or long thin TV console cabinet"),
    ("nightstand", "top-down floorplan symbol of a small square bedside nightstand table"),
    ("toilet", "top-down floorplan symbol of a small bathroom toilet with tank and bowl"),
    ("sink", "top-down floorplan symbol of a bathroom vanity cabinet with oval washbasin sink"),
    ("shower", "top-down floorplan symbol of a corner bathtub oval or square walk-in shower stall"),
    ("plant_ignore", "top-down floorplan drawing of a green potted plant or leaves"),
    ("noise_ignore", "top-down floorplan drawing of stairs steps, empty hallway, doorway swing, or wall border"),
]

ROOM_PATTERNS = [
    ("kitchen_dining", re.compile(r"(kitchen|dining|dinette|breakfast|scullery)", re.I)),
    ("living", re.compile(r"(living\s*room|livingroom|lounge|family\s*room|sitting\s*room|tv\s*room|\bliving)", re.I)),
    ("bedroom", re.compile(r"(master\s*bed|bed\s*room|bedroom|guest\s*room)", re.I)),
    ("bath", re.compile(r"(bath\s*room|bathroom|bath\d|\bbath\b|\bwc\b|restroom|ensuite|powder)", re.I)),
    ("laundry", re.compile(r"(laundry|utility|mudroom)", re.I)),
    ("closet", re.compile(r"(\bcloset\b|\bwic\b|walk-?in\s*closet|\bpantry\b|\bcl\b)", re.I)),
    ("hall", re.compile(r"(hallway|\bhall\b|\bentry\b|\bfoyer\b|corridor|lobby|vestibule)", re.I)),
    ("garage", re.compile(r"(garage|carport)", re.I)),
    ("outdoor", re.compile(r"(terrace|patio|balcony|garden|deck|porch|driveway|yard)", re.I)),
]

FURN_LABEL_PATTERNS = [
    ("sofa", re.compile(r"^(sofa|couch|settee)$", re.I)),
    ("bed", re.compile(r"^(bed|day\s*bed|single\s*bed|double\s*bed)$", re.I)),
    ("dining_table", re.compile(r"^(table|dining\s*table)$", re.I)),
    ("tv_cabinet", re.compile(r"^(tv\s*cab|tv\s*cabinet|tv\s*unit|console)$", re.I)),
    ("wardrobe", re.compile(r"^(wardrobe|closet\s*cab|cupboard)$", re.I)),
    ("cabinet", re.compile(r"^(cabinet)$", re.I)),
    ("desk", re.compile(r"^(desk|study\s*table|work\s*table)$", re.I)),
    ("nightstand", re.compile(r"^(night|nightstand|bedside)$", re.I)),
    ("sink", re.compile(r"^(sink|vanity|basin)$", re.I)),
    ("toilet", re.compile(r"^(wc|toilet)$", re.I)),
]

m2f_processor = None
m2f_model = None
cubi_yolo = None
clip_processor = None
clip_model = None
clip_text_features = None
clip_logit_scale = 100.0
ocr_reader = None


def get_models():
    global m2f_processor, m2f_model, cubi_yolo
    global clip_processor, clip_model, clip_text_features, clip_logit_scale, ocr_reader

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
    if clip_model is None:
        print("Loading visual symbol classifier...")
        clip_id = "openai/clip-vit-base-patch32"
        clip_processor = CLIPProcessor.from_pretrained(clip_id)
        clip_model = CLIPModel.from_pretrained(clip_id)
        clip_model.eval()
        clip_logit_scale = float(clip_model.logit_scale.exp().item())
        with torch.no_grad():
            t_in = clip_processor(text=[p[1] for p in CLIP_PROMPTS], return_tensors="pt", padding=True)
            t_ft = clip_model.get_text_features(**t_in)
            if not isinstance(t_ft, torch.Tensor):
                t_ft = t_ft.pooler_output
            clip_text_features = t_ft / t_ft.norm(dim=-1, keepdim=True)
        print("Visual symbol classifier ready!")
    if ocr_reader is None:
        ocr_reader = RapidOCR()
    return m2f_processor, m2f_model, cubi_yolo, clip_processor, clip_model, clip_text_features, clip_logit_scale, ocr_reader


def _classify_ocr_text(txt):
    """Classifies OCR string into a structural room label or printed furniture symbol."""
    t = re.sub(r"[^a-zA-Z0-9\s/]+", " ", txt).strip()
    for ftype, pat in FURN_LABEL_PATTERNS:
        if pat.search(t):
            return ("furn", ftype, t)
    for rtype, pat in ROOM_PATTERNS:
        if pat.search(t):
            return ("room", rtype, t)
    return (None, None, t)


def _subdivide_room_geodesic(rmask, anchors):
    """Partitions open-plan multi-label rooms using downsampled BFS geodesic distance around walls."""
    h, w = rmask.shape
    if not anchors:
        return [("unknown", rmask, "")]
    if len(anchors) == 1:
        return [(anchors[0][0], rmask, anchors[0][1])]
    step = 4
    sh, sw = max(1, (h + step - 1) // step), max(1, (w + step - 1) // step)
    s_mask = cv2.resize(rmask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0
    ys_s, xs_s = np.where(s_mask)
    if len(xs_s) == 0:
        return [(anchors[0][0], rmask, anchors[0][1])]
    zmap = np.full((sh, sw), -1, dtype=np.int32)
    q = deque()
    for idx, (_rtype, _clean_t, (cx, cy), _) in enumerate(anchors):
        scx = min(sw - 1, max(0, cx // step))
        scy = min(sh - 1, max(0, cy // step))
        if not s_mask[scy, scx]:
            j = int(np.argmin((xs_s - scx) ** 2 + (ys_s - scy) ** 2))
            scy, scx = int(ys_s[j]), int(xs_s[j])
        if zmap[scy, scx] == -1:
            zmap[scy, scx] = idx
            q.append((scy, scx, idx))
    while q:
        y, x, z = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < sh and 0 <= nx < sw and s_mask[ny, nx] and zmap[ny, nx] == -1:
                zmap[ny, nx] = z
                q.append((ny, nx, z))
    full_z = cv2.resize(zmap.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return [
        (anchors[i][0], rmask & (full_z == i), anchors[i][1])
        for i in range(len(anchors))
        if int((rmask & (full_z == i)).sum()) > 200
    ]


def _wall_contact_sides(box_c, wall_mask, orig_h, orig_w, pad=6):
    """Returns count of wall-adjacent sides and per-side wall contact fractions."""
    x1, y1, x2, y2 = [int(round(v)) for v in box_c]
    s = {
        "N": float((wall_mask[max(0, y1 - pad):max(0, y1), max(0, x1):min(orig_w, x2)] > 0).mean()) if y1 > 0 else 0.0,
        "S": float((wall_mask[min(orig_h, y2):min(orig_h, y2 + pad), max(0, x1):min(orig_w, x2)] > 0).mean()) if y2 < orig_h else 0.0,
        "W": float((wall_mask[max(0, y1):min(orig_h, y2), max(0, x1 - pad):max(0, x1)] > 0).mean()) if x1 > 0 else 0.0,
        "E": float((wall_mask[max(0, y1):min(orig_h, y2), min(orig_w, x2):min(orig_w, x2 + pad)] > 0).mean()) if x2 < orig_w else 0.0,
    }
    return sum(1 for v in s.values() if v >= 0.35), s


def _nearest_wall_dir(box, wall_mask, orig_w, orig_h, model_file=None):
    """Returns 'N', 'S', 'W', or 'E' indicating which wall edge is closest to box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    pad = max(14, int(min(orig_w, orig_h) * 0.035))
    scores = {
        "N": float((wall_mask[max(0, y1 - pad):max(0, y1), max(0, x1):min(orig_w, x2)] > 0).mean()) if y1 > 0 else 0.0,
        "S": float((wall_mask[min(orig_h, y2):min(orig_h, y2 + pad), max(0, x1):min(orig_w, x2)] > 0).mean()) if y2 < orig_h else 0.0,
        "W": float((wall_mask[max(0, y1):min(orig_h, y2), max(0, x1 - pad):max(0, x1)] > 0).mean()) if x1 > 0 else 0.0,
        "E": float((wall_mask[max(0, y1):min(orig_h, y2), min(orig_w, x2):min(orig_w, x2 + pad)] > 0).mean()) if x2 < orig_w else 0.0,
    }
    # For non-square beds, headboard rests against one of the short ends perpendicular to length
    if model_file == "Bed1.glb":
        bw_b = max(1, x2 - x1)
        bh_b = max(1, y2 - y1)
        if bh_b >= bw_b * 1.28 and max(scores["N"], scores["S"]) >= 0.25:
            scores = {"N": scores["N"], "S": scores["S"]}
        elif bw_b >= bh_b * 1.28 and max(scores["W"], scores["E"]) >= 0.25:
            scores = {"W": scores["W"], "E": scores["E"]}
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

    processor, wall_net, det_yolo, c_proc, c_mod, txt_feat, lscale, ocr_eng = get_models()

    # 0. Extract printed room/furniture labels & mask text so labels never become false objects
    ocr_res, _ = ocr_eng(img, use_cls=False)
    text_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    room_anchors = []
    furn_anchors = []
    for box, txt, _cf in (ocr_res or []):
        xs_o = [int(q[0]) for q in box]
        ys_o = [int(q[1]) for q in box]
        x1_o = max(0, min(xs_o) - 2)
        y1_o = max(0, min(ys_o) - 2)
        x2_o = min(orig_w, max(xs_o) + 2)
        y2_o = min(orig_h, max(ys_o) + 2)
        text_mask[y1_o:y2_o, x1_o:x2_o] = 255
        kind, lbl, clean_t = _classify_ocr_text(txt)
        cx_o, cy_o = (x1_o + x2_o) // 2, (y1_o + y2_o) // 2
        if kind == "room":
            room_anchors.append((lbl, clean_t, (cx_o, cy_o), [x1_o, y1_o, x2_o, y2_o]))
        elif kind == "furn":
            furn_anchors.append((lbl, clean_t, (cx_o, cy_o), [x1_o, y1_o, x2_o, y2_o]))

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

    # Clean tiny fragments and detached exterior logo/watermark wall islands
    num_l, labels, stats, _ = cv2.connectedComponentsWithStats(wall_mask, 8)
    clean_wall_mask = np.zeros_like(wall_mask)
    if num_l > 1:
        cc_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_l)]
        max_cc_area = max(cc_areas)
        main_cc = 1 + int(np.argmax(cc_areas))
        mx1, my1, mw_c, mh_c = [int(v) for v in stats[main_cc, :4]]
        mx2, my2 = mx1 + mw_c, my1 + mh_c
        for l_idx in range(1, num_l):
            area_c = int(stats[l_idx, cv2.CC_STAT_AREA])
            if area_c < 55:
                continue
            bx_c, by_c, bw_c, bh_c = [int(v) for v in stats[l_idx, :4]]
            gap_c = max(0, mx1 - (bx_c + bw_c), bx_c - mx2, my1 - (by_c + bh_c), by_c - my2)
            if area_c < max_cc_area * 0.12 and gap_c > max(12, min(orig_w, orig_h) * 0.025):
                continue
            if area_c < max_cc_area * 0.08 and float((text_mask[labels == l_idx] > 0).mean()) >= 0.45:
                continue
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
        elif name.startswith('door') and bconf >= 0.38:
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
                    if seg_h and abs(sy1 - py) <= 8:
                        py = sy1
                        break
                door_seg = ([px1, py], [px2, py], max(ph, 6.0))
            else:
                px = (px1 + px2) / 2.0
                for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                    if not seg_h and abs(sx1 - px) <= 8:
                        px = sx1
                        break
                door_seg = ([px, py1], [px, py2], max(pw, 6.0))
        else:
            best_score = 1e9
            for sx1, sy1, sx2, sy2, seg_h in wall_segments:
                if seg_h:
                    gap_x = max(0.0, max(bx1 - sx2, sx1 - bx2))
                    if gap_x <= 14.0:
                        d = min(abs(sy1 - by1), abs(sy1 - by2), abs(sy1 - (by1 + by2) / 2.0))
                        score = d + gap_x * 0.35
                        if d <= 14.0 and score < best_score:
                            best_score = score
                            door_seg = ([bx1, sy1], [bx2, sy1], 8.0)
                else:
                    gap_y = max(0.0, max(by1 - sy2, sy1 - by2))
                    if gap_y <= 14.0:
                        d = min(abs(sx1 - bx1), abs(sx1 - bx2), abs(sx1 - (bx1 + bx2) / 2.0))
                        score = d + gap_y * 0.35
                        if d <= 14.0 and score < best_score:
                            best_score = score
                            door_seg = ([sx1, by1], [sx1, by2], 8.0)

        # Ensure every door actually connects to at least one wall near its endpoints
        if door_seg is not None:
            st_d, en_d, _th_d = door_seg
            ep_pad = max(10, int(min(orig_w, orig_h) * 0.02))
            has_wall_anchor = False
            for ex_d, ey_d in (st_d, en_d):
                xi, yi = int(round(ex_d)), int(round(ey_d))
                patch_w = clean_wall_mask[
                    max(0, yi - ep_pad):min(orig_h, yi + ep_pad + 1),
                    max(0, xi - ep_pad):min(orig_w, xi + ep_pad + 1),
                ]
                if patch_w.size > 0 and np.any(patch_w > 0):
                    has_wall_anchor = True
                    break
            if not has_wall_anchor:
                door_seg = None

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
        wall_dir = _nearest_wall_dir(box_coords, clean_wall_mask, orig_w, orig_h, model_file=model_file)
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

    # Build enclosed building interior mask via thin doorway crossbars
    bridge_mask = clean_wall_mask.copy()
    for _, b, _ in raw_doors:
        dx1, dy1, dx2, dy2 = [int(round(v)) for v in b]
        cv2.line(bridge_mask, (dx1, (dy1 + dy2) // 2), (dx2, (dy1 + dy2) // 2), 255, 2)
        cv2.line(bridge_mask, ((dx1 + dx2) // 2, dy1), ((dx1 + dx2) // 2, dy2), 255, 2)
    k_bridge = max(9, int(min(orig_w, orig_h) * 0.022) | 1)
    closed_walls = cv2.morphologyEx(bridge_mask, cv2.MORPH_CLOSE, np.ones((k_bridge, k_bridge), np.uint8))
    pad_inv = cv2.copyMakeBorder((closed_walls == 0).astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
    ff_mask = np.zeros((orig_h + 4, orig_w + 4), dtype=np.uint8)
    cv2.floodFill(pad_inv, ff_mask, (0, 0), 128)
    interior_mask = (pad_inv[1:-1, 1:-1] == 255) & (clean_wall_mask == 0)

    if int(interior_mask.sum()) < int(orig_w * orig_h * 0.04) and wall_polygons:
        wx1 = min(w["box"][0] for w in wall_polygons)
        wy1 = min(w["box"][1] for w in wall_polygons)
        wx2 = max(w["box"][2] for w in wall_polygons)
        wy2 = max(w["box"][3] for w in wall_polygons)
        interior_mask[wy1:wy2, wx1:wx2] = True
        interior_mask[clean_wall_mask > 0] = False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    wall_dil = cv2.dilate(clean_wall_mask, np.ones((5, 5), np.uint8))
    inv_rooms = (interior_mask & (wall_dil == 0)).astype(np.uint8) * 255
    nr, rlab, rstats, _ = cv2.connectedComponentsWithStats(inv_rooms, 4)
    unit_scale = (800.0 * 600.0) / max(1.0, float(orig_w * orig_h))
    len_scale = math.sqrt(unit_scale)

    zones = []
    for ri in range(1, nr):
        rx, ry, rw, rh, rarea = [int(v) for v in rstats[ri]]
        norm_room_area = rarea * unit_scale
        if norm_room_area < 700:
            continue
        rmask = rlab == ri
        b_m = float(img[:, :, 0][rmask].mean())
        g_m = float(img[:, :, 1][rmask].mean())
        r_m = float(img[:, :, 2][rmask].mean())
        s_m = float(hsv[:, :, 1][rmask].mean())

        if (g_m > max(r_m, b_m) + 6 and s_m >= 18) or (r_m > b_m + 35 and s_m >= 50):
            continue
        overlaps_garage = False
        for sp_nm, _sp_cf, sb in space_boxes:
            ix = max(0.0, min(rx + rw, sb[2]) - max(rx, sb[0]))
            iy = max(0.0, min(ry + rh, sb[3]) - max(ry, sb[1]))
            ov_r = (ix * iy) / max(1.0, rw * rh)
            ov_s = (ix * iy) / max(1.0, (sb[2] - sb[0]) * (sb[3] - sb[1]))
            if max(ov_r, ov_s) >= 0.45 and "garage" in sp_nm and norm_room_area > 18000:
                overlaps_garage = True
                break
        if overlaps_garage:
            continue

        anchors_in = [
            a for a in room_anchors
            if np.any(rmask[max(0, a[2][1] - 8):min(orig_h, a[2][1] + 9), max(0, a[2][0] - 8):min(orig_w, a[2][0] + 9)])
        ]
        if not anchors_in:
            wc_in = [
                ("bath", ft_w, pt_w, fb_w)
                for fl_w, ft_w, pt_w, fb_w in furn_anchors
                if fl_w == "toilet" and re.search(r"^wc$", ft_w, re.I)
                and np.any(rmask[max(0, pt_w[1] - 8):min(orig_h, pt_w[1] + 9), max(0, pt_w[0] - 8):min(orig_w, pt_w[0] + 9)])
            ]
            if wc_in:
                anchors_in = wc_in
        for rtype, zm, lbl_txt in _subdivide_room_geodesic(rmask, anchors_in):
            zys, zxs = np.where(zm)
            zna = len(zxs) * unit_scale
            if zna < 700 or (rtype == "unknown" and zna < 3000):
                continue
            if rtype == "unknown":
                rtype = "bath" if zna < 5500 else ("bedroom" if zna <= 35000 else "living")
            zones.append({
                "id": len(zones),
                "rtype": rtype,
                "mask": zm,
                "label": lbl_txt,
                "na": zna,
                "center": [float(zxs.mean()), float(zys.mean())],
                "box": [int(zxs.min()), int(zys.min()), int(zxs.max()), int(zys.max())],
            })

    # Compute semantic room centers for 3D quick camera presets
    canvas_rooms = {}
    for z in zones:
        rt = z["rtype"]
        if rt == "living" and ("living" not in canvas_rooms or z["na"] > canvas_rooms["living"][1]):
            canvas_rooms["living"] = (to_canvas_pt(z["center"]), z["na"])
        elif rt == "kitchen_dining":
            if "kitchen" in z["label"].lower() or "kitchen" not in canvas_rooms or z["na"] > canvas_rooms["kitchen"][1]:
                canvas_rooms["kitchen"] = (to_canvas_pt(z["center"]), z["na"])
        elif rt == "bath" and ("bath" not in canvas_rooms or z["na"] > canvas_rooms["bath"][1]):
            canvas_rooms["bath"] = (to_canvas_pt(z["center"]), z["na"])
        elif rt == "hall" and ("entry" not in canvas_rooms or z["na"] > canvas_rooms["entry"][1]):
            canvas_rooms["entry"] = (to_canvas_pt(z["center"]), z["na"])
    bed_zones = sorted([z for z in zones if z["rtype"] == "bedroom" and z["na"] >= 8000], key=lambda zz: -zz["na"])
    if bed_zones:
        master_z = next((z for z in bed_zones if "master" in z["label"].lower() or "1" in z["label"]), bed_zones[0])
        canvas_rooms["master_bed"] = (to_canvas_pt(master_z["center"]), master_z["na"])
        other_beds = [z for z in bed_zones if z["id"] != master_z["id"]]
        if other_beds:
            canvas_rooms["bed_2"] = (to_canvas_pt(other_beds[0]["center"]), other_beds[0]["na"])
    canvas_rooms = {k: v[0] for k, v in canvas_rooms.items()}

    # Compute physical door-width scale reference (invariant to image resolution & white margins)
    door_spans = [
        math.hypot(d["end"][0] - d["start"][0], d["end"][1] - d["start"][1])
        for d in doors
        if math.hypot(d["end"][0] - d["start"][0], d["end"][1] - d["start"][1]) >= 8.0
    ]
    if door_spans:
        door_px = float(np.median(door_spans))
    else:
        ys_w, xs_w = np.where(clean_wall_mask > 0)
        b_dim = min(int(xs_w.max() - xs_w.min()), int(ys_w.max() - ys_w.min())) if xs_w.size > 0 else min(orig_w, orig_h)
        door_px = max(18.0, b_dim * 0.085)

    # Build global OCR-masked, wall-masked, door-masked proposal candidates
    door_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for _, db_raw, _ in raw_doors:
        cv2.rectangle(door_mask, (int(db_raw[0]), int(db_raw[1])), (int(db_raw[2]), int(db_raw[3])), 255, -1)
    door_dil = cv2.dilate(door_mask, np.ones((5, 5), np.uint8)) if np.any(door_mask > 0) else door_mask

    furn_zone_map = np.full((orig_h, orig_w), -1, dtype=np.int32)
    furnishable_mask = np.zeros((orig_h, orig_w), dtype=bool)
    for z in zones:
        if z["rtype"] not in ("closet", "garage", "outdoor", "laundry"):
            furnishable_mask |= z["mask"]
        furn_zone_map[z["mask"]] = z["id"]

    gray_clean = gray.copy()
    gray_clean[text_mask > 0] = 250
    line_mask = cv2.adaptiveThreshold(gray_clean, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 11)
    line_mask[text_mask > 0] = 0
    line_mask[wall_dil > 0] = 0
    line_mask[door_dil > 0] = 0
    valid_global = furnishable_mask & (text_mask == 0) & (wall_dil == 0) & (door_dil == 0)

    raw_props = []
    # Track Y: Explicit YOLO fixed-furniture inside furnishable rooms
    for lbl_y, bconf_y, b_y in yolo_furniture_boxes:
        cx_i = min(orig_w - 1, max(0, int((b_y[0] + b_y[2]) / 2.0)))
        cy_i = min(orig_h - 1, max(0, int((b_y[1] + b_y[3]) / 2.0)))
        if furnishable_mask[cy_i, cx_i]:
            na_y = (max(1.0, b_y[2] - b_y[0]) * max(1.0, b_y[3] - b_y[1])) * unit_scale
            raw_props.append(("yolo", [float(v) for v in b_y], na_y, lbl_y, bconf_y))

    # Track L: Line-art contours (handles white tiled baths, shaded plans, and line drawings uniformly)
    lsub = cv2.morphologyEx((line_mask & valid_global).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(lsub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x_c, y_c, bw_c, bh_c = cv2.boundingRect(cnt)
        d_area_c = (bw_c * bh_c) / max(1.0, door_px * door_px)
        d_min_c = min(bw_c, bh_c) / max(1.0, door_px)
        if 0.22 <= d_area_c <= 14.0 and d_min_c >= 0.30:
            sub_bgr = img[y_c:y_c + bh_c, x_c:x_c + bw_c]
            if float(sub_bgr[:, :, 1].mean()) > max(float(sub_bgr[:, :, 2].mean()), float(sub_bgr[:, :, 0].mean())) + 10 and d_area_c < 2.2:
                continue
            na_c = (bw_c * bh_c) * unit_scale
            raw_props.append(("line", [float(x_c), float(y_c), float(x_c + bw_c), float(y_c + bh_c)], na_c, None, 0.90))

    # Track B & Track A per zone
    for z in zones:
        if z["rtype"] in ("closet", "garage", "outdoor", "laundry"):
            continue
        zm_v = z["mask"] & valid_global
        z_v30 = float(np.percentile(gray[z["mask"]], 30))
        z_sm = float(hsv[:, :, 1][z["mask"]].mean())
        if z_v30 <= 243:
            thr = max(242, int(z_v30) + 10)
            bfg = ((gray_clean >= thr) & (hsv[:, :, 1] < 26) & zm_v).astype(np.uint8) * 255
            bfg = cv2.morphologyEx(bfg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            nb, _blab, bst, _ = cv2.connectedComponentsWithStats(bfg, 8)
            for bi in range(1, nb):
                fx, fy, fw, fh, fa = [int(v) for v in bst[bi]]
                d_area_b = fa / max(1.0, door_px * door_px)
                d_min_b = min(fw, fh) / max(1.0, door_px)
                if 0.20 <= d_area_b <= 12.0 and d_min_b >= 0.30 and (fa / max(1.0, fw * fh)) >= 0.42:
                    raw_props.append(("bright", [float(fx), float(fy), float(fx + fw), float(fy + fh)], fa * unit_scale, None, 0.90))
        has_zone_cands = any(
            z["mask"][min(orig_h - 1, max(0, int((b[1] + b[3]) / 2.0))), min(orig_w - 1, max(0, int((b[0] + b[2]) / 2.0)))]
            for _, b, _, _, _ in raw_props
        )
        if z_sm < 10 and not has_zone_cands:
            col_fg = ((hsv[:, :, 1] >= 12) & (hsv[:, :, 2] > 95) & zm_v).astype(np.uint8) * 255
            col_fg = cv2.morphologyEx(col_fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            nc, clab, cstats, _ = cv2.connectedComponentsWithStats(col_fg, 8)
            for ci in range(1, nc):
                fx, fy, fw, fh, fa = [int(v) for v in cstats[ci]]
                d_area_col = fa / max(1.0, door_px * door_px)
                d_min_col = min(fw, fh) / max(1.0, door_px)
                if not (0.20 <= d_area_col <= 11.0 and d_min_col >= 0.25):
                    continue
                comp_m = clab == ci
                if float(img[:, :, 1][comp_m].mean()) > max(float(img[:, :, 2][comp_m].mean()), float(img[:, :, 0][comp_m].mean())) + 12 and d_area_col < 2.0:
                    continue
                raw_props.append(("color", [float(fx), float(fy), float(fx + fw), float(fy + fh)], fa * unit_scale, None, 0.88))

    # Track O: Explicit enclosed CAD chambers labeled with OCR 'BED'
    for flbl, _ft, (fx, fy), _fb in furn_anchors:
        if flbl == "bed" and 0 <= fy < orig_h and 0 <= fx < orig_w:
            if not any(b[0] <= fx <= b[2] and b[1] <= fy <= b[3] and ((b[2] - b[0]) * (b[3] - b[1])) / max(1.0, door_px * door_px) >= 2.2 for _, b, _na, _, _ in raw_props):
                rid_at = int(rlab[fy, fx]) if rlab[fy, fx] > 0 else 0
                if rid_at > 0 and rstats[rid_at, cv2.CC_STAT_AREA] / max(1.0, door_px * door_px) <= 18.0:
                    rx, ry, rw, rh, _ = [int(v) for v in rstats[rid_at]]
                    raw_props.append(("ocr_box", [float(rx), float(ry), float(rx + rw), float(ry + rh)], (rw * rh) * unit_scale, "bed", 0.94))

    src_rank = {"yolo": 0, "ocr_box": 1, "line": 2, "bright": 3, "color": 4}
    raw_props.sort(key=lambda c: (src_rank.get(c[0], 9), -((c[1][2] - c[1][0]) * (c[1][3] - c[1][1]))))
    props = []
    for src, b, na, hint_lbl, conf_val in raw_props:
        cx_i = min(orig_w - 1, max(0, int((b[0] + b[2]) / 2.0)))
        cy_i = min(orig_h - 1, max(0, int((b[1] + b[3]) / 2.0)))
        zid = int(furn_zone_map[cy_i, cx_i])
        if zid < 0:
            continue
        z = zones[zid]
        if src != "ocr_box":
            if any(_boxes_overlap_ratio(b, db) > 0.25 for db in accepted_door_boxes):
                continue
            if z["na"] >= 10000:
                if (b[2] - b[0]) > (z["box"][2] - z["box"][0]) * 0.86 and (b[3] - b[1]) > (z["box"][3] - z["box"][1]) * 0.86:
                    continue
        if any(_boxes_overlap_ratio(b, kb["box"]) > 0.42 for kb in props):
            continue
        props.append({"src": src, "box": b, "na": na, "hint": hint_lbl, "conf": conf_val, "zid": zid, "rtype": z["rtype"]})

    # Batched CLIP classification across all proposal crops
    clip_probs = []
    if props:
        crops = []
        for pr in props:
            bx1, by1, bx2, by2 = [int(round(v)) for v in pr["box"]]
            bw_p, bh_p = max(1, bx2 - bx1), max(1, by2 - by1)
            px, py = max(8, int(bw_p * 0.28)), max(8, int(bh_p * 0.28))
            crops.append(Image.fromarray(img[max(0, by1 - py):min(orig_h, by2 + py), max(0, bx1 - px):min(orig_w, bx2 + px), ::-1]))
        with torch.no_grad():
            im_in = c_proc(images=crops, return_tensors="pt")
            im_ft = c_mod.get_image_features(**im_in)
            if not isinstance(im_ft, torch.Tensor):
                im_ft = im_ft.pooler_output
            im_ft = im_ft / im_ft.norm(dim=-1, keepdim=True)
            clip_probs = (lscale * (im_ft @ txt_feat.T)).softmax(dim=-1)

    classified_items = []
    living_mid_indices = []
    bed_zones_seen = set()
    bed_centers = []
    bath_zone_cands = {}
    wall_pad_px = max(5, min(14, int(round(door_px * 0.18))))
    ocr_pad = max(14, int(round(door_px * 0.35)))

    for idx, pr in enumerate(props):
        b = pr["box"]
        zid = pr["zid"]
        rtype = pr["rtype"]
        na = pr["na"]
        bw = max(1.0, b[2] - b[0])
        bh = max(1.0, b[3] - b[1])
        d_w, d_h = bw / door_px, bh / door_px
        d_min, d_max = min(d_w, d_h), max(d_w, d_h)
        d_area = d_w * d_h
        aspect = d_max / max(1e-4, d_min)
        ws_cnt, ws_sides = _wall_contact_sides(b, clean_wall_mask, orig_h, orig_w, pad=wall_pad_px)
        cp = {CLIP_PROMPTS[j][0]: float(clip_probs[idx, j]) for j in range(len(CLIP_PROMPTS))}

        ocr_hits_inside = [
            fl for fl, _ft, (fx, fy), _fb in furn_anchors
            if b[0] <= fx <= b[2] and b[1] <= fy <= b[3]
        ]
        ocr_hits_near = [
            fl for fl, _ft, (fx, fy), _fb in furn_anchors
            if b[0] - ocr_pad <= fx <= b[2] + ocr_pad and b[1] - ocr_pad <= fy <= b[3] + ocr_pad
        ]
        ocr_hit = ocr_hits_inside[0] if ocr_hits_inside else (ocr_hits_near[0] if ocr_hits_near else None)

        if cp["plant_ignore"] >= 0.26 or (cp["noise_ignore"] >= 0.36 and not ocr_hit and not pr["hint"]):
            continue

        # Skip built-in kitchen counters early (even when 'SINK' is printed along the counter)
        if rtype == "kitchen_dining":
            if (
                (cp["kitchen_counter"] >= 0.85 and cp["dining_table"] < 0.04)
                or (cp["kitchen_counter"] >= 0.45 and ws_cnt >= 1)
                or ws_cnt >= 2
                or ("sink" in ocr_hits_near and d_area >= 1.65)
                or (ws_cnt >= 1 and d_area >= 1.80 and aspect >= 2.0 and d_min >= 0.85)
            ):
                continue

        if rtype == "bath":
            bath_zone_cands.setdefault(zid, []).append({
                "box": b, "na": na, "d_w": d_w, "d_h": d_h, "d_min": d_min, "d_max": d_max,
                "d_area": d_area, "aspect": aspect, "ws_cnt": ws_cnt, "cp": cp,
                "hint": pr["hint"], "ocr_hit": ocr_hit, "conf": pr["conf"],
            })
            continue

        label = pr["hint"]
        if label is None and ocr_hit:
            if ocr_hit == "dining_table":
                label = "table"
            elif ocr_hit == "desk" and d_area < 0.85:
                label = "chair"
            elif ocr_hit in ("wardrobe", "tv_cabinet", "desk", "bed", "sofa", "nightstand", "toilet", "sink"):
                if ocr_hit == "bed" and zid in bed_zones_seen:
                    label = "desk" if 0.85 <= d_area <= 3.6 else None
                else:
                    label = ocr_hit
            elif ocr_hit == "cabinet":
                label = "wardrobe" if rtype == "bedroom" else "tv_cabinet"

        if label is None:
            bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            if rtype == "bedroom":
                near_existing_bed = (zid in bed_zones_seen) or any(
                    math.hypot(bcx - bx_e, bcy - by_e) < door_px * 3.0 for bx_e, by_e in bed_centers
                )
                if not near_existing_bed and d_area >= 2.30 and d_max >= 1.95 and d_min >= 1.05 and aspect <= 2.35:
                    label = "bed"
                    bed_zones_seen.add(zid)
                    bed_centers.append((bcx, bcy))
                elif aspect >= 2.0 and d_min <= 1.15 and d_area >= 0.70:
                    label = "wardrobe"
                elif d_area <= 1.15 and aspect < 1.75 and ws_cnt >= 1 and d_max <= 1.15:
                    label = "nightstand"
                elif 0.85 <= d_area <= 3.60 and aspect < 2.20:
                    label = "desk"
                elif 0.25 <= d_area < 0.85 and aspect <= 1.65:
                    label = "chair"
            elif rtype == "kitchen_dining":
                if aspect >= 2.05 and d_min <= 0.72 and d_area >= 0.80:
                    label = "tv_cabinet"
                elif d_area >= 1.45 and ws_cnt <= 1:
                    label = "table"
                elif 0.25 <= d_area < 1.45 and aspect <= 1.65:
                    label = "chair"
            elif rtype == "living":
                if aspect >= 2.05 and d_min <= 0.72 and d_area >= 0.60 and (ws_cnt >= 1 or aspect >= 2.75):
                    label = "tv_cabinet"
                elif d_area >= 1.70 and (d_area >= 1.95 or aspect >= 1.65):
                    living_mid_indices.append((len(classified_items), b))
                    label = "sofa"
                elif 0.50 <= d_area < 1.95 and aspect <= 1.38 and (cp["chair"] >= cp["coffee_table"] or ws_cnt >= 1 or aspect <= 1.15):
                    label = "chair"
                elif 0.65 <= d_area < 1.95 and ws_cnt == 0:
                    label = "coffee_table"
                elif 0.35 <= d_area < 1.95:
                    label = "chair"
            elif rtype == "hall":
                if aspect >= 2.05 and d_min <= 0.72 and d_area >= 0.55 and ws_cnt >= 1:
                    label = "wardrobe"
        elif label == "bed":
            bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            bed_zones_seen.add(zid)
            bed_centers.append((bcx, bcy))

        if label:
            classified_items.append({"label": label, "box": b, "na": na, "d_area": d_area, "aspect": aspect, "conf": pr["conf"]})

    # Per-zone bathroom fixture disambiguation (shower/tub vs toilet vs vanity/corner sink)
    for _zid_b, bcands in bath_zone_cands.items():
        has_toilet = False
        shower_boxes = []
        remaining = []
        for c in bcands:
            b_c = c["box"]
            if c["hint"] in ("toilet", "sink", "shower"):
                lbl_b = c["hint"]
            elif c["ocr_hit"] == "toilet":
                lbl_b = "toilet"
            elif c["aspect"] >= 1.75 and c["d_min"] <= 0.72 and c["d_max"] >= 1.15 and c["ws_cnt"] >= 1:
                lbl_b = "sink"
            elif (
                (c["d_area"] >= 1.35 and c["aspect"] <= 1.75)
                or (c["d_area"] >= 1.50 and c["d_min"] >= 0.90 and c["aspect"] <= 2.35)
                or (c["d_area"] >= 1.15 and c["d_min"] >= 0.98 and c["aspect"] <= 1.30)
                or (c["ws_cnt"] >= 2 and c["aspect"] <= 1.18 and 0.65 <= c["d_area"] <= 1.35)
            ):
                if any(_boxes_overlap_ratio(b_c, sb) > 0.25 or math.hypot((b_c[0] + b_c[2] - sb[0] - sb[2]) / 2.0, (b_c[1] + b_c[3] - sb[1] - sb[3]) / 2.0) < door_px * 0.9 for sb in shower_boxes):
                    continue
                lbl_b = "shower"
                shower_boxes.append(b_c)
            else:
                remaining.append(c)
                continue

            if lbl_b == "toilet":
                has_toilet = True
            classified_items.append({"label": lbl_b, "box": b_c, "na": c["na"], "d_area": c["d_area"], "aspect": c["aspect"], "conf": c["conf"]})

        if remaining:
            if not has_toilet:
                # Prefer standalone 1-wall fixture first, then larger/more toilet-like fixture
                best_t = max(
                    range(len(remaining)),
                    key=lambda i_r: (
                        1 if remaining[i_r]["ws_cnt"] <= 1 else 0,
                        round(remaining[i_r]["d_area"], 2),
                        remaining[i_r]["cp"]["toilet"] - remaining[i_r]["cp"]["sink"],
                    ),
                )
            else:
                best_t = -1
            for i_r, c in enumerate(remaining):
                lbl_b = "toilet" if i_r == best_t else "sink"
                classified_items.append({"label": lbl_b, "box": c["box"], "na": c["na"], "d_area": c["d_area"], "aspect": c["aspect"], "conf": c["conf"]})

    # Disambiguate central coffee table sandwiched between parallel facing sofas in Living Room
    coffee_table_boxes = []
    if len(living_mid_indices) >= 3:
        for idx_m, bm in living_mid_indices:
            has_l = any(ba[2] <= bm[0] + 8 and min(ba[3], bm[3]) - max(ba[1], bm[1]) > 15 for _, ba in living_mid_indices)
            has_r = any(bb[0] >= bm[2] - 8 and min(bb[3], bm[3]) - max(bb[1], bm[1]) > 15 for _, bb in living_mid_indices)
            has_t = any(ba[3] <= bm[1] + 8 and min(ba[2], bm[2]) - max(ba[0], bm[0]) > 15 for _, ba in living_mid_indices)
            has_b = any(bb[1] >= bm[3] - 8 and min(bb[2], bm[2]) - max(bb[0], bm[0]) > 15 for _, bb in living_mid_indices)
            if (has_l and has_r) or (has_t and has_b):
                classified_items[idx_m]["label"] = "coffee_table"
                coffee_table_boxes.append(bm)
    # Any lone smaller sofa candidate (< 1.95 door_px^2) in living without a coffee table becomes coffee_table
    for idx_m, bm in living_mid_indices:
        if classified_items[idx_m]["label"] == "sofa" and classified_items[idx_m]["d_area"] < 1.95:
            classified_items[idx_m]["label"] = "coffee_table"
            coffee_table_boxes.append(bm)

    dining_boxes = []
    for item in classified_items:
        lbl = item["label"]
        box = item["box"]
        aspect = item["aspect"]
        d_area = item["d_area"]
        model_file = MODEL_MAP.get(lbl)
        if not model_file:
            continue
        ang_override = None
        if lbl == "coffee_table" or (lbl == "table" and d_area <= 1.35 and aspect <= 1.6 and not any(_boxes_overlap_ratio(box, ctb) > 0 for ctb in coffee_table_boxes)):
            model_file = "TableS2.glb"
        elif lbl == "table":
            model_file = "TableB1.glb"
            dining_boxes.append(box)
        elif lbl == "sofa" and coffee_table_boxes:
            cx_s, cy_s = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            near_ct = min(coffee_table_boxes, key=lambda ct: math.hypot((ct[0] + ct[2]) / 2.0 - cx_s, (ct[1] + ct[3]) / 2.0 - cy_s))
            tx_ct, ty_ct = (near_ct[0] + near_ct[2]) / 2.0, (near_ct[1] + near_ct[3]) / 2.0
            if math.hypot(tx_ct - cx_s, ty_ct - cy_s) <= door_px * 4.5:
                ang_override = math.atan2(-(ty_ct - cy_s), tx_ct - cx_s)
        append_furniture(lbl, box, model_file, confidence=round(item["conf"], 2), angle_rad=ang_override)

    # Add 4 dining chairs cleanly around each Dining Table if not already surrounded by chairs
    ch_half = max(5.5, door_px * 0.16)
    for db in dining_boxes:
        dcx, dcy = (db[0] + db[2]) / 2.0, (db[1] + db[3]) / 2.0
        dw, dh = db[2] - db[0], db[3] - db[1]
        existing_near_chairs = sum(
            1 for f in furniture
            if f["label"] == "chair" and math.hypot(f["center"][0] - dcx, f["center"][1] - dcy) <= max(dw, dh) * 0.9
        )
        if existing_near_chairs >= 2:
            continue
        if dw >= dh:
            dx_off, dy_off = max(door_px * 0.30, dw * 0.22), max(door_px * 0.36, dh * 0.44)
            for sx in (-1.0, 1.0):
                for py, ang in ((dcy - dy_off, 0.0), (dcy + dy_off, math.pi)):
                    px = dcx + sx * dx_off
                    cbox = [px - ch_half, py - ch_half, px + ch_half, py + ch_half]
                    if 0 <= int(py) < orig_h and 0 <= int(px) < orig_w and clean_wall_mask[int(py), int(px)] == 0:
                        append_furniture("chair", cbox, "Chair1.glb", confidence=0.86, angle_rad=ang)
        else:
            dx_off, dy_off = max(door_px * 0.36, dw * 0.44), max(door_px * 0.30, dh * 0.22)
            for sy in (-1.0, 1.0):
                for px, ang in ((dcx - dx_off, math.pi / 2.0), (dcx + dx_off, -math.pi / 2.0)):
                    py = dcy + sy * dy_off
                    cbox = [px - ch_half, py - ch_half, px + ch_half, py + ch_half]
                    if 0 <= int(py) < orig_h and 0 <= int(px) < orig_w and clean_wall_mask[int(py), int(px)] == 0:
                        append_furniture("chair", cbox, "Chair1.glb", confidence=0.86, angle_rad=ang)

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
        "rooms": canvas_rooms,
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
