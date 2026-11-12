import cv2
import numpy as np
import math

CATEGORY_MAP = {
    'wall': 'wall',
    'wall external': 'wall',
    'class_31': 'wall',
    'class_32': 'wall',
    
    'window': 'window',
    'glass': 'window',
    'window regular': 'window',
    'sliding window': 'window',
    'bay_window': 'window',
    'blind_window': 'window',
    'class_34': 'window',
    
    'door': 'door',
    'door swing beside': 'door',
    'door swing opposite': 'door',
    'sliding door': 'door',
    'single_door': 'door',
    'double_door': 'door',
    'sliding_door': 'door',
    'opening_symbol': 'door',
    'class_35': 'door',
    
    'bed': 'furniture',
    'bedside_cupboard': 'furniture',
    'nightstand': 'furniture',
    'sofa': 'furniture',
    'table': 'furniture',
    'chair': 'furniture',
    'tv_cabinet': 'furniture',
    'wardrobe': 'furniture',
    'high_cabinet': 'furniture',
    'half_height_cabinet': 'furniture',
    'desk': 'furniture',
    
    'bath': 'furniture',
    'bath_tub': 'furniture',
    'shower': 'furniture',
    'sink': 'furniture',
    'toilet': 'furniture',
    'squat_toilet': 'furniture',
    'urinal': 'furniture',
    'gas_stove': 'furniture',
    'refrigerator': 'furniture',
    'washing_machine': 'furniture',
    
    'stair': 'stairs',
    'elevator': 'stairs',
    'escalator': 'stairs',
    'railing': 'stairs',
    'parking': 'other',
}

def get_category(label: str) -> str:
    return CATEGORY_MAP.get(label.lower(), 'other')

def extract_oriented_box(polygon_or_box):
    """
    Extracts center, length, thickness, and rotation from polygon points or bounding box.
    """
    if len(polygon_or_box) >= 3 and isinstance(polygon_or_box[0], (list, tuple)):
        pts = np.array(polygon_or_box, dtype=np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect
        if h > w:
            w, h = h, w
            angle += 90.0
    else:
        # Bounding box [x1, y1, x2, y2]
        x1, y1, x2, y2 = polygon_or_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        angle = 0.0
        if h > w:
            w, h = h, w
            angle = 90.0

    return {
        "center": [round(float(cx), 2), round(float(cy), 2)],
        "width": round(max(float(w), 8.0), 2),
        "depth": round(max(float(h), 4.0), 2),
        "angle_deg": round(float(angle % 180), 2),
        "angle_rad": round(float(math.radians(angle % 180)), 4)
    }

def trace_skeleton_edges(skel_binary, min_edge_len=8.0, approx_epsilon=2.5):
    """
    Traces 1px skeleton lines into polylines and simplifies them using approxPolyDP.
    """
    h, w = skel_binary.shape
    skel = (skel_binary > 0).astype(np.uint8)
    
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel) * skel
    
    is_node = (neighbor_count == 1) | (neighbor_count >= 3)
    node_coords = np.column_stack(np.where(is_node)) # (y, x)
    node_set = set((int(x), int(y)) for y, x in node_coords)
    
    consumed = np.zeros_like(skel, dtype=bool)
    started = set()
    raw_edges = []
    
    def get_nbrs(x, y):
        nbrs = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and skel[ny, nx]:
                    nbrs.append((nx, ny))
        return nbrs

    for sx, sy in node_set:
        for nx, ny in get_nbrs(sx, sy):
            edge_key = (sx, sy, nx, ny)
            if edge_key in started or consumed[ny, nx]:
                continue
            
            chain = [(sx, sy)]
            prev_x, prev_y = sx, sy
            cur_x, cur_y = nx, ny
            
            while True:
                chain.append((cur_x, cur_y))
                if (cur_x, cur_y) in node_set:
                    break
                consumed[cur_y, cur_x] = True
                nbrs = [p for p in get_nbrs(cur_x, cur_y) if p != (prev_x, prev_y)]
                if not nbrs:
                    break
                prev_x, prev_y = cur_x, cur_y
                cur_x, cur_y = nbrs[0]
            
            ex, ey = chain[-1]
            rev_key = (ex, ey, chain[-2][0], chain[-2][1]) if len(chain) > 1 else (ex, ey, sx, sy)
            started.add(edge_key)
            started.add(rev_key)
            
            if len(chain) >= 2:
                pts = np.array(chain, dtype=np.int32).reshape((-1, 1, 2))
                approx = cv2.approxPolyDP(pts, approx_epsilon, closed=False)
                simplified = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                
                total_len = sum(
                    math.hypot(simplified[i+1][0] - simplified[i][0], simplified[i+1][1] - simplified[i][1])
                    for i in range(len(simplified)-1)
                )
                if total_len >= min_edge_len:
                    raw_edges.append({"points": simplified, "closed": False})

    # Catch closed isolated loops
    visited_loops = np.zeros_like(skel, dtype=bool)
    for y in range(h):
        for x in range(w):
            if not skel[y, x] or consumed[y, x] or visited_loops[y, x] or (x, y) in node_set:
                continue
            nbrs = get_nbrs(x, y)
            if len(nbrs) != 2:
                continue
            
            loop_chain = [(x, y)]
            visited_loops[y, x] = True
            prev_x, prev_y = x, y
            cur_x, cur_y = nbrs[0]
            is_true_loop = True
            
            while not (cur_x == x and cur_y == y):
                loop_chain.append((cur_x, cur_y))
                visited_loops[cur_y, cur_x] = True
                c_nbrs = get_nbrs(cur_x, cur_y)
                nxt = [p for p in c_nbrs if p != (prev_x, prev_y)]
                if not nxt:
                    is_true_loop = False
                    break
                prev_x, prev_y = cur_x, cur_y
                cur_x, cur_y = nxt[0]
                
            if len(loop_chain) >= 4:
                pts = np.array(loop_chain, dtype=np.int32).reshape((-1, 1, 2))
                approx = cv2.approxPolyDP(pts, approx_epsilon, closed=is_true_loop)
                simplified = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                raw_edges.append({"points": simplified, "closed": is_true_loop})

    return raw_edges

def straighten_and_snap_edges(edges):
    """
    Straightens near-horizontal and near-vertical wall segments into clean orthogonal
    architectural polylines and snaps colinear axes.
    """
    clean_edges = []
    for edge in edges:
        pts = [list(p) for p in edge["points"]]
        if len(pts) < 2:
            continue
        for _ in range(2):
            for i in range(len(pts) - 1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                length = math.hypot(dx, dy)
                if length < 1e-3:
                    continue
                if abs(dy) <= max(4.0, length * 0.12):
                    avg_y = int(round((pts[i][1] + pts[i+1][1]) / 2.0))
                    pts[i][1] = avg_y
                    pts[i+1][1] = avg_y
                elif abs(dx) <= max(4.0, length * 0.12):
                    avg_x = int(round((pts[i][0] + pts[i+1][0]) / 2.0))
                    pts[i][0] = avg_x
                    pts[i+1][0] = avg_x

        simplified = [pts[0]]
        for i in range(1, len(pts) - 1):
            p0, p1, p2 = simplified[-1], pts[i], pts[i+1]
            if (p0[0] == p1[0] == p2[0]) or (p0[1] == p1[1] == p2[1]):
                continue
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 3.0:
                continue
            simplified.append(p1)
        if math.hypot(pts[-1][0] - simplified[-1][0], pts[-1][1] - simplified[-1][1]) >= 2.0 or len(simplified) == 1:
            simplified.append(pts[-1])
        if len(simplified) >= 2:
            clean_edges.append({"points": simplified, "closed": edge.get("closed", False)})

    h_ys = []
    v_xs = []
    for e in clean_edges:
        pts = e["points"]
        for i in range(len(pts) - 1):
            if pts[i][1] == pts[i+1][1] and abs(pts[i+1][0] - pts[i][0]) >= 10:
                h_ys.append(pts[i][1])
            elif pts[i][0] == pts[i+1][0] and abs(pts[i+1][1] - pts[i][1]) >= 10:
                v_xs.append(pts[i][0])

    def snap_coord(val, coords, tol=3):
        near = [c for c in coords if abs(c - val) <= tol]
        if near:
            return int(round(sum(near) / len(near)))
        return val

    for e in clean_edges:
        pts = e["points"]
        for i in range(len(pts) - 1):
            if pts[i][1] == pts[i+1][1]:
                sy = snap_coord(pts[i][1], h_ys, 3)
                pts[i][1] = sy
                pts[i+1][1] = sy
            elif pts[i][0] == pts[i+1][0]:
                sx = snap_coord(pts[i][0], v_xs, 3)
                pts[i][0] = sx
                pts[i+1][0] = sx

    return clean_edges

def extract_walls_from_mask(mask, min_edge_len=8.0):
    """
    Takes a binary wall mask (255 = wall, 0 = background),
    runs thinning, traces polylines, and straightens orthogonal wall segments.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    skeleton = cv2.ximgproc.thinning(closed, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    raw_edges = trace_skeleton_edges(skeleton, min_edge_len=min_edge_len)
    edges = straighten_and_snap_edges(raw_edges)
    return edges, skeleton
