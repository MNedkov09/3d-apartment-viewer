import cv2
import numpy as np

def trace_skeleton_graph(skeleton):
    """
    Given a binary 1px skeleton image (0 and 255),
    extract continuous chains of coordinates and simplify them into polylines.
    """
    h, w = skeleton.shape
    skel = (skeleton > 0).astype(np.uint8)
    
    # Kernel for counting 8-neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel, -1, kernel) * skel
    
    # Nodes are endpoints (1 neighbor) or junctions (>= 3 neighbors)
    is_node = (neighbor_count == 1) | (neighbor_count >= 3)
    
    node_coords = np.column_stack(np.where(is_node)) # (y, x)
    node_set = set((int(x), int(y)) for y, x in node_coords)
    
    consumed = np.zeros_like(skel, dtype=bool)
    started = set()
    edges = []
    
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
            
            # Simplify chain using cv2.approxPolyDP
            if len(chain) >= 2:
                pts = np.array(chain, dtype=np.int32).reshape((-1, 1, 2))
                approx = cv2.approxPolyDP(pts, epsilon=3.0, closed=False)
                simplified = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]
                edges.append({"points": simplified, "closed": False})
                
    return edges

# Quick test
canvas = np.zeros((200, 200), dtype=np.uint8)
cv2.line(canvas, (20, 20), (180, 20), 255, 1)
cv2.line(canvas, (180, 20), (180, 180), 255, 1)
edges = trace_skeleton_graph(canvas)
print(f"Traced {len(edges)} edge(s):", edges)
