import math
import cv2
import numpy as np

def clean_edges(edges, min_length=8.0):
    cleaned = []
    for edge in edges:
        pts = edge["points"]
        if len(pts) < 2:
            continue
        total_len = sum(math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1]) for i in range(len(pts)-1))
        if total_len >= min_length:
            cleaned.append(edge)
    return cleaned
