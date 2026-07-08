"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import { useOpenCv, OPENCV_URL } from "./useOpenCv";
import Viewer3D from "./Viewer3D";

/* ---------- Pipeline constants ---------- */
const CANVAS_W = 800;
const CANVAS_H = 600;
const THRESH_BLOCK = 25;
const THRESH_C = 8;
const MIN_AREA = 500;
const PAD = 20;
const SIMPLIFY_EPSILON = 2;
const CORNER_ANGLE_THRESHOLD = 25;

/* ---------- OpenCV helpers ---------- */

function zhangSuenThinning(data, width, height) {
  const img = new Uint8Array(data);
  const at = (x, y) =>
    x < 0 || y < 0 || x >= width || y >= height ? 0 : img[y * width + x];

  let changed = true;
  while (changed) {
    changed = false;
    for (const step of [0, 1]) {
      const toClear = [];
      for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
          if (!at(x, y)) continue;
          const p2 = at(x, y - 1),
            p3 = at(x + 1, y - 1),
            p4 = at(x + 1, y),
            p5 = at(x + 1, y + 1);
          const p6 = at(x, y + 1),
            p7 = at(x - 1, y + 1),
            p8 = at(x - 1, y),
            p9 = at(x - 1, y - 1);
          const n = [p2, p3, p4, p5, p6, p7, p8, p9];
          const B = n.reduce((a, b) => a + b, 0);
          if (B < 2 || B > 6) continue;
          let A = 0;
          for (let i = 0; i < 8; i++)
            if (n[i] === 0 && n[(i + 1) % 8] === 1) A++;
          if (A !== 1) continue;
          if (step === 0) {
            if (p2 * p4 * p6 !== 0 || p4 * p6 * p8 !== 0) continue;
          } else {
            if (p2 * p4 * p8 !== 0 || p2 * p6 * p8 !== 0) continue;
          }
          toClear.push(y * width + x);
        }
      }
      if (toClear.length) {
        changed = true;
        toClear.forEach((i) => (img[i] = 0));
      }
    }
  }
  return img;
}

function neighbors8(skeleton, cols, rows, x, y) {
  const result = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      const nx = x + dx,
        ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
      if (skeleton[ny * cols + nx]) result.push([nx, ny]);
    }
  }
  return result;
}

function traceSkeletonGraph(skeleton, cols, rows) {
  const degree = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (!skeleton[y * cols + x]) continue;
      degree[y * cols + x] = neighbors8(skeleton, cols, rows, x, y).length;
    }
  }
  const isNode = (x, y) => {
    const d = degree[y * cols + x];
    return d === 1 || d >= 3;
  };

  const consumed = new Uint8Array(cols * rows);
  const startedKeys = new Set();
  const rawEdges = [];
  const nodeSet = new Set();

  const walk = (sx, sy, nx, ny) => {
    const points = [[sx, sy]];
    let prevX = sx,
      prevY = sy;
    let curX = nx,
      curY = ny;
    while (true) {
      points.push([curX, curY]);
      if (isNode(curX, curY)) break;
      consumed[curY * cols + curX] = 1;
      const nbrs = neighbors8(skeleton, cols, rows, curX, curY);
      const next = nbrs.find(([ax, ay]) => !(ax === prevX && ay === prevY));
      if (!next) break;
      prevX = curX;
      prevY = curY;
      [curX, curY] = next;
    }
    return points;
  };

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      if (!skeleton[y * cols + x] || !isNode(x, y)) continue;
      nodeSet.add(`${x},${y}`);
      for (const [nx, ny] of neighbors8(skeleton, cols, rows, x, y)) {
        const key = `${x},${y}->${nx},${ny}`;
        if (startedKeys.has(key) || consumed[ny * cols + nx]) continue;
        const chain = walk(x, y, nx, ny);
        const [ex, ey] = chain[chain.length - 1];
        const [px, py] = chain[chain.length - 2] || [x, y];
        startedKeys.add(key);
        startedKeys.add(`${ex},${ey}->${px},${py}`);
        rawEdges.push({ points: chain, closed: false });
      }
    }
  }

  const visitedLoop = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const idx = y * cols + x;
      if (!skeleton[idx] || consumed[idx] || visitedLoop[idx]) continue;
      if (isNode(x, y)) continue;
      const nbrs = neighbors8(skeleton, cols, rows, x, y);
      if (nbrs.length !== 2) continue;
      const points = [[x, y]];
      visitedLoop[idx] = 1;
      let prevX = x,
        prevY = y;
      let [curX, curY] = nbrs[0];
      while (!(curX === x && curY === y)) {
        points.push([curX, curY]);
        visitedLoop[curY * cols + curX] = 1;
        const cNbrs = neighbors8(skeleton, cols, rows, curX, curY);
        const next = cNbrs.find(([ax, ay]) => !(ax === prevX && ay === prevY));
        if (!next) break;
        prevX = curX;
        prevY = curY;
        [curX, curY] = next;
      }
      rawEdges.push({ points, closed: true });
    }
  }

  const nodes = [...nodeSet].map((s) => s.split(",").map(Number));
  return { edges: rawEdges, nodes };
}

function simplifyChain(points, closed) {
  if (points.length < 3) return points;
  const flat = [];
  for (const [x, y] of points) flat.push(x, y);
  const mat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flat);
  const approx = new cv.Mat();
  try {
    cv.approxPolyDP(mat, approx, SIMPLIFY_EPSILON, closed);
    const out = [];
    for (let i = 0; i < approx.rows; i++) {
      out.push([approx.data32S[i * 2], approx.data32S[i * 2 + 1]]);
    }
    return out;
  } finally {
    mat.delete();
    approx.delete();
  }
}

function angleBetween(p0, p1, p2) {
  const v1x = p1[0] - p0[0],
    v1y = p1[1] - p0[1];
  const v2x = p2[0] - p1[0],
    v2y = p2[1] - p1[1];
  const len1 = Math.hypot(v1x, v1y) || 1;
  const len2 = Math.hypot(v2x, v2y) || 1;
  const dot = (v1x * v2x + v1y * v2y) / (len1 * len2);
  return Math.acos(Math.max(-1, Math.min(1, dot))) * (180 / Math.PI);
}

function splitEdgeAtCorners(edge) {
  const { points, closed } = edge;
  if (points.length < 3) return { edges: [edge], cornerNodes: [] };

  const n = points.length;
  const last = closed ? n : n - 1;
  const cornerIdx = new Set();
  for (let i = 1; i < last; i++) {
    const angle = angleBetween(points[i - 1], points[i], points[(i + 1) % n]);
    if (angle > CORNER_ANGLE_THRESHOLD) cornerIdx.add(i);
  }
  if (cornerIdx.size === 0) return { edges: [edge], cornerNodes: [] };

  const cornerNodes = [...cornerIdx].map((i) => points[i]);
  const subEdges = [];
  let current = [points[0]];
  for (let i = 1; i < n; i++) {
    current.push(points[i]);
    if (cornerIdx.has(i)) {
      subEdges.push({ points: current, closed: false });
      current = [points[i]];
    }
  }
  if (closed) current.push(points[0]);
  if (current.length > 1) subEdges.push({ points: current, closed: false });

  return { edges: subEdges, cornerNodes };
}

function buildSkeleton(canvas, image) {
  const ctx = canvas.getContext("2d");
  const s = Math.min(CANVAS_W / image.width, CANVAS_H / image.height);
  const w = image.width * s;
  const h = image.height * s;
  const ox = (CANVAS_W - w) / 2;
  const oy = (CANVAS_H - h) / 2;
  ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
  ctx.drawImage(image, ox, oy, w, h);

  const src = cv.imread(canvas);
  const gray = new cv.Mat();
  const blurred = new cv.Mat();
  const mask = new cv.Mat();
  const labels = new cv.Mat();
  const stats = new cv.Mat();
  const centroids = new cv.Mat();

  let rows, cols;
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const mRows = gray.rows,
      mCols = gray.cols;
    let top = mRows,
      bottom = 0,
      left = mCols,
      right = 0;
    const grayData = gray.data;
    for (let y = 0; y < mRows; y++) {
      for (let x = 0; x < mCols; x++) {
        if (grayData[y * mCols + x] < 248) {
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }

    const margin = 30;
    top = Math.max(0, top - margin);
    bottom = Math.min(mRows - 1, bottom + margin);
    left = Math.max(0, left - margin);
    right = Math.min(mCols - 1, right + margin);

    const cw = right - left + 1;
    const ch = bottom - top + 1;
    const cropped = cv.Mat.zeros(ch, cw, cv.CV_8UC1);
    gray.roi = new cv.Rect(left, top, cw, ch);
    gray.copyTo(cropped);
    gray.roi = new cv.Rect(0, 0, mCols, mRows);

    const padded = cv.Mat.zeros(ch + 2 * PAD, cw + 2 * PAD, cv.CV_8UC1);
    for (let y = 0; y < padded.rows; y++) {
      for (let x = 0; x < padded.cols; x++) {
        padded.data[y * padded.cols + x] = 255;
      }
    }
    padded.roi = new cv.Rect(PAD, PAD, cw, ch);
    cropped.copyTo(padded);
    padded.roi = new cv.Rect(0, 0, padded.cols, padded.rows);

    cv.GaussianBlur(padded, blurred, new cv.Size(3, 3), 0);

    cv.adaptiveThreshold(
      blurred,
      mask,
      255,
      cv.ADAPTIVE_THRESH_GAUSSIAN_C,
      cv.THRESH_BINARY_INV,
      THRESH_BLOCK,
      THRESH_C,
    );

    const numLabels = cv.connectedComponentsWithStats(
      mask,
      labels,
      stats,
      centroids,
      8,
      cv.CV_32S,
    );
    ((rows = mask.rows), (cols = mask.cols));
    const labelData = labels.data32S;

    const keep = new Uint8Array(numLabels);
    for (let i = 1; i < numLabels; i++) {
      if (stats.intAt(i, cv.CC_STAT_AREA) > MIN_AREA) keep[i] = 1;
    }
    const wallData = new Uint8Array(rows * cols);
    for (let p = 0; p < rows * cols; p++) {
      wallData[p] = keep[labelData[p]] ? 1 : 0;
    }

    const skeleton = zhangSuenThinning(wallData, cols, rows);

    const border = PAD;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (
          y < border ||
          y >= rows - border ||
          x < border ||
          x >= cols - border
        ) {
          skeleton[y * cols + x] = 0;
        }
      }
    }

    return { skeleton, cols, rows };
  } finally {
    src.delete();
    gray.delete();
    blurred.delete();
    mask.delete();
    labels.delete();
    stats.delete();
    centroids.delete();
  }
}

function extractWallGraph(canvas, image) {
  const { skeleton, cols, rows } = buildSkeleton(canvas, image);
  const { edges: rawEdges, nodes: junctionNodes } = traceSkeletonGraph(
    skeleton,
    cols,
    rows,
  );

  const finalEdges = [];
  const cornerNodeSet = new Set();
  for (const raw of rawEdges) {
    const simplified = {
      points: simplifyChain(raw.points, raw.closed),
      closed: raw.closed,
    };
    const { edges: split, cornerNodes } = splitEdgeAtCorners(simplified);
    finalEdges.push(...split);
    for (const [x, y] of cornerNodes) cornerNodeSet.add(`${x},${y}`);
  }

  const cornerNodes = [...cornerNodeSet].map((s) => s.split(",").map(Number));
  const allNodes = [...junctionNodes, ...cornerNodes];

  const border = PAD + 5;
  const validNodeSet = new Set(
    allNodes
      .filter(
        ([nx, ny]) =>
          nx >= border &&
          nx < cols - border &&
          ny >= border &&
          ny < rows - border,
      )
      .map(([nx, ny]) => `${nx},${ny}`),
  );
  const filteredEdges = finalEdges.filter((e) =>
    e.points.every(
      ([px, py]) =>
        px >= border &&
        px < cols - border &&
        py >= border &&
        py < rows - border,
    ),
  );
  const filteredNodes = allNodes.filter(([nx, ny]) =>
    validNodeSet.has(`${nx},${ny}`),
  );

  return { edges: filteredEdges, nodes: filteredNodes };
}

/* ---------- Styles ---------- */
const CANVAS_STYLE = {
  border: "1px solid var(--line)",
  display: "block",
  marginTop: 12,
  background: "#e8e8e8",
  maxWidth: "100%",
};

const SETTINGS_STYLE = {
  display: "flex",
  gap: 20,
  flexWrap: "wrap",
  alignItems: "flex-end",
  padding: 12,
  border: "1px solid var(--line)",
  borderRadius: 6,
  margin: "12px 0",
};

const LABEL_STYLE = {
  display: "flex",
  flexDirection: "column",
  fontSize: 13,
};

const INPUT_STYLE = { width: 90, padding: "2px 4px" };

/* ---------- Component ---------- */

export default function FloorplanViewer({ imageSrc, blueprintTitle }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const { status: cvStatus } = useOpenCv();

  const [edges, setEdges] = useState(null);
  const [nodes, setNodes] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [view, setView] = useState("2d");
  const [pixelsPerMeter, setPixelsPerMeter] = useState(50);
  const [wallHeight, setWallHeight] = useState(2.7);
  const [wallThickness, setWallThickness] = useState(0.15);
  const [error, setError] = useState("");

  const runDetection = useCallback(async (img) => {
    if (cvStatus !== "ready") return;
    setProcessing(true);
    setError("");
    try {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const { edges: detectedEdges, nodes: detectedNodes } = extractWallGraph(
        canvas,
        img,
      );

      // Draw overlay on canvas — must use the same scale/offset as buildSkeleton
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
      const s = Math.min(CANVAS_W / img.width, CANVAS_H / img.height);
      const w = img.width * s;
      const h = img.height * s;
      const ox = (CANVAS_W - w) / 2;
      const oy = (CANVAS_H - h) / 2;
      ctx.drawImage(img, ox, oy, w, h);

      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const edge of detectedEdges) {
        if (edge.points.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(edge.points[0][0], edge.points[0][1]);
        for (let i = 1; i < edge.points.length; i++) {
          ctx.lineTo(edge.points[i][0], edge.points[i][1]);
        }
        if (edge.closed) ctx.closePath();
        ctx.stroke();
      }

      ctx.fillStyle = "#1565c0";
      for (const [nx, ny] of detectedNodes) {
        ctx.beginPath();
        ctx.arc(nx, ny, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      setEdges(detectedEdges);
      setNodes(detectedNodes);
    } catch (err) {
      console.error("Floorplan detection failed:", err);
      setError(err.message || "Detection failed");
    } finally {
      setProcessing(false);
    }
  }, [cvStatus]);

  useEffect(() => {
    if (!imageSrc || cvStatus !== "ready" || view !== "2d") return;

    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      imgRef.current = image;
      runDetection(image);
    };
    image.onerror = () => {
      setError("Failed to load image");
      setProcessing(false);
    };
    image.src = imageSrc;

    return () => {
      if (imgRef.current?.src === imageSrc) imgRef.current = null;
    };
  }, [imageSrc, cvStatus, view, runDetection]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {cvStatus === "loading" && <p>Loading OpenCV.js...</p>}
      {cvStatus === "error" && (
        <p style={{ color: "red" }}>OpenCV failed to load — 2D detection unavailable. 3D viewer will not render.</p>
      )}
      {error && <p style={{ color: "red" }}>{error}</p>}
      {processing && <p>Processing floorplan...</p>}

      {edges && (
        <div style={SETTINGS_STYLE}>
          <label style={LABEL_STYLE}>
            Scale (px/m)
            <input
              type="number"
              min="1"
              value={pixelsPerMeter}
              onChange={(e) => setPixelsPerMeter(Number(e.target.value) || 1)}
              style={INPUT_STYLE}
            />
          </label>
          <label style={LABEL_STYLE}>
            Wall height (m)
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={wallHeight}
              onChange={(e) => setWallHeight(Number(e.target.value) || 0.1)}
              style={INPUT_STYLE}
            />
          </label>
          <label style={LABEL_STYLE}>
            Wall thickness (m)
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={wallThickness}
              onChange={(e) => setWallThickness(Number(e.target.value) || 0.01)}
              style={INPUT_STYLE}
            />
          </label>
          <button onClick={() => setView(view === "2d" ? "3d" : "2d")}>
            {view === "2d" ? "Generate 3D →" : "← Back to 2D"}
          </button>
        </div>
      )}

      {view === "2d" && (
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={CANVAS_STYLE}
        />
      )}

      {view === "3d" && edges && (
        <Viewer3D
          edges={edges}
          pixelsPerMeter={pixelsPerMeter}
          wallHeightMeters={wallHeight}
          wallThicknessMeters={wallThickness}
        />
      )}
    </div>
  );
}
