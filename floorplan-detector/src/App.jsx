import { useRef, useState, useEffect } from "react";
import Viewer3D from "./Viewer3D";
import { OPENCV_URL } from "./hooks/useOpenCv";

const CANVAS_W = 800;
const CANVAS_H = 600;

/* ---------- OpenCV.js loader ---------- */
// OpenCV.js is Emscripten-compiled. It checks window.Module for
// onRuntimeInitialized. Set it BEFORE the script loads.
let cvReady = false;
let cvPromise = null; // shared promise for concurrent callers

function loadCv() {
  if (cvReady) return Promise.resolve();
  if (cvPromise) return cvPromise; // dedupe: caller waits for the same load
  cvPromise = new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn) => (val) => {
      if (settled) return;
      settled = true;
      fn(val);
    };
    const timeoutId = setTimeout(() => {
      if (!settled) {
        cvPromise = null;
        reject(new Error("OpenCV.js load timed out"));
      }
    }, 30000);

    window.Module = window.Module || {};
    window.Module.onRuntimeInitialized = () => {
      clearTimeout(timeoutId);
      cvReady = true;
      cvPromise = null; // allow future retries
      console.log("OpenCV ready");
      settle(resolve)();
    };
    window.Module.onAbort = (err) => {
      clearTimeout(timeoutId);
      cvPromise = null;
      console.error("OpenCV WASM abort:", err);
      settle(reject)(new Error("OpenCV failed to initialize"));
    };
    const script = document.createElement("script");
    script.src = OPENCV_URL;
    script.onerror = () => {
      clearTimeout(timeoutId);
      cvPromise = null;
      settle(reject)(new Error("Failed to load OpenCV.js"));
    };
    document.head.appendChild(script);
  });
  return cvPromise;
}

/* ---------- Wall mask extraction ---------- */
const THRESH_BLOCK = 25; // adaptive threshold block size (odd)
const THRESH_C = 8; // adaptive threshold constant
const MIN_AREA = 500; // min area (px^2) to keep a component as wall
const PAD = 20; // px — white padding around cropped content before thresholding

/**
 * Build a binary mask of wall-stroke pixels (dropping text by stroke
 * thickness via true morphological opening — NOT the erode-then-keep-
 * whole-component approach, which lets thin text riding on a thick wall
 * survive intact), then thin it to a 1px centerline skeleton.
 */
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

    // Find the bounding box of non-white content (the floor plan).
    // White padding (255) creates a frame when inverted; we want to
    // exclude it entirely by cropping to the content region first.
    const mRows = gray.rows,
      mCols = gray.cols;
    let top = mRows, bottom = 0, left = mCols, right = 0;
    const grayData = gray.data;
    for (let y = 0; y < mRows; y++) {
      for (let x = 0; x < mCols; x++) {
        if (grayData[y * mCols + x] < 248) { // not near-white
          if (y < top) top = y;
          if (y > bottom) bottom = y;
          if (x < left) left = x;
          if (x > right) right = x;
        }
      }
    }

    // Crop to content bounding box (with a generous margin to push the
    // crop edge away from the skeleton, preventing boundary artifacts).
    const margin = 30;
    top = Math.max(0, top - margin);
    bottom = Math.min(mRows - 1, bottom + margin);
    left = Math.max(0, left - margin);
    right = Math.min(mCols - 1, right + margin);

    // Extract the content bounding box.
    const cw = right - left + 1;
    const ch = bottom - top + 1;
    const cropped = cv.Mat.zeros(ch, cw, cv.CV_8UC1);
    gray.roi = new cv.Rect(left, top, cw, ch);
    gray.copyTo(cropped);
    gray.roi = new cv.Rect(0, 0, mCols, mRows); // restore

    // Pad the crop with white (255) on all sides so the crop edge never
    // touches floor-plan content.  Without this, the adaptive threshold
    // sees the crop boundary as a strong edge and skeletonization produces
    // lines along the crop perimeter, which the graph tracer interprets as
    // nodes at the corners.
    const padded = cv.Mat.zeros(ch + 2 * PAD, cw + 2 * PAD, cv.CV_8UC1);
    // Fill with white (255)
    for (let y = 0; y < padded.rows; y++) {
      for (let x = 0; x < padded.cols; x++) {
        padded.data[y * padded.cols + x] = 255;
      }
    }
    // Paste the cropped content into the center of the white padding
    padded.roi = new cv.Rect(PAD, PAD, cw, ch);
    cropped.copyTo(padded);
    padded.roi = new cv.Rect(0, 0, padded.cols, padded.rows); // restore

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
    rows = mask.rows,
      cols = mask.cols;
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

    // Strip a border from the skeleton matching the padding size.
    // The padded image has PAD px of white around the content, so the
    // skeleton won't reach the outer edge (all white).  Strip exactly
    // PAD px to remove any skeleton pixels that touch the crop boundary.
    const border = PAD;
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        if (y < border || y >= rows - border || x < border || x >= cols - border) {
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

// Zhang-Suen thinning: strips boundary pixels from a stroke until only its
// 1px-wide medial skeleton remains, preserving connectivity through
// corners, T-junctions, curves, and door gaps.
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

/* ---------- Skeleton -> graph of polylines (straight AND curved) ---------- */
// NODES = pixels where the skeleton branches (degree >= 3, a junction) or
// ends (degree == 1, a dead end). EDGES = the ordered pixel chain
// connecting one node to another. Each edge is simplified independently
// with cv.approxPolyDP — a straight run collapses to 2 points, a curved
// run keeps however many points it needs to stay within tolerance. Same
// code path either way; no "is this straight or curved" branching.
const SIMPLIFY_EPSILON = 2; // px — how tightly polylines hug the skeleton

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

  const consumed = new Uint8Array(cols * rows); // interior (degree-2) pixels already used by an edge
  const startedKeys = new Set(); // guards against walking the same edge from both ends
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
      if (isNode(curX, curY)) break; // reached another node — edge complete
      consumed[curY * cols + curX] = 1;
      const nbrs = neighbors8(skeleton, cols, rows, curX, curY);
      const next = nbrs.find(([ax, ay]) => !(ax === prevX && ay === prevY));
      if (!next) break; // guard against malformed chains
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

  // Leftover pixels belong to closed loops with no junctions at all (e.g.
  // a simple curved outer wall with no branching interior walls).
  const visitedLoop = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const idx = y * cols + x;
      if (!skeleton[idx] || consumed[idx] || visitedLoop[idx]) continue;
      if (isNode(x, y)) continue;
      const nbrs = neighbors8(skeleton, cols, rows, x, y);
      if (nbrs.length !== 2) continue; // malformed isolated fragment, skip
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

/**
 * Simplify one traced pixel chain into a polyline via approxPolyDP.
 * A straight wall run collapses to 2 points; a curved run keeps as many
 * points as needed to stay within SIMPLIFY_EPSILON of the original curve.
 */
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

const CORNER_ANGLE_THRESHOLD = 25; // degrees — turns sharper than this become nodes

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

/** Split a simplified polyline wherever it turns sharper than the
 * threshold. Each split point becomes a new node. Gentle bends stay
 * inside a single edge, so curved walls aren't shattered into segments. */
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

  // Filter out nodes within the padding zone (PAD=20px) plus a small margin
  // from the crop boundary — these are artifacts from the crop edge /
  // canvas padding, not real floor plan features.
  const border = PAD + 5; // 25px from padded crop edge
  const validNodeSet = new Set(
    allNodes
      .filter(([nx, ny]) => nx >= border && nx < cols - border && ny >= border && ny < rows - border)
      .map(([nx, ny]) => `${nx},${ny}`),
  );
  const filteredEdges = finalEdges.filter((e) =>
    e.points.every(([px, py]) => px >= border && px < cols - border && py >= border && py < rows - border),
  );
  const filteredNodes = allNodes.filter(([nx, ny]) => validNodeSet.has(`${nx},${ny}`));

  return { edges: filteredEdges, nodes: filteredNodes };
}

/* ---------- React component ---------- */
function App() {
  const canvasRef = useRef(null);
  const [fileName, setFileName] = useState("");
  const [cvLoading, setCvLoading] = useState(false);
  const [blobUrl, setBlobUrl] = useState("");
  const [cvError, setCvError] = useState("");
  const [stats, setStats] = useState(null); // { edgeCount, nodeCount }
  const [edges, setEdges] = useState(null); // wall graph edges, kept for 3D generation
  const [view, setView] = useState("2d"); // "2d" | "3d"
  const [pixelsPerMeter, setPixelsPerMeter] = useState(50);
  const [wallHeight, setWallHeight] = useState(2.7);
  const [wallThickness, setWallThickness] = useState(0.15);
  const imgRef = useRef(null);
  const blobUrlRef = useRef("");
  const uploadDone = useRef(false);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  useEffect(() => {
    const url = blobUrl;
    if (!url || !cvReady) return;

    let cancelled = false;
    if (uploadDone.current) URL.revokeObjectURL(blobUrlRef.current);
    blobUrlRef.current = url;

    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      uploadDone.current = true;
      imgRef.current = image;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      const s = Math.min(CANVAS_W / image.width, CANVAS_H / image.height);
      const w = image.width * s;
      const h = image.height * s;
      const ox = (CANVAS_W - w) / 2;
      const oy = (CANVAS_H - h) / 2;
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
      ctx.drawImage(image, ox, oy, w, h);

      const { edges, nodes } = extractWallGraph(canvas, image);

      // Draw each edge's simplified polyline — straight runs render as one
      // clean segment, curved runs render as a smooth multi-point path.
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      for (const edge of edges) {
        if (edge.points.length < 2) continue;
        ctx.beginPath();
        ctx.moveTo(edge.points[0][0], edge.points[0][1]);
        for (let i = 1; i < edge.points.length; i++) {
          ctx.lineTo(edge.points[i][0], edge.points[i][1]);
        }
        if (edge.closed) ctx.closePath();
        ctx.stroke();
      }

      // Mark junctions/endpoints so you can visually verify the graph
      ctx.fillStyle = "#1565c0";
      for (const [nx, ny] of nodes) {
        ctx.beginPath();
        ctx.arc(nx, ny, 3, 0, Math.PI * 2);
        ctx.fill();
      }

      setEdges(edges);
      setStats({ edgeCount: edges.length, nodeCount: nodes.length });
      console.log("Edges:", edges.length, "Nodes:", nodes.length, edges);
    };

    image.onerror = () => {
      if (cancelled) return;
      uploadDone.current = false;
      console.error("Failed to load image from file");
      URL.revokeObjectURL(url);
    };

    image.src = url;

    return () => {
      cancelled = true;
    };
  }, [blobUrl, cvReady]);

  const handleFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setStats(null);
    setCvError("");
    uploadDone.current = false;

    if (!cvReady) {
      setCvLoading(true);
      try {
        await loadCv();
      } catch (err) {
        setCvError(err.message);
        setCvLoading(false);
        return;
      }
      setCvLoading(false);
    }

    const url = URL.createObjectURL(file);
    setBlobUrl(url);
  };

  const retryCvLoad = async () => {
    setCvLoading(true);
    setCvError("");
    cvPromise = null; // allow a fresh load
    try {
      await loadCv();
    } catch (err) {
      setCvError(err.message);
    }
    setCvLoading(false);
  };

  return (
    <div style={{ padding: 24, fontFamily: "sans-serif" }}>
      <h1>2D Floor Plan &rarr; Wall Graph</h1>
      <p style={{ color: "#555", maxWidth: 640 }}>
        Traces the skeleton into a graph of nodes (junctions/endpoints, blue
        dots) and edges (wall runs, red). Each edge is simplified independently,
        so straight walls collapse to clean lines and curved walls keep their
        shape — same code path for both.
      </p>

      <input type="file" accept="image/*" onChange={handleFile} />
      {cvLoading && <p>Loading OpenCV.js...</p>}
      {cvError && (
        <p style={{ color: "red" }}>
          {cvError} <button onClick={retryCvLoad}>Retry</button>
        </p>
      )}
      {fileName && <p>Loaded: {fileName}</p>}
      {stats && (
        <p>
          {stats.edgeCount} wall edges, {stats.nodeCount} junctions/endpoints
        </p>
      )}

      {edges && (
        <div
          style={{
            display: "flex",
            gap: 20,
            flexWrap: "wrap",
            alignItems: "flex-end",
            padding: 12,
            border: "1px solid #ddd",
            borderRadius: 6,
            margin: "12px 0",
          }}
        >
          <label
            style={{ display: "flex", flexDirection: "column", fontSize: 13 }}
          >
            Scale (pixels per meter)
            <input
              type="number"
              min="1"
              value={pixelsPerMeter}
              onChange={(e) => setPixelsPerMeter(Number(e.target.value) || 1)}
              style={{ width: 90 }}
            />
          </label>
          <label
            style={{ display: "flex", flexDirection: "column", fontSize: 13 }}
          >
            Wall height (m)
            <input
              type="number"
              step="0.1"
              min="0.1"
              value={wallHeight}
              onChange={(e) => setWallHeight(Number(e.target.value) || 0.1)}
              style={{ width: 90 }}
            />
          </label>
          <label
            style={{ display: "flex", flexDirection: "column", fontSize: 13 }}
          >
            Wall thickness (m)
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={wallThickness}
              onChange={(e) => setWallThickness(Number(e.target.value) || 0.01)}
              style={{ width: 90 }}
            />
          </label>
          <button onClick={() => setView(view === "2d" ? "3d" : "2d")}>
            {view === "2d" ? "Generate 3D →" : "← Back to 2D detection"}
          </button>
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={CANVAS_W}
        height={CANVAS_H}
        style={{
          border: "1px solid #ccc",
          display: view === "2d" ? "block" : "none",
          marginTop: 12,
          background: "#e8e8e8",
        }}
      />

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

export default App;
