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

/* ---------- Wall-network isolation ---------- */
// "Keep only the single largest connected blob" assumed the whole wall
// network is always one unbroken piece. That's true for simple test
// drawings, but real architectural floorplans break it constantly — a
// doorway with no wall pixels bridging it, a rendering gap at a corner, a
// dashed section — any of which splits the wall network into several
// separate components. Picking just one then throws most of the house away
// (exactly what happened here: only the master bedroom corner survived).
//
// What's actually reliable across virtually every floorplan convention is
// that walls are drawn in a distinctly bolder stroke than everything else on
// the page — text labels, dimension lines, fixture icons, hatching, door
// swings. So we classify every sizeable component by its own stroke
// thickness (via a distance transform, same technique as the window/curve
// detection above) and keep *every* component that reads as wall-weight,
// however many separate pieces the wall network is actually split into.
function splitComponentsByThickness(mask, labels, stats, numLabels, rows, cols) {
  const dist = new cv.Mat();
  cv.distanceTransform(mask, dist, cv.DIST_L2, 5);
  const distData = dist.data32F;
  const labelData = labels.data32S;

  const sums = new Float64Array(numLabels);
  const counts = new Int32Array(numLabels);
  for (let p = 0; p < rows * cols; p++) {
    const lbl = labelData[p];
    if (lbl === 0) continue;
    sums[lbl] += distData[p];
    counts[lbl] += 1;
  }
  dist.delete();

  const entries = [];
  for (let i = 1; i < numLabels; i++) {
    if (!counts[i] || stats.intAt(i, cv.CC_STAT_AREA) <= MIN_AREA) continue;
    entries.push({ label: i, thickness: (sums[i] / counts[i]) * 2 });
  }

  if (entries.length < 2) {
    return new Set(entries.map((e) => e.label));
  }

  const sorted = [...entries].sort((a, b) => a.thickness - b.thickness);
  let maxGap = -1,
    gapIdx = 0;
  for (let i = 1; i < sorted.length; i++) {
    const gap = sorted[i].thickness - sorted[i - 1].thickness;
    if (gap > maxGap) {
      maxGap = gap;
      gapIdx = i;
    }
  }

  const thinCeiling = sorted[gapIdx - 1].thickness;
  // Require a genuinely bimodal split (a gap that's a large fraction of the
  // thin cluster's own thickness); otherwise treat every sizeable component
  // as wall-weight — safer than accidentally discarding real walls when
  // nothing thin stands out to separate from them.
  if (gapIdx === 0 || thinCeiling <= 0 || maxGap < thinCeiling * 0.6) {
    return new Set(sorted.map((e) => e.label));
  }

  return new Set(sorted.slice(gapIdx).map((e) => e.label));
}

function selectWallComponents(mask, labels, stats, numLabels, rows, cols) {
  const thickLabels = splitComponentsByThickness(
    mask,
    labels,
    stats,
    numLabels,
    rows,
    cols,
  );

  const wallData = new Uint8Array(rows * cols);
  const labelData = labels.data32S;
  for (let p = 0; p < rows * cols; p++) {
    if (thickLabels.has(labelData[p])) wallData[p] = 1;
  }
  return wallData;
}

/* ---------- Window detection (holes in the wall stroke) ---------- */
// On the reference symbol, a window isn't a separate line floating near a wall —
// it's a break in the wall's solid fill: the ink stays connected (thin strokes
// continue along both wall faces) but a gap opens up between them. That gap is an
// *interior hole* in the wall's connected-component shape. We find it directly
// with cv.findContours' hierarchy (RETR_CCOMP marks any contour that has a parent
// as a hole inside a filled region) instead of trying to track line thickness
// point-by-point along the skeleton, which washes out once the window sits in the
// middle of an otherwise-thick, unbroken wall run.

const WINDOW_MIN_ASPECT = 2.2; // long/short ratio required for a hole to read as a window slit, not a stray nick
const WINDOW_MAX_SHORT_FACTOR = 1.6; // hole's short side must be <= this * wall thickness
const WINDOW_MIN_LONG_FACTOR = 1.4; // hole's long side must be >= this * wall thickness
const WINDOW_MAX_LONG_PX = 220; // hard ceiling so a room interior is never mistaken for a window

function estimateWallThickness(wallMat, rows, cols) {
  const dist = new cv.Mat();
  cv.distanceTransform(wallMat, dist, cv.DIST_L2, 5);
  const distData = dist.data32F;
  const vals = [];
  for (let p = 0; p < rows * cols; p++) {
    if (distData[p] > 0.5) vals.push(distData[p]);
  }
  dist.delete();
  if (!vals.length) return 4;
  vals.sort((a, b) => a - b);
  const idx = Math.min(vals.length - 1, Math.floor(vals.length * 0.9));
  return vals[idx] * 2;
}

function findWindowHoles(wallMat, rows, cols) {
  const wallThickness = estimateWallThickness(wallMat, rows, cols);

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(
    wallMat,
    contours,
    hierarchy,
    cv.RETR_CCOMP,
    cv.CHAIN_APPROX_SIMPLE,
  );

  const windows = [];
  const hData = hierarchy.data32S; // rows of [next, previous, firstChild, parent]
  for (let i = 0; i < contours.size(); i++) {
    const parent = hData[i * 4 + 3];
    if (parent === -1) continue; // outer boundary, not a hole

    const contour = contours.get(i);
    const area = cv.contourArea(contour);
    if (area < 4) {
      contour.delete();
      continue;
    }
    const rect = cv.minAreaRect(contour);
    contour.delete();

    const long = Math.max(rect.size.width, rect.size.height);
    const short = Math.min(rect.size.width, rect.size.height);
    if (
      short <= 0 ||
      long <= 0 ||
      long / short < WINDOW_MIN_ASPECT ||
      short > wallThickness * WINDOW_MAX_SHORT_FACTOR ||
      long < wallThickness * WINDOW_MIN_LONG_FACTOR ||
      long > WINDOW_MAX_LONG_PX
    ) {
      continue;
    }

    // OpenCV's rotated-rect angle convention differs across builds/versions, so
    // derive the long-axis direction from which side is longer rather than
    // trusting the angle's sign. If windows come out rotated 90 degrees off in
    // practice, this is the spot to flip.
    const angleRad = (rect.angle * Math.PI) / 180;
    const dirRad =
      rect.size.width >= rect.size.height ? angleRad : angleRad + Math.PI / 2;
    const hx = Math.cos(dirRad) * (long / 2);
    const hy = Math.sin(dirRad) * (long / 2);

    windows.push({
      start: [rect.center.x - hx, rect.center.y - hy],
      end: [rect.center.x + hx, rect.center.y + hy],
      width: short,
    });
  }

  contours.delete();
  hierarchy.delete();
  return windows;
}

/* ---------- OpenCV segmentation ---------- */

function segmentMasks(canvas, image) {
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

    const wallData = selectWallComponents(mask, labels, stats, numLabels, rows, cols);

    const wallMat = new cv.Mat(rows, cols, cv.CV_8UC1);
    wallMat.data.set(wallData);
    const windows = findWindowHoles(wallMat, rows, cols);
    wallMat.delete();

    return { wallData, windows, cols, rows };
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

function skeletonizeAndTrace(binaryData, cols, rows) {
  const skeleton = zhangSuenThinning(binaryData, cols, rows);

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

  return traceSkeletonGraph(skeleton, cols, rows);
}

/* ---------- Keeping windows and walls mutually exclusive ---------- */
// A window hole leaves the wall's skeleton bifurcated into two thin stub
// lines hugging the sides of the opening (see findWindowHoles above) —
// without this step, both stubs come back as ordinary wall edges and the
// window still reads as solid wall. We drop those stub artifacts, then
// stitch a single straight edge across the gap so the wall PATH stays
// continuous (so a normal, solid wall gets built there) — Viewer3D is what
// actually carves the window-height opening out of that solid wall later.

const WINDOW_EXCLUDE_PAD = 4; // px: margin so stub artifacts hugging a window are reliably caught
const WINDOW_SNAP_TOLERANCE = 10; // px: snap a window's bridge endpoint onto a real wall endpoint if one is close by

function pointInWindowBox(pt, win, pad) {
  const dx = win.end[0] - win.start[0];
  const dy = win.end[1] - win.start[1];
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len,
    uy = dy / len;
  const nx = -uy,
    ny = ux;
  const cx = (win.start[0] + win.end[0]) / 2;
  const cy = (win.start[1] + win.end[1]) / 2;
  const relX = pt[0] - cx,
    relY = pt[1] - cy;
  const along = relX * ux + relY * uy;
  const across = relX * nx + relY * ny;
  const halfLen = len / 2 + pad;
  const halfWidth = (win.width || 4) / 2 + pad;
  return Math.abs(along) <= halfLen && Math.abs(across) <= halfWidth;
}

function edgeMostlyInsideWindows(edge, windows, pad) {
  if (edge.points.length === 0 || windows.length === 0) return false;
  for (const win of windows) {
    let insideCount = 0;
    for (const pt of edge.points) {
      if (pointInWindowBox(pt, win, pad)) insideCount++;
    }
    if (insideCount / edge.points.length > 0.6) return true;
  }
  return false;
}

function snapToNearestEndpoint(pt, edges, tolerance) {
  let best = pt;
  let bestDist = tolerance;
  for (const e of edges) {
    const candidates = [e.points[0], e.points[e.points.length - 1]];
    for (const cand of candidates) {
      const d = Math.hypot(cand[0] - pt[0], cand[1] - pt[1]);
      if (d < bestDist) {
        bestDist = d;
        best = cand;
      }
    }
  }
  return best;
}

/* ---------- Door-swing arc rejection ---------- */
// Door swing arcs are drawn as a thin quarter-circle touching the wall at
// both ends, so they get traced as their own branch off the wall skeleton —
// same mechanism that pulled window mullions in before that was fixed. A
// real wall path is either straight or has one isolated sharp corner; an arc
// is a long run of many *small* turns that all curve the same direction.
// IMPORTANT: this has to run on the RAW (dense, unsimplified) skeleton
// chain, not the corner-simplification output — a small door arc collapses
// to only a handful of points after approxPolyDP, which isn't enough
// samples to see a curve in. Resampling the raw chain at a fixed spacing
// also means classification doesn't depend on how many pixels the arc
// happens to span on screen.

const ARC_RESAMPLE_STEP_PX = 8; // spacing between smoothed sample points along the raw skeleton chain
const ARC_MIN_SAMPLES = 4; // need several resampled points before curvature is even measurable
const ARC_MAX_AVG_TURN = (45 * Math.PI) / 180; // radians (~45°) per resampled step; real corners jump close to 90° in one step, curve steps stay well under this
const ARC_MIN_TOTAL_ABS_TURN = (40 * Math.PI) / 180; // radians (~40°): total curvature must be substantial, not just skeleton jitter on a straight run
const ARC_MAX_SINGLE_TURN_FRACTION = 0.35; // no single resampled turn may account for more than this share of the total — real corners concentrate ~all their turning in one step, a curve spreads it out

// Smooth + resample the raw skeleton chain at an even spacing. This has to be
// a genuine moving-average (not just picking/interpolating between the two
// nearest raw points), otherwise per-pixel skeleton noise on an otherwise
// straight wall passes straight through untouched and gets mistaken for
// curvature. Averaging every raw point within +-step/2 of each sample
// position filters that noise out while a real curve's larger-scale bend
// survives fine.
function smoothedResample(points, closed, step) {
  const pts = closed ? [...points, points[0]] : points;
  if (pts.length < 2) return pts;

  const cum = [0];
  for (let i = 1; i < pts.length; i++) {
    cum.push(
      cum[i - 1] +
        Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]),
    );
  }
  const total = cum[cum.length - 1];
  if (total < step) return pts;

  const halfWin = step / 2;
  const result = [];
  for (let d = 0; d <= total; d += step) {
    let sx = 0,
      sy = 0,
      count = 0;
    for (let i = 0; i < pts.length; i++) {
      if (cum[i] >= d - halfWin && cum[i] <= d + halfWin) {
        sx += pts[i][0];
        sy += pts[i][1];
        count++;
      }
    }
    if (count === 0) {
      let bestI = 0,
        bestDist = Infinity;
      for (let i = 0; i < pts.length; i++) {
        const dd = Math.abs(cum[i] - d);
        if (dd < bestDist) {
          bestDist = dd;
          bestI = i;
        }
      }
      result.push([pts[bestI][0], pts[bestI][1]]);
    } else {
      result.push([sx / count, sy / count]);
    }
  }
  return result;
}

function turningAnglesOpen(points) {
  const turns = [];
  for (let i = 1; i < points.length - 1; i++) {
    const [ax, ay] = points[i - 1];
    const [bx, by] = points[i];
    const [cx, cy] = points[i + 1];
    const v1x = bx - ax,
      v1y = by - ay;
    const v2x = cx - bx,
      v2y = cy - by;
    const cross = v1x * v2y - v1y * v2x;
    const dot = v1x * v2x + v1y * v2y;
    turns.push(Math.atan2(cross, dot));
  }
  return turns;
}

function isDoorSwingArc(rawPoints, closed) {
  if (rawPoints.length < 4) return false;

  const resampled = smoothedResample(rawPoints, closed, ARC_RESAMPLE_STEP_PX);
  if (resampled.length < ARC_MIN_SAMPLES) return false;

  const turns = turningAnglesOpen(resampled);
  if (turns.length === 0) return false;
  const sumAbs = turns.reduce((s, t) => s + Math.abs(t), 0);
  if (sumAbs < 1e-6) return false;
  const avgAbs = sumAbs / turns.length;
  const maxAbs = Math.max(...turns.map((t) => Math.abs(t)));

  // Direction doesn't matter here on purpose: several arcs meeting (e.g. door
  // swings from adjacent openings touching in the middle, like a quatrefoil)
  // reverse curvature at each meeting point, which would break a
  // same-direction check. What's still true regardless of shape is that a
  // curve spreads its turning across many small-to-moderate steps, none of
  // which dominates — while real wall geometry concentrates nearly all its
  // turning into one sharp, isolated corner.
  return (
    avgAbs < ARC_MAX_AVG_TURN &&
    sumAbs > ARC_MIN_TOTAL_ABS_TURN &&
    maxAbs < sumAbs * ARC_MAX_SINGLE_TURN_FRACTION
  );
}


function extractWallGraph(canvas, image) {
  const {
    wallData,
    windows: rawWindows,
    cols,
    rows,
  } = segmentMasks(canvas, image);

  const { edges: rawWallEdges, nodes: wallJunctionNodes } = skeletonizeAndTrace(
    wallData,
    cols,
    rows,
  );

  const finalEdges = [];
  const cornerNodeSet = new Set();
  for (const raw of rawWallEdges) {
    if (isDoorSwingArc(raw.points, raw.closed)) continue; // curved door-swing line, not a wall
    const simplified = {
      points: simplifyChain(raw.points, raw.closed),
      closed: raw.closed,
    };
    const { edges: split, cornerNodes } = splitEdgeAtCorners(simplified);
    finalEdges.push(...split);
    for (const [x, y] of cornerNodes) cornerNodeSet.add(`${x},${y}`);
  }

  const cornerNodes = [...cornerNodeSet].map((s) => s.split(",").map(Number));
  const allNodes = [...wallJunctionNodes, ...cornerNodes];

  const border = PAD + 5;
  const inBounds = ([nx, ny]) =>
    nx >= border && nx < cols - border && ny >= border && ny < rows - border;

  const windows = rawWindows.filter(
    (win) => inBounds(win.start) && inBounds(win.end),
  );

  // Strip the thin ring-stub edges that trace around each window hole — these
  // are a skeletonization artifact of the hole, not real wall.
  const wallEdgesOnly = finalEdges.filter(
    (edge) => !edgeMostlyInsideWindows(edge, windows, WINDOW_EXCLUDE_PAD),
  );

  // Re-bridge across each window with a single straight edge so the wall path
  // stays unbroken (solid wall gets built there; the opening itself is cut in
  // Viewer3D at window height, not by leaving a full floor-to-ceiling gap
  // here). Snap bridge endpoints onto real wall endpoints nearby so the joint
  // lines up exactly instead of leaving a hairline seam.
  const bridgeEdges = windows.map((win) => ({
    points: [
      snapToNearestEndpoint(win.start, wallEdgesOnly, WINDOW_SNAP_TOLERANCE),
      snapToNearestEndpoint(win.end, wallEdgesOnly, WINDOW_SNAP_TOLERANCE),
    ],
    closed: false,
  }));

  const combinedEdges = [...wallEdgesOnly, ...bridgeEdges];

  const validNodeSet = new Set(
    allNodes
      .filter(inBounds)
      .filter(
        (n) => !windows.some((win) => pointInWindowBox(n, win, WINDOW_EXCLUDE_PAD)),
      )
      .map(([nx, ny]) => `${nx},${ny}`),
  );
  const filteredEdges = combinedEdges.filter((e) => e.points.every(inBounds));
  const filteredNodes = allNodes.filter(([nx, ny]) =>
    validNodeSet.has(`${nx},${ny}`),
  );

  return { edges: filteredEdges, nodes: filteredNodes, windows };
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
  const [windows, setWindows] = useState(null);
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
      const {
        edges: detectedEdges,
        nodes: detectedNodes,
        windows: detectedWindows,
      } = extractWallGraph(canvas, img);

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

      ctx.strokeStyle = "#00c853";
      ctx.lineWidth = 4;
      for (const win of detectedWindows) {
        ctx.beginPath();
        ctx.moveTo(win.start[0], win.start[1]);
        ctx.lineTo(win.end[0], win.end[1]);
        ctx.stroke();
      }

      setEdges(detectedEdges);
      setNodes(detectedNodes);
      setWindows(detectedWindows);
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
          windows={windows}
          pixelsPerMeter={pixelsPerMeter}
          wallHeightMeters={wallHeight}
          wallThicknessMeters={wallThickness}
        />
      )}
    </div>
  );
}