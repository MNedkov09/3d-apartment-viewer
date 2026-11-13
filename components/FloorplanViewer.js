"use client";

import { useRef, useState, useEffect, useCallback } from "react";
import Viewer3D from "./Viewer3D";

const CANVAS_W = 800;
const CANVAS_H = 600;

const CANVAS_STYLE = {
  border: "1px solid var(--line)",
  maxWidth: "100%",
  display: "block",
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

export default function FloorplanViewer({ imageSrc, blueprintTitle }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);

  const [edges, setEdges] = useState(null);
  const [windows, setWindows] = useState([]);
  const [doors, setDoors] = useState([]);
  const [furniture, setFurniture] = useState([]);
  const [fixtures, setFixtures] = useState([]);
  const [rooms, setRooms] = useState({});
  const [stats, setStats] = useState(null);
  const [showWalls, setShowWalls] = useState(true);
  const [showFurniture, setShowFurniture] = useState(true);
  const [showDoors, setShowDoors] = useState(true);
  const [showWindows, setShowWindows] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [view, setView] = useState("2d");
  const [pixelsPerMeter, setPixelsPerMeter] = useState(22);
  const [wallHeight, setWallHeight] = useState(2.5);
  const [wallThickness, setWallThickness] = useState(0.15);
  const [error, setError] = useState("");

  const cachedDetectionsRef = useRef(new Map());

  const drawOverlay = useCallback(
    (img, edgeList, winList, doorList, furnList, fixList) => {
      const canvas = canvasRef.current;
      if (!canvas || !img) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);

      const s = Math.min(CANVAS_W / img.width, CANVAS_H / img.height);
      const w = img.width * s;
      const h = img.height * s;
      const ox = (CANVAS_W - w) / 2;
      const oy = (CANVAS_H - h) / 2;
      ctx.drawImage(img, ox, oy, w, h);

      // 1. Walls (Red)
      if (showWalls && edgeList && edgeList.length > 0) {
        ctx.strokeStyle = "#e5484d";
        ctx.lineWidth = 3;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        for (const edge of edgeList) {
          if (edge.points.length < 2) continue;
          ctx.beginPath();
          ctx.moveTo(edge.points[0][0], edge.points[0][1]);
          for (let i = 1; i < edge.points.length; i++) {
            ctx.lineTo(edge.points[i][0], edge.points[i][1]);
          }
          if (edge.closed) ctx.closePath();
          ctx.stroke();
        }
      }

      // 2. Windows (Bright Cyan)
      if (showWindows && winList && winList.length > 0) {
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 5;
        ctx.lineCap = "round";
        for (const win of winList) {
          if (!win.start || !win.end) continue;
          ctx.beginPath();
          ctx.moveTo(win.start[0], win.start[1]);
          ctx.lineTo(win.end[0], win.end[1]);
          ctx.stroke();
        }
      }

      // 3. Doors (Amber / Orange)
      if (showDoors && doorList && doorList.length > 0) {
        ctx.strokeStyle = "#ff9800";
        ctx.lineWidth = 4;
        ctx.lineCap = "round";
        for (const door of doorList) {
          if (!door.start || !door.end) continue;
          ctx.beginPath();
          ctx.moveTo(door.start[0], door.start[1]);
          ctx.lineTo(door.end[0], door.end[1]);
          ctx.stroke();
        }
      }

      // 4. Unified Furniture & Fixtures (Emerald bounding boxes with labels)
      const allFurn = [...(furnList || []), ...(fixList || [])];
      if (showFurniture && allFurn.length > 0) {
        ctx.lineWidth = 1.5;
        for (const f of allFurn) {
          const bx = f.canvas_box || [0, 0, 0, 0];
          const bw = bx[2] - bx[0];
          const bh = bx[3] - bx[1];
          if (bw <= 0 || bh <= 0) continue;

          ctx.strokeStyle = "rgba(16, 185, 129, 0.95)";
          ctx.fillStyle = "rgba(16, 185, 129, 0.14)";
          ctx.fillRect(bx[0], bx[1], bw, bh);
          ctx.strokeRect(bx[0], bx[1], bw, bh);

          const label = (f.label || "item").replace(/_/g, " ").toUpperCase();
          const textW = label.length * 6.2 + 8;
          ctx.fillStyle = "rgba(16, 185, 129, 0.95)";
          ctx.fillRect(bx[0], Math.max(0, bx[1] - 14), textW, 14);
          ctx.fillStyle = "#ffffff";
          ctx.font = "bold 9px monospace";
          ctx.fillText(label, bx[0] + 4, Math.max(10, bx[1] - 3));
        }
      }
    },
    [showWalls, showWindows, showDoors, showFurniture],
  );

  // Re-draw when layer visibility toggles change
  useEffect(() => {
    if (imgRef.current && edges && view === "2d") {
      drawOverlay(imgRef.current, edges, windows, doors, furniture, fixtures);
    }
  }, [showWalls, showWindows, showDoors, showFurniture, edges, windows, doors, furniture, fixtures, view, drawOverlay]);

  const runDetection = useCallback(
    async (img) => {
      setProcessing(true);
      setError("");
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, CANVAS_W, CANVAS_H);
        const s = Math.min(CANVAS_W / img.width, CANVAS_H / img.height);
        const w = img.width * s;
        const h = img.height * s;
        const ox = (CANVAS_W - w) / 2;
        const oy = (CANVAS_H - h) / 2;
        ctx.drawImage(img, ox, oy, w, h);
      }

      try {
        // Export native-resolution image so detection runs on unpadded original dimensions
        const offscreen = document.createElement("canvas");
        offscreen.width = img.naturalWidth || img.width;
        offscreen.height = img.naturalHeight || img.height;
        const offCtx = offscreen.getContext("2d");
        offCtx.drawImage(img, 0, 0);

        const blob = await new Promise((resolve) =>
          offscreen.toBlob(resolve, "image/png"),
        );
        const formData = new FormData();
        formData.append("file", blob, "plan.png");
        formData.append("conf", "0.12");
        formData.append("use_hybrid", "true");
        formData.append("canvas_w", CANVAS_W.toString());
        formData.append("canvas_h", CANVAS_H.toString());

        const res = await fetch("/api/floorplan/detect", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          throw new Error(`Floorplan analysis failed (status ${res.status}).`);
        }

        const data = await res.json();
        const detectedEdges = data.wall_edges || [];
        const detectedWindows = data.windows || [];
        const detectedDoors = data.doors || [];
        const detectedFurniture = data.furniture || [];
        const detectedFixtures = data.fixtures || [];
        const detectedRooms = data.rooms || {};

        const resultObj = {
          edges: detectedEdges,
          windows: detectedWindows,
          doors: detectedDoors,
          furniture: detectedFurniture,
          fixtures: detectedFixtures,
          rooms: detectedRooms,
          stats: data.stats,
        };
        cachedDetectionsRef.current.set(imageSrc, resultObj);

        setEdges(detectedEdges);
        setWindows(detectedWindows);
        setDoors(detectedDoors);
        setFurniture(detectedFurniture);
        setFixtures(detectedFixtures);
        setRooms(detectedRooms);
        setStats(data.stats);

        drawOverlay(
          img,
          detectedEdges,
          detectedWindows,
          detectedDoors,
          detectedFurniture,
          detectedFixtures,
        );
      } catch (err) {
        console.error("Floorplan detection error:", err);
        setError(err.message || "Failed to analyze floorplan.");
      } finally {
        setProcessing(false);
      }
    },
    [drawOverlay, imageSrc],
  );

  useEffect(() => {
    if (!imageSrc) return;

    if (cachedDetectionsRef.current.has(imageSrc)) {
      const cached = cachedDetectionsRef.current.get(imageSrc);
      setEdges(cached.edges);
      setWindows(cached.windows);
      setDoors(cached.doors);
      setFurniture(cached.furniture);
      setFixtures(cached.fixtures);
      setRooms(cached.rooms || {});
      setStats(cached.stats);

      if (view === "2d") {
        if (imgRef.current && imgRef.current.src === imageSrc) {
          drawOverlay(
            imgRef.current,
            cached.edges,
            cached.windows,
            cached.doors,
            cached.furniture,
            cached.fixtures,
          );
        } else {
          const image = new Image();
          image.crossOrigin = "anonymous";
          image.onload = () => {
            imgRef.current = image;
            drawOverlay(
              image,
              cached.edges,
              cached.windows,
              cached.doors,
              cached.furniture,
              cached.fixtures,
            );
          };
          image.src = imageSrc;
        }
      }
      return;
    }

    if (view !== "2d") return;

    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      imgRef.current = image;
      runDetection(image);
    };
    image.onerror = () => {
      setError("Failed to load blueprint image.");
      setProcessing(false);
    };
    image.src = imageSrc;

    return () => {
      if (imgRef.current?.src === imageSrc) imgRef.current = null;
    };
  }, [imageSrc, view, drawOverlay, runDetection]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Layer & Analysis Toolbar */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: 10,
          padding: "10px 14px",
          background: "var(--paper-raised, #f5f5f5)",
          border: "1px solid var(--line, #ddd)",
          borderRadius: 6,
          fontSize: 13,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600 }}>Layers:</span>
          <label style={{ fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={showWalls}
              onChange={(e) => setShowWalls(e.target.checked)}
            />
            <span style={{ color: "#e5484d", fontWeight: 600 }}>Walls</span>
          </label>
          <label style={{ fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={showWindows}
              onChange={(e) => setShowWindows(e.target.checked)}
            />
            <span style={{ color: "#00acc1", fontWeight: 600 }}>Windows</span>
          </label>
          <label style={{ fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={showDoors}
              onChange={(e) => setShowDoors(e.target.checked)}
            />
            <span style={{ color: "#ff9800", fontWeight: 600 }}>Doors</span>
          </label>
          <label style={{ fontSize: 12, cursor: "pointer", display: "flex", alignItems: "center", gap: 4 }}>
            <input
              type="checkbox"
              checked={showFurniture}
              onChange={(e) => setShowFurniture(e.target.checked)}
            />
            <span style={{ color: "#10b981", fontWeight: 600 }}>Furniture</span>
          </label>
        </div>

        {stats && (
          <div
            style={{
              fontSize: 12,
              fontFamily: "var(--font-mono, monospace)",
              color: "var(--ink-muted, #555)",
            }}
          >
            {stats.wall_edges} walls · {stats.doors} doors · {stats.windows} windows · {(stats.furniture || 0) + (stats.fixtures || 0)} furniture
          </div>
        )}
      </div>

      {error && <p style={{ color: "red", fontSize: 12 }}>{error}</p>}
      {processing && (
        <p style={{ color: "var(--accent, #e5484d)", fontWeight: 600, fontSize: 13 }}>
          Extracting floorplan geometry...
        </p>
      )}

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
          doors={doors}
          furniture={furniture}
          fixtures={fixtures}
          rooms={rooms}
          pixelsPerMeter={pixelsPerMeter}
          wallHeightMeters={wallHeight}
          wallThicknessMeters={wallThickness}
          stats={stats}
        />
      )}
    </div>
  );
}
