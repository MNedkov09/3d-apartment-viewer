import { useEffect, useRef, useMemo, useCallback, useState } from "react";
import {
  Engine,
  Scene,
  HemisphericLight,
  DirectionalLight,
  Vector3,
  Color3,
  Color4,
  PBRMaterial,
  StandardMaterial,
  DynamicTexture,
  MeshBuilder,
  ArcRotateCamera,
  ShadowGenerator,
  SceneLoader,
  Matrix,
  CSG,
  Quaternion,
  TransformNode,
  Plane,
} from "@babylonjs/core";
import { GLTFFileLoader } from "@babylonjs/loaders";
import { buildWallMeshes } from "./wallGeometry";

// Register the GLTF loader once at module load time
SceneLoader.RegisterPlugin(new GLTFFileLoader());
SceneLoader.ShowLoadingScreen = false;

function rayGroundIntersect(scene, cam, canvas, flrY, ptrX, ptrY) {
  const ray = scene.createPickingRay(
    ptrX ?? canvas.clientWidth / 2,
    ptrY ?? canvas.clientHeight / 2,
    Matrix.Identity(),
    cam,
  );
  const plane = Plane.FromPositionAndNormal(
    new Vector3(0, flrY, 0),
    new Vector3(0, 1, 0),
  );
  const distance = ray.intersectsPlane(plane);
  if (distance === null || distance < 0) return null;
  return ray.origin.add(ray.direction.scale(distance));
}

function computeBounds(edges, pixelsPerMeter) {
  if (!edges || edges.length === 0) {
    return { cx: 0, cz: 0, w: 10, d: 10 };
  }
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const edge of edges) {
    for (const [x, y] of edge.points) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  const scale = 1 / pixelsPerMeter;
  return {
    cx: ((minX + maxX) / 2) * scale,
    cz: ((minY + maxY) / 2) * scale,
    w: Math.max((maxX - minX) * scale, 4),
    d: Math.max((maxY - minY) * scale, 4),
  };
}

function createFloorMaterial(style = "wood", scene) {
  const mat = new PBRMaterial(`floorMat_${style}`, scene);
  const tex = new DynamicTexture(`floorTex_${style}`, { width: 512, height: 512 }, scene, true);
  const ctx = tex.getContext();

  if (style === "wood") {
    // Warm Scandinavian Oak Hardwood Parquet
    ctx.fillStyle = "#dfc8a5";
    ctx.fillRect(0, 0, 512, 512);

    const plankH = 32;
    for (let y = 0; y < 512; y += plankH) {
      const plankW = 128;
      const xOffset = ((y / plankH) % 2) * 64;
      for (let x = -64; x < 512 + 64; x += plankW) {
        const tone = 205 + Math.floor(Math.sin(x * 12 + y * 7) * 12);
        ctx.fillStyle = `rgb(${tone + 18}, ${tone}, ${tone - 26})`;
        ctx.fillRect(x + xOffset, y, plankW - 2, plankH - 2);

        // Fine grain lines
        ctx.fillStyle = "rgba(105, 75, 35, 0.08)";
        ctx.fillRect(x + xOffset, y + 6, plankW - 2, 2);
        ctx.fillRect(x + xOffset, y + 16, plankW - 2, 2);
      }
      ctx.fillStyle = "#9c815e";
      ctx.fillRect(0, y + plankH - 2, 512, 2);
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.42;
    mat.metallic = 0.02;
  } else if (style === "tile") {
    // Modern Architectural Neutral Tile
    ctx.fillStyle = "#b0b4b9";
    ctx.fillRect(0, 0, 512, 512);
    const tileSize = 128;
    for (let y = 0; y < 512; y += tileSize) {
      for (let x = 0; x < 512; x += tileSize) {
        const tone = 230 + Math.floor(Math.cos(x + y) * 8);
        ctx.fillStyle = `rgb(${tone}, ${tone}, ${tone})`;
        ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
      }
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.28;
    mat.metallic = 0.08;
  } else {
    // Minimalist Contemporary Studio Concrete
    mat.albedoColor = new Color3(0.90, 0.90, 0.88);
    mat.roughness = 0.65;
    mat.metallic = 0.02;
  }
  return mat;
}

function createBathroomFixtures(fixtures, opts, scene) {
  if (!fixtures || fixtures.length === 0) return [];
  const { scale } = opts;

  const ceramicMat = new PBRMaterial("ceramicMat", scene);
  ceramicMat.albedoColor = new Color3(0.96, 0.96, 0.96);
  ceramicMat.metallic = 0.05;
  ceramicMat.roughness = 0.15;

  const chromeMat = new PBRMaterial("chromeMat", scene);
  chromeMat.albedoColor = new Color3(0.85, 0.85, 0.85);
  chromeMat.metallic = 0.95;
  chromeMat.roughness = 0.1;

  const glassMat = new PBRMaterial("showerGlassMat", scene);
  glassMat.albedoColor = new Color3(0.7, 0.9, 1.0);
  glassMat.alpha = 0.35;
  glassMat.metallic = 0.05;
  glassMat.roughness = 0.05;

  const fixtureMeshes = [];
  fixtures.forEach((f, i) => {
    const canvasPt = f.canvas_center || f.center;
    if (!canvasPt) return;
    const cx = canvasPt[0] * scale;
    const cz = canvasPt[1] * scale;

    if (f.label === "toilet") {
      const base = MeshBuilder.CreateCylinder(
        `toilet_bowl_${i}`,
        { diameterTop: 0.38, diameterBottom: 0.28, height: 0.42 },
        scene,
      );
      base.position.set(cx, 0.21, cz);
      base.material = ceramicMat;
      base.castShadow = true;
      base.receiveShadow = true;

      const cistern = MeshBuilder.CreateBox(
        `toilet_tank_${i}`,
        { width: 0.42, depth: 0.22, height: 0.44 },
        scene,
      );
      cistern.position.set(cx, 0.52, cz - 0.14);
      cistern.material = ceramicMat;
      cistern.castShadow = true;
      cistern.receiveShadow = true;
      fixtureMeshes.push(base, cistern);
    } else if (f.label === "sink") {
      const vanity = MeshBuilder.CreateBox(
        `vanity_${i}`,
        { width: 0.6, depth: 0.45, height: 0.82 },
        scene,
      );
      vanity.position.set(cx, 0.41, cz);
      vanity.material = ceramicMat;
      vanity.castShadow = true;
      vanity.receiveShadow = true;

      const tap = MeshBuilder.CreateCylinder(
        `tap_${i}`,
        { diameter: 0.04, height: 0.16 },
        scene,
      );
      tap.position.set(cx, 0.86, cz - 0.14);
      tap.material = chromeMat;
      tap.castShadow = true;
      fixtureMeshes.push(vanity, tap);
    } else if (f.label === "shower") {
      const tray = MeshBuilder.CreateBox(
        `shower_tray_${i}`,
        { width: 0.85, depth: 0.85, height: 0.08 },
        scene,
      );
      tray.position.set(cx, 0.04, cz);
      tray.material = ceramicMat;
      tray.receiveShadow = true;

      const glass = MeshBuilder.CreateBox(
        `shower_glass_${i}`,
        { width: 0.85, depth: 0.04, height: 2.0 },
        scene,
      );
      glass.position.set(cx, 1.04, cz + 0.4);
      glass.material = glassMat;
      glass.castShadow = true;
      fixtureMeshes.push(tray, glass);
    }
  });

  return fixtureMeshes;
}

function cutWindowOpenings(windows, wallMeshes, opts, scene) {
  const { scale, wallThickness, sillHeight, windowHeight, wallHeight } = opts;
  const remainingMeshes = [...wallMeshes];

  for (const win of windows) {
    const startX = win.start[0] * scale;
    const startZ = win.start[1] * scale;
    const endX = win.end[0] * scale;
    const endZ = win.end[1] * scale;
    const dX = endX - startX;
    const dZ = endZ - startZ;
    const openingLength = Math.hypot(dX, dZ);
    if (openingLength < 0.05) continue;

    const centerX = (startX + endX) / 2;
    const centerZ = (startZ + endZ) / 2;
    const centerY = Math.min(sillHeight + windowHeight / 2, wallHeight - 0.2);
    const angle = Math.atan2(dZ, dX);

    const cutter = MeshBuilder.CreateBox(
      "windowCutter",
      {
        width: openingLength + 0.04,
        height: windowHeight,
        depth: wallThickness * 2.5,
      },
      scene,
    );
    cutter.position.set(centerX, centerY, centerZ);
    cutter.rotation.y = angle;
    cutter.computeWorldMatrix(true);

    const cutterBounds = cutter.getBoundingInfo().boundingBox;

    for (let i = 0; i < remainingMeshes.length; i++) {
      const wallMesh = remainingMeshes[i];
      if (!wallMesh || wallMesh.isDisposed()) continue;

      wallMesh.computeWorldMatrix(true);
      const wallBounds = wallMesh.getBoundingInfo().boundingBox;
      const overlaps =
        wallBounds.maximumWorld.x >= cutterBounds.minimumWorld.x &&
        wallBounds.minimumWorld.x <= cutterBounds.maximumWorld.x &&
        wallBounds.maximumWorld.y >= cutterBounds.minimumWorld.y &&
        wallBounds.minimumWorld.y <= cutterBounds.maximumWorld.y &&
        wallBounds.maximumWorld.z >= cutterBounds.minimumWorld.z &&
        wallBounds.minimumWorld.z <= cutterBounds.maximumWorld.z;
      if (!overlaps) continue;

      try {
        const wallCSG = CSG.FromMesh(wallMesh);
        const cutterCSG = CSG.FromMesh(cutter);
        const resultCSG = wallCSG.subtract(cutterCSG);
        const material = wallMesh.material;
        const name = wallMesh.name;
        const resultMesh = resultCSG.toMesh(name, material, scene);
        resultMesh.material = material;
        wallMesh.dispose();
        remainingMeshes[i] = resultMesh;
      } catch (e) {
        // Continue if CSG encounters collinear edge
      }
    }
    cutter.dispose();
  }
  return remainingMeshes;
}

function cutDoorOpenings(doors, wallMeshes, opts, scene) {
  const { scale, wallThickness, doorHeight = 2.1 } = opts;
  const remainingMeshes = [...wallMeshes];

  for (const door of doors) {
    if (!door.start || !door.end) continue;
    const startX = door.start[0] * scale;
    const startZ = door.start[1] * scale;
    const endX = door.end[0] * scale;
    const endZ = door.end[1] * scale;
    const dX = endX - startX;
    const dZ = endZ - startZ;
    const openingLength = Math.hypot(dX, dZ);
    if (openingLength < 0.05) continue;

    const centerX = (startX + endX) / 2;
    const centerZ = (startZ + endZ) / 2;
    const centerY = doorHeight / 2;
    const angle = Math.atan2(dZ, dX);

    const cutter = MeshBuilder.CreateBox(
      "doorCutter",
      {
        width: openingLength + 0.06,
        height: doorHeight,
        depth: wallThickness * 2.5,
      },
      scene,
    );
    cutter.position.set(centerX, centerY, centerZ);
    cutter.rotation.y = angle;
    cutter.computeWorldMatrix(true);

    const cutterBounds = cutter.getBoundingInfo().boundingBox;

    for (let i = 0; i < remainingMeshes.length; i++) {
      const wallMesh = remainingMeshes[i];
      if (!wallMesh || wallMesh.isDisposed()) continue;

      wallMesh.computeWorldMatrix(true);
      const wallBounds = wallMesh.getBoundingInfo().boundingBox;
      const overlaps =
        wallBounds.maximumWorld.x >= cutterBounds.minimumWorld.x &&
        wallBounds.minimumWorld.x <= cutterBounds.maximumWorld.x &&
        wallBounds.maximumWorld.y >= cutterBounds.minimumWorld.y &&
        wallBounds.minimumWorld.y <= cutterBounds.maximumWorld.y &&
        wallBounds.maximumWorld.z >= cutterBounds.minimumWorld.z &&
        wallBounds.minimumWorld.z <= cutterBounds.maximumWorld.z;
      if (!overlaps) continue;

      try {
        const wallCSG = CSG.FromMesh(wallMesh);
        const cutterCSG = CSG.FromMesh(cutter);
        const resultCSG = wallCSG.subtract(cutterCSG);
        const material = wallMesh.material;
        const name = wallMesh.name;
        const resultMesh = resultCSG.toMesh(name, material, scene);
        resultMesh.material = material;
        wallMesh.dispose();
        remainingMeshes[i] = resultMesh;
      } catch (e) {
        // Continue if CSG encounters collinear edge
      }
    }
    cutter.dispose();
  }
  return remainingMeshes;
}

async function fetchUserModels() {
  try {
    const res = await fetch("/api/models?limit=100");
    if (!res.ok) return [];
    const data = await res.json();
    return (data.models || []).map((m) => ({
      name: m.name,
      url: m.file_path,
      thumbnail: m.thumbnail_path,
    }));
  } catch {
    return [];
  }
}

function exportSceneJSON(
  edges,
  windows,
  doors,
  placedModels,
  wallHeightMeters,
  wallThicknessMeters,
  sillHeightMeters,
  windowHeightMeters,
  cameraDistance,
) {
  if (!edges || edges.length === 0) {
    throw new Error("Cannot export: no floorplan edges found");
  }

  const placedData = placedModels.map((p) => {
    const rotation = Array.isArray(p.rotation)
      ? p.rotation
      : [0, p.rotation ?? 0, 0];
    return {
      name: p.name,
      url: p.url,
      position: [p.position[0], p.position[1] ?? 0, p.position[2]],
      rotation,
    };
  });

  const sceneData = {
    version: 1,
    floorplan: {
      edges,
      windows: windows || [],
      doors: doors || [],
      wallHeightMeters: wallHeightMeters ?? 2.7,
      wallThicknessMeters: wallThicknessMeters ?? 0.15,
      sillHeightMeters: sillHeightMeters ?? 0.9,
      windowHeightMeters: windowHeightMeters ?? 1.2,
      cameraDistance: cameraDistance ?? 10,
    },
    placedModels: placedData,
    exportedAt: new Date().toISOString(),
  };

  const blob = new Blob([JSON.stringify(sceneData, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const timestamp = new Date().toISOString().replace(/[T:]/g, "-").slice(0, 19);
  a.download = `scene-${timestamp}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

async function importSceneJSON(file) {
  let text = await file.text();
  if (!text.trim()) throw new Error("Empty file");
  const data = JSON.parse(text);
  if (!data.floorplan || !Array.isArray(data.floorplan.edges)) {
    throw new Error("Invalid floorplan file format");
  }
  return data;
}

function ModelPanel({
  availableModels,
  placedModels,
  draggedModel,
  onPlaceModel,
  onExportScene,
  onImportScene,
  onRemovePlacedModel,
  onClearAllModels,
  onAutoFurnish,
  furnitureCount = 0,
  isAutoFurnishing = false,
}) {
  return (
    <div
      style={{
        width: 220,
        background: "rgba(24, 27, 32, 0.88)",
        backdropFilter: "blur(12px)",
        color: "#fff",
        padding: "12px 14px",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 8,
        fontSize: 12,
        fontFamily: "var(--font-mono, monospace)",
        borderLeft: "1px solid rgba(255,255,255,0.1)",
      }}
    >
      {furnitureCount > 0 && onAutoFurnish && (
        <button
          onClick={onAutoFurnish}
          disabled={isAutoFurnishing}
          style={{
            background: isAutoFurnishing ? "#6b7280" : "#10b981",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            padding: "9px 12px",
            fontSize: 12,
            fontWeight: 600,
            cursor: isAutoFurnishing ? "wait" : "pointer",
            textAlign: "center",
            boxShadow: "0 2px 6px rgba(16, 185, 129, 0.3)",
            transition: "all 0.15s ease",
          }}
        >
          {isAutoFurnishing ? "⏳ Furnishing..." : `✨ Auto-Furnish (${furnitureCount})`}
        </button>
      )}

      {placedModels.length > 0 && onClearAllModels && (
        <button
          onClick={onClearAllModels}
          style={{
            background: "rgba(239, 68, 68, 0.15)",
            color: "#fca5a5",
            border: "1px solid rgba(239, 68, 68, 0.3)",
            borderRadius: 6,
            padding: "5px 10px",
            fontSize: 11,
            fontWeight: 500,
            cursor: "pointer",
            textAlign: "center",
          }}
        >
          🗑️ Clear Furniture ({placedModels.length})
        </button>
      )}

      <div
        style={{
          fontWeight: 600,
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "rgba(255,255,255,0.5)",
          marginTop: 4,
        }}
      >
        Models Catalog
      </div>
      {availableModels.length === 0 && (
        <div style={{ color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
          No models yet.
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 180, overflowY: "auto" }}>
        {availableModels.map((m, i) => (
          <div
            key={i}
            onClick={() => onPlaceModel(m)}
            style={{
              padding: "5px 8px",
              borderRadius: 4,
              cursor: "pointer",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid transparent",
              display: "flex",
              alignItems: "center",
              gap: 8,
              transition: "background 0.1s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.12)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.06)";
            }}
          >
            {m.thumbnail ? (
              <img
                src={m.thumbnail}
                alt=""
                width={32}
                height={32}
                style={{ flexShrink: 0, objectFit: "contain" }}
                onError={(e) => {
                  e.currentTarget.style.display = "none";
                }}
              />
            ) : (
              <div style={{ width: 16, height: 16, flexShrink: 0 }} />
            )}
            <span
              style={{
                flex: 1,
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                fontSize: 11,
              }}
            >
              {m.name}
            </span>
            <span
              style={{
                fontSize: 10,
                background: "rgba(255,255,255,0.15)",
                padding: "1px 5px",
                borderRadius: 3,
              }}
            >
              +
            </span>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 6,
          fontWeight: 600,
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "rgba(255,255,255,0.5)",
        }}
      >
        Scene State
      </div>
      <div style={{ display: "flex", gap: 6 }}>
        <button
          onClick={onExportScene}
          disabled={placedModels.length === 0}
          style={{
            flex: 1,
            padding: "6px 0",
            background: "rgba(255,255,255,0.1)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: 4,
            cursor: placedModels.length === 0 ? "not-allowed" : "pointer",
            opacity: placedModels.length === 0 ? 0.4 : 1,
            fontSize: 10,
          }}
        >
          💾 Export
        </button>
        <label
          style={{
            flex: 1,
            padding: "6px 0",
            background: "rgba(255,255,255,0.1)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: 4,
            cursor: "pointer",
            fontSize: 10,
            textAlign: "center",
          }}
        >
          📂 Import
          <input
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onImportScene(file);
              e.target.value = "";
            }}
          />
        </label>
      </div>

      {placedModels.length > 0 && (
        <>
          <div
            style={{
              marginTop: 6,
              fontWeight: 600,
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "rgba(255,255,255,0.5)",
            }}
          >
            In Scene ({placedModels.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 3, maxHeight: 150, overflowY: "auto" }}>
            {placedModels.map((m, i) => {
              const isDragging = draggedModel && draggedModel.mesh === m.mesh;
              return (
                <div
                  key={`placed-${i}`}
                  style={{
                    padding: "4px 6px",
                    borderRadius: 4,
                    background: isDragging
                      ? "rgba(255,255,255,0.2)"
                      : "rgba(255,255,255,0.06)",
                    border: isDragging
                      ? "1px solid rgba(255,255,255,0.4)"
                      : "1px solid transparent",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 11,
                  }}
                >
                  <span
                    style={{
                      flex: 1,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {m.name}
                  </span>
                  <button
                    onPointerDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onRemovePlacedModel(m);
                    }}
                    title="Remove from scene"
                    type="button"
                    style={{
                      background: "rgba(255,80,80,0.25)",
                      color: "#fff",
                      border: "none",
                      borderRadius: 3,
                      width: 16,
                      height: 16,
                      fontSize: 9,
                      cursor: "pointer",
                    }}
                  >
                    ✕
                  </button>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export default function Viewer3D({
  edges,
  windows = [],
  doors = [],
  furniture = [],
  fixtures = [],
  pixelsPerMeter = 50,
  wallHeightMeters = 2.7,
  wallThicknessMeters = 0.15,
  sillHeightMeters = 0.9,
  windowHeightMeters = 1.2,
}) {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const wallMeshesRef = useRef([]);
  const keysRef = useRef(new Set());
  const floorMeshRef = useRef(null);
  const floorMatRef = useRef(null);
  const draggedMeshRef = useRef(null);
  const selectedMeshRef = useRef(null);
  const dragOffsetRef = useRef(new Vector3());
  const dragStartPosRef = useRef(new Vector3());
  const isDraggingRef = useRef(false);
  const modelRotationRef = useRef(new Map());
  const yawAngleRef = useRef(new Map());
  const meshToModelRef = useRef(new Map());
  const floorYRef = useRef(-0.05);
  const removePlacedModelRef = useRef(() => {});

  const edgesRef = useRef([]);
  const windowsRef = useRef([]);
  const doorsRef = useRef([]);
  const containerRef = useRef(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [placedModels, setPlacedModels] = useState([]);
  const [draggedModel, setDraggedModel] = useState(null);
  const [isAutoFurnishing, setIsAutoFurnishing] = useState(false);
  const [floorStyle, setFloorStyle] = useState("wood");
  const [wallMode, setWallMode] = useState("full"); // "full" (2.7m) | "cutaway" (1.3m)

  useEffect(() => {
    let cancelled = false;
    fetchUserModels().then((apiModels) => {
      if (cancelled) return;
      setAvailableModels(apiModels);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const bounds = useMemo(
    () => computeBounds(edges, pixelsPerMeter),
    [edges, pixelsPerMeter],
  );

  const cameraDistance = useMemo(() => {
    if (!edges || edges.length === 0) return 10;
    let maxExtent = 1;
    const scale = 1 / pixelsPerMeter;
    for (const edge of edges) {
      for (const [x, y] of edge.points) {
        maxExtent = Math.max(
          maxExtent,
          Math.abs(x * scale),
          Math.abs(y * scale),
        );
      }
    }
    return Math.max(maxExtent * 1.6, 8);
  }, [edges, pixelsPerMeter]);

  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen();
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
      requestAnimationFrame(() => {
        engineRef.current?.resize();
      });
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  const setViewpoint = useCallback(
    (mode) => {
      const cam = cameraRef.current;
      if (!cam) return;
      if (mode === "top") {
        cam.alpha = -Math.PI / 2;
        cam.beta = 0.05;
        cam.radius = cameraDistance * 1.15;
        cam.target.set(bounds.cx, 0, bounds.cz);
      } else if (mode === "iso") {
        cam.alpha = -Math.PI / 4;
        cam.beta = Math.PI / 3;
        cam.radius = cameraDistance;
        cam.target.set(bounds.cx, wallHeightMeters / 2, bounds.cz);
      } else if (mode === "walk") {
        cam.alpha = -Math.PI / 4;
        cam.beta = Math.PI / 2 - 0.05;
        cam.radius = 4;
        cam.target.set(bounds.cx, 1.6, bounds.cz);
      } else if (mode === "living") {
        cam.alpha = -Math.PI / 3;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.55;
        cam.target.set(bounds.cx - bounds.w * 0.2, 1.2, bounds.cz + bounds.d * 0.15);
      } else if (mode === "bed") {
        cam.alpha = -Math.PI / 1.5;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.55;
        cam.target.set(bounds.cx + bounds.w * 0.25, 1.2, bounds.cz - bounds.d * 0.2);
      }
    },
    [bounds, cameraDistance, wallHeightMeters],
  );

  // Robust model importer: binds imported meshes to a dedicated TransformNode
  const loadSingleModel = useCallback(
    async (url, spawnPoint, rotationY = 0, name = "Furniture") => {
      const scene = sceneRef.current;
      if (!scene) return null;

      try {
        const result = await SceneLoader.ImportMeshAsync("", "", url, scene);
        const { meshes, transformNodes } = result;
        if (!meshes || meshes.length === 0) return null;

        const modelId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
        const modelRoot = new TransformNode(modelId, scene);

        const topNodes = [
          ...transformNodes.filter((t) => !t.parent),
          ...meshes.filter((m) => !m.parent),
        ];
        topNodes.forEach((node) => {
          node.parent = modelRoot;
        });

        // Compute local bounding box to align model bottom with ground
        let minLocalY = Infinity;
        for (const m of meshes) {
          if (m.getTotalVertices && m.getTotalVertices() > 0) {
            const b = m.getBoundingInfo().boundingBox;
            if (b.minimum.y < minLocalY) minLocalY = b.minimum.y;
            m.castShadow = true;
            m.receiveShadow = true;
            if (m.material instanceof PBRMaterial) {
              m.material.environmentIntensity = 0.8;
            }
          }
        }

        const yOffset = isFinite(minLocalY) ? -minLocalY : 0;
        modelRoot.position.set(spawnPoint.x, spawnPoint.y + yOffset, spawnPoint.z);
        if (rotationY) {
          modelRoot.rotation.y = rotationY;
        }

        const entry = {
          name,
          position: [spawnPoint.x - bounds.cx, yOffset, spawnPoint.z - bounds.cz],
          rotation: [0, rotationY, 0],
          scaling: [1, 1, 1],
          url,
          mesh: modelRoot,
        };

        yawAngleRef.current.set(modelRoot, rotationY);
        modelRotationRef.current.set(modelRoot, {
          initialQuaternion: Quaternion.RotationYawPitchRoll(rotationY, 0, 0),
        });

        for (const m of meshes) {
          meshToModelRef.current.set(m, { entry, root: modelRoot });
        }

        return entry;
      } catch (err) {
        console.warn("Failed to load model:", url, err);
        return null;
      }
    },
    [bounds],
  );

  const placeModelInScene = useCallback(
    async (apiModel) => {
      const cam = cameraRef.current;
      const scene = sceneRef.current;
      const canvas = canvasRef.current;
      if (!scene || !cam || !canvas) return;

      const centerHit = rayGroundIntersect(
        scene,
        cam,
        canvas,
        floorYRef.current,
        canvas.clientWidth / 2,
        canvas.clientHeight / 2,
      );
      if (!centerHit) return;

      const url = apiModel.url;
      const name = apiModel.name || url.split("/").pop().replace(/\.[^.]+$/, "");
      const entry = await loadSingleModel(url, centerHit, 0, name);
      if (entry) {
        setPlacedModels((prev) => [...prev, entry]);
      }
    },
    [loadSingleModel],
  );

  const handleAutoFurnish = useCallback(async () => {
    if (!furniture || furniture.length === 0 || !sceneRef.current || isAutoFurnishing) return;
    setIsAutoFurnishing(true);
    const scale = 1 / pixelsPerMeter;

    const added = [];
    for (const f of furniture) {
      if (!f.model_file) continue;
      const url = `/models/${f.model_file}`;
      const name = (f.label || "Furniture").replace(/_/g, " ");
      const canvasPt = f.canvas_center || f.center;
      if (!canvasPt) continue;
      const worldX = canvasPt[0] * scale;
      const worldZ = canvasPt[1] * scale;
      const spawnPoint = new Vector3(worldX, 0, worldZ);
      const angle = f.angle_rad || 0;

      const entry = await loadSingleModel(url, spawnPoint, angle, name);
      if (entry) {
        added.push(entry);
      }
    }

    if (added.length > 0) {
      setPlacedModels((prev) => [...prev, ...added]);
    }
    setIsAutoFurnishing(false);
  }, [furniture, pixelsPerMeter, loadSingleModel, isAutoFurnishing]);

  const removePlacedModel = useCallback((entryToRemove) => {
    const root =
      entryToRemove.mesh ||
      [...meshToModelRef.current.values()].find(
        (mapping) => mapping.entry === entryToRemove,
      )?.root;
    if (!root) return;

    for (const [mesh, mapping] of meshToModelRef.current.entries()) {
      if (mapping.entry === entryToRemove || mapping.root === root) {
        meshToModelRef.current.delete(mesh);
      }
    }

    modelRotationRef.current.delete(root);
    yawAngleRef.current.delete(root);

    if (selectedMeshRef.current === root) selectedMeshRef.current = null;
    if (draggedMeshRef.current === root) {
      draggedMeshRef.current = null;
      isDraggingRef.current = false;
      setDraggedModel(null);
      cameraRef.current?.attachControl(canvasRef.current, true);
    }

    try {
      root.dispose(false, true);
    } catch (e) {}

    setPlacedModels((prev) => prev.filter((p) => p.mesh !== root));
  }, []);

  const clearAllModels = useCallback(() => {
    const roots = new Set();
    for (const { root } of meshToModelRef.current.values()) {
      roots.add(root);
    }
    roots.forEach((root) => {
      try {
        root.dispose(false, true);
      } catch (e) {}
    });
    meshToModelRef.current.clear();
    modelRotationRef.current.clear();
    yawAngleRef.current.clear();
    selectedMeshRef.current = null;
    draggedMeshRef.current = null;
    isDraggingRef.current = false;
    setDraggedModel(null);
    setPlacedModels([]);
  }, []);

  const handleExportScene = useCallback(() => {
    exportSceneJSON(
      edgesRef.current,
      windowsRef.current,
      doorsRef.current,
      placedModels,
      wallHeightMeters,
      wallThicknessMeters,
      sillHeightMeters,
      windowHeightMeters,
      cameraDistance,
    );
  }, [
    placedModels,
    wallHeightMeters,
    wallThicknessMeters,
    sillHeightMeters,
    windowHeightMeters,
    cameraDistance,
  ]);

  const handleImportScene = useCallback(
    async (file) => {
      try {
        const sceneData = await importSceneJSON(file);
        if (sceneData.placedModels && sceneData.placedModels.length > 0) {
          clearAllModels();
          for (const m of sceneData.placedModels) {
            const worldX = (m.position?.[0] ?? 0) + bounds.cx;
            const worldZ = (m.position?.[2] ?? 0) + bounds.cz;
            const angle = m.rotation?.[1] ?? 0;
            const entry = await loadSingleModel(m.url, new Vector3(worldX, 0, worldZ), angle, m.name);
            if (entry) {
              setPlacedModels((prev) => [...prev, entry]);
            }
          }
        }
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      }
    },
    [bounds, clearAllModels, loadSingleModel],
  );

  const handleKeyDown = useCallback((e) => {
    keysRef.current.add(e.key.toLowerCase());

    if (e.key === "Delete" || e.key === "Backspace") {
      const root = isDraggingRef.current
        ? draggedMeshRef.current
        : selectedMeshRef.current;
      if (!root) return;
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      e.preventDefault();

      const mapping = [...meshToModelRef.current.values()].find((m) => m.root === root);
      if (mapping) {
        removePlacedModelRef.current(mapping.entry);
      }
      return;
    }

    if (e.key === "r" || e.key === "R") {
      const root = isDraggingRef.current
        ? draggedMeshRef.current
        : selectedMeshRef.current;
      if (!root) return;

      e.preventDefault();
      const delta = e.shiftKey ? -Math.PI / 4 : Math.PI / 4;
      const prevYaw = yawAngleRef.current.get(root) ?? (root.rotation.y || 0);
      const newYaw = prevYaw + delta;
      root.rotation.y = newYaw;
      yawAngleRef.current.set(root, newYaw);

      const mapping = [...meshToModelRef.current.values()].find((m) => m.root === root);
      if (mapping?.entry?.rotation) {
        mapping.entry.rotation[1] = newYaw;
      }
    }
  }, []);

  const handleKeyUp = useCallback((e) => {
    keysRef.current.delete(e.key.toLowerCase());
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [handleKeyDown, handleKeyUp]);

  useEffect(() => {
    removePlacedModelRef.current = removePlacedModel;
  }, [removePlacedModel]);

  // Main 3D Babylon.js Scene Initialization
  useEffect(() => {
    if (!canvasRef.current || !edges || edges.length === 0) return;

    if (sceneRef.current) {
      wallMeshesRef.current.forEach((m) => m.dispose());
      sceneRef.current.dispose();
      engineRef.current?.dispose();
    }

    const engine = new Engine(canvasRef.current, true, {
      preserveDrawingBuffer: false,
      stencil: false,
    });
    engineRef.current = engine;

    const scene = new Scene(engine);
    sceneRef.current = scene;
    scene.clearColor = new Color4(0.94, 0.96, 0.98, 1);

    const effectiveWallHeight = wallMode === "cutaway" ? 1.3 : wallHeightMeters;

    const camera = new ArcRotateCamera(
      "camera",
      -Math.PI / 4,
      Math.PI / 3,
      cameraDistance,
      new Vector3(bounds.cx, effectiveWallHeight / 2, bounds.cz),
      scene,
    );
    cameraRef.current = camera;
    camera.attachControl(canvasRef.current, true);
    camera.lowerRadiusLimit = 2;
    camera.upperRadiusLimit = 150;
    camera.lowerBetaLimit = 0.05;
    camera.upperBetaLimit = Math.PI / 2;
    camera.alphaRotationSpeed = 10;
    camera.betaRotationSpeed = 10;

    const wasdSpeed = 8;

    // 1. Natural Sky Hemisphere Light
    const hemiLight = new HemisphericLight(
      "hemiLight",
      new Vector3(0, 1, 0),
      scene,
    );
    hemiLight.intensity = 0.72;
    hemiLight.diffuse = new Color3(1.0, 0.98, 0.95);
    hemiLight.groundColor = new Color3(0.40, 0.38, 0.35);

    // 2. Warm Sun Directional Light with soft shadows
    const sunDir = new Vector3(-0.6, -1.2, -0.7).normalize();
    const dirLight = new DirectionalLight("dirLight", sunDir, scene);
    dirLight.position = new Vector3(bounds.cx + 12, 18, bounds.cz + 12);
    dirLight.intensity = 1.25;
    dirLight.diffuse = new Color3(1.0, 0.97, 0.90);

    const shadowGen = new ShadowGenerator(2048, dirLight);
    shadowGen.useBlurExponentialShadowMap = true;
    shadowGen.blurKernel = 28;
    shadowGen.darkness = 0.45;
    shadowGen.bias = 0.002;

    // 3. Interior Fill Light
    const fillLight = new DirectionalLight(
      "fillLight",
      new Vector3(0.6, -0.8, 0.7).normalize(),
      scene,
    );
    fillLight.intensity = 0.35;
    fillLight.diffuse = new Color3(0.88, 0.92, 1.0);

    // 4. Floor with Procedural Material
    const halfThick = wallThicknessMeters / 2;
    const floor = MeshBuilder.CreateGround(
      "floor",
      {
        width: bounds.w + halfThick * 4,
        height: bounds.d + halfThick * 4,
        subdivisions: 2,
        onReady: (mesh) => {
          mesh.receiveShadows = true;
        },
      },
      scene,
    );
    const floorY = -0.05;
    floor.position = new Vector3(bounds.cx, floorY, bounds.cz);
    floorYRef.current = floorY;

    const floorMat = createFloorMaterial(floorStyle, scene);
    floor.material = floorMat;
    floorMeshRef.current = floor;
    floorMatRef.current = floorMat;

    // 5. Build Wall Geometry
    wallMeshesRef.current = [];
    const wallOpts = {
      thickness: wallThicknessMeters,
      height: effectiveWallHeight,
      scale: 1 / pixelsPerMeter,
    };
    const wallMeshes = buildWallMeshes(edges, wallOpts, scene);

    const wallMat = new PBRMaterial("wallMat", scene);
    wallMat.albedoColor = new Color3(0.94, 0.93, 0.90);
    wallMat.metallic = 0.0;
    wallMat.roughness = 0.88;
    wallMeshes.forEach((mesh) => {
      mesh.material = wallMat;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    // 6. CSG Window and Door Cutouts
    const cutWithWindows =
      windows && windows.length > 0 && wallMode !== "cutaway"
        ? cutWindowOpenings(
            windows,
            wallMeshes,
            {
              scale: 1 / pixelsPerMeter,
              wallThickness: wallThicknessMeters,
              sillHeight: sillHeightMeters,
              windowHeight: windowHeightMeters,
              wallHeight: effectiveWallHeight,
            },
            scene,
          )
        : wallMeshes;

    const cutWallMeshes =
      doors && doors.length > 0
        ? cutDoorOpenings(
            doors,
            cutWithWindows,
            {
              scale: 1 / pixelsPerMeter,
              wallThickness: wallThicknessMeters,
              doorHeight: Math.min(2.1, effectiveWallHeight),
            },
            scene,
          )
        : cutWithWindows;

    cutWallMeshes.forEach((mesh) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
    wallMeshesRef.current = cutWallMeshes;

    // 7. Procedural Bathroom Fixtures
    createBathroomFixtures(fixtures, { scale: 1 / pixelsPerMeter }, scene);

    edgesRef.current = edges;
    windowsRef.current = windows;
    doorsRef.current = doors;

    // 8. Interaction & Picking Handlers
    const _forward = Vector3.Zero();
    const _right = Vector3.Zero();
    const _targetDelta = Vector3.Zero();

    const onWheel = (e) => {
      if (document.activeElement === canvasRef.current) {
        e.preventDefault();
      }
    };
    canvasRef.current?.ownerDocument?.addEventListener("wheel", onWheel, {
      passive: false,
    });

    const onContextMenu = (e) => e.preventDefault();
    canvasRef.current.addEventListener("contextmenu", onContextMenu);

    const onPointerDown = (e) => {
      if (e.button !== 0) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const ptrX = e.clientX - rect.left;
      const ptrY = e.clientY - rect.top;

      const hit = scene.pick(ptrX, ptrY, (mesh) =>
        meshToModelRef.current.has(mesh),
      );
      if (!hit.hit) {
        selectedMeshRef.current = null;
        return;
      }
      const mapping = meshToModelRef.current.get(hit.pickedMesh);
      if (!mapping) return;

      const root = mapping.root;
      selectedMeshRef.current = root;
      camera.detachControl();

      const hitPoint = rayGroundIntersect(
        scene,
        camera,
        canvasRef.current,
        floorYRef.current,
        ptrX,
        ptrY,
      );
      if (!hitPoint) {
        camera.attachControl(canvasRef.current, true);
        return;
      }

      isDraggingRef.current = true;
      draggedMeshRef.current = root;
      dragOffsetRef.current.copyFrom(hitPoint);
      dragStartPosRef.current.copyFrom(root.position);
      setDraggedModel(mapping.entry);

      canvasRef.current.style.cursor = "grabbing";
    };

    const onPointerMove = (e) => {
      if (!isDraggingRef.current) return;
      const mesh = draggedMeshRef.current;
      if (!mesh) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const ptrX = e.clientX - rect.left;
      const ptrY = e.clientY - rect.top;

      const hitPoint = rayGroundIntersect(
        scene,
        camera,
        canvasRef.current,
        floorYRef.current,
        ptrX,
        ptrY,
      );
      if (!hitPoint) return;

      const deltaX = hitPoint.x - dragOffsetRef.current.x;
      const deltaZ = hitPoint.z - dragOffsetRef.current.z;

      mesh.position.x = dragStartPosRef.current.x + deltaX;
      mesh.position.z = dragStartPosRef.current.z + deltaZ;

      const mapping = [...meshToModelRef.current.values()].find((m) => m.root === mesh);
      if (mapping) {
        mapping.entry.position[0] = mesh.position.x - bounds.cx;
        mapping.entry.position[2] = mesh.position.z - bounds.cz;
      }
    };

    const onPointerUp = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;
      const mesh = draggedMeshRef.current;
      cameraRef.current?.attachControl(canvasRef.current, true);
      draggedMeshRef.current = null;
      setDraggedModel(null);
      if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
    };

    canvasRef.current.addEventListener("pointerdown", onPointerDown);
    canvasRef.current.ownerDocument.addEventListener("pointermove", onPointerMove);
    canvasRef.current.ownerDocument.addEventListener("pointerup", onPointerUp);

    engine.runRenderLoop(() => {
      const keys = keysRef.current;
      _forward.copyFrom(camera.getDirection(Vector3.Forward()));
      _forward.y = 0;
      _forward.normalize();

      _right.copyFrom(camera.getDirection(Vector3.Right()));
      _right.y = 0;
      _right.normalize();

      const step = (wasdSpeed * engine.getDeltaTime()) / 1000;
      _targetDelta.set(0, 0, 0);
      if (keys.has("w")) _targetDelta.addInPlace(_forward.scale(step));
      if (keys.has("s")) _targetDelta.addInPlace(_forward.scale(-step));
      if (keys.has("a")) _targetDelta.addInPlace(_right.scale(-step));
      if (keys.has("d")) _targetDelta.addInPlace(_right.scale(step));

      if (_targetDelta.length() > 0) {
        camera.target.addInPlace(_targetDelta);
      }

      scene.render();
    });

    return () => {
      canvasRef.current?.removeEventListener("contextmenu", onContextMenu);
      canvasRef.current?.ownerDocument?.removeEventListener("wheel", onWheel);
      canvasRef.current?.removeEventListener("pointerdown", onPointerDown);
      canvasRef.current?.ownerDocument?.removeEventListener("pointermove", onPointerMove);
      canvasRef.current?.ownerDocument?.removeEventListener("pointerup", onPointerUp);
      wallMeshesRef.current.forEach((m) => m.dispose());
      floor.dispose();
      wallMat.dispose();
      floorMat.dispose();
      dirLight.dispose();
      hemiLight.dispose();
      fillLight.dispose();
      shadowGen.dispose();
      camera.dispose();
      scene.dispose();
      engine.dispose();
    };
  }, [
    edges,
    windows,
    doors,
    fixtures,
    pixelsPerMeter,
    cameraDistance,
    bounds,
    wallHeightMeters,
    wallThicknessMeters,
    sillHeightMeters,
    windowHeightMeters,
    floorStyle,
    wallMode,
  ]);

  useEffect(() => {
    const handleResize = () => {
      engineRef.current?.resize();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: isFullscreen ? "100vh" : 540,
        display: "flex",
        background: "#0f172a",
        cursor: draggedModel ? "grabbing" : "crosshair",
        position: "relative",
        borderRadius: 8,
        overflow: "hidden",
        border: "1px solid var(--line, #334155)",
      }}
    >
      <div style={{ flex: 1, position: "relative", minHeight: 0 }}>
        <canvas
          ref={canvasRef}
          tabIndex={0}
          style={{ width: "100%", height: "100%", border: "none", outline: "none" }}
        />

        {/* Viewpoints & Material Preset Bar */}
        <div
          style={{
            position: "absolute",
            top: 10,
            left: 10,
            display: "flex",
            alignItems: "center",
            gap: 6,
            flexWrap: "wrap",
            background: "rgba(15, 23, 42, 0.75)",
            backdropFilter: "blur(8px)",
            padding: "5px 8px",
            borderRadius: 6,
            border: "1px solid rgba(255,255,255,0.15)",
            fontSize: 11,
            color: "#fff",
            zIndex: 10,
          }}
        >
          <span style={{ fontWeight: 600, color: "rgba(255,255,255,0.7)", marginRight: 2 }}>View:</span>
          <button
            onClick={() => setViewpoint("iso")}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            🏠 3D Iso
          </button>
          <button
            onClick={() => setViewpoint("top")}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            📐 Top Plan
          </button>
          <button
            onClick={() => setViewpoint("walk")}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            🚶 Walkthrough
          </button>
          <button
            onClick={() => setViewpoint("living")}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            🛋️ Living
          </button>
          <button
            onClick={() => setViewpoint("bed")}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
            }}
          >
            🛏️ Bed
          </button>

          <span style={{ width: 1, height: 16, background: "rgba(255,255,255,0.2)", margin: "0 2px" }} />

          <button
            onClick={() => setWallMode(wallMode === "full" ? "cutaway" : "full")}
            style={{
              background: wallMode === "cutaway" ? "#0284c7" : "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              padding: "4px 8px",
              cursor: "pointer",
              fontWeight: 500,
            }}
          >
            ✂️ {wallMode === "cutaway" ? "Cutaway Walls" : "Full Walls"}
          </button>

          <select
            value={floorStyle}
            onChange={(e) => setFloorStyle(e.target.value)}
            style={{
              background: "rgba(255,255,255,0.12)",
              color: "#fff",
              border: "1px solid rgba(255,255,255,0.2)",
              borderRadius: 4,
              padding: "3px 6px",
              fontSize: 11,
              cursor: "pointer",
            }}
          >
            <option value="wood" style={{ color: "#000" }}>🪵 Oak Parquet</option>
            <option value="tile" style={{ color: "#000" }}>🏛️ Modern Tile</option>
            <option value="loft" style={{ color: "#000" }}>⚪ Studio Grey</option>
          </select>
        </div>

        <button
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            background: "rgba(15, 23, 42, 0.75)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: 6,
            padding: "5px 10px",
            fontSize: 11,
            cursor: "pointer",
            zIndex: 10,
          }}
        >
          {isFullscreen ? "⤓ Exit Fullscreen" : "⤢ Fullscreen"}
        </button>

        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: 10,
            background: "rgba(15, 23, 42, 0.75)",
            backdropFilter: "blur(6px)",
            color: "rgba(255,255,255,0.9)",
            padding: "5px 10px",
            borderRadius: 6,
            fontSize: 11,
            border: "1px solid rgba(255,255,255,0.15)",
          }}
        >
          WASD to move · Left-click model to drag · <strong>R</strong> to rotate · <strong>Del</strong> to remove · Scroll to zoom
        </div>
      </div>

      <ModelPanel
        availableModels={availableModels}
        placedModels={placedModels}
        draggedModel={draggedModel}
        onPlaceModel={placeModelInScene}
        onExportScene={handleExportScene}
        onImportScene={handleImportScene}
        onRemovePlacedModel={removePlacedModel}
        onClearAllModels={clearAllModels}
        onAutoFurnish={handleAutoFurnish}
        furnitureCount={furniture ? furniture.filter((f) => f.model_file).length : 0}
        isAutoFurnishing={isAutoFurnishing}
      />
    </div>
  );
}
