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
  MeshBuilder,
  ArcRotateCamera,
  ShadowGenerator,
  SceneLoader,
  Matrix,
  CSG,
  Quaternion,
  Space,
} from "@babylonjs/core";
import { Plane } from "@babylonjs/core";
import { GLTFFileLoader } from "@babylonjs/loaders";
import { buildWallMeshes } from "./wallGeometry";

// Register the GLTF loader once at module load time (global Babylon.js singleton).
SceneLoader.RegisterPlugin(new GLTFFileLoader());

// SceneLoader.Append shows Babylon's default black-screen-with-logo loading
// UI automatically while an asset loads. For small GLTF files this loads
// fast enough that all you see is a jarring flash. This flag turns that
// automatic behavior off entirely — you can add your own loading indicator
// separately if you want one, but this stops the flash.
SceneLoader.ShowLoadingScreen = false;

/**
 * Cast a ray from the camera through a given canvas pixel and intersect it
 * with the horizontal ground plane at height `flrY`. Used for BOTH placing
 * a new model (ray through canvas center) and dragging an existing one
 * (ray through the cursor) — same math, different pixel coordinates.
 */
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
    w: (maxX - minX) * scale,
    d: (maxY - minY) * scale,
  };
}
/**
 * Carve rectangular window openings into already-built wall meshes.
 *
 * The floorplan pipeline gives us each window as a straight segment in image
 * pixel space (`start`/`end`, same coordinate space as `edges`) plus its
 * opening width across the wall's thickness. It intentionally does NOT leave
 * a gap in the wall graph at the window — the wall path is bridged back to
 * continuous there so buildWallMeshes produces a normal, solid, full-height
 * wall segment. This function is what actually turns that solid stretch into
 * an opening: for each window, it builds a box sized to the window's plan
 * footprint and a configurable sill/head height, then CSG-subtracts that box
 * from any wall mesh whose bounding box overlaps it. This keeps the wall
 * solid below the sill and above the head, exactly as if a real window had
 * been installed in a solid wall.
 *
 * Runs once per (re)build, not per frame — CSG is too slow for real-time use,
 * but is fine for a handful of one-off subtractions when the scene loads.
 */
function cutWindowOpenings(windows, wallMeshes, opts, scene) {
  const { scale, wallThickness, sillHeight, windowHeight, wallHeight } = opts;
  const headHeight = sillHeight + windowHeight;
  if (headHeight >= wallHeight) {
    console.warn(
      "Window head height reaches the ceiling — check sillHeightMeters/windowHeightMeters vs wallHeightMeters.",
    );
  }

  const remainingMeshes = [...wallMeshes];

  for (const win of windows) {
    const startX = win.start[0] * scale;
    const startZ = win.start[1] * scale;
    const endX = win.end[0] * scale;
    const endZ = win.end[1] * scale;
    const dX = endX - startX;
    const dZ = endZ - startZ;
    const openingLength = Math.hypot(dX, dZ);
    if (openingLength < 0.05) continue; // degenerate, skip

    const centerX = (startX + endX) / 2;
    const centerZ = (startZ + endZ) / 2;
    const centerY = sillHeight + windowHeight / 2;
    const angle = Math.atan2(dZ, dX);

    // Cutter is intentionally oversized a little in every direction so it
    // fully punches through the wall face instead of leaving razor-thin
    // leftover slivers from floating-point misses at the boundary.
    const cutter = MeshBuilder.CreateBox(
      "windowCutter",
      {
        width: openingLength + 0.04,
        height: windowHeight,
        depth: wallThickness * 2,
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

      const wallCSG = CSG.FromMesh(wallMesh);
      const cutterCSG = CSG.FromMesh(cutter);
      const resultCSG = wallCSG.subtract(cutterCSG);
      const material = wallMesh.material;
      const name = wallMesh.name;
      const resultMesh = resultCSG.toMesh(name, material, scene);
      resultMesh.material = material;

      wallMesh.dispose();
      remainingMeshes[i] = resultMesh;
    }

    cutter.dispose();
  }

  return remainingMeshes;
}

/**
 * Fetch models from the API endpoint. The backend merges seeded models
 * (from public/models/*.glb) with user-uploaded models, so every client
 * gets the full catalog automatically after a fresh clone.
 */
async function fetchUserModels() {
  try {
    const res = await fetch("/api/models?limit=100");
    if (!res.ok) return [];
    const data = await res.json();
    // Keep thumbnail alongside name/url so the sidebar can render a
    // preview icon instead of a bare text row.
    return (data.models || []).map((m) => ({
      name: m.name,
      url: m.file_path,
      thumbnail: m.thumbnail_path,
    }));
  } catch {
    return [];
  }
}

/**
 * Export scene state as a JSON file.
 * Serializes floorplan (edges, windows) and placed models (name, position, rotation).
 */
function exportSceneJSON(
  edges,
  windows,
  placedModels,
  wallHeightMeters,
  wallThicknessMeters,
  sillHeightMeters,
  windowHeightMeters,
  cameraDistance,
) {
  // Validate input data
  if (!edges || edges.length === 0) {
    throw new Error("Cannot export: no floorplan edges found");
  }
  if (!placedModels || placedModels.length === 0) {
    throw new Error("Cannot export: no models placed in scene");
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
      edges: edges,
      windows: windows || [],
      wallHeightMeters: wallHeightMeters ?? 2.8,
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

/**
 * Import scene state from a JSON file.
 * Returns parsed scene data or throws on error.
 */
async function importSceneJSON(file) {
  // Validate file type
  if (
    file.type &&
    !file.type.startsWith("application/json") &&
    !file.type.includes("json")
  ) {
    // Also check extension as fallback
    if (!file.name.endsWith(".json")) {
      throw new Error("Invalid file type: expected a JSON file");
    }
  }

  let text;
  try {
    text = await file.text();
  } catch (readErr) {
    throw new Error("Failed to read file: " + readErr.message);
  }

  if (!text.trim()) {
    throw new Error("Empty file");
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch (parseErr) {
    throw new Error("Invalid JSON format: " + parseErr.message);
  }

  // Validate required structure
  if (!data.version) {
    throw new Error("Missing version field in scene file");
  }
  if (data.version !== 1) {
    throw new Error(
      `Unsupported scene file version: ${data.version} (expected 1)`,
    );
  }
  if (!data.floorplan) {
    throw new Error("Missing floorplan data in scene file");
  }
  if (!Array.isArray(data.floorplan.edges)) {
    throw new Error("Invalid floorplan: edges array is required");
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
}) {
  return (
    <div
      style={{
        width: 200,
        background: "rgba(0,0,0,0.6)",
        color: "#fff",
        padding: "10px 12px",
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        fontSize: 12,
        fontFamily: "var(--font-mono, monospace)",
      }}
    >
      <div
        style={{
          fontWeight: 600,
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "rgba(255,255,255,0.5)",
          marginBottom: 2,
        }}
      >
        Available
      </div>
      {availableModels.length === 0 && (
        <div style={{ color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
          No models yet.
        </div>
      )}
      {availableModels.map((m, i) => (
        <div
          key={i}
          onClick={() => onPlaceModel(m)}
          style={{
            padding: "6px 8px",
            borderRadius: 4,
            cursor: "pointer",
            background: "rgba(255,255,255,0.05)",
            border: "1px solid transparent",
            transition: "background 0.1s",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.1)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "rgba(255,255,255,0.05)";
          }}
        >
          {m.thumbnail ? (
            <img
              src={m.thumbnail}
              alt=""
              width={40}
              height={40}
              style={{ flexShrink: 0, objectFit: "contain" }}
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          ) : (
            <div style={{ width: 20, height: 20, flexShrink: 0 }} />
          )}
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
          <span
            style={{
              fontSize: 9,
              background: "rgba(255,255,255,0.15)",
              padding: "1px 5px",
              borderRadius: 3,
              flexShrink: 0,
            }}
          >
            +
          </span>
        </div>
      ))}

      {/* Scene import/export — always visible, independent of whether
          any models are currently placed. Import needs to work on an
          empty scene (that's the point of loading a saved one); Export
          is disabled rather than hidden when there's nothing to save. */}
      <div
        style={{
          marginTop: 8,
          fontWeight: 600,
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          color: "rgba(255,255,255,0.5)",
        }}
      >
        Scene
      </div>
      <div style={{ display: "flex", gap: 4, marginTop: 4 }}>
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
            fontFamily: "var(--font-mono, monospace)",
          }}
          title={
            placedModels.length === 0
              ? "Place at least one model to export"
              : "Export scene state as JSON"
          }
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
            fontFamily: "var(--font-mono, monospace)",
            textAlign: "center",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
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
              marginTop: 8,
              fontWeight: 600,
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "rgba(255,255,255,0.5)",
            }}
          >
            In Scene
          </div>
          {placedModels.map((m, i) => {
            const isDragging = draggedModel && draggedModel.mesh === m.mesh;
            return (
              <div
                key={`placed-${i}`}
                style={{
                  padding: "6px 8px",
                  borderRadius: 4,
                  background: isDragging
                    ? "rgba(255,255,255,0.15)"
                    : "rgba(255,255,255,0.05)",
                  border: isDragging
                    ? "1px solid rgba(255,255,255,0.3)"
                    : "1px solid transparent",
                  transition: "background 0.1s",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 6,
                  color: "rgba(255,255,255,0.7)",
                  fontStyle: "italic",
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
                {isDragging && (
                  <span
                    style={{
                      fontSize: 9,
                      background: "rgba(255,255,255,0.2)",
                      padding: "1px 5px",
                      borderRadius: 3,
                      flexShrink: 0,
                    }}
                  >
                    Drag
                  </span>
                )}
                {onRemovePlacedModel && (
                  <button
                    onPointerDown={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      onRemovePlacedModel(m);
                    }}
                    title="Remove from scene"
                    type="button"
                    style={{
                      background: "rgba(255,80,80,0.2)",
                      color: "#fff",
                      border: "1px solid rgba(255,80,80,0.4)",
                      borderRadius: 3,
                      width: 18,
                      height: 18,
                      fontSize: 10,
                      lineHeight: 1,
                      padding: 0,
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      flexShrink: 0,
                      fontStyle: "normal",
                    }}
                  >
                    ✕
                  </button>
                )}
              </div>
            );
          })}
        </>
      )}

      {(availableModels.length > 0 || placedModels.length > 0) && (
        <div
          style={{
            marginTop: "auto",
            paddingTop: 8,
            borderTop: "1px solid rgba(255,255,255,0.1)",
            color: "rgba(255,255,255,0.35)",
            fontSize: 10,
            lineHeight: 1.5,
          }}
        >
          Click + to place a model.
          <br />
          Drag in the scene or reposition later.
        </div>
      )}
    </div>
  );
}

/**
 * Full 3D viewer. Takes the `edges` graph produced by the skeleton-tracing
 * step and renders it as walls you can orbit/walk around.
 */
export default function Viewer3D({
  edges,
  windows = [],
  pixelsPerMeter = 50,
  wallHeightMeters = 2.7,
  wallThicknessMeters = 0.15,
  sillHeightMeters = 0.9,
  windowHeightMeters = 1.2,
}) {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const wallMeshesRef = useRef([]);
  const keysRef = useRef(new Set());
  const floorMeshRef = useRef(null);
  const floorMatRef = useRef(null);
  const draggedMeshRef = useRef(null);
  const selectedMeshRef = useRef(null);
  const dragOffsetRef = useRef(new Vector3());
  const dragStartPosRef = useRef(new Vector3());
  const isDraggingRef = useRef(false);
  // Track accumulated Y rotation angle (in radians) per root mesh object
  const modelRotationRef = useRef(new Map());
  const removePlacedModelRef = useRef(() => {});
  // Tracks the CURRENT yaw angle (radians) per root mesh — separate from
  // modelRotationRef, which stores the base quaternion R rotates FROM.
  // This is the actual value written into entry.rotation for export.
  const yawAngleRef = useRef(new Map());
  // Maps EVERY mesh in a placed model's hierarchy (root + all descendants)
  // to { entry, root }. Registering only the root used to be the bug: a
  // GLTF import's root is often an invisible transform node with zero
  // vertices, so clicking the visible furniture actually hits a CHILD
  // mesh that was never in this map — scene.pick's predicate failed for
  // it every time, so dragging silently never started.
  const meshToModelRef = useRef(new Map());
  const floorYRef = useRef(-0.05);
  const cameraRef = useRef(null);

  // Store edges/windows as refs so export can access them
  const edgesRef = useRef([]);
  const windowsRef = useRef([]);
  const containerRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const [availableModels, setAvailableModels] = useState([]);
  const [placedModels, setPlacedModels] = useState([]);
  const [draggedModel, setDraggedModel] = useState(null);

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

  // Bounds computed from edges — must be available before rebuildFromImport
  // which references bounds.cx/cz for model positioning.
  const bounds = useMemo(
    () => computeBounds(edges, pixelsPerMeter),
    [edges, pixelsPerMeter],
  );

  // FIX: cameraDistance moved up here — handleExportScene (below) depends on
  // it in its useCallback dependency array. Previously cameraDistance was
  // declared much further down the file (right before the big scene-setup
  // effect), which meant handleExportScene referenced a `const` that hadn't
  // been initialized yet in this render pass — a TDZ ReferenceError
  // ("Cannot access 'cameraDistance' before initialization") on every render.
  const toggleFullscreen = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (!document.fullscreenElement) {
      el.requestFullscreen().catch((err) => {
        console.warn("Fullscreen request failed:", err.message);
      });
    } else {
      document.exitFullscreen();
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      const active = document.fullscreenElement === containerRef.current;
      setIsFullscreen(active);
      // The canvas's actual pixel size changes when the browser goes
      // fullscreen, but Babylon doesn't know that on its own — it only
      // resizes on the window "resize" event, which the Fullscreen API
      // doesn't reliably fire. Kick the engine explicitly.
      requestAnimationFrame(() => {
        engineRef.current?.resize();
      });
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

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
    return Math.max(maxExtent * 1.8, 10);
  }, [edges, pixelsPerMeter]);

  // Rebuild scene from imported JSON data.
  // FIX: this must be declared before handleImportScene (below), which
  // depends on it — same TDZ issue as cameraDistance above.
  const rebuildFromImport = useCallback(
    (sceneData) => {
      const scene = sceneRef.current;
      const canvas = canvasRef.current;
      if (!scene || !canvas) {
        alert("Cannot import: viewer not ready.");
        return;
      }

      if (sceneData.version !== 1) {
        alert("Unsupported scene file version");
        return;
      }

      const fp = sceneData.floorplan;
      const wallHeight = fp.wallHeightMeters ?? wallHeightMeters;
      const wallThickness = fp.wallThicknessMeters ?? wallThicknessMeters;
      const sillHeight = fp.sillHeightMeters ?? sillHeightMeters;
      const windowHeight = fp.windowHeightMeters ?? windowHeightMeters;
      const cameraDist = fp.cameraDistance ?? 10;

      // Rebuild the 3D scene with imported floorplan settings
      const rootsToDispose = new Set();
      for (const { root } of meshToModelRef.current.values()) {
        rootsToDispose.add(root);
      }
      rootsToDispose.forEach((root) => {
        try {
          root.dispose(false, true);
        } catch (e) {
          /* already disposed */
        }
      });
      modelRotationRef.current.clear();
      yawAngleRef.current.clear();
      selectedMeshRef.current = null;
      draggedMeshRef.current = null;
      isDraggingRef.current = false;
      setDraggedModel(null);

      // Rebuild the 3D scene with imported floorplan settings
      // Clear existing wall meshes
      wallMeshesRef.current.forEach((m) => {
        try {
          m.dispose();
        } catch (e) {
          /* already disposed */
        }
      });
      wallMeshesRef.current = [];
      meshToModelRef.current.clear();
      // Clear existing wall meshes
      wallMeshesRef.current.forEach((m) => {
        try {
          m.dispose();
        } catch (e) {
          /* already disposed */
        }
      });
      wallMeshesRef.current = [];
      meshToModelRef.current.clear();

      // Update refs with imported data
      // Update refs with imported data
      edgesRef.current = fp.edges;
      windowsRef.current = fp.windows || [];

      // Bounds must come from the IMPORTED edges, not the outer `bounds`
      // memo (which reflects whatever edges are currently loaded from
      // props). Using stale bounds here was sizing/positioning the floor
      // for the wrong floorplan.
      const importedBounds = computeBounds(fp.edges, pixelsPerMeter);

      // Dispose the previous floor mesh/material before building a new
      // one. Without this, the old floor was never removed — Babylon
      // doesn't enforce unique mesh names, so a second "floor" mesh just
      // stacked on top of (or beside) the first, which is what made the
      // floor look wrong after import.
      if (floorMeshRef.current) {
        try {
          floorMeshRef.current.dispose();
        } catch (e) {
          /* already disposed */
        }
      }
      if (floorMatRef.current) {
        try {
          floorMatRef.current.dispose();
        } catch (e) {
          /* already disposed */
        }
      }

      // Rebuild walls with imported settings
      const halfThick = wallThickness / 2;
      const floor = MeshBuilder.CreateGround(
        "floor",
        {
          width: importedBounds.w + halfThick * 2,
          height: importedBounds.d + halfThick * 2,
          subdivisions: 2,
          onReady: (mesh) => {
            mesh.receiveShadows = true;
          },
        },
        scene,
      );
      const floorY = -0.05;
      floor.position = new Vector3(
        importedBounds.cx,
        floorY,
        importedBounds.cz,
      );
      floorYRef.current = floorY;

      const floorMat = new StandardMaterial("floorMat", scene);
      floorMat.diffuseColor = new Color3(0.9, 0.9, 0.9);
      floorMat.specularColor = new Color3(0.2, 0.2, 0.2);
      floorMat.emissiveColor = new Color3(0.3, 0.3, 0.3);
      floor.material = floorMat;

      floorMeshRef.current = floor;
      floorMatRef.current = floorMat;

      const wallOpts = {
        thickness: wallThickness,
        height: wallHeight,
        scale: 1 / pixelsPerMeter,
      };
      const wallMeshes = buildWallMeshes(fp.edges, wallOpts, scene);

      const wallMat = new PBRMaterial("wallMat", scene);
      wallMat.diffuseColor = new Color3(0.91, 0.89, 0.84);
      wallMat.metallic = 0.0;
      wallMat.roughness = 0.8;
      wallMeshes.forEach((mesh) => {
        mesh.material = wallMat;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      });

      const cutWallMeshes =
        fp.windows && fp.windows.length > 0
          ? cutWindowOpenings(
              fp.windows,
              wallMeshes,
              {
                scale: 1 / pixelsPerMeter,
                wallThickness,
                sillHeight,
                windowHeight,
                wallHeight,
              },
              scene,
            )
          : wallMeshes;
      cutWallMeshes.forEach((mesh) => {
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      });

      wallMeshesRef.current = cutWallMeshes;

      // Update camera if distance changed
      if (Math.abs(cameraDist - cameraRef.current?.radius) > 0.1) {
        cameraRef.current.radius = cameraDist;
      }

      // Re-place models from saved positions/rotations (sequential chain)
      if (sceneData.placedModels && sceneData.placedModels.length > 0) {
        const newPlaced = [];

        const loadChain = sceneData.placedModels.reduce(
          (promiseChain, saved) => {
            return promiseChain.then(() => {
              return new Promise((resolve) => {
                const apiModel =
                  availableModels.find((m) => m.url === saved.url) ||
                  availableModels.find((m) => m.name === saved.name);
                if (!apiModel) {
                  console.warn(`Model not found: ${saved.name}, skipping`);
                  resolve();
                  return;
                }

                // Target position from saved data (offset from bounds center)
                const targetX = (saved.position?.[0] ?? 0) + importedBounds.cx;
                const targetZ = (saved.position?.[2] ?? 0) + importedBounds.cz;
                const targetY = saved.position?.[1] ?? 0;

                const url = apiModel.url;
                const name = url
                  .split("/")
                  .pop()
                  .replace(/\.[^.]+$/, "");
                const meshCountBefore = scene.meshes.length;

                SceneLoader.Append(
                  "",
                  url,
                  scene,
                  (mesh) => {
                    const newMeshes = scene.meshes.slice(meshCountBefore);
                    if (newMeshes.length === 0) {
                      resolve();
                      return;
                    }

                    const root =
                      newMeshes.find((m) => m.parent === null) || newMeshes[0];

                    for (const m of newMeshes) {
                      if (!m.material && (m.getTotalVertices?.() ?? 0) > 0) {
                        const mat = new PBRMaterial(
                          `${name}_mat_${m.uniqueId}`,
                          scene,
                        );
                        mat.diffuseColor = new Color3(0.6, 0.4, 0.2);
                        mat.metallic = 0.1;
                        mat.roughness = 0.7;
                        m.material = mat;
                      }
                      m.castShadow = true;
                      m.receiveShadow = true;
                    }

                    // Set position to the saved position
                    root.position.set(targetX, targetY, targetZ);

                    // Apply saved rotation via rotationQuaternion, not Euler .rotation.
                    // GLTF-imported roots come back from the loader with a non-null
                    // rotationQuaternion already assigned (from coordinate system conversion).
                    // The saved yaw was captured relative to this initial quaternion, so we
                    // must MULTIPLY by it rather than overwrite, or the model will appear
                    // rotated incorrectly (typically +180 degrees from what was saved).
                    const savedYaw = saved.rotation?.[1] ?? 0;
                    const loaderInitialQuat = root.rotationQuaternion;
                    const savedRotationQuat = Quaternion.RotationYawPitchRoll(
                      savedYaw,
                      0,
                      0,
                    );
                    // Multiply: loader's initial quaternion THEN the saved rotation
                    root.rotationQuaternion =
                      loaderInitialQuat.multiply(savedRotationQuat);
                    modelRotationRef.current.set(root, {
                      initialQuaternion: root.rotationQuaternion,
                    });

                    const entry = {
                      name,
                      position: [
                        root.position.x - importedBounds.cx,
                        0,
                        root.position.z - importedBounds.cz,
                      ],
                      rotation: [0, savedYaw, 0],
                      scaling: [1, 1, 1],
                      url,
                      mesh: root,
                    };

                    yawAngleRef.current.set(root, savedYaw);

                    for (const m of newMeshes) {
                      meshToModelRef.current.set(m, { entry, root });
                    }

                    newPlaced.push({ ...entry });
                    resolve();
                  },
                  undefined,
                  () => {
                    resolve();
                  },
                );
              });
            });
          },
          Promise.resolve(), // seed with resolved promise
        );

        loadChain.then(() => {
          setPlacedModels(newPlaced);
          alert(
            `Scene imported: ${newPlaced.length} models loaded successfully.`,
          );
        });
      } else {
        setPlacedModels([]);
        alert(`Scene imported: No models to load.`);
      }
    },
    [
      availableModels,
      bounds,
      pixelsPerMeter,
      wallHeightMeters,
      wallThicknessMeters,
      sillHeightMeters,
      windowHeightMeters,
    ],
  );

  // Export handler — called from ModelPanel sidebar
  const handleExportScene = useCallback(() => {
    // Use refs for edges/windows (stored during scene creation)
    try {
      exportSceneJSON(
        edgesRef.current,
        windowsRef.current,
        placedModels,
        wallHeightMeters,
        wallThicknessMeters,
        sillHeightMeters,
        windowHeightMeters,
        cameraDistance,
      );
    } catch (err) {
      console.warn("Export skipped:", err.message);
    }
  }, [
    placedModels,
    wallHeightMeters,
    wallThicknessMeters,
    sillHeightMeters,
    windowHeightMeters,
    cameraDistance,
  ]);

  // Import handler — called from ModelPanel file input
  const handleImportScene = useCallback(
    async (file) => {
      try {
        const sceneData = await importSceneJSON(file);
        rebuildFromImport(sceneData);
      } catch (err) {
        console.error("Import failed:", err);
        alert(`Import failed: ${err.message}`);
      }
      // Reset file input so the same file can be re-selected
    },
    [rebuildFromImport],
  );

  const _lastRotatedAt = useRef(new Map()); // debounce R key rotations

  const handleKeyDown = useCallback((e) => {
    keysRef.current.add(e.key.toLowerCase());

    // Delete / Backspace: remove the currently selected model from the
    // scene. Selection is set on click (see onPointerDown) and persists
    // after you release the mouse — you don't need to be actively
    // dragging, just to have clicked a model beforehand. If the user is
    // dragging, prefer the dragged root so Delete works immediately mid-drag.
    if (e.key === "Delete" || e.key === "Backspace") {
      const root = isDraggingRef.current
        ? draggedMeshRef.current
        : selectedMeshRef.current;
      console.log("[Viewer3D] delete key", {
        key: e.key,
        isDragging: isDraggingRef.current,
        hasDraggedMesh: !!draggedMeshRef.current,
        hasSelectedMesh: !!selectedMeshRef.current,
        rootName: root?.name ?? null,
      });
      if (!root) return;

      // Don't let Backspace delete a model while the user is typing in an
      // input/textarea elsewhere on the page (title field, etc).
      const tag = document.activeElement?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;

      e.preventDefault();

      // meshToModelRef maps every mesh (root + descendants) to { entry, root }.
      // The root itself is always registered, so this lookup will succeed
      // for anything that's actually a placed, selected model.
      const mapping = meshToModelRef.current.get(root);
      if (mapping) {
        console.log("[Viewer3D] deleting model", {
          name: mapping.entry?.name,
          url: mapping.entry?.url,
        });
        removePlacedModelRef.current(mapping.entry);
      } else {
        console.log("[Viewer3D] delete key had no model mapping", {
          rootName: root?.name ?? null,
        });
      }
      return;
    }

    // R key: rotate model 90° — only when actively dragging (left click held on model)
    if (
      (e.key === "r" || e.key === "R") &&
      isDraggingRef.current &&
      draggedMeshRef.current
    ) {
      const now = performance.now();
      const last = _lastRotatedAt.current.get("r") ?? 0;
      if (now - last < 150) return;
      _lastRotatedAt.current.set("r", now);

      e.preventDefault();
      e.stopPropagation();

      const root = isDraggingRef.current
        ? draggedMeshRef.current
        : selectedMeshRef.current;
      if (!root) return;

      const delta = e.shiftKey ? -(Math.PI / 2) : Math.PI / 2;
      const stored = modelRotationRef.current.get(root);
      if (!stored) return;

      const deltaQuat = Quaternion.RotationYawPitchRoll(delta, 0, 0);
      root.rotationQuaternion = stored.initialQuaternion.multiply(deltaQuat);

      // Update the stored quaternion so subsequent rotations build from the new orientation
      modelRotationRef.current.set(root, {
        initialQuaternion: root.rotationQuaternion,
      });

      // Track the running yaw angle and write it into entry.rotation so
      // export actually captures it. Mutating index [1] in place (rather
      // than replacing the array) means this update is visible through
      // both the live `entry` object and its shallow-copied counterpart
      // in the placedModels state array, since they share the same
      // rotation array reference.
      const prevYaw = yawAngleRef.current.get(root) ?? 0;
      const newYaw = prevYaw + delta;
      yawAngleRef.current.set(root, newYaw);

      const mapping = meshToModelRef.current.get(root);
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

  const MIN_MODEL_SPACING = 0.4;

  function isSpawnBlocked(sx, sz) {
    if (edges && edges.length > 0) {
      for (const edge of edges) {
        for (let i = 0; i < edge.points.length - 1; i++) {
          const [ax, ay] = edge.points[i];
          const [bx, by] = edge.points[i + 1];
          const scale = 1 / pixelsPerMeter;
          const waX = ax * scale,
            waZ = ay * scale;
          const wbX = bx * scale,
            wbZ = by * scale;
          const dx = wbX - waX,
            dz = wbZ - waZ;
          const lenSq = dx * dx + dz * dz;
          let t = lenSq > 0 ? ((sx - waX) * dx + (sz - waZ) * dz) / lenSq : 0;
          t = Math.max(0, Math.min(1, t));
          const closestX = waX + t * dx,
            closestZ = waZ + t * dz;
          const distX = sx - closestX,
            distZ = sz - closestZ;
          if (distX * distX + distZ * distZ < 0.25) return true;
        }
      }
    }
    for (const m of placedModels) {
      const dx = sx - m.position[0],
        dz = sz - m.position[2];
      if (dx * dx + dz * dz < (MIN_MODEL_SPACING * 2) ** 2) return true;
    }
    return false;
  }

  /**
   * Load a single GLTF model into the scene at the given ground position.
   *
   * FIX: previously this picked "the first mesh in the scene that isn't
   * the floor or a wall" — which breaks two ways:
   *   1. GLTF imports usually add an invisible root transform node
   *      (0 vertices) as part of the hierarchy. If THAT gets treated as
   *      "the model," it's literally unclickable — there's no geometry
   *      for scene.pick to hit, so drag can never start.
   *   2. Once more than one model is placed, `scene.meshes` contains
   *      every mesh in the whole scene, so "first non-floor/wall mesh"
   *      can grab an OLD model instead of the one you just loaded.
   *
   * Fix: snapshot the mesh count before loading, so we know exactly
   * which meshes THIS load call added. Find the true root (parent ===
   * null) among those new meshes to move as one unit, but register
   * EVERY mesh in that hierarchy in meshToModelRef — so clicking any
   * visible child correctly resolves back to "drag the root."
   */
  const lastPlaceAtRef = useRef(0);
  const placeModelInScene = useCallback(
    (apiModel) => {
      const now = performance.now();
      if (now - lastPlaceAtRef.current < 250) return; // swallow duplicate fires
      lastPlaceAtRef.current = now;

      const cam = cameraRef.current;
      const scene = sceneRef.current;
      const canvas = canvasRef.current;
      if (!scene || !cam || !canvas) return;

      // Cast a ray from the camera through the CENTER of the canvas and
      // intersect it with the ground plane — this is what "where the
      // camera is looking" actually means. The previous version used
      // `cam.target`, which is just the ArcRotateCamera's orbit pivot
      // point — it only happens to match the true screen-center ground
      // intersection in specific cases (e.g. looking straight down), and
      // drifts as soon as the camera's pitch/angle differs, or once WASD
      // has moved the target around. Reusing rayGroundIntersect here
      // means placement and dragging now share the exact same, correct
      // ray math.
      const centerHit = rayGroundIntersect(
        scene,
        cam,
        canvas,
        floorYRef.current,
        canvas.clientWidth / 2,
        canvas.clientHeight / 2,
      );
      if (!centerHit) {
        console.warn(
          "Placement failed: camera isn't looking at the ground (ray never hit the floor plane). Try tilting the camera down.",
        );
        return;
      }

      let spawnPoint = centerHit;

      // If the exact center-of-view point is blocked (on a wall, or
      // stacked on an existing model), nudge outward along the camera's
      // right vector in a few small steps rather than failing outright —
      // repeated clicks with the camera in the same position would
      // otherwise always target the identical point and always collide.
      if (isSpawnBlocked(spawnPoint.x, spawnPoint.z)) {
        const right = cam.getDirection(Vector3.Right());
        right.y = 0;
        right.normalize();
        let found = false;
        for (let step = 1; step <= 5; step++) {
          const candidate = centerHit.add(right.scale(step * 0.6));
          if (!isSpawnBlocked(candidate.x, candidate.z)) {
            spawnPoint = candidate;
            found = true;
            break;
          }
        }
        if (!found) {
          console.warn(
            "Placement blocked: no clear spot near the camera's view center. Try orbiting to a more open part of the room.",
          );
          return;
        }
      }

      const url = apiModel.url;
      const name = url
        .split("/")
        .pop()
        .replace(/\.[^.]+$/, "");
      const meshCountBefore = scene.meshes.length;

      SceneLoader.Append(
        "",
        url,
        scene,
        () => {
          const newMeshes = scene.meshes.slice(meshCountBefore);
          if (newMeshes.length === 0) return;

          const root = newMeshes.find((m) => m.parent === null) || newMeshes[0];

          for (const m of newMeshes) {
            if (!m.material && (m.getTotalVertices?.() ?? 0) > 0) {
              const mat = new PBRMaterial(`${name}_mat_${m.uniqueId}`, scene);
              mat.diffuseColor = new Color3(0.6, 0.4, 0.2);
              mat.metallic = 0.1;
              mat.roughness = 0.7;
              m.material = mat;
            }
            m.castShadow = true;
            m.receiveShadow = true;
          }

          root.position = spawnPoint;

          const entry = {
            name,
            position: [spawnPoint.x - bounds.cx, 0, spawnPoint.z - bounds.cz],
            rotation: [0, 0, 0],
            scaling: [1, 1, 1],
            url,
            mesh: root,
          };

          for (const m of newMeshes) {
            meshToModelRef.current.set(m, { entry, root });
          }

          yawAngleRef.current.set(root, 0);

          setPlacedModels((prev) => [...prev, entry]);
        },
        undefined,
        () => {
          // GLTF loader warnings can fire alongside success — ignore them.
        },
      );
    },
    [bounds, placedModels],
  );

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
      if (canvasRef.current) canvasRef.current.style.cursor = "crosshair";
    }

    try {
      root.dispose(false, true);
    } catch (e) {
      // already disposed
    }

    setPlacedModels((prev) => prev.filter((p) => p.mesh !== root));
  }, []);

  useEffect(() => {
    removePlacedModelRef.current = removePlacedModel;
  }, [removePlacedModel]);

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
    scene.clearColor = new Color4(0.53, 0.81, 0.92, 1);

    const camera = new ArcRotateCamera(
      "camera",
      -Math.PI / 4,
      Math.PI / 3,
      cameraDistance,
      new Vector3(bounds.cx, wallHeightMeters / 2, bounds.cz),
      scene,
    );
    cameraRef.current = camera;
    camera.attachControl(canvasRef.current, true);
    camera.keysUp = [];
    camera.keysDown = [];
    camera.keysLeft = [];
    camera.keysRight = [];
    camera.lowerRadiusLimit = 2;
    camera.upperRadiusLimit = 200;
    camera.lowerBetaLimit = 0.1;
    camera.upperBetaLimit = Math.PI / 2;
    camera.alphaRotationSpeed = 12;
    camera.betaRotationSpeed = 12;

    const wasdSpeed = 8;

    const hemiLight = new HemisphericLight(
      "hemiLight",
      new Vector3(0, 1, 0),
      scene,
    );
    hemiLight.intensity = 0.6;
    hemiLight.diffuse = new Color3(1, 1, 1);
    hemiLight.groundColor = new Color3(0.3, 0.3, 0.3);

    const dirLight = new DirectionalLight(
      "dirLight",
      new Vector3(
        cameraDistance,
        cameraDistance * 1.5,
        cameraDistance,
      ).normalize(),
      scene,
    );
    dirLight.intensity = 1;

    const shadowGen = new ShadowGenerator(1024, dirLight);
    shadowGen.useBlurExponentialShadowMap = true;
    shadowGen.blurKernel = 16;

    const halfThick = wallThicknessMeters / 2;
    const floor = MeshBuilder.CreateGround(
      "floor",
      {
        width: bounds.w + halfThick * 2,
        height: bounds.d + halfThick * 2,
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

    const floorMat = new StandardMaterial("floorMat", scene);
    floorMat.diffuseColor = new Color3(0.9, 0.9, 0.9);
    floorMat.specularColor = new Color3(0.2, 0.2, 0.2);
    floorMat.emissiveColor = new Color3(0.3, 0.3, 0.3);
    floor.material = floorMat;

    floorMeshRef.current = floor;
    floorMatRef.current = floorMat;

    wallMeshesRef.current = [];
    const wallOpts = {
      thickness: wallThicknessMeters,
      height: wallHeightMeters,
      scale: 1 / pixelsPerMeter,
    };
    const wallMeshes = buildWallMeshes(edges, wallOpts, scene);

    const wallMat = new PBRMaterial("wallMat", scene);
    wallMat.diffuseColor = new Color3(0.91, 0.89, 0.84);
    wallMat.metallic = 0.0;
    wallMat.roughness = 0.8;
    wallMeshes.forEach((mesh) => {
      mesh.material = wallMat;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    // Carve window openings into the solid wall meshes we just built. This
    // runs after material assignment so CSG.toMesh can carry the material
    // straight through; shadow flags don't survive the rebuild, so they're
    // reapplied below on whatever mesh array comes back (a mix of untouched
    // originals and CSG results).
    const cutWallMeshes =
      windows && windows.length > 0
        ? cutWindowOpenings(
            windows,
            wallMeshes,
            {
              scale: 1 / pixelsPerMeter,
              wallThickness: wallThicknessMeters,
              sillHeight: sillHeightMeters,
              windowHeight: windowHeightMeters,
              wallHeight: wallHeightMeters,
            },
            scene,
          )
        : wallMeshes;
    cutWallMeshes.forEach((mesh) => {
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    wallMeshesRef.current = cutWallMeshes;

    // Store edges/windows in refs so export can access them
    edgesRef.current = edges;
    windowsRef.current = windows;

    const ourMeshes = [...cutWallMeshes];

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

    // (rayGroundIntersect is now a module-level helper, shared with
    // placeModelInScene — see top of file)

    const onPointerDown = (e) => {
      if (e.button !== 0) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const ptrX = e.clientX - rect.left;
      const ptrY = e.clientY - rect.top;

      const hit = scene.pick(ptrX, ptrY, (mesh) =>
        meshToModelRef.current.has(mesh),
      );
      if (!hit.hit) {
        // Clicked empty space — deselect
        selectedMeshRef.current = null;
        return;
      }
      const picked = hit.pickedMesh;
      if (!picked) return;
      const mapping = meshToModelRef.current.get(picked);
      if (!mapping) return;

      // Select this mesh for rotation (R key works on selected mesh)
      selectedMeshRef.current = mapping.root;

      // Drag the ROOT, not whatever child mesh was actually clicked —
      // moving the root cascades the whole hierarchy, and it's also
      // guaranteed to be the same object as what we stored in
      // placedModels' `mesh` field for the sidebar's isDragging check.
      const root = mapping.root;

      // detachControl actually disables camera input; the previous
      // `camera.pointerInputEnabled = false` was not a real Babylon.js
      // property, so it silently did nothing — the camera kept orbiting
      // underneath the drag.
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
      selectedMeshRef.current = root;
      dragOffsetRef.current.copyFrom(hitPoint);
      dragStartPosRef.current.copyFrom(root.position);
      setDraggedModel(mapping.entry);
      console.log("[Viewer3D] drag start", {
        name: mapping.entry?.name,
        rootName: root.name,
      });

      // Always store the current quaternion as the base for next rotation
      modelRotationRef.current.set(root, {
        initialQuaternion: root.rotationQuaternion ?? Quaternion.Identity(),
      });

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

      const floorY = floorYRef.current;
      const deltaX = hitPoint.x - dragOffsetRef.current.x;
      const deltaZ = hitPoint.z - dragOffsetRef.current.z;

      mesh.position.set(
        dragStartPosRef.current.x + deltaX,
        floorY,
        dragStartPosRef.current.z + deltaZ,
      );

      const mapping = meshToModelRef.current.get(mesh);
      if (mapping) {
        mapping.entry.position[0] = mesh.position.x - bounds.cx;
        mapping.entry.position[1] = 0;
        mapping.entry.position[2] = mesh.position.z - bounds.cz;
      }
    };

    const onPointerUp = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;

      const mesh = draggedMeshRef.current;

      const cam = cameraRef.current;
      if (cam) {
        cam.attachControl(canvasRef.current, true);
      }

      if (mesh) {
        const mapping = meshToModelRef.current.get(mesh);
        if (mapping) {
          mapping.entry.position[0] = mesh.position.x - bounds.cx;
          mapping.entry.position[1] = 0;
          mapping.entry.position[2] = mesh.position.z - bounds.cz;
        }
        setDraggedModel(null);
      }

      // Clear refs AFTER using them above
      console.log("[Viewer3D] drag end", {
        rootName: mesh?.name ?? null,
      });
      draggedMeshRef.current = null;
      canvasRef.current.style.cursor = "crosshair";
    };

    canvasRef.current.addEventListener("pointerdown", onPointerDown);
    canvasRef.current.ownerDocument.addEventListener(
      "pointermove",
      onPointerMove,
    );
    canvasRef.current.ownerDocument.addEventListener("pointerup", onPointerUp);
    // keydown/keyup handled by window listeners (handleKeyDown/handleKeyUp)

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
      canvasRef.current?.ownerDocument?.removeEventListener(
        "pointermove",
        onPointerMove,
      );
      canvasRef.current?.ownerDocument?.removeEventListener(
        "pointerup",
        onPointerUp,
      );
      // keydown/keyup cleanup handled by window listener effect above
      if (document.pointerLockElement === canvasRef.current) {
        document.exitPointerLock();
      }
      ourMeshes.forEach((m) => m.dispose());
      floor.dispose();
      wallMat.dispose();
      floorMat.dispose();
      dirLight.dispose();
      hemiLight.dispose();
      shadowGen.dispose();
      camera.dispose();
      scene.dispose();
      engine.dispose();
    };
  }, [
    edges,
    windows,
    pixelsPerMeter,
    cameraDistance,
    bounds,
    wallHeightMeters,
    wallThicknessMeters,
    sillHeightMeters,
    windowHeightMeters,
  ]);

  useEffect(() => {
    const handleResize = () => {
      if (engineRef.current) {
        engineRef.current.resize();
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: isFullscreen ? "100vh" : 500,
        display: "flex",
        background: "#87ceeb",
        cursor: draggedModel ? "grabbing" : "crosshair",
        position: "relative",
      }}
    >
      <div
        style={{
          flex: 1,
          position: "relative",
          minHeight: 0,
        }}
      >
        <canvas
          ref={canvasRef}
          tabIndex={0}
          style={{
            width: "100%",
            height: "100%",
            border: "none",
            outline: "none",
          }}
        />
        {(!edges || edges.length === 0) && (
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              color: "#fff",
              fontSize: 14,
            }}
          >
            Loading 3D scene...
          </div>
        )}
        <button
          onClick={toggleFullscreen}
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            background: "rgba(0,0,0,0.5)",
            color: "#fff",
            border: "1px solid rgba(255,255,255,0.3)",
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "sans-serif",
          }}
        >
          {isFullscreen ? "⤓ Exit" : "⤢ Fullscreen"}
        </button>
        <div
          style={{
            position: "absolute",
            bottom: 8,
            left: 8,
            background: "rgba(0,0,0,0.5)",
            color: "#fff",
            padding: "4px 8px",
            borderRadius: 4,
            fontSize: 12,
            fontFamily: "sans-serif",
          }}
        >
          WASD to move · Click model to select · R to rotate · Del to remove ·
          Drag to reposition · Scroll to zoom
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
      />
    </div>
  );
}
