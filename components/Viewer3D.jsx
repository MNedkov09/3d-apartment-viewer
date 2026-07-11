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
} from "@babylonjs/core";
import { Plane } from "@babylonjs/core";
import { GLTFFileLoader } from "@babylonjs/loaders";
import { buildWallMeshes } from "./wallGeometry";

// Register the GLTF loader once at module load time (global Babylon.js singleton).
SceneLoader.RegisterPlugin(new GLTFFileLoader());

// Default models to show when the user has no uploaded models yet.
const DEFAULT_MODELS = [
  {
    url: "/models/placeholder.glb",
    position: [0, 0, 0],
    scaling: [1, 1, 1],
  },
  {
    url: "/models/placeholder_copy.glb",
    position: [2, 0, 2],
    scaling: [1, 1, 1],
  },
  {
    url: "/models/placeholder_copy_2.glb",
    position: [-2, 0, -2],
    scaling: [1, 1, 1],
  },
];

/**
 * Fetch models from the API for the current user.
 * Returns an array of { url, position, scaling, name } objects, or null on error.
 */
async function fetchUserModels() {
  try {
    const res = await fetch("/api/models?limit=100");
    if (!res.ok) return null;
    const data = await res.json();
    return (data.models || []).map((m) => ({
      url: m.file_path,
      position: [0, 0, 0],
      scaling: [1, 1, 1],
      name: m.name,
      thumbnail: m.thumbnail_path,
    }));
  } catch {
    return null;
  }
}

/**
 * Sidebar panel listing loaded models. Click a model to start dragging it
 * in the 3D scene; its current position updates live while dragging.
 */
function ModelPanel({ models, draggedModel, onModelClick }) {
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
        Models
      </div>
      {models.length === 0 && (
        <div style={{ color: "rgba(255,255,255,0.4)", fontStyle: "italic" }}>
          Loading...
        </div>
      )}
      {models.map((m, i) => {
        const isDragging = draggedModel && draggedModel.mesh === m.mesh;
        return (
          <div
            key={i}
            onClick={() => onModelClick(m)}
            style={{
              padding: "6px 8px",
              borderRadius: 4,
              cursor: "pointer",
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
            }}
            onMouseEnter={(e) => {
              if (!isDragging)
                e.currentTarget.style.background = "rgba(255,255,255,0.1)";
            }}
            onMouseLeave={(e) => {
              if (!isDragging)
                e.currentTarget.style.background = "rgba(255,255,255,0.05)";
            }}
          >
            <span
              style={{
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
                  marginLeft: 6,
                }}
              >
                Drag
              </span>
            )}
          </div>
        );
      })}
      {models.length > 0 && (
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
          Click a model or drag it
          <br />
          directly in the scene.
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
  pixelsPerMeter = 50,
  wallHeightMeters = 2.7,
  wallThicknessMeters = 0.15,
}) {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const wallMeshesRef = useRef([]);
  const keysRef = useRef(new Set());
  const draggedMeshRef = useRef(null);
  const dragOffsetRef = useRef(new Vector3());
  const dragStartPosRef = useRef(new Vector3());
  // Unified drag state — used by BOTH the sidebar click (onModelClick) and
  // the in-scene pointer drag (onPointerDown/Move/Up), so a drag started
  // from either entry point behaves identically and cleans up correctly.
  const isDraggingRef = useRef(false);
  const meshToModelRef = useRef(new Map());
  const floorYRef = useRef(-0.05);
  const cameraRef = useRef(null);

  // Track loaded models for the sidebar panel.
  const [loadedModels, setLoadedModels] = useState([]);
  const [draggedModel, setDraggedModel] = useState(null);

  // Fetch user models on mount and merge with defaults.
  useEffect(() => {
    let cancelled = false;
    fetchUserModels().then((apiModels) => {
      if (cancelled) return;
      const modelsToLoad = apiModels && apiModels.length > 0 ? apiModels : DEFAULT_MODELS;
      setLoadedModels(modelsToLoad);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // Click a model in the panel to start dragging it in the scene.
  const onModelClick = useCallback((modelEntry) => {
    if (!modelEntry.mesh) return;
    const mesh = modelEntry.mesh;
    const cam = cameraRef.current;
    const scene = sceneRef.current;
    if (!cam || !scene) return;
    const flrY = floorYRef.current;

    // Disable camera so it doesn't fight the drag.  Use detachControl() to
    // fully remove Babylon's internal event listeners (not just isActive),
    // which is required for reliable re-attachment after a sidebar-initiated
    // drag that never calls pointerdown on the canvas.
    cam.detachControl();

    // Compute camera→ground intersection (canvas center, since there's no
    // actual pointer event from the sidebar click).
    const ray = scene.createPickingRay(
      canvasRef.current.clientWidth / 2,
      canvasRef.current.clientHeight / 2,
      Matrix.Identity(),
      cam,
    );
    const dir = ray?.direction;
    let hitPoint = cam.position.clone();
    if (dir && Math.abs(dir.y) > 1e-6) {
      const t = (flrY - cam.position.y) / dir.y;
      if (t >= 0) {
        hitPoint = new Vector3(
          cam.position.x + dir.x * t,
          flrY,
          cam.position.z + dir.z * t,
        );
      }
    }

    draggedMeshRef.current = mesh;
    dragOffsetRef.current.copyFrom(hitPoint);
    dragStartPosRef.current.copyFrom(mesh.position);
    isDraggingRef.current = true;
    setDraggedModel(modelEntry);
  }, []);

  const handleKeyDown = useCallback((e) => {
    keysRef.current.add(e.key.toLowerCase());
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

  const bounds = useMemo(() => {
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
  }, [edges, pixelsPerMeter]);

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

  // ---- Scene lifecycle: only depends on `edges` ----
  // When edges change the scene is rebuilt (floor/walls resize).
  // Model loading is handled separately by the "load models" effect below.
  useEffect(() => {
    if (!canvasRef.current || !edges || edges.length === 0) return;

    // Guard against stale async callbacks (SceneLoader.Append) from previous
    // effect runs.  When edges/bounds change the effect re-runs, disposes the
    // old scene, and the pending GLTF callback would otherwise operate on a
    // disposed scene → "Error in onSuccess callback".
    if (sceneRef.current) {
      wallMeshesRef.current.forEach((m) => m.dispose());
      sceneRef.current.dispose();
      engineRef.current?.dispose();
    }
    let disposed = false;

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

    wallMeshesRef.current = [];
    const wallOpts = {
      thickness: wallThicknessMeters,
      height: wallHeightMeters,
      scale: 1 / pixelsPerMeter,
    };
    const wallMeshes = buildWallMeshes(edges, wallOpts, scene);
    wallMeshesRef.current = wallMeshes;

    const ourMeshes = [...wallMeshes];

    const wallMat = new PBRMaterial("wallMat", scene);
    wallMat.diffuseColor = new Color3(0.91, 0.89, 0.84);
    wallMat.metallic = 0.0;
    wallMat.roughness = 0.8;
    wallMeshes.forEach((mesh) => {
      mesh.material = wallMat;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

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

    // ---- Drag-and-drop for model meshes ----
    // Both entry points (sidebar click via onModelClick, and direct scene
    // click via onPointerDown below) share isDraggingRef / draggedMeshRef /
    // dragOffsetRef, so a drag started either way is tracked identically
    // and always gets cleaned up by onPointerUp.

    // Intersect a camera ray (from canvas pixel) with the ground plane Y=flrY.
    // Returns the 3D point on the ground, or null if the ray is parallel/away.

    function rayGroundIntersect(cam, flrY, ptrX, ptrY) {
      const ray = scene.createPickingRay(
        ptrX ?? canvasRef.current.clientWidth / 2,
        ptrY ?? canvasRef.current.clientHeight / 2,
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

    const onPointerDown = (e) => {
      if (e.button !== 0) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const ptrX = e.clientX - rect.left;
      const ptrY = e.clientY - rect.top;

      // Pick the mesh under the cursor.
      const hit = scene.pick(ptrX, ptrY, (mesh) =>
        meshToModelRef.current.has(mesh),
      );
      if (!hit.hit) return;
      const picked = hit.pickedMesh;
      if (!picked) return;
      const modelEntry = meshToModelRef.current.get(picked);
      if (!modelEntry) return;

      // Disable camera so Babylon can't process pointer events (orbit/zoom)
      // during drag — isActive alone is insufficient for left-drag orbit.
      camera.detachControl();

      // Cast ray from camera through cursor onto ground plane to get initial hit.
      const hitPoint = rayGroundIntersect(
        camera,
        floorYRef.current,
        ptrX,
        ptrY,
      );
      if (!hitPoint) {
        camera.attachControl(canvasRef.current, true);
        return;
      }

      isDraggingRef.current = true;
      draggedMeshRef.current = picked;
      dragOffsetRef.current.copyFrom(hitPoint); // now stores the START hit point
      dragStartPosRef.current.copyFrom(picked.position); // and the START mesh position
      setDraggedModel(modelEntry);
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
        camera,
        floorYRef.current,
        ptrX,
        ptrY,
      );
      if (!hitPoint) return;

      // Snap Y to floor level.
      const floorY = floorYRef.current;
      const deltaX = hitPoint.x - dragOffsetRef.current.x;
      const deltaZ = hitPoint.z - dragOffsetRef.current.z;

      mesh.position.set(
        dragStartPosRef.current.x - deltaX,
        floorY,
        dragStartPosRef.current.z - deltaZ,
      );

      // Update the model entry's position via ref (no React state update).
      const entry = meshToModelRef.current.get(mesh);
      if (entry) {
        entry.position[0] = mesh.position.x - bounds.cx;
        entry.position[1] = 0; // snapped to floor
        entry.position[2] = mesh.position.z - bounds.cz;
      }
    };

    const onPointerUp = () => {
      if (!isDraggingRef.current) return;
      isDraggingRef.current = false;

      // Always re-enable camera interaction, regardless of which path ended
      // the drag (sidebar click vs. in-scene pointerdown).  The sidebar path
      // never calls detachControl() before dragging — we use isActive=false
      // there — so this is essential for unlocking the camera after a drop.
      const cam = cameraRef.current;
      if (cam) {
        cam.isActive = true;
        if (!cam.inputs.attachedToElement) {
          cam.attachControl(canvasRef.current, true);
        }
      }

      // Sync final position into the model entry.
      const mesh = draggedMeshRef.current;
      if (mesh) {
        const entry = meshToModelRef.current.get(mesh);
        if (entry) {
          entry.position[0] = mesh.position.x - bounds.cx;
          entry.position[1] = 0;
          entry.position[2] = mesh.position.z - bounds.cz;
        }
        draggedMeshRef.current = null;
        setDraggedModel(null);
      }
      canvasRef.current.style.cursor = "crosshair";
    };

    canvasRef.current.addEventListener("pointerdown", onPointerDown);
    canvasRef.current.ownerDocument.addEventListener(
      "pointermove",
      onPointerMove,
    );
    canvasRef.current.ownerDocument.addEventListener("pointerup", onPointerUp);

    engine.runRenderLoop(() => {
      const keys = keysRef.current;
      const target = camera.target;
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
      disposed = true;
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
    pixelsPerMeter,
    cameraDistance,
    bounds,
    wallHeightMeters,
    wallThicknessMeters,
  ]);

  // ---- Load models: independent of scene lifecycle ----
  // Fetches models from the API (or falls back to defaults) and loads them
  // into the existing scene. Runs on mount and whenever the model list changes.
  useEffect(() => {
    if (!sceneRef.current) return;

    const modelsToLoad = loadedModels.length > 0 ? loadedModels : DEFAULT_MODELS;
    const scene = sceneRef.current;
    let disposed = false;

    const modelMeshes = [];

    for (const { url, position, scaling } of modelsToLoad) {
      const name = url.split("/").pop().replace(/\.[^.]+$/, "");
      const prevMeshCount = scene.meshes.length;

      SceneLoader.Append(
        "",
        url,
        scene,
        () => {
          if (disposed) return;
          const allMeshes = scene.meshes;
          const newMeshes = allMeshes.slice(prevMeshCount);
          const floor = scene.getMeshByName("floor");
          const modelMeshesInRange = newMeshes.filter(
            (m) => m !== floor && !wallMeshesRef.current.includes(m),
          );
          if (modelMeshesInRange.length === 0) {
            console.warn(`GLTF loaded but produced no visible geometry: ${url}`);
            return;
          }
          modelMeshesInRange.forEach((m) => {
            if (!m || !m.material) {
              const mat = new PBRMaterial(`${m?.name || "unknown"}_mat`, scene);
              mat.diffuseColor = new Color3(0.6, 0.4, 0.2);
              mat.metallic = 0.1;
              mat.roughness = 0.7;
              if (m) m.material = mat;
            }
            if (m) {
              m.position = new Vector3(
                bounds.cx + position[0],
                position[1],
                bounds.cz + position[2],
              );
              m.scaling = new Vector3(scaling[0], scaling[1], scaling[2]);
              m.castShadow = true;
              m.receiveShadow = true;
              modelMeshes.push(m);
              meshToModelRef.current.set(m, {
                name,
                position: [...position],
                scaling: [...scaling],
                url,
              });
            }
          });
          setLoadedModels(
            [...meshToModelRef.current.entries()].map(([mesh, entry]) => ({
              ...entry,
              mesh,
            })),
          );
        },
        undefined,
        (_scene, message) =>
          console.error(`GLTF load failed: ${url}: ${message}`),
      );
    }
    // Dispose model meshes on unload / re-fetch
    return () => {
      disposed = true;
      modelMeshes.forEach((m) => m.dispose());
    };
  }, [loadedModels, bounds]);

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
      style={{
        width: "100%",
        height: 500,
        display: "flex",
        background: "#87ceeb",
        cursor: draggedModel ? "grabbing" : "crosshair",
      }}
    >
      {/* 3D canvas area */}
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
          WASD to move · Left-click to orbit · Drag models to reposition ·
          Scroll to zoom
        </div>
      </div>
      {/* Model sidebar */}
      <ModelPanel
        models={loadedModels}
        draggedModel={draggedModel}
        onModelClick={onModelClick}
      />
    </div>
  );
}
