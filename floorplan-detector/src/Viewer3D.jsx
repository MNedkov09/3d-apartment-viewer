import { useEffect, useRef, useMemo, useState, useCallback } from "react";
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
} from "@babylonjs/core";
import { buildWallMeshes } from "./wallGeometry";

/**
 * Full 3D viewer. Takes the `edges` graph produced by the skeleton-tracing
 * step and renders it as walls you can orbit/walk around.
 *
 * @param {Array<{points: Array<[number,number]>, closed: boolean}>} edges
 * @param {number} pixelsPerMeter - calibration ratio from the 2D image (see scale-calibration step)
 * @param {number} wallHeightMeters
 * @param {number} wallThicknessMeters
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
  const [isLoading, setIsLoading] = useState(true);
  const [cursorLocked, setCursorLocked] = useState(false);
  const keysRef = useRef(new Set());

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

  // Compute plan bounds so Floor and camera share the same origin.
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
    const padding = 1;
    return {
      cx: ((minX + maxX) / 2) * scale,
      cz: ((minY + maxY) / 2) * scale,
      w: (maxX - minX) * scale + padding * 2,
      d: (maxY - minY) * scale + padding * 2,
    };
  }, [edges, pixelsPerMeter]);

  // Rough camera framing based on plan size, so the scene doesn't spawn
  // absurdly zoomed in/out regardless of how large the uploaded image is.
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

  // Create and dispose the Babylon scene
  useEffect(() => {
    if (!canvasRef.current || !edges || edges.length === 0) return;

    // Dispose previous scene if any
    if (sceneRef.current) {
      wallMeshesRef.current.forEach((m) => m.dispose());
      sceneRef.current.dispose();
      engineRef.current?.dispose();
    }

    // Create engine and scene
    const engine = new Engine(canvasRef.current, true, {
      preserveDrawingBuffer: false,
      stencil: false,
    });
    engineRef.current = engine;

    const scene = new Scene(engine);
    sceneRef.current = scene;

    // Clear background (sky blue)
    scene.clearColor = new Color4(0.53, 0.81, 0.92, 1);

    // ---- Camera: WASD + mouse orbit ----
    const camera = new ArcRotateCamera(
      "camera",
      -Math.PI / 4, // alpha (horizontal angle)
      Math.PI / 3, // beta (vertical angle, 60 degrees down)
      cameraDistance, // radius (distance from target)
      new Vector3(0, wallHeightMeters / 2, 0), // target
      scene,
    );
    camera.attachControl(canvasRef.current, true);
    // Clear built-in WASD bindings so our render loop handles movement
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

    const wasdSpeed = 8; // meters per second

    // ---- Lighting ----
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

    // ---- Shadows ----
    const shadowGen = new ShadowGenerator(1024, dirLight);
    shadowGen.useBlurExponentialShadowMap = true;
    shadowGen.blurKernel = 16;

    // ---- Floor ----
    const floor = MeshBuilder.CreatePlane(
      "floor",
      {
        width: bounds.w,
        height: bounds.d,
        sideOrientation: MeshBuilder.DOUBLESIDE,
      },
      scene,
    );
    floor.position = new Vector3(bounds.cx, 0, bounds.cz);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadows = true;

    const floorMat = new StandardMaterial("floorMat", scene);
    floorMat.diffuseColor = new Color3(0.86, 0.86, 0.86);
    floorMat.specularColor = new Color3(0.1, 0.1, 0.1);
    floor.material = floorMat;

    // ---- Walls ----
    // Clear previous wall meshes
    wallMeshesRef.current.forEach((m) => m.dispose());
    wallMeshesRef.current = [];

    const wallOpts = {
      thickness: wallThicknessMeters,
      height: wallHeightMeters,
      scale: 1 / pixelsPerMeter,
    };

    const wallMeshes = buildWallMeshes(edges, wallOpts, scene);
    wallMeshesRef.current = wallMeshes;

    // Capture our meshes locally so cleanup disposes the right ones
    // (in StrictMode the ref gets overwritten by the next effect before
    // the previous cleanup runs).
    const ourMeshes = [...wallMeshes];

    // Apply PBR material to each wall
    const wallMat = new PBRMaterial("wallMat", scene);
    wallMat.diffuseColor = new Color3(0.91, 0.89, 0.84); // #e8e2d5 warm wall color
    wallMat.metallic = 0.0;
    wallMat.roughness = 0.8;
    wallMeshes.forEach((mesh) => {
      mesh.material = wallMat;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    // ---- Render loop ----
    setIsLoading(false);

    // WASD movement helper
    const _forward = Vector3.Zero();
    const _right = Vector3.Zero();
    const _targetDelta = Vector3.Zero();

    // Prevent page scroll when canvas is focused
    const onWheel = (e) => {
      if (document.activeElement === canvasRef.current) {
        e.preventDefault();
      }
    };
    document.addEventListener("wheel", onWheel, { passive: false });

    // Update cursor lock state
    const onPointerLockChange = () => {
      setCursorLocked(document.pointerLockElement === canvasRef.current);
    };
    document.addEventListener("pointerlockchange", onPointerLockChange);

    // Prevent context menu from interfering with right-click orbit
    const onContextMenu = (e) => e.preventDefault();
    canvasRef.current.addEventListener("contextmenu", onContextMenu);

    engine.runRenderLoop(() => {
      // WASD: move camera target (camera follows at fixed radius)
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

    // Cleanup on unmount
    return () => {
      canvasRef.current?.removeEventListener("click", onCanvasClick);
      canvasRef.current?.removeEventListener("contextmenu", onContextMenu);
      document.removeEventListener("wheel", onWheel);
      document.removeEventListener("pointerlockchange", onPointerLockChange);
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
    wallHeightMeters,
    wallThicknessMeters,
    bounds,
    cameraDistance,
  ]);

  // Handle window resize
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
        background: "#87ceeb",
        cursor: "crosshair",
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
      {isLoading && (
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
        {cursorLocked
          ? "ESC to unlock · WASD to move · Right-click to orbit"
          : "Click to lock cursor · WASD to move · Mouse to orbit"}
      </div>
    </div>
  );
}
