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

export const FLOOR_MATERIALS = [
  { id: "wood", name: "Oak Parquet", desc: "Scandinavian natural oak planks", color: "#d7be98" },
  { id: "walnut", name: "Dark Walnut", desc: "Espresso rich luxury wood", color: "#4a3525" },
  { id: "marble", name: "Carrara Marble", desc: "Italian polished white stone", color: "#f3f4f6" },
  { id: "tile", name: "Architectural Tile", desc: "Contemporary porcelain tiles", color: "#c5c9ce" },
  { id: "concrete", name: "Slate Concrete", desc: "Minimalist industrial finish", color: "#9ca3af" },
  { id: "terracotta", name: "Tuscan Terracotta", desc: "Warm rustic clay tiles", color: "#b86146" },
];

export const WALL_FINISHES = [
  { id: "white", name: "Modern White", desc: "Pure architectural smooth plaster", color: "#f8fafc" },
  { id: "greige", name: "Warm Greige", desc: "Soft Scandinavian warm neutral", color: "#dfdad0" },
  { id: "sage", name: "Sage Green", desc: "Serene calm natural accent", color: "#8f9f91" },
  { id: "navy", name: "Midnight Navy", desc: "Bold dramatic luxury statement", color: "#243342" },
  { id: "brick", name: "Loft Brick", desc: "Exposed architectural red brick", color: "#b2523e" },
  { id: "charcoal", name: "Studio Charcoal", desc: "Contemporary industrial matte", color: "#2f3542" },
];

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

export function createFloorMaterial(style = "wood", scene) {
  const mat = new PBRMaterial(`floorMat_${style}_${Date.now()}`, scene);
  const tex = new DynamicTexture(`floorTex_${style}_${Date.now()}`, { width: 512, height: 512 }, scene, true);
  const ctx = tex.getContext();

  if (style === "wood") {
    // Warm Scandinavian Oak Hardwood Parquet
    ctx.fillStyle = "#d7be98";
    ctx.fillRect(0, 0, 512, 512);
    const plankH = 32;
    for (let y = 0; y < 512; y += plankH) {
      const plankW = 128;
      const xOffset = ((y / plankH) % 2) * 64;
      for (let x = -64; x < 512 + 64; x += plankW) {
        const tone = 205 + Math.floor(Math.sin(x * 12 + y * 7) * 12);
        ctx.fillStyle = `rgb(${tone + 18}, ${tone}, ${tone - 26})`;
        ctx.fillRect(x + xOffset, y, plankW - 2, plankH - 2);
        ctx.fillStyle = "rgba(105, 75, 35, 0.09)";
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
    mat.roughness = 0.38;
    mat.metallic = 0.02;
  } else if (style === "walnut") {
    // Rich Dark Walnut Parquet
    ctx.fillStyle = "#4a3525";
    ctx.fillRect(0, 0, 512, 512);
    const plankH = 32;
    for (let y = 0; y < 512; y += plankH) {
      const plankW = 128;
      const xOffset = ((y / plankH) % 2) * 64;
      for (let x = -64; x < 512 + 64; x += plankW) {
        const tone = 65 + Math.floor(Math.sin(x * 12 + y * 7) * 8);
        ctx.fillStyle = `rgb(${tone + 16}, ${tone + 4}, ${tone - 10})`;
        ctx.fillRect(x + xOffset, y, plankW - 2, plankH - 2);
        ctx.fillStyle = "rgba(20, 10, 5, 0.25)";
        ctx.fillRect(x + xOffset, y + 8, plankW - 2, 2);
      }
      ctx.fillStyle = "#2c1d12";
      ctx.fillRect(0, y + plankH - 2, 512, 2);
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.32;
    mat.metallic = 0.04;
  } else if (style === "marble") {
    // Carrara White Italian Polished Marble
    ctx.fillStyle = "#f4f5f7";
    ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = "rgba(180, 185, 195, 0.35)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(30, 0); ctx.bezierCurveTo(120, 180, 200, 240, 310, 512);
    ctx.moveTo(250, 0); ctx.bezierCurveTo(340, 140, 420, 300, 490, 512);
    ctx.stroke();
    ctx.strokeStyle = "rgba(160, 165, 175, 0.2)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, 150); ctx.bezierCurveTo(160, 200, 320, 120, 512, 240);
    ctx.stroke();
    // Grout tile grid
    ctx.strokeStyle = "rgba(200, 205, 215, 0.4)";
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, 255, 255);
    ctx.strokeRect(257, 1, 254, 255);
    ctx.strokeRect(1, 257, 255, 254);
    ctx.strokeRect(257, 257, 254, 254);
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 3;
    mat.albedoTexture.vScale = 3;
    mat.roughness = 0.18;
    mat.metallic = 0.08;
  } else if (style === "tile") {
    // Modern Architectural Grey Tile
    ctx.fillStyle = "#c5c9ce";
    ctx.fillRect(0, 0, 512, 512);
    const tileSize = 128;
    for (let y = 0; y < 512; y += tileSize) {
      for (let x = 0; x < 512; x += tileSize) {
        const tone = 225 + Math.floor(Math.cos(x + y) * 6);
        ctx.fillStyle = `rgb(${tone}, ${tone}, ${tone})`;
        ctx.fillRect(x + 2, y + 2, tileSize - 4, tileSize - 4);
      }
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.28;
    mat.metallic = 0.06;
  } else if (style === "terracotta") {
    // Tuscan Terracotta Warm Clay
    ctx.fillStyle = "#b86146";
    ctx.fillRect(0, 0, 512, 512);
    const tileSize = 128;
    for (let y = 0; y < 512; y += tileSize) {
      for (let x = 0; x < 512; x += tileSize) {
        const tone = 180 + Math.floor(Math.sin(x * 7 + y) * 12);
        ctx.fillStyle = `rgb(${tone + 10}, ${Math.floor(tone * 0.52)}, ${Math.floor(tone * 0.38)})`;
        ctx.fillRect(x + 3, y + 3, tileSize - 6, tileSize - 6);
      }
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.45;
    mat.metallic = 0.02;
  } else {
    // Minimalist Contemporary Slate Concrete
    ctx.fillStyle = "#9ca3af";
    ctx.fillRect(0, 0, 512, 512);
    for (let i = 0; i < 400; i++) {
      const rx = Math.random() * 512;
      const ry = Math.random() * 512;
      const tone = 140 + Math.floor(Math.random() * 30);
      ctx.fillStyle = `rgba(${tone}, ${tone}, ${tone}, 0.15)`;
      ctx.fillRect(rx, ry, 3, 3);
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 4;
    mat.albedoTexture.vScale = 4;
    mat.roughness = 0.55;
    mat.metallic = 0.02;
  }
  return mat;
}

export function createWallMaterial(style = "white", scene) {
  const mat = new PBRMaterial(`wallMat_${style}_${Date.now()}`, scene);

  if (style === "brick") {
    const tex = new DynamicTexture(`brickTex_${Date.now()}`, { width: 512, height: 512 }, scene, true);
    const ctx = tex.getContext();
    ctx.fillStyle = "#c8c4be";
    ctx.fillRect(0, 0, 512, 512);
    const rowH = 32;
    for (let y = 0; y < 512; y += rowH) {
      const brickW = 80;
      const xOffset = ((y / rowH) % 2) * 40;
      for (let x = -40; x < 512 + 40; x += brickW) {
        const tone = 150 + Math.floor(Math.sin(x * 17 + y) * 20);
        ctx.fillStyle = `rgb(${tone + 30}, ${Math.floor(tone * 0.55)}, ${Math.floor(tone * 0.45)})`;
        ctx.fillRect(x + xOffset + 2, y + 2, brickW - 4, rowH - 4);
      }
    }
    tex.update();
    mat.albedoTexture = tex;
    mat.albedoTexture.uScale = 3;
    mat.albedoTexture.vScale = 2;
    mat.roughness = 0.85;
    mat.metallic = 0.0;
  } else if (style === "greige") {
    mat.albedoColor = new Color3(0.88, 0.86, 0.82);
    mat.roughness = 0.88;
    mat.metallic = 0.0;
  } else if (style === "sage") {
    mat.albedoColor = new Color3(0.58, 0.65, 0.59);
    mat.roughness = 0.85;
    mat.metallic = 0.0;
  } else if (style === "navy") {
    mat.albedoColor = new Color3(0.14, 0.20, 0.26);
    mat.roughness = 0.82;
    mat.metallic = 0.02;
  } else if (style === "charcoal") {
    mat.albedoColor = new Color3(0.18, 0.21, 0.25);
    mat.roughness = 0.85;
    mat.metallic = 0.0;
  } else {
    // Pure Architectural White
    mat.albedoColor = new Color3(0.96, 0.96, 0.95);
    mat.roughness = 0.88;
    mat.metallic = 0.0;
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
        { diameterTop: 0.30, diameterBottom: 0.22, height: 0.40 },
        scene,
      );
      base.position.set(cx, 0.20, cz);
      base.material = ceramicMat;
      base.castShadow = true;
      base.receiveShadow = true;

      const cistern = MeshBuilder.CreateBox(
        `toilet_tank_${i}`,
        { width: 0.34, depth: 0.18, height: 0.38 },
        scene,
      );
      cistern.position.set(cx, 0.48, cz - 0.12);
      cistern.material = ceramicMat;
      cistern.castShadow = true;
      cistern.receiveShadow = true;
      fixtureMeshes.push(base, cistern);
    } else if (f.label === "sink") {
      const vanity = MeshBuilder.CreateBox(
        `vanity_${i}`,
        { width: 0.42, depth: 0.35, height: 0.78 },
        scene,
      );
      vanity.position.set(cx, 0.39, cz);
      vanity.material = ceramicMat;
      vanity.castShadow = true;
      vanity.receiveShadow = true;

      const tap = MeshBuilder.CreateCylinder(
        `tap_${i}`,
        { diameter: 0.03, height: 0.14 },
        scene,
      );
      tap.position.set(cx, 0.82, cz - 0.10);
      tap.material = chromeMat;
      tap.castShadow = true;
      fixtureMeshes.push(vanity, tap);
    } else if (f.label === "shower") {
      const tray = MeshBuilder.CreateBox(
        `shower_tray_${i}`,
        { width: 0.80, depth: 0.80, height: 0.06 },
        scene,
      );
      tray.position.set(cx, 0.03, cz);
      tray.material = ceramicMat;
      tray.receiveShadow = true;

      // Realistic modern low-profile shower glass screen (height: 1.25m, width: 0.75m)
      const glass = MeshBuilder.CreateBox(
        `shower_glass_${i}`,
        { width: 0.75, depth: 0.02, height: 1.25 },
        scene,
      );
      // Bottom of glass rests neatly on the tray: y = 0.06 + 1.25/2 = 0.685m
      glass.position.set(cx, 0.685, cz + 0.35);
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
        // Collinear edge handling
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
        // Collinear edge handling
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

function RightSidePanel({
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
  floorStyle,
  wallStyle,
  onApplyFloorMaterial,
  onApplyWallMaterial,
  stats,
}) {
  const [activeTab, setActiveTab] = useState("furniture"); // "furniture" | "materials" | "scene"

  return (
    <div
      style={{
        width: 310,
        background: "rgba(18, 22, 31, 0.96)",
        backdropFilter: "blur(16px)",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        fontSize: 12,
        fontFamily: "var(--font-mono, monospace)",
        borderLeft: "1px solid rgba(255,255,255,0.12)",
        zIndex: 10,
        height: "100%",
        userSelect: "none",
        overflow: "hidden",
      }}
    >
      {/* Top Panel Tabs */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid rgba(255,255,255,0.1)",
          background: "rgba(12, 15, 22, 0.75)",
          flexShrink: 0,
        }}
      >
        <button
          onClick={() => setActiveTab("furniture")}
          style={{
            flex: 1,
            padding: "11px 4px",
            background: activeTab === "furniture" ? "rgba(255,255,255,0.12)" : "transparent",
            color: activeTab === "furniture" ? "#fff" : "rgba(255,255,255,0.6)",
            border: "none",
            borderBottom: activeTab === "furniture" ? "2px solid #38bdf8" : "2px solid transparent",
            cursor: "pointer",
            fontSize: 11,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 4,
            transition: "all 0.15s ease",
          }}
        >
          🪑 Furniture
        </button>
        <button
          onClick={() => setActiveTab("materials")}
          style={{
            flex: 1,
            padding: "11px 4px",
            background: activeTab === "materials" ? "rgba(255,255,255,0.12)" : "transparent",
            color: activeTab === "materials" ? "#fff" : "rgba(255,255,255,0.6)",
            border: "none",
            borderBottom: activeTab === "materials" ? "2px solid #38bdf8" : "2px solid transparent",
            cursor: "pointer",
            fontSize: 11,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 4,
            transition: "all 0.15s ease",
          }}
        >
          🎨 Materials
        </button>
        <button
          onClick={() => setActiveTab("scene")}
          style={{
            flex: 1,
            padding: "11px 4px",
            background: activeTab === "scene" ? "rgba(255,255,255,0.12)" : "transparent",
            color: activeTab === "scene" ? "#fff" : "rgba(255,255,255,0.6)",
            border: "none",
            borderBottom: activeTab === "scene" ? "2px solid #38bdf8" : "2px solid transparent",
            cursor: "pointer",
            fontSize: 11,
            fontWeight: 600,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 4,
            transition: "all 0.15s ease",
          }}
        >
          💾 Scene
        </button>
      </div>

      {/* Tab 1: Furniture Catalog & Placed Items (Flex-expanded to fill entire height!) */}
      {activeTab === "furniture" && (
        <div
          style={{
            flex: 1,
            padding: "12px",
            display: "flex",
            flexDirection: "column",
            gap: 12,
            minHeight: 0,
            overflow: "hidden",
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
                padding: "10px 14px",
                fontSize: 12,
                fontWeight: 600,
                cursor: isAutoFurnishing ? "wait" : "pointer",
                textAlign: "center",
                boxShadow: "0 2px 8px rgba(16, 185, 129, 0.35)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
                flexShrink: 0,
              }}
            >
              {isAutoFurnishing ? "⏳ Positioning Models..." : `✨ Auto-Furnish (${furnitureCount})`}
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
                padding: "6px 10px",
                fontSize: 11,
                fontWeight: 500,
                cursor: "pointer",
                textAlign: "center",
                flexShrink: 0,
              }}
            >
              🗑️ Clear Furniture ({placedModels.length})
            </button>
          )}

          {/* Models Catalog: Generous flex container with larger preview cards */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              flex: placedModels.length > 0 ? "1.2" : "1",
              minHeight: 0,
            }}
          >
            <div
              style={{
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "rgba(255,255,255,0.5)",
                marginBottom: 6,
                flexShrink: 0,
              }}
            >
              Models Catalog ({availableModels.length})
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 6,
                flex: 1,
                minHeight: 0,
                overflowY: "auto",
                paddingRight: 4,
              }}
            >
              {availableModels.map((m, i) => (
                <div
                  key={i}
                  onClick={() => onPlaceModel(m)}
                  style={{
                    padding: "8px 10px",
                    borderRadius: 6,
                    cursor: "pointer",
                    background: "rgba(255,255,255,0.06)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    transition: "all 0.15s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.14)";
                    e.currentTarget.style.borderColor = "rgba(56, 189, 248, 0.4)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "rgba(255,255,255,0.06)";
                    e.currentTarget.style.borderColor = "rgba(255,255,255,0.08)";
                  }}
                >
                  {m.thumbnail ? (
                    <img
                      src={m.thumbnail}
                      alt=""
                      width={44}
                      height={44}
                      style={{
                        flexShrink: 0,
                        objectFit: "contain",
                        borderRadius: 5,
                        background: "rgba(0,0,0,0.3)",
                        padding: 2,
                      }}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                    />
                  ) : (
                    <div style={{ width: 44, height: 44, flexShrink: 0, background: "rgba(255,255,255,0.1)", borderRadius: 5 }} />
                  )}
                  <span
                    style={{
                      flex: 1,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontSize: 12,
                      fontWeight: 600,
                    }}
                  >
                    {m.name}
                  </span>
                  <span
                    style={{
                      fontSize: 11,
                      color: "#38bdf8",
                      padding: "4px 8px",
                      background: "rgba(56, 189, 248, 0.18)",
                      borderRadius: 4,
                      fontWeight: 600,
                      flexShrink: 0,
                    }}
                  >
                    + Add
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* In Scene Section: Flex container filling the bottom area with zero empty void */}
          {placedModels.length > 0 && (
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                flex: "1",
                minHeight: 0,
                borderTop: "1px solid rgba(255,255,255,0.1)",
                paddingTop: 8,
              }}
            >
              <div
                style={{
                  fontWeight: 600,
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: "0.08em",
                  color: "rgba(255,255,255,0.5)",
                  marginBottom: 6,
                  flexShrink: 0,
                }}
              >
                In Scene ({placedModels.length})
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 5,
                  flex: 1,
                  minHeight: 0,
                  overflowY: "auto",
                  paddingRight: 4,
                }}
              >
                {placedModels.map((m, i) => {
                  const isDragging = draggedModel && draggedModel.mesh === m.mesh;
                  return (
                    <div
                      key={`placed-${i}`}
                      style={{
                        padding: "7px 10px",
                        borderRadius: 6,
                        background: isDragging
                          ? "rgba(56, 189, 248, 0.25)"
                          : "rgba(255,255,255,0.06)",
                        border: isDragging
                          ? "1px solid rgba(56, 189, 248, 0.5)"
                          : "1px solid rgba(255,255,255,0.06)",
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        gap: 8,
                        fontSize: 12,
                      }}
                    >
                      <span
                        style={{
                          flex: 1,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                          fontWeight: 500,
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
                          background: "rgba(239, 68, 68, 0.25)",
                          color: "#fca5a5",
                          border: "none",
                          borderRadius: 4,
                          width: 22,
                          height: 22,
                          fontSize: 11,
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          transition: "background 0.15s ease",
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Materials Palette */}
      {activeTab === "materials" && (
        <div
          style={{
            flex: 1,
            padding: "12px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 14,
            minHeight: 0,
          }}
        >
          <div style={{ fontSize: 11, color: "rgba(255,255,255,0.6)", lineHeight: 1.4 }}>
            💡 <strong>Drag & Drop</strong> any swatch onto the 3D floor or walls, or click <strong>Apply</strong>.
          </div>

          {/* Flooring Materials */}
          <div>
            <div
              style={{
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#38bdf8",
                marginBottom: 6,
              }}
            >
              🪵 Floor Finishes
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {FLOOR_MATERIALS.map((mat) => {
                const isSelected = floorStyle === mat.id;
                return (
                  <div
                    key={mat.id}
                    draggable={true}
                    onDragStart={(e) => {
                      e.dataTransfer.setData(
                        "application/json",
                        JSON.stringify({ type: "floor", id: mat.id, name: mat.name }),
                      );
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "7px 10px",
                      borderRadius: 6,
                      background: isSelected ? "rgba(56, 189, 248, 0.15)" : "rgba(255,255,255,0.06)",
                      border: isSelected ? "1px solid #38bdf8" : "1px solid rgba(255,255,255,0.08)",
                      cursor: "grab",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <div
                      style={{
                        width: 26,
                        height: 26,
                        borderRadius: 4,
                        background: mat.color,
                        boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.3)",
                        flexShrink: 0,
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#fff" }}>{mat.name}</div>
                      <div style={{ fontSize: 9, color: "rgba(255,255,255,0.5)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                        {mat.desc}
                      </div>
                    </div>
                    <button
                      onClick={() => onApplyFloorMaterial(mat.id)}
                      style={{
                        background: isSelected ? "#0284c7" : "rgba(255,255,255,0.12)",
                        color: "#fff",
                        border: "none",
                        borderRadius: 4,
                        padding: "4px 8px",
                        fontSize: 10,
                        cursor: "pointer",
                      }}
                    >
                      {isSelected ? "Active" : "Apply"}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Wall Finishes */}
          <div>
            <div
              style={{
                fontWeight: 600,
                fontSize: 11,
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "#a78bfa",
                marginBottom: 6,
              }}
            >
              🧱 Wall Finishes
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
              {WALL_FINISHES.map((fin) => {
                const isSelected = wallStyle === fin.id;
                return (
                  <div
                    key={fin.id}
                    draggable={true}
                    onDragStart={(e) => {
                      e.dataTransfer.setData(
                        "application/json",
                        JSON.stringify({ type: "wall", id: fin.id, name: fin.name }),
                      );
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      padding: "7px 10px",
                      borderRadius: 6,
                      background: isSelected ? "rgba(167, 139, 250, 0.15)" : "rgba(255,255,255,0.06)",
                      border: isSelected ? "1px solid #a78bfa" : "1px solid rgba(255,255,255,0.08)",
                      cursor: "grab",
                      transition: "all 0.15s ease",
                    }}
                  >
                    <div
                      style={{
                        width: 26,
                        height: 26,
                        borderRadius: 4,
                        background: fin.color,
                        boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.3)",
                        flexShrink: 0,
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, fontWeight: 600, color: "#fff" }}>{fin.name}</div>
                      <div style={{ fontSize: 9, color: "rgba(255,255,255,0.5)", textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                        {fin.desc}
                      </div>
                    </div>
                    <button
                      onClick={() => onApplyWallMaterial(fin.id)}
                      style={{
                        background: isSelected ? "#7c3aed" : "rgba(255,255,255,0.12)",
                        color: "#fff",
                        border: "none",
                        borderRadius: 4,
                        padding: "4px 8px",
                        fontSize: 10,
                        cursor: "pointer",
                      }}
                    >
                      {isSelected ? "Active" : "Apply"}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: Scene State & Metrics */}
      {activeTab === "scene" && (
        <div
          style={{
            flex: 1,
            padding: "12px",
            overflowY: "auto",
            display: "flex",
            flexDirection: "column",
            gap: 12,
            minHeight: 0,
          }}
        >
          <div
            style={{
              fontWeight: 600,
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "rgba(255,255,255,0.5)",
            }}
          >
            Scene Presets & Backup
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            <button
              onClick={onExportScene}
              style={{
                flex: 1,
                padding: "8px 0",
                background: "rgba(255,255,255,0.1)",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              💾 Export JSON
            </button>
            <label
              style={{
                flex: 1,
                padding: "8px 0",
                background: "rgba(255,255,255,0.1)",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: 6,
                cursor: "pointer",
                fontSize: 11,
                fontWeight: 600,
                textAlign: "center",
              }}
            >
              📂 Import JSON
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

          <div
            style={{
              fontWeight: 600,
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.08em",
              color: "rgba(255,255,255,0.5)",
              marginTop: 6,
            }}
          >
            Architectural Metrics
          </div>

          <div
            style={{
              background: "rgba(255,255,255,0.05)",
              borderRadius: 6,
              padding: "10px",
              display: "flex",
              flexDirection: "column",
              gap: 6,
              fontSize: 11,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "rgba(255,255,255,0.6)" }}>Scale Ratio:</span>
              <span style={{ fontWeight: 600 }}>32 px / meter</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "rgba(255,255,255,0.6)" }}>Apartment Footprint:</span>
              <span style={{ fontWeight: 600 }}>~16.2m × 11.2m</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "rgba(255,255,255,0.6)" }}>Wall Height:</span>
              <span style={{ fontWeight: 600 }}>2.70 meters</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "rgba(255,255,255,0.6)" }}>Placed Furniture:</span>
              <span style={{ fontWeight: 600 }}>{placedModels.length} models</span>
            </div>
            {stats && (
              <>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "rgba(255,255,255,0.6)" }}>Wall Segments:</span>
                  <span style={{ fontWeight: 600 }}>{stats.wall_edges || 0}</span>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "rgba(255,255,255,0.6)" }}>Openings:</span>
                  <span style={{ fontWeight: 600 }}>{(stats.windows || 0) + (stats.doors || 0)}</span>
                </div>
              </>
            )}
          </div>
        </div>
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
  pixelsPerMeter = 32,
  wallHeightMeters = 2.7,
  wallThicknessMeters = 0.15,
  sillHeightMeters = 0.9,
  windowHeightMeters = 1.2,
  stats,
}) {
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const wallMeshesRef = useRef([]);
  const keysRef = useRef(new Set());
  const floorMeshRef = useRef(null);
  const draggedMeshRef = useRef(null);
  const selectedMeshRef = useRef(null);
  const dragOffsetRef = useRef(new Vector3());
  const dragStartPosRef = useRef(new Vector3());
  const isDraggingRef = useRef(false);
  const meshToModelRef = useRef(new Map());
  const floorYRef = useRef(0);
  const edgesRef = useRef(edges);
  const windowsRef = useRef(windows);
  const doorsRef = useRef(doors);
  const modelRotationRef = useRef(new Map());
  const yawAngleRef = useRef(new Map());
  const containerRef = useRef(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [placedModels, setPlacedModels] = useState([]);
  const [draggedModel, setDraggedModel] = useState(null);
  const [isAutoFurnishing, setIsAutoFurnishing] = useState(false);
  const [floorStyle, setFloorStyle] = useState("wood");
  const [wallStyle, setWallStyle] = useState("white");
  const [viewpoint, setViewpointState] = useState("iso");
  const [toastMessage, setToastMessage] = useState(null);

  const toastTimeoutRef = useRef(null);
  const showToast = useCallback((msg) => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setToastMessage(msg);
    toastTimeoutRef.current = setTimeout(() => {
      setToastMessage(null);
    }, 2500);
  }, []);

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
    const diag = Math.hypot(bounds.w, bounds.d);
    return Math.max(diag * 1.1, 8);
  }, [bounds]);

  const toggleFullscreen = useCallback(async () => {
    const el = containerRef.current;
    if (!el) return;
    try {
      if (!document.fullscreenElement) {
        await el.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch (e) {
      console.warn("Fullscreen toggle error:", e);
    }
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(document.fullscreenElement === containerRef.current);
      setTimeout(() => engineRef.current?.resize(), 100);
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  const setViewpoint = useCallback(
    (mode) => {
      const cam = cameraRef.current;
      if (!cam) return;
      setViewpointState(mode);

      if (mode === "top") {
        cam.alpha = -Math.PI / 2;
        cam.beta = 0.04;
        cam.radius = cameraDistance * 1.15;
        cam.target.set(bounds.cx, 0, bounds.cz);
      } else if (mode === "iso") {
        cam.alpha = -Math.PI / 4;
        cam.beta = Math.PI / 3.4;
        cam.radius = cameraDistance * 1.1;
        cam.target.set(bounds.cx, wallHeightMeters / 2, bounds.cz);
      } else if (mode === "walk") {
        cam.alpha = -Math.PI / 4;
        cam.beta = Math.PI / 2.05;
        cam.radius = 0.5;
        cam.target.set(bounds.cx, 1.45, bounds.cz);
      } else if (mode === "living") {
        cam.alpha = -Math.PI / 3;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.45;
        cam.target.set(bounds.cx - bounds.w * 0.28, 1.2, bounds.cz + bounds.d * 0.18);
      } else if (mode === "kitchen") {
        cam.alpha = -Math.PI / 1.3;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.42;
        cam.target.set(bounds.cx + bounds.w * 0.24, 1.2, bounds.cz + bounds.d * 0.18);
      } else if (mode === "master_bed") {
        cam.alpha = -Math.PI / 1.5;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.42;
        cam.target.set(bounds.cx + bounds.w * 0.26, 1.2, bounds.cz - bounds.d * 0.25);
      } else if (mode === "bed_2") {
        cam.alpha = -Math.PI / 3;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.42;
        cam.target.set(bounds.cx - bounds.w * 0.05, 1.2, bounds.cz - bounds.d * 0.25);
      } else if (mode === "bath") {
        cam.alpha = -Math.PI / 2.4;
        cam.beta = Math.PI / 3.0;
        cam.radius = cameraDistance * 0.32;
        cam.target.set(bounds.cx - bounds.w * 0.06, 1.2, bounds.cz - bounds.d * 0.02);
      } else if (mode === "entry") {
        cam.alpha = -Math.PI / 2;
        cam.beta = Math.PI / 3.2;
        cam.radius = cameraDistance * 0.38;
        cam.target.set(bounds.cx - bounds.w * 0.02, 1.2, bounds.cz + bounds.d * 0.38);
      }
    },
    [bounds, cameraDistance, wallHeightMeters],
  );

  const applyFloorMaterial = useCallback((styleKey) => {
    const scene = sceneRef.current;
    if (!scene || !floorMeshRef.current) return;
    const newMat = createFloorMaterial(styleKey, scene);
    floorMeshRef.current.material = newMat;
    setFloorStyle(styleKey);
    const mObj = FLOOR_MATERIALS.find((m) => m.id === styleKey);
    showToast(`✨ Applied ${mObj?.name || styleKey} to Floor`);
  }, [showToast]);

  const applyWallMaterial = useCallback((styleKey) => {
    const scene = sceneRef.current;
    if (!scene || !wallMeshesRef.current) return;
    const newMat = createWallMaterial(styleKey, scene);
    wallMeshesRef.current.forEach((m) => {
      if (m && !m.isDisposed()) m.material = newMat;
    });
    setWallStyle(styleKey);
    const wObj = WALL_FINISHES.find((w) => w.id === styleKey);
    showToast(`✨ Applied ${wObj?.name || styleKey} to Walls`);
  }, [showToast]);

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
        showToast(`Added ${name} to scene`);
      }
    },
    [loadSingleModel, showToast],
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
      showToast(`✨ Automatically furnished ${added.length} items`);
    }
    setIsAutoFurnishing(false);
  }, [furniture, isAutoFurnishing, pixelsPerMeter, loadSingleModel, showToast]);

  const clearAllModels = useCallback(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    for (const [mesh, mapping] of meshToModelRef.current.entries()) {
      mesh.dispose();
      mapping.root.dispose();
    }
    meshToModelRef.current.clear();
    setPlacedModels([]);
    showToast("Cleared all furniture");
  }, [showToast]);

  const removePlacedModel = useCallback((modelEntry) => {
    if (!modelEntry || !modelEntry.mesh) return;
    for (const [mesh, mapping] of meshToModelRef.current.entries()) {
      if (mapping.entry === modelEntry || mapping.root === modelEntry.mesh) {
        meshToModelRef.current.delete(mesh);
      }
    }
    modelEntry.mesh.dispose();
    setPlacedModels((prev) => prev.filter((m) => m !== modelEntry));
    showToast(`Removed ${modelEntry.name}`);
  }, [showToast]);

  const handleExportScene = useCallback(() => {
    try {
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
      showToast("💾 Scene JSON exported successfully");
    } catch (err) {
      alert(`Export failed: ${err.message}`);
    }
  }, [
    cameraDistance,
    placedModels,
    wallHeightMeters,
    wallThicknessMeters,
    sillHeightMeters,
    windowHeightMeters,
    showToast,
  ]);

  const handleImportScene = useCallback(
    async (file) => {
      try {
        const sceneData = await importSceneJSON(file);
        clearAllModels();
        for (const m of sceneData.placedModels) {
          const worldX = m.position[0] + bounds.cx;
          const worldZ = m.position[2] + bounds.cz;
          const angle = Array.isArray(m.rotation) ? m.rotation[1] : 0;
          const entry = await loadSingleModel(m.url, new Vector3(worldX, 0, worldZ), angle, m.name);
          if (entry) {
            setPlacedModels((prev) => [...prev, entry]);
          }
        }
        showToast("📂 Scene loaded successfully");
      } catch (err) {
        alert(`Import failed: ${err.message}`);
      }
    },
    [bounds, clearAllModels, loadSingleModel, showToast],
  );

  // Keyboard navigation & model manipulation
  useEffect(() => {
    const onKeyDown = (e) => {
      const tag = document.activeElement?.tagName?.toLowerCase();
      if (tag === "input" || tag === "textarea") return;

      keysRef.current.add(e.key.toLowerCase());

      if (e.key === "r" || e.key === "R") {
        const root = selectedMeshRef.current || draggedMeshRef.current;
        if (root) {
          e.preventDefault();
          const currentYaw = yawAngleRef.current.get(root) ?? 0;
          const newYaw = currentYaw + Math.PI / 4;
          yawAngleRef.current.set(root, newYaw);
          root.rotation.y = newYaw;

          const mapping = [...meshToModelRef.current.values()].find((m) => m.root === root);
          if (mapping) {
            mapping.entry.rotation = [0, newYaw, 0];
          }
        }
      }

      if (e.key === "Delete" || e.key === "Backspace") {
        const root = selectedMeshRef.current;
        if (root) {
          e.preventDefault();
          const mapping = [...meshToModelRef.current.values()].find((m) => m.root === root);
          if (mapping) {
            removePlacedModel(mapping.entry);
            selectedMeshRef.current = null;
          }
        }
      }
    };

    const onKeyUp = (e) => {
      keysRef.current.delete(e.key.toLowerCase());
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [removePlacedModel]);

  // Main 3D Babylon.js Scene Initialization (Always full walls at 2.7m ceiling height)
  useEffect(() => {
    if (!canvasRef.current) return;

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
    scene.clearColor = new Color4(0.08, 0.11, 0.18, 1);

    const camera = new ArcRotateCamera(
      "camera",
      -Math.PI / 4,
      Math.PI / 3.4,
      cameraDistance * 1.1,
      new Vector3(bounds.cx, wallHeightMeters / 2, bounds.cz),
      scene,
    );
    cameraRef.current = camera;
    camera.attachControl(canvasRef.current, true);
    camera.lowerRadiusLimit = 0.1; // Allows true human eye-level walkthrough
    camera.upperRadiusLimit = 150;
    camera.lowerBetaLimit = 0.04;
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

    // 5. Build Wall Geometry (Full 2.7m walls with gap-free overlapping corners)
    wallMeshesRef.current = [];
    const wallOpts = {
      thickness: wallThicknessMeters,
      height: wallHeightMeters,
      scale: 1 / pixelsPerMeter,
    };
    const wallMeshes = buildWallMeshes(edges, wallOpts, scene);

    const wallMat = createWallMaterial(wallStyle, scene);
    wallMeshes.forEach((mesh) => {
      mesh.material = wallMat;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });

    // 6. CSG Window and Door Cutouts
    const cutWithWindows =
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

    const cutWallMeshes =
      doors && doors.length > 0
        ? cutDoorOpenings(
            doors,
            cutWithWindows,
            {
              scale: 1 / pixelsPerMeter,
              wallThickness: wallThicknessMeters,
              doorHeight: Math.min(2.1, wallHeightMeters),
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
  ]);

  // Drag and drop onto 3D Canvas
  const handleCanvasDragOver = (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
  };

  const handleCanvasDrop = (e) => {
    e.preventDefault();
    const dataStr = e.dataTransfer.getData("application/json");
    if (!dataStr) return;
    try {
      const data = JSON.parse(dataStr);
      const scene = sceneRef.current;
      const canvas = canvasRef.current;
      if (!scene || !canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const hit = scene.pick(x, y);

      if (data.type === "floor") {
        applyFloorMaterial(data.id);
      } else if (data.type === "wall") {
        applyWallMaterial(data.id);
      } else if (hit.hit && hit.pickedMesh) {
        if (hit.pickedMesh === floorMeshRef.current) {
          applyFloorMaterial(data.id);
        } else {
          applyWallMaterial(data.id);
        }
      }
    } catch (err) {
      console.warn("Drag drop parse error", err);
    }
  };

  useEffect(() => {
    const handleResize = () => {
      engineRef.current?.resize();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const btnStyle = (active) => ({
    background: active ? "#0284c7" : "rgba(255,255,255,0.12)",
    color: "#fff",
    border: "none",
    borderRadius: 4,
    padding: "4px 8px",
    cursor: "pointer",
    fontSize: 11,
    fontWeight: active ? 600 : 400,
    boxShadow: active ? "0 2px 6px rgba(2, 132, 199, 0.4)" : "none",
    transition: "all 0.15s ease",
  });

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: isFullscreen ? "100vh" : 580,
        display: "flex",
        background: "#080b12",
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
          onDragOver={handleCanvasDragOver}
          onDrop={handleCanvasDrop}
          style={{ width: "100%", height: "100%", border: "none", outline: "none", display: "block" }}
        />

        {/* Top Header Bar: Unified Flex Container (Zero Overlap!) */}
        <div
          style={{
            position: "absolute",
            top: 10,
            left: 10,
            right: 10,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 8,
            pointerEvents: "none",
            zIndex: 20,
          }}
        >
          {/* Left Pill: Camera Modes & Room Focus Presets */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              flexWrap: "wrap",
              background: "rgba(15, 23, 42, 0.88)",
              backdropFilter: "blur(12px)",
              padding: "5px 10px",
              borderRadius: 8,
              border: "1px solid rgba(255,255,255,0.15)",
              boxShadow: "0 4px 14px rgba(0,0,0,0.35)",
              pointerEvents: "auto",
            }}
          >
            <span style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginRight: 1 }}>
              View:
            </span>
            <button onClick={() => setViewpoint("iso")} style={btnStyle(viewpoint === "iso")}>
              🏠 3D Iso
            </button>
            <button onClick={() => setViewpoint("top")} style={btnStyle(viewpoint === "top")}>
              📐 Top Plan
            </button>
            <button onClick={() => setViewpoint("walk")} style={btnStyle(viewpoint === "walk")}>
              🚶 Walkthrough
            </button>

            <span style={{ width: 1, height: 16, background: "rgba(255,255,255,0.2)", margin: "0 4px" }} />

            <span style={{ fontSize: 11, fontWeight: 600, color: "rgba(255,255,255,0.5)", marginRight: 1 }}>
              Rooms:
            </span>
            <button onClick={() => setViewpoint("living")} style={btnStyle(viewpoint === "living")}>
              🛋️ Living
            </button>
            <button onClick={() => setViewpoint("kitchen")} style={btnStyle(viewpoint === "kitchen")}>
              🍳 Kitchen
            </button>
            <button onClick={() => setViewpoint("master_bed")} style={btnStyle(viewpoint === "master_bed")}>
              🛏️ Master Bed
            </button>
            <button onClick={() => setViewpoint("bed_2")} style={btnStyle(viewpoint === "bed_2")}>
              🛏️ Bed 2
            </button>
            <button onClick={() => setViewpoint("bath")} style={btnStyle(viewpoint === "bath")}>
              🚿 Bath
            </button>
            <button onClick={() => setViewpoint("entry")} style={btnStyle(viewpoint === "entry")}>
              🚪 Entry
            </button>
          </div>

          {/* Right Pill: Fullscreen Button (Isolated with zero overlap) */}
          <div style={{ pointerEvents: "auto", flexShrink: 0 }}>
            <button
              onClick={toggleFullscreen}
              title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
              style={{
                background: "rgba(15, 23, 42, 0.88)",
                backdropFilter: "blur(12px)",
                color: "#fff",
                border: "1px solid rgba(255,255,255,0.2)",
                borderRadius: 8,
                padding: "6px 12px",
                fontSize: 11,
                fontWeight: 600,
                cursor: "pointer",
                boxShadow: "0 4px 14px rgba(0,0,0,0.35)",
                display: "flex",
                alignItems: "center",
                gap: 5,
              }}
            >
              {isFullscreen ? "⤓ Exit Fullscreen" : "⤢ Fullscreen"}
            </button>
          </div>
        </div>

        {/* Floating Toast Notification */}
        {toastMessage && (
          <div
            style={{
              position: "absolute",
              top: 58,
              left: "50%",
              transform: "translateX(-50%)",
              background: "rgba(15, 23, 42, 0.92)",
              backdropFilter: "blur(8px)",
              color: "#38bdf8",
              border: "1px solid rgba(56, 189, 248, 0.4)",
              borderRadius: 20,
              padding: "6px 16px",
              fontSize: 12,
              fontWeight: 600,
              boxShadow: "0 4px 16px rgba(0,0,0,0.4)",
              zIndex: 30,
              pointerEvents: "none",
            }}
          >
            {toastMessage}
          </div>
        )}

        {/* Bottom Help Bar */}
        <div
          style={{
            position: "absolute",
            bottom: 10,
            left: 10,
            background: "rgba(15, 23, 42, 0.85)",
            backdropFilter: "blur(8px)",
            color: "rgba(255,255,255,0.9)",
            padding: "5px 12px",
            borderRadius: 6,
            fontSize: 11,
            border: "1px solid rgba(255,255,255,0.15)",
          }}
        >
          WASD to walk · Left-click model to drag · <strong>R</strong> to rotate · <strong>Del</strong> to remove · Drag materials onto scene
        </div>
      </div>

      {/* Right Side Panel: Furniture, Materials, and Scene Backups */}
      <RightSidePanel
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
        floorStyle={floorStyle}
        wallStyle={wallStyle}
        onApplyFloorMaterial={applyFloorMaterial}
        onApplyWallMaterial={applyWallMaterial}
        stats={stats}
      />
    </div>
  );
}
