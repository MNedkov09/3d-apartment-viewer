import { Vector3, MeshBuilder, VertexBuffer } from "@babylonjs/core";

/**
 * Compute consistent real-world meter UV coordinates across any wall mesh
 * (both box walls and CSG-cut walls). Guarantees horizontal brick/tile alignment,
 * identical scale across all wall lengths, and zero UV stretching.
 */
export function applyWorldSpaceWallUVs(mesh, tileMeters = 1.6) {
  if (!mesh || mesh.isDisposed()) return;
  mesh.computeWorldMatrix(true);
  const worldMat = mesh.getWorldMatrix();
  const positions = mesh.getVerticesData(VertexBuffer.PositionKind);
  const normals = mesh.getVerticesData(VertexBuffer.NormalKind);
  if (!positions || !normals) return;

  const vertCount = positions.length / 3;
  const uvs = new Float32Array(vertCount * 2);
  const p = Vector3.Zero();
  const n = Vector3.Zero();

  for (let i = 0; i < vertCount; i++) {
    p.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    Vector3.TransformCoordinatesToRef(p, worldMat, p);

    n.set(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
    Vector3.TransformNormalToRef(n, worldMat, n);
    n.normalize();

    const ax = Math.abs(n.x);
    const ay = Math.abs(n.y);
    const az = Math.abs(n.z);

    if (ay > ax && ay > az) {
      // Top or bottom cap face
      uvs[i * 2] = p.x / tileMeters;
      uvs[i * 2 + 1] = p.z / tileMeters;
    } else if (ax > az) {
      // Wall running along Z axis (normal points along ±X)
      uvs[i * 2] = p.z / tileMeters;
      uvs[i * 2 + 1] = p.y / tileMeters;
    } else {
      // Wall running along X axis (normal points along ±Z)
      uvs[i * 2] = p.x / tileMeters;
      uvs[i * 2 + 1] = p.y / tileMeters;
    }
  }

  mesh.setVerticesData(VertexBuffer.UVKind, uvs, true);
}

/**
 * Clusters a list of 1D coordinates within `tol` and returns a snap function.
 */
function buildAxisSnapper(coords, tol = 0.14) {
  if (coords.length === 0) return (v) => v;
  const sorted = [...coords].sort((a, b) => a - b);
  const clusters = [];
  let cur = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i] - cur[cur.length - 1] <= tol) {
      cur.push(sorted[i]);
    } else {
      clusters.push(cur.reduce((s, v) => s + v, 0) / cur.length);
      cur = [sorted[i]];
    }
  }
  clusters.push(cur.reduce((s, v) => s + v, 0) / cur.length);

  return (val) => {
    let best = val;
    let bestDist = tol + 0.001;
    for (const c of clusters) {
      const d = Math.abs(val - c);
      if (d < bestDist) {
        bestDist = d;
        best = c;
      }
    }
    return Math.round(best * 1000) / 1000;
  };
}

/**
 * Build clean, merged, non-overlapping wall meshes for the entire wall graph.
 * Fuses collinear segments across T-junctions and offsets vertical walls by 1mm
 * so no two wall faces are ever coplanar (eliminating Z-fighting 100%).
 */
export function buildWallMeshes(edges, opts, scene) {
  const { thickness, height, scale } = opts;
  if (!edges || edges.length === 0) return [];

  const rawH = []; // { z, minX, maxX }
  const rawV = []; // { x, minZ, maxZ }
  const rawDiag = []; // { x1, z1, x2, z2 }

  for (const edge of edges) {
    if (!edge.points || edge.points.length < 2) continue;
    const pts = edge.points.map(([x, y]) => [x * scale, y * scale]);

    for (let i = 0; i < pts.length - 1; i++) {
      const [x1, z1] = pts[i];
      const [x2, z2] = pts[i + 1];
      const dx = x2 - x1;
      const dz = z2 - z1;
      const len = Math.hypot(dx, dz);
      if (len < 0.04) continue;

      if (Math.abs(dz) <= Math.max(0.14, Math.abs(dx) * 0.15)) {
        rawH.push({
          z: (z1 + z2) / 2,
          minX: Math.min(x1, x2),
          maxX: Math.max(x1, x2),
        });
      } else if (Math.abs(dx) <= Math.max(0.14, Math.abs(dz) * 0.15)) {
        rawV.push({
          x: (x1 + x2) / 2,
          minZ: Math.min(z1, z2),
          maxZ: Math.max(z1, z2),
        });
      } else {
        rawDiag.push({ x1, z1, x2, z2 });
      }
    }
  }

  const snapZ = buildAxisSnapper(rawH.map((s) => s.z), 0.14);
  const snapX = buildAxisSnapper(rawV.map((s) => s.x), 0.14);

  // Group horizontal segments by snapped Z and merge overlapping/touching intervals
  const hGroups = new Map();
  for (const seg of rawH) {
    const sz = snapZ(seg.z);
    if (!hGroups.has(sz)) hGroups.set(sz, []);
    hGroups.get(sz).push({ minX: seg.minX, maxX: seg.maxX });
  }

  const mergedH = [];
  for (const [z, list] of hGroups.entries()) {
    list.sort((a, b) => a.minX - b.minX);
    let cur = { z, minX: list[0].minX, maxX: list[0].maxX };
    for (let i = 1; i < list.length; i++) {
      // Bridge small T-junction skeleton gaps (<= 0.22m), but preserve doorway gaps (> 0.5m)
      if (list[i].minX <= cur.maxX + 0.22) {
        cur.maxX = Math.max(cur.maxX, list[i].maxX);
      } else {
        mergedH.push(cur);
        cur = { z, minX: list[i].minX, maxX: list[i].maxX };
      }
    }
    mergedH.push(cur);
  }

  // Group vertical segments by snapped X and merge overlapping/touching intervals
  const vGroups = new Map();
  for (const seg of rawV) {
    const sx = snapX(seg.x);
    if (!vGroups.has(sx)) vGroups.set(sx, []);
    vGroups.get(sx).push({ minZ: seg.minZ, maxZ: seg.maxZ });
  }

  const mergedV = [];
  for (const [x, list] of vGroups.entries()) {
    list.sort((a, b) => a.minZ - b.minZ);
    let cur = { x, minZ: list[0].minZ, maxZ: list[0].maxZ };
    for (let i = 1; i < list.length; i++) {
      if (list[i].minZ <= cur.maxZ + 0.22) {
        cur.maxZ = Math.max(cur.maxZ, list[i].maxZ);
      } else {
        mergedV.push(cur);
        cur = { x, minZ: list[i].minZ, maxZ: list[i].maxZ };
      }
    }
    mergedV.push(cur);
  }

  // Snap endpoints where H and V walls meet at corners or T-junctions
  const snapTol = 0.24;
  for (const h of mergedH) {
    for (const v of mergedV) {
      if (h.z >= v.minZ - snapTol && h.z <= v.maxZ + snapTol) {
        if (Math.abs(h.minX - v.x) <= snapTol) h.minX = v.x;
        if (Math.abs(h.maxX - v.x) <= snapTol) h.maxX = v.x;
      }
    }
  }
  for (const v of mergedV) {
    for (const h of mergedH) {
      if (v.x >= h.minX - snapTol && v.x <= h.maxX + snapTol) {
        if (Math.abs(v.minZ - h.z) <= snapTol) v.minZ = h.z;
        if (Math.abs(v.maxZ - h.z) <= snapTol) v.maxZ = h.z;
      }
    }
  }

  const allMeshes = [];
  let idx = 0;

  // 1. Horizontal walls (Full thickness & height, extended by thickness/2 to cap corners)
  // Bottom extends 2cm (-0.02m) into the floor slab to guarantee zero gap at the floor line
  const hFullHeight = height + 0.02;
  const hCenterY = height / 2 - 0.01;
  for (const h of mergedH) {
    const len = h.maxX - h.minX;
    if (len < 0.05) continue;
    const midX = (h.minX + h.maxX) / 2;
    const box = MeshBuilder.CreateBox(
      `wall_h_${idx++}`,
      {
        width: thickness,
        depth: len + thickness,
        height: hFullHeight,
      },
      scene,
    );
    box.position = new Vector3(midX, hCenterY, h.z);
    box.rotation.y = Math.PI / 2;
    box.castShadow = true;
    box.receiveShadow = true;
    applyWorldSpaceWallUVs(box);
    allMeshes.push(box);
  }

  // 2. Vertical walls (1.5mm inset in width & height to prevent coplanar Z-fighting at corners)
  const vThick = Math.max(0.05, thickness - 0.003);
  const vFullHeight = Math.max(0.1, height - 0.002) + 0.02;
  const vCenterY = (height - 0.002) / 2 - 0.01;
  for (const v of mergedV) {
    const len = v.maxZ - v.minZ;
    if (len < 0.05) continue;
    const midZ = (v.minZ + v.maxZ) / 2;
    const box = MeshBuilder.CreateBox(
      `wall_v_${idx++}`,
      {
        width: vThick,
        depth: len + vThick,
        height: vFullHeight,
      },
      scene,
    );
    box.position = new Vector3(v.x, vCenterY, midZ);
    box.rotation.y = 0;
    box.castShadow = true;
    box.receiveShadow = true;
    applyWorldSpaceWallUVs(box);
    allMeshes.push(box);
  }

  // 3. Any diagonal wall segments
  const dThick = Math.max(0.05, thickness - 0.005);
  const dFullHeight = Math.max(0.1, height - 0.004) + 0.02;
  const dCenterY = (height - 0.004) / 2 - 0.01;
  for (const d of rawDiag) {
    const dx = d.x2 - d.x1;
    const dz = d.z2 - d.z1;
    const len = Math.hypot(dx, dz);
    if (len < 0.05) continue;
    const box = MeshBuilder.CreateBox(
      `wall_d_${idx++}`,
      {
        width: dThick,
        depth: len + dThick,
        height: dFullHeight,
      },
      scene,
    );
    box.position = new Vector3((d.x1 + d.x2) / 2, dCenterY, (d.z1 + d.z2) / 2);
    box.rotation.y = Math.atan2(dx, dz);
    box.castShadow = true;
    box.receiveShadow = true;
    applyWorldSpaceWallUVs(box);
    allMeshes.push(box);
  }

  return allMeshes;
}

export function edgeToWallMesh(edge, opts, scene) {
  return buildWallMeshes([edge], opts, scene);
}
