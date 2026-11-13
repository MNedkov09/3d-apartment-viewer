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
 * Parse raw skeleton edges, drop false architectural artifacts, cluster orthogonal
 * axes, merge collinear segments across T-junctions, and snap L-corners / T-junctions.
 */
function extractMergedWallSegments(edges, scale) {
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
        const z = (z1 + z2) / 2;
        const minX = Math.min(x1, x2);
        const maxX = Math.max(x1, x2);
        rawH.push({ z, minX, maxX });
      } else if (Math.abs(dx) <= Math.max(0.14, Math.abs(dz) * 0.15)) {
        const x = (x1 + x2) / 2;
        const minZ = Math.min(z1, z2);
        const maxZ = Math.max(z1, z2);
        rawV.push({ x, minZ, maxZ });
      } else {
        rawDiag.push({ x1, z1, x2, z2 });
      }
    }
  }

  const axisTol = Math.max(0.14, 5.5 * scale);
  const bridgeGap = Math.max(0.22, 8.5 * scale);
  const cornerTol = Math.max(0.32, 14.0 * scale);

  const snapZ = buildAxisSnapper(rawH.map((s) => s.z), axisTol);
  const snapX = buildAxisSnapper(rawV.map((s) => s.x), axisTol);

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
      if (list[i].minX <= cur.maxX + bridgeGap) {
        cur.maxX = Math.max(cur.maxX, list[i].maxX);
      } else {
        mergedH.push(cur);
        cur = { z, minX: list[i].minX, maxX: list[i].maxX };
      }
    }
    mergedH.push(cur);
  }

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
      if (list[i].minZ <= cur.maxZ + bridgeGap) {
        cur.maxZ = Math.max(cur.maxZ, list[i].maxZ);
      } else {
        mergedV.push(cur);
        cur = { x, minZ: list[i].minZ, maxZ: list[i].maxZ };
      }
    }
    mergedV.push(cur);
  }

  // Simultaneously snap endpoints where H and V walls meet at L-corners or T-junctions
  for (let pass = 0; pass < 2; pass++) {
    for (const h of mergedH) {
      for (const v of mergedV) {
        const nearX = v.x >= h.minX - cornerTol && v.x <= h.maxX + cornerTol;
        const nearZ = h.z >= v.minZ - cornerTol && h.z <= v.maxZ + cornerTol;
        if (nearX && nearZ) {
          if (Math.abs(h.minX - v.x) <= cornerTol) h.minX = v.x;
          if (Math.abs(h.maxX - v.x) <= cornerTol) h.maxX = v.x;
          if (Math.abs(v.minZ - h.z) <= cornerTol) v.minZ = h.z;
          if (Math.abs(v.maxZ - h.z) <= cornerTol) v.maxZ = h.z;
        }
      }
    }
  }

  return { mergedH, mergedV, rawDiag };
}

/**
 * Build clean, merged, non-overlapping wall meshes for the entire wall graph.
 * Fuses collinear segments across T-junctions and offsets vertical walls by 1mm
 * so no two wall faces are ever coplanar (eliminating Z-fighting 100%).
 */
export function buildWallMeshes(edges, opts, scene) {
  const { thickness, height, scale } = opts;
  if (!edges || edges.length === 0) return [];

  const { mergedH, mergedV, rawDiag } = extractMergedWallSegments(edges, scale);

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

function subtractIntervals(minVal, maxVal, cuts) {
  if (!cuts || cuts.length === 0) return [[minVal, maxVal]];
  const sorted = [...cuts].sort((a, b) => a[0] - b[0]);
  const res = [];
  let cursor = minVal;
  for (const [cMin, cMax] of sorted) {
    if (cMax <= cursor) continue;
    if (cMin >= maxVal) break;
    if (cMin > cursor + 0.02) {
      res.push([cursor, Math.min(cMin, maxVal)]);
    }
    cursor = Math.max(cursor, cMax);
  }
  if (cursor < maxVal - 0.02) {
    res.push([cursor, maxVal]);
  }
  return res;
}

/**
 * Build 2D XZ solid wall collider boxes matching the final walls with door openings subtracted.
 */
export function buildWallColliders(edges, doors, opts) {
  const { thickness, scale } = opts;
  if (!edges || edges.length === 0) return [];

  const { mergedH, mergedV, rawDiag } = extractMergedWallSegments(edges, scale);
  const halfT = thickness / 2;

  const doorBoxes = (doors || [])
    .filter((d) => d.start && d.end)
    .map((d) => {
      const x1 = d.start[0] * scale;
      const z1 = d.start[1] * scale;
      const x2 = d.end[0] * scale;
      const z2 = d.end[1] * scale;
      const dx = x2 - x1;
      const dz = z2 - z1;
      const len = Math.hypot(dx, dz);
      if (len < 0.05) return null;
      const cx = (x1 + x2) / 2;
      const cz = (z1 + z2) / 2;
      const rotY = Math.atan2(dx, dz);
      const cw = thickness * 3.5;
      const cd = len + 0.22;
      const hx = (Math.abs(cw * Math.cos(rotY)) + Math.abs(cd * Math.sin(rotY))) / 2;
      const hz = (Math.abs(cw * Math.sin(rotY)) + Math.abs(cd * Math.cos(rotY))) / 2;
      return { minX: cx - hx, maxX: cx + hx, minZ: cz - hz, maxZ: cz + hz };
    })
    .filter(Boolean);

  const colliders = [];

  for (const h of mergedH) {
    if (h.maxX - h.minX < 0.05) continue;
    const minZ = h.z - halfT;
    const maxZ = h.z + halfT;
    const cuts = doorBoxes
      .filter((d) => d.maxZ >= minZ && d.minZ <= maxZ)
      .map((d) => [d.minX, d.maxX]);
    for (const [sx, ex] of subtractIntervals(h.minX - halfT, h.maxX + halfT, cuts)) {
      colliders.push({ minX: sx, maxX: ex, minZ, maxZ });
    }
  }

  for (const v of mergedV) {
    if (v.maxZ - v.minZ < 0.05) continue;
    const minX = v.x - halfT;
    const maxX = v.x + halfT;
    const cuts = doorBoxes
      .filter((d) => d.maxX >= minX && d.minX <= maxX)
      .map((d) => [d.minZ, d.maxZ]);
    for (const [sz, ez] of subtractIntervals(v.minZ - halfT, v.maxZ + halfT, cuts)) {
      colliders.push({ minX, maxX, minZ: sz, maxZ: ez });
    }
  }

  for (const d of rawDiag) {
    const len = Math.hypot(d.x2 - d.x1, d.z2 - d.z1);
    if (len < 0.05) continue;
    const steps = Math.max(2, Math.ceil(len / (thickness * 0.8)));
    for (let s = 0; s < steps; s++) {
      const t = (s + 0.5) / steps;
      const px = d.x1 + (d.x2 - d.x1) * t;
      const pz = d.z1 + (d.z2 - d.z1) * t;
      colliders.push({ minX: px - halfT, maxX: px + halfT, minZ: pz - halfT, maxZ: pz + halfT });
    }
  }

  return colliders;
}

/**
 * Compute world-space XZ footprint bounds for a placed furniture model at (rootX, rootZ) rotated by yaw.
 */
function getRotatedFootprintBounds(rootX, rootZ, footprint, yaw) {
  const { cx = 0, cz = 0, halfX = 0.25, halfZ = 0.25 } = footprint || {};
  const cosY = Math.cos(yaw || 0);
  const sinY = Math.sin(yaw || 0);
  // Babylon left-handed Y rotation: x' = x*cos(y) + z*sin(y), z' = -x*sin(y) + z*cos(y)
  const worldCenterX = rootX + cx * cosY + cz * sinY;
  const worldCenterZ = rootZ - cx * sinY + cz * cosY;
  const effHalfX = Math.abs(halfX * cosY) + Math.abs(halfZ * sinY);
  const effHalfZ = Math.abs(halfX * sinY) + Math.abs(halfZ * cosY);
  return {
    offsetX: worldCenterX - rootX,
    offsetZ: worldCenterZ - rootZ,
    effHalfX,
    effHalfZ,
    minX: worldCenterX - effHalfX,
    maxX: worldCenterX + effHalfX,
    minZ: worldCenterZ - effHalfZ,
    maxZ: worldCenterZ + effHalfZ,
  };
}

/**
 * Return true if the rotated model footprint at (x, z) overlaps any wall collider.
 */
export function hasWallCollision(x, z, footprint, yaw, colliders, clearance = 0.005) {
  if (!colliders || colliders.length === 0) return false;
  const b = getRotatedFootprintBounds(x, z, footprint, yaw);
  const minX = b.minX;
  const maxX = b.maxX;
  const minZ = b.minZ;
  const maxZ = b.maxZ;
  for (const wall of colliders) {
    if (
      maxX > wall.minX - clearance &&
      minX < wall.maxX + clearance &&
      maxZ > wall.minZ - clearance &&
      minZ < wall.maxZ + clearance
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Push a furniture root position (x, z) cleanly outside any colliding wall colliders
 * and keep it inside the apartment perimeter bounds.
 */
export function pushOutOfWalls(x, z, footprint, yaw, colliders, roomBounds, clearance = 0.015) {
  let rx = x;
  let rz = z;
  const b = getRotatedFootprintBounds(rx, rz, footprint, yaw);

  for (let pass = 0; pass < 8; pass++) {
    let moved = false;
    if (colliders && colliders.length > 0) {
      for (const wall of colliders) {
        const wMinX = wall.minX - clearance;
        const wMaxX = wall.maxX + clearance;
        const wMinZ = wall.minZ - clearance;
        const wMaxZ = wall.maxZ + clearance;

        const curMinX = rx + b.offsetX - b.effHalfX;
        const curMaxX = rx + b.offsetX + b.effHalfX;
        const curMinZ = rz + b.offsetZ - b.effHalfZ;
        const curMaxZ = rz + b.offsetZ + b.effHalfZ;

        if (curMaxX > wMinX && curMinX < wMaxX && curMaxZ > wMinZ && curMinZ < wMaxZ) {
          const pushLeft = curMaxX - wMinX;
          const pushRight = wMaxX - curMinX;
          const pushBack = curMaxZ - wMinZ;
          const pushForward = wMaxZ - curMinZ;
          const minXMove = pushLeft < pushRight ? -pushLeft : pushRight;
          const minZMove = pushBack < pushForward ? -pushBack : pushForward;

          if (Math.abs(minXMove) <= Math.abs(minZMove)) {
            rx += minXMove;
          } else {
            rz += minZMove;
          }
          moved = true;
        }
      }
    }

    if (roomBounds) {
      const clampedMinX = roomBounds.minX + b.effHalfX - b.offsetX;
      const clampedMaxX = roomBounds.maxX - b.effHalfX - b.offsetX;
      const clampedMinZ = roomBounds.minZ + b.effHalfZ - b.offsetZ;
      const clampedMaxZ = roomBounds.maxZ - b.effHalfZ - b.offsetZ;
      if (clampedMinX <= clampedMaxX) rx = Math.max(clampedMinX, Math.min(clampedMaxX, rx));
      if (clampedMinZ <= clampedMaxZ) rz = Math.max(clampedMinZ, Math.min(clampedMaxZ, rz));
    }

    if (!moved) break;
  }

  return { x: rx, z: rz };
}

/**
 * Move furniture smoothly from (curX, curZ) toward (targetX, targetZ) with substepping
 * so models slide cleanly along walls and never tunnel across thin walls.
 */
export function moveWithWallCollision(curX, curZ, targetX, targetZ, footprint, yaw, colliders, roomBounds) {
  const dx = targetX - curX;
  const dz = targetZ - curZ;
  const dist = Math.hypot(dx, dz);
  if (dist < 1e-5) return { x: curX, z: curZ };

  const maxStep = 0.04;
  const steps = Math.max(1, Math.min(60, Math.ceil(dist / maxStep)));
  let rx = curX;
  let rz = curZ;

  for (let s = 0; s < steps; s++) {
    const nextX = rx + dx / steps;
    const resX = pushOutOfWalls(nextX, rz, footprint, yaw, colliders, roomBounds);
    rx = resX.x;
    rz = resX.z;

    const nextZ = rz + dz / steps;
    const resZ = pushOutOfWalls(rx, nextZ, footprint, yaw, colliders, roomBounds);
    rx = resZ.x;
    rz = resZ.z;
  }

  return { x: rx, z: rz };
}
