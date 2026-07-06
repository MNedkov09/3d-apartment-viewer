import { Vector3, Mesh, MeshBuilder } from "@babylonjs/core";

/**
 * Offset a polyline into a wall "footprint" ribbon: two parallel lines
 * (left/right of the centerline) at +/- half wall thickness, joined into
 * one closed polygon.
 *
 * Normal direction at each point is computed from the AVERAGE of the
 * incoming and outgoing segment directions. This is a simple, cheap
 * approximation of a proper miter join — not perfect at very sharp
 * corners, but it doesn't blow up in length the way a true line-line
 * intersection miter can at near-180-degree bends, which matters more
 * for an MVP than corner perfection.
 *
 * Works identically for straight edges (2 points) and curved edges
 * (many points) — same math either way.
 *
 * @param {Array<[number, number]>} points - polyline in plan-space units (already scaled)
 * @param {number} thickness - wall thickness in the same units as points
 * @param {boolean} closed - whether this polyline is a closed loop
 * @returns {Array<[number, number]>} footprint polygon (left side forward, right side reversed)
 */

/**
 * Build one extruded wall mesh from a single traced edge.
 * Each segment of the edge becomes a box mesh.
 *
 * @param {{points: Array<[number,number]>, closed: boolean}} edge
 * @param {object} opts - { thickness, height, scale } all in consistent units
 * @param {BABYLON.Scene} scene
 * @returns {BABYLON.Mesh|null}
 */
export function edgeToWallMesh(edge, opts, scene) {
  const { thickness, height, scale } = opts;
  if (!edge.points || edge.points.length < 2) return null;

  const scaledPoints = edge.points.map(([x, y]) => [x * scale, y * scale]);

  const meshes = [];
  for (let i = 0; i < scaledPoints.length - 1; i++) {
    const [x1, y1] = scaledPoints[i];
    const [x2, y2] = scaledPoints[i + 1];
    const dx = x2 - x1;
    const dz = y2 - y1;
    const len = Math.hypot(dx, dz);
    if (len < 0.0001) continue;

    const midX = (x1 + x2) / 2;
    const midZ = (y1 + y2) / 2;

    // Box: width=thickness (along wall perpendicular), depth=len (along wall), height=height
    const box = MeshBuilder.CreateBox(`wall_seg_${i}`, {
      width: thickness,
      depth: len,
      height: height,
      scene,
    });
    box.position = new Vector3(midX, height / 2, midZ);
    box.rotation.y = Math.atan2(dx, dz);
    box.castShadow = true;
    box.receiveShadow = true;
    meshes.push(box);
  }

  return meshes.length > 0 ? meshes : null;
}

/**
 * Build wall meshes for an entire wall graph (all edges from the
 * skeleton-tracing step).
 *
 * @param {Array<{points: Array<[number,number]>, closed: boolean}>} edges
 * @param {object} opts - { thickness, height, scale }
 * @param {BABYLON.Scene} scene
 * @returns {BABYLON.Mesh[]}
 */
export function buildWallMeshes(edges, opts, scene) {
  const allMeshes = [];
  for (const edge of edges) {
    const meshes = edgeToWallMesh(edge, opts, scene);
    if (meshes) allMeshes.push(...meshes);
  }
  return allMeshes;
}
