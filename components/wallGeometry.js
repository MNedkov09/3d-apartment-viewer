import { Vector3, MeshBuilder } from "@babylonjs/core";

/**
 * Build one extruded wall mesh from a single traced edge.
 * Each segment of the edge becomes a box mesh.
 */
export function edgeToWallMesh(edge, opts, scene) {
  const { thickness, height, scale } = opts;
  if (!edge.points || edge.points.length < 2) return [];

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

  return meshes;
}

/**
 * Build wall meshes for an entire wall graph (all edges from the
 * skeleton-tracing step).
 */
export function buildWallMeshes(edges, opts, scene) {
  const allMeshes = [];
  for (const edge of edges) {
    allMeshes.push(...edgeToWallMesh(edge, opts, scene));
  }
  return allMeshes;
}
