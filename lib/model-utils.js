import crypto from "crypto";
import path from "path";

// 3D model formats supported by the viewer.
export const MODEL_EXTENSIONS = new Set([".glb", ".gltf"]);

const MAX_MODEL_SIZE = 50 * 1024 * 1024; // 50 MB for 3D models (larger than blueprints)

/**
 * Validate a model file's extension and magic bytes.
 * Returns { ok: true } on success, or { ok: false, error: string } on failure.
 */
export function validateModelFile(fileBuffer, filename) {
  const ext = path.extname(filename).toLowerCase();

  if (!MODEL_EXTENSIONS.has(ext)) {
    return { ok: false, error: `Unsupported model format: ${ext}. Use .glb or .gltf.` };
  }

  // Check file size
  if (fileBuffer.length > MAX_MODEL_SIZE) {
    return { ok: false, error: "Model file is larger than 50 MB." };
  }

  // Validate magic bytes based on format
  if (ext === ".glb") {
    const header = fileBuffer.toString("latin1", 0, 4);
    if (header !== "glTF") {
      return { ok: false, error: "Invalid GLB file: missing glTF magic bytes." };
    }
    // Validate version field (bytes 4-7 should be 2 for GLB 2.0)
    const version = fileBuffer.readUInt32LE(4);
    if (version !== 2) {
      return { ok: false, error: `Unsupported GLB version: ${version}. Only version 2 is supported.` };
    }
  } else if (ext === ".gltf") {
    const text = fileBuffer.toString("latin1");
    // glTF files start with whitespace then a JSON object
    const trimmed = text.trimStart();
    if (!trimmed.startsWith("{")) {
      return { ok: false, error: "Invalid GLTF file: must be valid JSON." };
    }
  }

  return { ok: true };
}

/**
 * Generate a safe filename for storing model files.
 * Format: {timestamp}-{uuid}{ext}
 */
export function generateModelFilename(originalName) {
  const ext = path.extname(originalName).toLowerCase();
  const timestamp = Date.now();
  const uuid = crypto.randomBytes(8).toString("hex");
  return `${timestamp}-${uuid}${ext}`;
}

/**
 * Extract polygon count from a GLB file's binary data.
 * Returns the total triangle count across all meshes, or 0 if extraction fails.
 */
export function extractPolygonCount(fileBuffer) {
  try {
    // GLB header: magic(4) + version(4) + length(4) + chunk1_len(4) + chunk1_type(4) = 20 bytes
    const headerSize = 20;
    if (fileBuffer.length < headerSize) return 0;

    // First chunk is JSON metadata
    const jsonChunkLength = fileBuffer.readUInt32LE(headerSize);
    const jsonChunkType = fileBuffer.readUInt32LE(headerSize + 4);

    // GLB chunk type 0x4E4F534A = "JSON"
    if (jsonChunkType !== 0x4e4f534a) return 0;

    const jsonStart = headerSize + 8;
    const jsonEnd = jsonStart + jsonChunkLength;
    const jsonString = fileBuffer.toString("utf8", jsonStart, jsonEnd);

    const metadata = JSON.parse(jsonString);

    // Navigate to meshes in the glTF structure: scene → scenes[0] → nodes → meshes
    if (!metadata.scenes || !Array.isArray(metadata.scenes)) return 0;

    // Resolve the default scene (metadata.scene is the index into metadata.scenes).
    const defaultSceneIndex = metadata.scene != null ? metadata.scene : 0;
    const sceneRoot = metadata.scenes?.[defaultSceneIndex];
    if (!sceneRoot || !Array.isArray(sceneRoot.nodes)) return 0;

    // Recursively traverse all nodes starting from `nodeIndices`, collecting mesh indices.
    function collectMeshes(nodeIndices) {
      const meshes = [];
      for (const nodeIndex of nodeIndices) {
        const node = metadata.nodes?.[nodeIndex];
        if (!node) continue;
        // Each mesh in glTF is a node with a `mesh` property (index into metadata.meshes).
        if (node.mesh != null) meshes.push(node.mesh);
        // Recurse into children.
        if (Array.isArray(node.children)) {
          meshes.push(...collectMeshes(node.children));
        }
      }
      return meshes;
    }

    const meshIndices = collectMeshes(sceneRoot.nodes);

    let totalTriangles = 0;

    for (const meshIndex of meshIndices) {
      const mesh = metadata.meshes?.[meshIndex];
      if (!mesh || !Array.isArray(mesh.primitives)) continue;

      for (const primitive of mesh.primitives) {
        // Each triangle is defined by 3 indices.
        // Use the `counts` field from glTF2 extensions, falling back to accessor.count / 3.
        const accessorIndex = primitive.indices;
        if (accessorIndex != null && metadata.accessors?.[accessorIndex]) {
          const accessor = metadata.accessors[accessorIndex];
          // Triangle count = number of indices / 3 (each triangle has 3 vertices)
          totalTriangles += Math.floor(accessor.count / 3);
        }
      }
    }

    return totalTriangles;
  } catch {
    // Return 0 on any parsing error
    return 0;
  }
}

/**
 * Generate a simple SVG thumbnail placeholder for a model.
 * Returns an SVG string that can be converted to PNG or served directly.
 */
export function generateModelThumbnailSvg(modelName, polygonCount) {
  const safeName = modelName.replace(/[^a-zA-Z0-9\s]/g, "").slice(0, 20);
  const displayName = safeName || "Model";

  // Calculate dimensions based on polygon count (log scale for visual variety)
  const width = 256;
  const height = 256;
  const boxSize = Math.max(40, Math.min(120, 40 + Math.log2(polygonCount || 1) * 10));

  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" fill="#f5f5f5"/>
  <rect x="${(width - boxSize) / 2}" y="${(height - boxSize) / 2}" width="${boxSize}" height="${boxSize}" fill="#e0e0e0" stroke="#bdbdbd" stroke-width="2"/>
  <text x="${width / 2}" y="${height - 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#757575">${displayName}</text>
  ${polygonCount > 0 ? `<text x="${width / 2}" y="${height - 10}" text-anchor="middle" font-family="monospace" font-size="10" fill="#9e9e9e">${polygonCount.toLocaleString()} triangles</text>` : ""}
</svg>`;
}

export default {
  validateModelFile,
  generateModelFilename,
  extractPolygonCount,
  generateModelThumbnailSvg,
};
