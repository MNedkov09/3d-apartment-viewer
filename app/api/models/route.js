import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import db from "@/lib/db";
import { getCurrentUser } from "@/lib/session";
import {
  validateModelFile,
  generateModelFilename,
  extractPolygonCount,
} from "@/lib/model-utils";

export const runtime = "nodejs";

const MODELS_DIR = path.join(process.cwd(), "public", "models", "uploaded");
const THUMBNAILS_DIR = path.join(
  process.cwd(),
  "public",
  "models",
  "thumbnails",
);
const MAX_SIZE = 50 * 1024 * 1024; // 50 MB

// GET /api/models -> list models (public read, filtered by user_id only if logged in)
export async function GET(request) {
  const user = await getCurrentUser();

  const { searchParams } = new URL(request.url);
  const category = searchParams.get("category");
  const search = searchParams.get("search");
  const page = parseInt(searchParams.get("page") || "1", 10);
  const limit = Math.min(parseInt(searchParams.get("limit") || "20", 10), 100);
  const offset = (page - 1) * limit;

  // Build WHERE clause dynamically.
  // Seeded models (source = 'seeded') always appear to everyone.
  // When logged in, also include the user's uploaded models (source != 'seeded').
  // When not logged in, only seeded models are visible.
  const whereClauses = [];
  const params = [];

  if (user) {
    whereClauses.push("(source = 'seeded' OR user_id = ?)");
    params.push(user.id);
  } else {
    whereClauses.push("source = 'seeded'");
  }

  if (category && category !== "all") {
    whereClauses.push("category = ?");
    params.push(category);
  }

  if (search) {
    whereClauses.push("(name LIKE ? OR description LIKE ?)");
    const searchPattern = `%${search}%`;
    params.push(searchPattern, searchPattern);
  }

  const whereClause = whereClauses.join(" AND ");

  // Get total count for pagination
  const countRow = db
    .prepare(`SELECT COUNT(*) as total FROM models WHERE ${whereClause}`)
    .get(...params);
  const total = countRow.total;
  const totalPages = Math.ceil(total / limit);

  // Get paginated results
  const rows = db
    .prepare(
      `SELECT id, name, description, category, tags, file_path, thumbnail_path, mime_type, size, polygon_count, created_at, source
         FROM models
        WHERE ${whereClause}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?`,
    )
    .all(...params, limit, offset);

  return NextResponse.json({
    models: rows,
    pagination: {
      page,
      limit,
      total,
      totalPages,
    },
  });
}

// POST /api/models -> upload a new 3D model
export async function POST(request) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json(
      { error: "Log in to upload a model." },
      { status: 401 },
    );
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: "Invalid upload." }, { status: 400 });
  }

  const file = form.get("file");
  const name = (form.get("name") || "").toString().trim();
  const description = (form.get("description") || "").toString().trim();
  const category =
    (form.get("category") || "uncategorized").toString().trim() ||
    "uncategorized";

  if (!file || typeof file === "string") {
    return NextResponse.json(
      { error: "No file was attached." },
      { status: 400 },
    );
  }
  if (file.size === 0) {
    return NextResponse.json({ error: "The file is empty." }, { status: 400 });
  }
  if (file.size > MAX_SIZE) {
    return NextResponse.json(
      { error: "File is larger than 50 MB." },
      { status: 413 },
    );
  }

  const ext = path.extname(file.name).toLowerCase();
  if (![".glb", ".gltf"].includes(ext)) {
    return NextResponse.json(
      { error: "Unsupported file type. Use .glb or .gltf." },
      { status: 415 },
    );
  }

  // Validate actual file content
  const buffer = Buffer.from(await file.arrayBuffer());
  const validation = validateModelFile(buffer, file.name);
  if (!validation.ok) {
    return NextResponse.json({ error: validation.error }, { status: 415 });
  }

  // Ensure directories exist
  if (!fs.existsSync(MODELS_DIR)) fs.mkdirSync(MODELS_DIR, { recursive: true });
  if (!fs.existsSync(THUMBNAILS_DIR))
    fs.mkdirSync(THUMBNAILS_DIR, { recursive: true });

  const safeName = generateModelFilename(file.name);
  const modelName = name || file.name.replace(/\.[^.]+$/, "");

  // Save the model file
  const modelDiskPath = path.join(MODELS_DIR, safeName);
  fs.writeFileSync(modelDiskPath, buffer);

  const publicModelPath = `/models/uploaded/${safeName}`;

  // Extract polygon count for GLB files
  let polygonCount = 0;
  if (ext === ".glb") {
    polygonCount = extractPolygonCount(buffer);
  }

  // Generate a simple SVG thumbnail placeholder
  const thumbnailSvg = generateThumbnailPlaceholder(modelName, polygonCount);
  const thumbnailFileName = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.svg`;
  const thumbnailDiskPath = path.join(THUMBNAILS_DIR, thumbnailFileName);
  fs.writeFileSync(thumbnailDiskPath, thumbnailSvg);

  const publicThumbnailPath = `/models/thumbnails/${thumbnailFileName}`;

  // Insert into database (uploaded models — source is left as default 'seeded')
  const info = db
    .prepare(
      `INSERT INTO models (user_id, name, description, category, file_path, thumbnail_path, mime_type, size, polygon_count, source)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'uploaded')`,
    )
    .run(
      user.id,
      modelName,
      description,
      category,
      publicModelPath,
      publicThumbnailPath,
      `model/${ext === ".glb" ? "gltf-binary" : "json"}`,
      file.size,
      polygonCount,
    );

  const model = db
    .prepare(
      `SELECT id, name, description, category, tags, file_path, thumbnail_path, mime_type, size, polygon_count, created_at
         FROM models WHERE id = ?`,
    )
    .get(info.lastInsertRowid);

  return NextResponse.json({ model }, { status: 201 });
}

// Helper to generate a simple SVG thumbnail placeholder
function generateThumbnailPlaceholder(name, polygonCount) {
  const safeName = name.replace(/[^a-zA-Z0-9\s]/g, "").slice(0, 20) || "Model";
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256">
  <rect width="256" height="256" fill="#f5f5f5"/>
  <rect x="88" y="88" width="80" height="80" fill="#e0e0e0" stroke="#bdbdbd" stroke-width="2"/>
  <text x="128" y="210" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#757575">${safeName}</text>
  ${polygonCount > 0 ? `<text x="128" y="230" text-anchor="middle" font-family="monospace" font-size="10" fill="#9e9e9e">${polygonCount.toLocaleString()} triangles</text>` : ""}
</svg>`;
}
