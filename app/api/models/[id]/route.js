import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import db from "@/lib/db";
import { getCurrentUser } from "@/lib/session";
import { extractPolygonCount, generateModelThumbnailSvg } from "@/lib/model-utils";

export const runtime = "nodejs";

const MODELS_DIR = path.join(process.cwd(), "public", "uploads", "models");
const THUMBNAILS_DIR = path.join(process.cwd(), "public", "uploads", "thumbnails");

// GET /api/models/:id — get single model metadata (owner-only).
export async function GET(request, { params }) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Not logged in." }, { status: 401 });
  }

  const id = parseInt(params.id, 10);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid model ID." }, { status: 400 });
  }

  const row = db
    .prepare(
      `SELECT * FROM models WHERE id = ?`
    )
    .get(id);

  if (!row) {
    return NextResponse.json({ error: "Model not found." }, { status: 404 });
  }
  if (row.user_id !== user.id) {
    return NextResponse.json({ error: "Not authorized to view this model." }, { status: 403 });
  }

  return NextResponse.json({ model: row });
}

// PUT /api/models/:id — update name/description/category/tags (owner only).
export async function PUT(request, { params }) {
  const user = await getCurrentUser();
  if (!user) return NextResponse.json({ error: "Not logged in." }, { status: 401 });

  const id = parseInt(params.id, 10);

  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON." }, { status: 400 });
  }

  const { name, description, category, tags } = body;

  // Validate inputs.
  if (name !== undefined && (!name || typeof name !== "string" || name.trim().length === 0)) {
    return NextResponse.json({ error: "Name must be a non-empty string." }, { status: 400 });
  }
  if (description !== undefined && typeof description !== "string") {
    return NextResponse.json({ error: "Description must be a string." }, { status: 400 });
  }
  if (category !== undefined) {
    const cats = ["uncategorized", "furniture", "lighting", "structures", "decorations", "outdoor"];
    if (!cats.includes(category)) {
      return NextResponse.json(
        { error: `Category must be one of: ${cats.join(", ")}.` },
        { status: 400 }
      );
    }
  }
  if (tags !== undefined) {
    if (!Array.isArray(tags) || !tags.every((t) => typeof t === "string")) {
      return NextResponse.json({ error: "Tags must be an array of strings." }, { status: 400 });
    }
  }

  // Build dynamic UPDATE.
  const fields = [];
  const params_arr = [];
  if (name !== undefined) {
    fields.push("name = ?");
    params_arr.push(name.trim());
  }
  if (description !== undefined) {
    fields.push("description = ?");
    params_arr.push(description.trim());
  }
  if (category !== undefined) {
    fields.push("category = ?");
    params_arr.push(category);
  }
  if (tags !== undefined) {
    fields.push("tags = ?");
    params_arr.push(JSON.stringify(tags));
  }

  // If nothing to update, return current state.
  if (fields.length === 0) {
    const row = db.prepare("SELECT * FROM models WHERE id = ?").get(id);
    return NextResponse.json({ model: row });
  }

  params_arr.push(id);
  db.prepare(`UPDATE models SET ${fields.join(", ")} WHERE id = ?`).run(...params_arr);

  const updated = db
    .prepare("SELECT * FROM models WHERE id = ?")
    .get(id);

  return NextResponse.json({ model: updated });
}

// DELETE /api/models/:id — delete model and its files (owner only).
export async function DELETE(request, { params }) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Not logged in." }, { status: 401 });
  }

  const id = parseInt(params.id, 10);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid model ID." }, { status: 400 });
  }

  const row = db.prepare("SELECT user_id, file_path, thumbnail_path FROM models WHERE id = ?").get(id);
  if (!row) {
    return NextResponse.json({ error: "Model not found." }, { status: 404 });
  }
  if (row.user_id !== user.id) {
    return NextResponse.json({ error: "Not authorized to delete this model." }, { status: 403 });
  }

  // Remove physical files.
  if (row.file_path) {
    const fullPath = path.join(process.cwd(), row.file_path);
    if (fs.existsSync(fullPath)) fs.unlinkSync(fullPath);
  }
  if (row.thumbnail_path) {
    const thumbPath = path.join(process.cwd(), row.thumbnail_path);
    if (fs.existsSync(thumbPath)) fs.unlinkSync(thumbPath);
  }

  // Remove from blueprint_models junction table and the model itself.
  db.prepare("DELETE FROM blueprint_models WHERE model_id = ?").run(id);
  db.prepare("DELETE FROM models WHERE id = ?").run(id);

  return NextResponse.json({ message: "Model deleted." });
}

// POST /api/models/:id/thumbnail — regenerate thumbnail from GLB file.
export async function POST(request, { params }) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Not logged in." }, { status: 401 });
  }

  const id = parseInt(params.id, 10);
  if (isNaN(id)) {
    return NextResponse.json({ error: "Invalid model ID." }, { status: 400 });
  }

  // Check ownership.
  const row = db.prepare("SELECT user_id, file_path, name FROM models WHERE id = ?").get(id);
  if (!row) {
    return NextResponse.json({ error: "Model not found." }, { status: 404 });
  }
  if (row.user_id !== user.id) {
    return NextResponse.json({ error: "Not authorized to modify this model." }, { status: 403 });
  }

  // Read the GLB file.
  const fullPath = path.join(process.cwd(), row.file_path);
  if (!fs.existsSync(fullPath)) {
    return NextResponse.json({ error: "Model file not found on disk." }, { status: 404 });
  }
  const buffer = fs.readFileSync(fullPath);

  // Extract polygon count and generate SVG thumbnail.
  const polygonCount = extractPolygonCount(buffer);
  const svg = generateModelThumbnailSvg(row.name, polygonCount);

  // Save new thumbnail.
  if (!fs.existsSync(THUMBNAILS_DIR)) fs.mkdirSync(THUMBNAILS_DIR, { recursive: true });
  const thumbName = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}.svg`;
  const thumbDiskPath = path.join(THUMBNAILS_DIR, thumbName);
  fs.writeFileSync(thumbDiskPath, svg);

  const publicThumbPath = `/uploads/thumbnails/${thumbName}`;

  db.prepare("UPDATE models SET thumbnail_path = ? WHERE id = ?").run(publicThumbPath, id);

  return NextResponse.json({ message: "Thumbnail regenerated.", thumbnail_path: publicThumbPath });
}
