import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import db from "@/lib/db";
import { getCurrentUser } from "@/lib/session";

export const runtime = "nodejs";

const UPLOAD_DIR = path.join(process.cwd(), "public", "uploads");
const MAX_SIZE = 25 * 1024 * 1024; // 25 MB
// Floor plans (images/pdf) + common 3D model formats for the viewer.
const ALLOWED_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg",
  ".pdf", ".glb", ".gltf", ".obj", ".dxf",
]);

// GET /api/blueprints  -> most recent blueprints for the logged-in user
export async function GET() {
  const user = getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Not logged in." }, { status: 401 });
  }

  const rows = db
    .prepare(
      `SELECT id, title, original_name, file_path, mime_type, size, created_at
         FROM blueprints
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT 20`
    )
    .all(user.id);

  return NextResponse.json({ blueprints: rows });
}

// POST /api/blueprints  -> upload a local blueprint (multipart/form-data)
export async function POST(request) {
  const user = getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Log in to upload a blueprint." }, { status: 401 });
  }

  let form;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: "Invalid upload." }, { status: 400 });
  }

  const file = form.get("file");
  const title = (form.get("title") || "").toString().trim();

  if (!file || typeof file === "string") {
    return NextResponse.json({ error: "No file was attached." }, { status: 400 });
  }
  if (file.size === 0) {
    return NextResponse.json({ error: "The file is empty." }, { status: 400 });
  }
  if (file.size > MAX_SIZE) {
    return NextResponse.json({ error: "File is larger than 25 MB." }, { status: 413 });
  }

  const ext = path.extname(file.name).toLowerCase();
  if (!ALLOWED_EXT.has(ext)) {
    return NextResponse.json(
      { error: "Unsupported file type. Use an image, PDF, or 3D model." },
      { status: 415 }
    );
  }

  if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

  const safeName = `${Date.now()}-${crypto.randomBytes(6).toString("hex")}${ext}`;
  const diskPath = path.join(UPLOAD_DIR, safeName);
  const publicPath = `/uploads/${safeName}`;

  const buffer = Buffer.from(await file.arrayBuffer());
  fs.writeFileSync(diskPath, buffer);

  const info = db
    .prepare(
      `INSERT INTO blueprints (user_id, title, original_name, file_path, mime_type, size)
       VALUES (?, ?, ?, ?, ?, ?)`
    )
    .run(user.id, title || file.name, file.name, publicPath, file.type || null, file.size);

  const blueprint = db
    .prepare(
      `SELECT id, title, original_name, file_path, mime_type, size, created_at
         FROM blueprints WHERE id = ?`
    )
    .get(info.lastInsertRowid);

  return NextResponse.json({ blueprint }, { status: 201 });
}
