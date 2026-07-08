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
  const user = await getCurrentUser();
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

  // Drop entries whose file was deleted from disk
  const alive = rows.filter((r) => fs.existsSync(path.join(process.cwd(), "public", r.file_path)));
  return NextResponse.json({ blueprints: alive });
}

// POST /api/blueprints  -> upload a local blueprint (multipart/form-data)
export async function POST(request) {
  const user = await getCurrentUser();
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

  // Validate actual file content, not just the filename extension.
  // This prevents a malicious file renamed to an allowed extension from being stored.
  if (ext === ".svg") {
    const text = buffer.toString("latin1");
    if (!/^<svg[\s>]/i.test(text)) {
      return NextResponse.json({ error: "Invalid SVG file." }, { status: 415 });
    }
  }
  if (ext === ".pdf") {
    if (buffer.toString("latin1", 0, 4) !== "%PDF") {
      return NextResponse.json({ error: "Invalid PDF file." }, { status: 415 });
    }
  }
  if (ext === ".glb") {
    if (buffer.toString("latin1", 0, 4) !== "glTF") {
      return NextResponse.json({ error: "Invalid GLB file." }, { status: 415 });
    }
  }
  if (ext === ".gltf") {
    const text = buffer.toString("latin1");
    if (!/^\s*\{/.test(text)) {
      return NextResponse.json({ error: "Invalid GLTF file." }, { status: 415 });
    }
  }
  if (ext === ".png") {
    if (buffer[0] !== 0x89 || buffer[1] !== 0x50 || buffer[2] !== 0x4e || buffer[3] !== 0x47) {
      return NextResponse.json({ error: "Invalid PNG file." }, { status: 415 });
    }
  }
  if (ext === ".jpg" || ext === ".jpeg") {
    if (buffer[0] !== 0xff || buffer[1] !== 0xd8) {
      return NextResponse.json({ error: "Invalid JPEG file." }, { status: 415 });
    }
  }
  if (ext === ".webp") {
    if (buffer[0] !== 0x52 || buffer[1] !== 0x49 || buffer[2] !== 0x46 || buffer[3] !== 0x46) {
      return NextResponse.json({ error: "Invalid WebP file." }, { status: 415 });
    }
  }
  if (ext === ".gif") {
    if (buffer.toString("latin1", 0, 3) !== "GIF") {
      return NextResponse.json({ error: "Invalid GIF file." }, { status: 415 });
    }
  }

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

// DELETE /api/blueprints?id=...  ->  delete a single blueprint
// DELETE /api/blueprints?clear=1  ->  delete all blueprints for this user
export async function DELETE(request) {
  const user = await getCurrentUser();
  if (!user) {
    return NextResponse.json({ error: "Not logged in." }, { status: 401 });
  }

  const { searchParams } = new URL(request.url);

  // Clear all
  if (searchParams.get("clear") === "1") {
    const rows = db.prepare("SELECT file_path FROM blueprints WHERE user_id = ?").all(user.id);
    for (const r of rows) {
      const diskPath = path.join(process.cwd(), "public", r.file_path);
      if (fs.existsSync(diskPath)) fs.unlinkSync(diskPath);
    }
    db.prepare("DELETE FROM blueprints WHERE user_id = ?").run(user.id);
    return NextResponse.json({ ok: true });
  }

  const id = searchParams.get("id");
  if (!id) {
    return NextResponse.json({ error: "Missing id." }, { status: 400 });
  }

  const bp = db
    .prepare("SELECT * FROM blueprints WHERE id = ? AND user_id = ?")
    .get(id, user.id);
  if (!bp) {
    return NextResponse.json({ error: "Not found." }, { status: 404 });
  }

  // Remove file from disk
  const diskPath = path.join(process.cwd(), "public", bp.file_path);
  if (fs.existsSync(diskPath)) fs.unlinkSync(diskPath);

  db.prepare("DELETE FROM blueprints WHERE id = ?").run(id);
  return NextResponse.json({ ok: true });
}
