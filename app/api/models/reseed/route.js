import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import db from "@/lib/db";

export const runtime = "nodejs";

/**
 * POST /api/models/reseed
 * Rebuilds seeded models from the GLB files in public/models/.
 * - Deletes existing seeded records from the database.
 * - Scans public/models/ for *.glb files (skips subdirectories like uploaded/, thumbnails/).
 * - Re-inserts them as seeded models with fresh thumbnails.
 */
export async function POST() {
  const MODELS_DIR = path.join(process.cwd(), "public", "models");
  const THUMBNAILS_DIR = path.join(
    process.cwd(),
    "public",
    "models",
    "thumbnails",
  );

  // 1. Delete existing seeded records (and their thumbnails)
  const existingSeeded = db
    .prepare("SELECT thumbnail_path FROM models WHERE source = 'seeded'")
    .all();

  for (const row of existingSeeded) {
    if (row.thumbnail_path) {
      const thumbPath = path.join(process.cwd(), "public", row.thumbnail_path);
      if (fs.existsSync(thumbPath)) {
        try {
          fs.unlinkSync(thumbPath);
        } catch {
          // Ignore deletion errors
        }
      }
    }
  }

  db.prepare("DELETE FROM models WHERE source = 'seeded'").run();

  // 2. Scan for GLB files and re-seed
  if (!fs.existsSync(MODELS_DIR)) {
    return NextResponse.json({
      message: "Models directory not found.",
      reseeded: 0,
    });
  }

  const files = fs
    .readdirSync(MODELS_DIR)
    .filter((f) => f.toLowerCase().endsWith(".glb"))
    .map((f) => path.join(MODELS_DIR, f))
    .filter((fp) => fs.statSync(fp).isFile());

  const insert = db.prepare(
    "INSERT INTO models (user_id, name, description, category, file_path, thumbnail_path, mime_type, size, polygon_count, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );

  let reseeded = 0;

  const tx = db.transaction(() => {
    for (const fp of files) {
      const base = path.basename(fp).replace(/\.glb$/i, "");
      const displayName = base
        .replace(/_test$/, "")
        .replace(/([a-z])([A-Z])/g, "$1 $2");

      const ts = Date.now();
      const thumbName = `seed-${ts}-${crypto.randomBytes(4).toString("hex")}.svg`;

      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="#f5f5f5"/><rect x="88" y="88" width="80" height="80" fill="#e0e0e0" stroke="#bdbdbd" stroke-width="2"/><text x="128" y="210" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#757575">${displayName}</text></svg>`;

      if (!fs.existsSync(THUMBNAILS_DIR)) {
        fs.mkdirSync(THUMBNAILS_DIR, { recursive: true });
      }
      fs.writeFileSync(path.join(THUMBNAILS_DIR, thumbName), svg);

      insert.run(
        null,
        displayName.trim(),
        "Seeded sample model",
        "sample",
        `/models/${path.basename(fp)}`,
        `/models/thumbnails/${thumbName}`,
        "model/gltf-binary",
        fs.statSync(fp).size,
        0,
        "seeded",
      );

      reseeded++;
    }
  });

  tx();

  return NextResponse.json({
    message: `Reseeded ${reseeded} seeded models from public/models/.`,
    reseeded,
  });
}
