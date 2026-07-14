import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";
import crypto from "crypto";
import db from "@/lib/db";

export const runtime = "nodejs";

/**
 * POST /api/models/reseed
 * Rebuilds seeded model DB records from the GLB files in public/models/.
 * - Deletes existing seeded rows from the database (DB rows only).
 * - Scans public/models/ for *.glb files (skips subdirectories like uploaded/, thumbnails/).
 * - Re-inserts them as seeded models, pointing at the existing static thumbnails.
 *
 * IMPORTANT: seeded thumbnails in public/models/thumbnails/ are static source
 * assets shared by every reseed, NOT per-record generated files. They must
 * never be deleted from disk here — only uploaded-model thumbnails (if you
 * generate those elsewhere) are safe to treat as disposable. Deleting the
 * seeded SVGs on every reseed is what was silently wiping the thumbnail
 * files themselves, so later reseeds could never find them and quietly
 * skipped re-inserting those models altogether.
 */
export async function POST() {
  const MODELS_DIR = path.join(process.cwd(), "public", "models");
  const THUMBNAILS_DIR = path.join(
    process.cwd(),
    "public",
    "models",
    "thumbnails",
  );

  // 1. Delete existing seeded DB rows only. Do NOT touch files on disk —
  // seeded thumbnails are static assets, not generated artifacts tied to
  // a specific row's lifecycle.
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
  const skipped = [];

  const tx = db.transaction(() => {
    for (const fp of files) {
      const base = path.basename(fp).replace(/\.glb$/i, "");
      const displayName = base
        .replace(/_test$/, "")
        .replace(/([a-z])([A-Z])/g, "$1 $2");

      // Look for an existing SVG thumbnail with the same name as the model
      const thumbName = `${base}.svg`;
      const thumbPath = path.join(THUMBNAILS_DIR, thumbName);

      // Use the matching thumbnail if it exists, otherwise skip the model
      if (!fs.existsSync(thumbPath)) {
        skipped.push(path.basename(fp));
        continue;
      }

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

  if (skipped.length > 0) {
    console.warn(
      `Reseed: skipped ${skipped.length} model(s) with no matching thumbnail SVG:`,
      skipped,
    );
  }

  return NextResponse.json({
    message: `Reseeded ${reseeded} seeded models from public/models/.`,
    reseeded,
    skipped,
  });
}
