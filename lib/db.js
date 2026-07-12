import Database from "better-sqlite3";
import path from "path";
import fs from "fs";
import crypto from "crypto";

// Store the DB file under ./data (created on first run).
const dataDir = path.join(process.cwd(), "data");
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

const dbPath = path.join(dataDir, "app.db");

/** Generate a short, unique suffix for seeded file paths. */
function shortId() {
  return crypto.randomBytes(4).toString("hex");
}

/** Ensure the database schema supports nullable user_id for models. */
function ensureSchema(db) {
  // Check if source column exists
  let hasSourceCol = false;
  try {
    const cols = db.prepare("PRAGMA table_info(models)").all();
    hasSourceCol = cols.some((c) => c.name === "source");
  } catch {
    // Table might not exist yet
  }

  if (!hasSourceCol) {
    db.exec("ALTER TABLE models ADD COLUMN source TEXT DEFAULT 'seeded'");
    console.log("[db] Added 'source' column to models table.");
  }

  // Check if user_id allows NULL
  let allowsNull = false;
  try {
    const cols = db.prepare("PRAGMA table_info(models)").all();
    const userIdCol = cols.find((c) => c.name === "user_id");
    allowsNull = userIdCol && !userIdCol.notnull;
  } catch {}

  if (!allowsNull) {
    throw new Error(
      "DB_SCHEMA outdated - user_id must be nullable for seeded models. Please delete data/app.db and restart.",
    );
  }
}

/** Seed the models table from public/models/*.glb if no seeded records exist yet. */
function seedModels(db) {
  const count = db
    .prepare("SELECT COUNT(*) AS c FROM models WHERE source = 'seeded'")
    .get();
  if (count.c > 0) return; // already seeded

  const dir = path.join(process.cwd(), "public", "models");
  if (!fs.existsSync(dir)) return;

  const files = fs
    .readdirSync(dir)
    .filter((f) => f.toLowerCase().endsWith(".glb"))
    .map((f) => path.join(dir, f))
    // skip subdirectories (e.g. uploaded/, thumbnails/)
    .filter((fp) => fs.statSync(fp).isFile());

  const insert = db.prepare(
    "INSERT INTO models (user_id, name, description, category, file_path, thumbnail_path, mime_type, size, polygon_count, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
  );

  const tx = db.transaction(() => {
    for (const fp of files) {
      const base = path.basename(fp).replace(/\.glb$/i, "");
      const displayName = base
        .replace(/_test$/, "")
        .replace(/([a-z])([A-Z])/g, "$1 $2");
      const ts = Date.now();
      const thumbName = `seed-${ts}-${shortId()}.svg`;

      // minimal thumbnail SVG
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><rect width="256" height="256" fill="#f5f5f5"/><rect x="88" y="88" width="80" height="80" fill="#e0e0e0" stroke="#bdbdbd" stroke-width="2"/><text x="128" y="210" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#757575">${displayName}</text></svg>`;
      const thumbDir = path.join(
        process.cwd(),
        "public",
        "models",
        "thumbnails",
      );
      if (!fs.existsSync(thumbDir)) fs.mkdirSync(thumbDir, { recursive: true });
      fs.writeFileSync(path.join(thumbDir, thumbName), svg);

      insert.run(
        null, // user_id — seeded models have no owner
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
    }
  });
  tx();
  console.log(`[db] Seeded ${files.length} model(s) into the database.`);
}

// Check if existing DB has the old (broken) schema before deciding to migrate.
function needsSchemaMigration(existingDb) {
  try {
    const cols = existingDb.prepare("PRAGMA table_info(models)").all();
    const userIdCol = cols.find((c) => c.name === "user_id");
    // If user_id exists and is NOT NULL, migration is needed
    if (userIdCol && userIdCol.notnull) {
      return true;
    }
    return false;
  } catch {
    // Table doesn't exist yet — no migration needed
    return false;
  }
}

// Reuse a single connection across hot-reloads in dev.
let db = globalThis.__appDb;
if (!db) {
  // Only delete DB if it has the old broken schema (user_id was NOT NULL)
  if (fs.existsSync(dbPath)) {
    const tempDb = new Database(dbPath);
    const mustMigrate = needsSchemaMigration(tempDb);
    tempDb.close();

    if (mustMigrate) {
      fs.unlinkSync(dbPath);
      console.log(
        "[db] Migrated: deleted old database with NOT NULL user_id constraint.",
      );
    } else {
      console.log("[db] Existing database found, skipping migration.");
    }
  }

  db = new Database(dbPath);
  db.pragma("journal_mode = WAL");
  db.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id         INTEGER PRIMARY KEY AUTOINCREMENT,
      email      TEXT NOT NULL UNIQUE,
      name       TEXT NOT NULL,
      password   TEXT NOT NULL,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS sessions (
      token      TEXT PRIMARY KEY,
      user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS blueprints (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      title         TEXT NOT NULL,
      original_name TEXT NOT NULL,
      file_path     TEXT NOT NULL,
      mime_type     TEXT,
      size          INTEGER,
      created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_blueprints_user_recent
      ON blueprints(user_id, created_at DESC);
    
    CREATE TABLE IF NOT EXISTS models (
      id            INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id       INTEGER DEFAULT NULL REFERENCES users(id) ON DELETE CASCADE,
      name          TEXT NOT NULL,
      description   TEXT DEFAULT '',
      category      TEXT DEFAULT 'uncategorized',
      tags          TEXT DEFAULT '[]',
      file_path     TEXT NOT NULL,
      thumbnail_path TEXT,
      mime_type     TEXT DEFAULT 'model/gltf-binary',
      size          INTEGER,
      polygon_count INTEGER DEFAULT 0,
      source        TEXT DEFAULT 'seeded',
      created_at    TEXT NOT NULL DEFAULT (datetime('now'))
    );

    CREATE INDEX IF NOT EXISTS idx_models_user_recent
      ON models(user_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_models_category
      ON models(category);

    CREATE TABLE IF NOT EXISTS blueprint_models (
      blueprint_id INTEGER NOT NULL REFERENCES blueprints(id) ON DELETE CASCADE,
      model_id     INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
      PRIMARY KEY (blueprint_id, model_id)
    );
  `);
  globalThis.__appDb = db;

  // Ensure schema is compatible (checks/adds columns if needed), then auto-seed
  ensureSchema(db);

  // Auto-seed on fresh initialization (fresh DB or after migration)
  seedModels(db);
}

export default db;

// Export for external scripts (e.g. next/font or manual seeding)
export { seedModels };

export function close() {
  db.close();
  globalThis.__appDb = undefined;
}

if (process.env.NODE_ENV === "production") {
  process.on("SIGTERM", () => {
    db.close();
    process.exit(0);
  });
}
