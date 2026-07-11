import Database from "better-sqlite3";
import path from "path";
import fs from "fs";

// Store the DB file under ./data (created on first run).
const dataDir = path.join(process.cwd(), "data");
if (!fs.existsSync(dataDir)) fs.mkdirSync(dataDir, { recursive: true });

const dbPath = path.join(dataDir, "app.db");

// Reuse a single connection across hot-reloads in dev.
let db = globalThis.__appDb;
if (!db) {
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
      user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      name          TEXT NOT NULL,
      description   TEXT DEFAULT '',
      category      TEXT DEFAULT 'uncategorized',
      tags          TEXT DEFAULT '[]',
      file_path     TEXT NOT NULL,
      thumbnail_path TEXT,
      mime_type     TEXT DEFAULT 'model/gltf-binary',
      size          INTEGER,
      polygon_count INTEGER DEFAULT 0,
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
}

export default db;

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
