import { cookies } from "next/headers";
import crypto from "crypto";
import db from "./db";

export const SESSION_COOKIE = "session";

// Create a session row + return the token to set as a cookie.
export function createSession(userId) {
  const token = crypto.randomBytes(32).toString("hex");
  db.prepare("INSERT INTO sessions (token, user_id) VALUES (?, ?)").run(token, userId);
  return token;
}

export function destroySession(token) {
  if (!token) return;
  db.prepare("DELETE FROM sessions WHERE token = ?").run(token);
}

// Read the current user from the session cookie. Returns null if not logged in.
export async function getCurrentUser() {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  if (!token) return null;

  const row = db
    .prepare(
      `SELECT u.id, u.email, u.name, u.created_at
         FROM sessions s
         JOIN users u ON u.id = s.user_id
        WHERE s.token = ?`
    )
    .get(token);

  return row || null;
}

// Cookie options shared by login/register/logout routes.
export const cookieOptions = {
  httpOnly: true,
  sameSite: "lax",
  path: "/",
  secure: process.env.SESSION_SECURE === "1",
  maxAge: 60 * 60 * 24 * 30, // 30 days
};
