import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import db from "@/lib/db";
import { createSession, SESSION_COOKIE, cookieOptions } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(request) {
  let body;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid request." }, { status: 400 });
  }

  const email = (body.email || "").trim().toLowerCase();
  const password = body.password || "";

  if (!email || !password) {
    return NextResponse.json({ error: "Enter email and password." }, { status: 400 });
  }

  const user = db.prepare("SELECT * FROM users WHERE email = ?").get(email);
  // Same message for both cases so we don't leak which emails exist.
  if (!user || !bcrypt.compareSync(password, user.password)) {
    return NextResponse.json({ error: "Wrong email or password." }, { status: 401 });
  }

  const token = createSession(user.id);

  const res = NextResponse.json({
    user: { id: user.id, email: user.email, name: user.name },
  });
  res.cookies.set(SESSION_COOKIE, token, cookieOptions);
  return res;
}
