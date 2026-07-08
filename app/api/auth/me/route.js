import { NextResponse } from "next/server";
import { getCurrentUser } from "@/lib/session";

export const runtime = "nodejs";

export async function GET() {
  const user = getCurrentUser();
  return NextResponse.json({ user: user || null });
}
