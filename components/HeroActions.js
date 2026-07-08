"use client";

import Link from "next/link";

export default function HeroActions() {
  function openAuth(mode) {
    window.dispatchEvent(new CustomEvent("open-auth", { detail: mode }));
  }

  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginTop: 28 }}>
      <button className="btn btn-accent" onClick={() => openAuth("register")}>
        Create free account
      </button>
      <Link href="/upload" className="btn btn-ghost">
        Open the viewer
      </Link>
    </div>
  );
}
