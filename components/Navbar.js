"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import AuthModal from "./AuthModal";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/upload", label: "Upload & View" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const pathname = usePathname();
  const [user, setUser] = useState(null);
  const [modal, setModal] = useState(null); // 'login' | 'register' | null
  const [menuOpen, setMenuOpen] = useState(false);

  const loadUser = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/me", { cache: "no-store" });
      const data = await res.json();
      setUser(data.user);
    } catch {
      setUser(null);
    }
  }, []);

  useEffect(() => {
    loadUser();
    const openAuth = (e) => setModal(e.detail || "login");
    const authChanged = () => loadUser();
    window.addEventListener("open-auth", openAuth);
    window.addEventListener("auth-changed", authChanged);
    return () => {
      window.removeEventListener("open-auth", openAuth);
      window.removeEventListener("auth-changed", authChanged);
    };
  }, [loadUser]);

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.dispatchEvent(new CustomEvent("auth-changed"));
  }

  return (
    <>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 50,
          borderBottom: "1px solid var(--line)",
          background: "rgba(244,245,247,0.85)",
          backdropFilter: "blur(8px)",
        }}
      >
        <div
          className="container"
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            height: 64,
            gap: 24,
          }}
        >
          <Link href="/" style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontWeight: 600,
                fontSize: 19,
                letterSpacing: "-0.03em",
              }}
            >
              Plan<span style={{ color: "var(--accent)" }}>/</span>Space
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                letterSpacing: "0.12em",
                color: "var(--ink-faint)",
                textTransform: "uppercase",
              }}
            >
              v0.1
            </span>
          </Link>

          <button
            aria-label="Toggle menu"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((v) => !v)}
            style={{
              display: "none",
              background: "none",
              border: "1px solid var(--line-strong)",
              borderRadius: "var(--radius)",
              padding: "6px 10px",
              cursor: "pointer",
              fontSize: 18,
              lineHeight: 1,
              color: "var(--ink)",
            }}
            className="mobile-menu-btn"
          >
            {menuOpen ? "✕" : "☰"}
          </button>

          <nav
            style={{ display: "flex", gap: 6, alignItems: "center" }}
            className="nav-links"
          >
            {LINKS.map((l) => {
              const active = pathname === l.href;
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    padding: "8px 12px",
                    borderRadius: "var(--radius)",
                    color: active ? "var(--ink)" : "var(--ink-soft)",
                    background: active ? "var(--paper-sunk)" : "transparent",
                  }}
                >
                  {l.label}
                </Link>
              );
            })}
          </nav>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {user ? (
              <>
                <span
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 12,
                    color: "var(--ink-soft)",
                  }}
                >
                  {user.name}
                </span>
                <button className="btn btn-ghost" onClick={logout}>
                  Log out
                </button>
              </>
            ) : (
              <>
                <button className="btn btn-ghost" onClick={() => setModal("login")}>
                  Log in
                </button>
                <button className="btn btn-accent" onClick={() => setModal("register")}>
                  Register
                </button>
              </>
            )}
          </div>
        </div>
      </header>

      {modal && (
        <AuthModal
          mode={modal}
          onClose={() => setModal(null)}
          onSuccess={() => {
            setModal(null);
            window.dispatchEvent(new CustomEvent("auth-changed"));
          }}
        />
      )}

      <style>{`
        .mobile-menu-btn { display: none; }
        @media (max-width: 720px) {
          .nav-links {
            display: ${menuOpen ? "flex" : "none"} !important;
            position: absolute;
            top: 64px;
            left: 0;
            right: 0;
            flex-direction: column;
            background: var(--paper-raised);
            border-bottom: 1px solid var(--line);
            padding: 12px 24px;
            gap: 4px;
          }
          .mobile-menu-btn { display: inline-block !important; }
        }
      `}</style>
    </>
  );
}
