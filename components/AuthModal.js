"use client";

import { useState, useEffect } from "react";

export default function AuthModal({ mode = "login", onClose, onSuccess }) {
  const [tab, setTab] = useState(mode);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => setError(""), [tab]);

  async function submit() {
    setError("");
    setLoading(true);
    const endpoint = tab === "register" ? "/api/auth/register" : "/api/auth/login";
    const payload = tab === "register" ? { name, email, password } : { email, password };
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Something went wrong.");
      } else {
        onSuccess?.();
      }
    } catch {
      setError("Network error. Try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgba(22,24,29,0.45)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
    >
      <div
        className="card framed"
        onClick={(e) => e.stopPropagation()}
        style={{ width: "100%", maxWidth: 400, padding: 28 }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="eyebrow">{tab === "register" ? "New account" : "Sign in"}</span>
          <button
            onClick={onClose}
            aria-label="Close"
            style={{
              border: "none",
              background: "none",
              cursor: "pointer",
              fontFamily: "var(--font-mono)",
              fontSize: 16,
              color: "var(--ink-faint)",
            }}
          >
            ✕
          </button>
        </div>

        <h2 style={{ fontSize: 26, margin: "12px 0 20px" }}>
          {tab === "register" ? "Create your account" : "Welcome back"}
        </h2>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          {tab === "register" && (
            <div className="field">
              <label>Name</label>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Jane Architect"
                autoComplete="name"
              />
            </div>
          )}
          <div className="field">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@studio.com"
              autoComplete="email"
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </div>
          <div className="field">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete={tab === "register" ? "new-password" : "current-password"}
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          <button className="btn btn-accent" onClick={submit} disabled={loading} style={{ justifyContent: "center" }}>
            {loading ? "Working…" : tab === "register" ? "Create account" : "Log in"}
          </button>
        </div>

        <div
          style={{
            marginTop: 18,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            color: "var(--ink-soft)",
            textAlign: "center",
          }}
        >
          {tab === "register" ? (
            <>
              Already registered?{" "}
              <button onClick={() => setTab("login")} style={linkBtn}>
                Log in
              </button>
            </>
          ) : (
            <>
              No account yet?{" "}
              <button onClick={() => setTab("register")} style={linkBtn}>
                Register
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

const linkBtn = {
  border: "none",
  background: "none",
  cursor: "pointer",
  color: "var(--accent)",
  fontFamily: "var(--font-mono)",
  fontSize: 12,
  textDecoration: "underline",
};
