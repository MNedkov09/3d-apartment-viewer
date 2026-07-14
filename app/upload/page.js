"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import RecentPanel from "@/components/RecentPanel";
import Viewer from "@/components/Viewer";

export default function UploadPage() {
  const [user, setUser] = useState(undefined); // undefined = still checking
  const [recent, setRecent] = useState([]);
  const [active, setActive] = useState(null);
  const [loadingRecent, setLoadingRecent] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [msg, setMsg] = useState(null); // { type, text }
  const inputRef = useRef(null);
  // Guards the auto-reseed effect below so it fires at most once per
  // page load, no matter how many times `user` changes reference.
  const reseedRanRef = useRef(false);

  const checkAuth = useCallback(async () => {
    try {
      const res = await fetch("/api/auth/me", { cache: "no-store" });
      const data = await res.json();
      setUser(data.user);
      return data.user;
    } catch {
      setUser(null);
      return null;
    }
  }, []);

  const loadRecent = useCallback(async () => {
    setLoadingRecent(true);
    try {
      const res = await fetch("/api/blueprints", { cache: "no-store" });
      if (res.ok) {
        const data = await res.json();
        setRecent(data.blueprints);
        setActive((cur) => cur || data.blueprints[0] || null);
      }
    } finally {
      setLoadingRecent(false);
    }
  }, []);

  useEffect(() => {
    checkAuth().then((u) => {
      if (u) loadRecent();
    });
    const onAuthChange = () =>
      checkAuth().then((u) => (u ? loadRecent() : setRecent([])));
    window.addEventListener("auth-changed", onAuthChange);
    return () => window.removeEventListener("auth-changed", onAuthChange);
  }, [checkAuth, loadRecent]);

  // Auto-seed sample 3D models on page load — once per mount only.
  //
  // FIX: the original version dispatched "auth-changed" on success, which
  // re-ran checkAuth(), which called setUser() with a brand-new object
  // reference every time — and this effect depended on `user`, so it fired
  // again, reseeded again, dispatched again... an infinite loop that
  // showed up as endless auth/blueprints/reseed spam in the server log.
  //
  // Fix: reseedRanRef makes this run exactly once per page load, and on
  // success it refreshes the list directly via loadRecent() instead of
  // bouncing back through the auth-changed event.
  useEffect(() => {
    if (!user || reseedRanRef.current) return;
    reseedRanRef.current = true;
    let cancelled = false;
    async function reseed() {
      try {
        const res = await fetch("/api/models/reseed", { method: "POST" });
        const data = await res.json();
        if (!cancelled && res.ok && data.reseeded > 0) {
          loadRecent();
        }
      } catch {
        // silently ignore reseed failures
      }
    }
    reseed();
    return () => {
      cancelled = true;
    };
  }, [user, loadRecent]);

  async function deleteOne(bp) {
    setDeleting(true);
    try {
      const res = await fetch(`/api/blueprints?id=${bp.id}`, {
        method: "DELETE",
      });
      if (res.ok) {
        setRecent((r) => r.filter((x) => x.id !== bp.id));
        if (active?.id === bp.id) setActive(null);
      }
    } finally {
      setDeleting(false);
    }
  }

  async function clearAll() {
    setDeleting(true);
    try {
      const res = await fetch("/api/blueprints?clear=1", { method: "DELETE" });
      if (res.ok) {
        setRecent([]);
        setActive(null);
      }
    } finally {
      setDeleting(false);
    }
  }

  function pickFile(f) {
    if (!f) return;
    setFile(f);
    setMsg(null);
    if (!title) setTitle(f.name.replace(/\.[^.]+$/, ""));
  }

  async function upload() {
    if (!file) {
      setMsg({ type: "error", text: "Choose a file first." });
      return;
    }
    setUploading(true);
    setMsg(null);
    const fd = new FormData();
    fd.append("file", file);
    fd.append("title", title);
    try {
      const res = await fetch("/api/blueprints", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        setMsg({ type: "error", text: data.error || "Upload failed." });
      } else {
        setRecent((r) => [data.blueprint, ...r]);
        setActive(data.blueprint);
        setFile(null);
        setTitle("");
        if (inputRef.current) inputRef.current.value = "";
        setMsg({ type: "ok", text: "Saved to your recent blueprints." });
      }
    } catch {
      setMsg({ type: "error", text: "Network error during upload." });
    } finally {
      setUploading(false);
    }
  }

  // ---- Not logged in gate ----
  if (user === null) {
    return (
      <div
        className="container"
        style={{ padding: "80px 24px", textAlign: "center" }}
      >
        <span className="eyebrow" style={{ justifyContent: "center" }}>
          Restricted
        </span>
        <h1 style={{ fontSize: 34, marginTop: 16 }}>
          Log in to upload blueprints
        </h1>
        <p style={{ color: "var(--ink-soft)", marginTop: 12 }}>
          Your uploads are saved to your account, so you need to be signed in.
        </p>
        <div
          style={{
            display: "flex",
            gap: 12,
            justifyContent: "center",
            marginTop: 24,
          }}
        >
          <button
            className="btn btn-accent"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("open-auth", { detail: "register" }),
              )
            }
          >
            Register
          </button>
          <button
            className="btn btn-ghost"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("open-auth", { detail: "login" }),
              )
            }
          >
            Log in
          </button>
        </div>
      </div>
    );
  }

  // ---- Loading auth ----
  if (user === undefined) {
    return (
      <div className="container" style={{ padding: "80px 24px" }}>
        <p
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 13,
            color: "var(--ink-faint)",
          }}
        >
          Loading…
        </p>
      </div>
    );
  }

  // ---- Main app ----
  return (
    <div className="container" style={{ padding: "40px 24px 72px" }}>
      <span className="eyebrow">Upload &amp; View</span>
      <h1 style={{ fontSize: 32, marginTop: 14 }}>Your workspace</h1>
      <p style={{ color: "var(--ink-soft)", marginTop: 8, marginBottom: 28 }}>
        Signed in as {user.name}. Drop a blueprint below — it saves to your
        account and opens in the viewport.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(240px, 300px) 1fr",
          gap: 24,
          alignItems: "start",
        }}
        className="workspace-grid"
      >
        {/* Left: recent */}
        <RecentPanel
          items={recent}
          activeId={active?.id}
          onSelect={setActive}
          loading={loadingRecent}
          onClear={clearAll}
          onDelete={deleteOne}
        />

        {/* Right: uploader + viewer */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {/* Dropzone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pickFile(e.dataTransfer.files?.[0]);
            }}
            className="card"
            style={{
              padding: 20,
              borderStyle: "dashed",
              borderColor: dragging ? "var(--accent)" : "var(--line-strong)",
              background: dragging
                ? "var(--accent-wash)"
                : "var(--paper-raised)",
              transition: "border-color 0.15s ease, background 0.15s ease",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: 16,
                flexWrap: "wrap",
                alignItems: "flex-end",
              }}
            >
              <div style={{ flex: "1 1 200px", minWidth: 180 }}>
                <div className="field">
                  <label>Title (optional)</label>
                  <input
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    placeholder="e.g. Mladost 2-bed"
                  />
                </div>
              </div>

              <div style={{ flex: "1 1 200px", minWidth: 180 }}>
                <div className="field">
                  <label>File</label>
                  <input
                    ref={inputRef}
                    type="file"
                    accept=".png,.jpg,.jpeg,.webp,.gif,.svg,.pdf,.glb,.gltf,.obj,.dxf"
                    onChange={(e) => pickFile(e.target.files?.[0])}
                    style={{ padding: 8 }}
                  />
                </div>
              </div>

              <button
                className="btn btn-accent"
                onClick={upload}
                disabled={uploading}
              >
                {uploading ? "Uploading…" : "Upload blueprint"}
              </button>
            </div>

            <p
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: "var(--ink-faint)",
                marginTop: 12,
                marginBottom: 0,
              }}
            >
              {file
                ? `Selected: ${file.name}`
                : "Drag a file here, or browse. Images, PDF, or 3D models up to 25 MB."}
            </p>

            {msg && (
              <div
                className={`alert ${msg.type === "ok" ? "alert-ok" : "alert-error"}`}
                style={{ marginTop: 12 }}
              >
                {msg.text}
              </div>
            )}
          </div>

          {/* Viewer */}
          <Viewer blueprint={active} />
        </div>
      </div>

      <style>{`
        @media (max-width: 760px) {
          .workspace-grid { grid-template-columns: 1fr !important; }
        }
      `}</style>
    </div>
  );
}
