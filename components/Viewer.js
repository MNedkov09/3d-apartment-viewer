"use client";

const IMAGE_EXT = ["png", "jpg", "jpeg", "webp", "gif", "svg"];
const MODEL_EXT = ["glb", "gltf", "obj"];

function extOf(p = "") {
  return p.split(".").pop().toLowerCase();
}

function fmtSize(bytes) {
  if (!bytes) return "—";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(0)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

export default function Viewer({ blueprint }) {
  const ext = blueprint ? extOf(blueprint.file_path) : null;
  const isImage = IMAGE_EXT.includes(ext);
  const isPdf = ext === "pdf";
  const isModel = MODEL_EXT.includes(ext);

  return (
    <div
      className="framed"
      style={{
        border: "1px solid var(--line-strong)",
        borderRadius: "var(--radius)",
        background: "var(--paper-raised)",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        minHeight: 440,
      }}
    >
      {/* Stage header — reads like a drawing title block */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "10px 14px",
          borderBottom: "1px solid var(--line)",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          letterSpacing: "0.08em",
          textTransform: "uppercase",
          color: "var(--ink-faint)",
        }}
      >
        <span>Viewport</span>
        <span>{blueprint ? `${ext} · ${fmtSize(blueprint.size)}` : "no drawing loaded"}</span>
      </div>

      {/* Stage body */}
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 20,
          backgroundImage:
            "linear-gradient(var(--paper-sunk) 1px, transparent 1px), linear-gradient(90deg, var(--paper-sunk) 1px, transparent 1px)",
          backgroundSize: "20px 20px",
        }}
      >
        {!blueprint && (
          <div style={{ textAlign: "center", color: "var(--ink-faint)" }}>
            <div style={{ fontSize: 40, lineHeight: 1 }}>⌗</div>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: 12, marginTop: 10 }}>
              Upload a blueprint or pick a recent one to preview it here.
            </p>
          </div>
        )}

        {isImage && (
          <img
            src={blueprint.file_path}
            alt={blueprint.title}
            style={{
              maxWidth: "100%",
              maxHeight: 520,
              objectFit: "contain",
              boxShadow: "0 1px 0 var(--line)",
            }}
          />
        )}

        {isPdf && (
          <iframe
            src={blueprint.file_path}
            title={blueprint.title}
            style={{ width: "100%", height: 520, border: "1px solid var(--line)" }}
          />
        )}

        {isModel && (
          <div style={{ textAlign: "center", maxWidth: 380 }}>
            <div style={{ fontSize: 40, color: "var(--accent)" }}>◱</div>
            <p style={{ fontFamily: "var(--font-mono)", fontSize: 12, marginTop: 10, color: "var(--ink-soft)" }}>
              3D model ready: <strong>{blueprint.original_name}</strong>
            </p>
            {/*
              ── INTEGRATION POINT ──────────────────────────────────────────
              Mount your existing 3D apartment viewer here. Pass the model URL:
                <ApartmentViewer src={blueprint.file_path} />
              e.g. with react-three-fiber + @react-three/drei:
                npm i three @react-three/fiber @react-three/drei
                const { scene } = useGLTF(blueprint.file_path)
              ───────────────────────────────────────────────────────────────
            */}
            <p style={{ fontFamily: "var(--font-mono)", fontSize: 11, marginTop: 14, color: "var(--ink-faint)" }}>
              Wire your Three.js viewer into components/Viewer.js
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
