"use client";

function timeAgo(iso) {
  // SQLite datetime('now') is UTC without a "Z" — mark it so JS parses correctly.
  const then = new Date(iso.replace(" ", "T") + "Z").getTime();
  const s = Math.floor((Date.now() - then) / 1000);
  if (s < 60) return "just now";
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

export default function RecentPanel({ items, activeId, onSelect, loading }) {
  return (
    <aside
      className="card"
      style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid var(--line)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <span className="eyebrow" style={{ margin: 0 }}>
          Recent
        </span>
        <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--ink-faint)" }}>
          {items?.length || 0}
        </span>
      </div>

      <div style={{ maxHeight: 480, overflowY: "auto" }}>
        {loading && (
          <p style={{ padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--ink-faint)" }}>
            Loading…
          </p>
        )}

        {!loading && (!items || items.length === 0) && (
          <p style={{ padding: 16, fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--ink-faint)" }}>
            No blueprints yet. Your uploads will appear here.
          </p>
        )}

        {!loading &&
          items?.map((b) => {
            const active = b.id === activeId;
            return (
              <button
                key={b.id}
                onClick={() => onSelect(b)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  border: "none",
                  borderBottom: "1px solid var(--line)",
                  borderLeft: `3px solid ${active ? "var(--accent)" : "transparent"}`,
                  background: active ? "var(--accent-wash)" : "transparent",
                  padding: "12px 16px",
                  cursor: "pointer",
                  display: "block",
                }}
              >
                <div
                  style={{
                    fontSize: 14,
                    fontWeight: 500,
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {b.title}
                </div>
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: "var(--ink-faint)",
                    marginTop: 3,
                  }}
                >
                  {timeAgo(b.created_at)}
                </div>
              </button>
            );
          })}
      </div>
    </aside>
  );
}
